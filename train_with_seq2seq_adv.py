#!/usr/bin/env/python3
"""Recipe for training a sequence-to-sequence ASR system with librispeech.
The system employs an encoder, a decoder, and an attention mechanism
between them. Decoding is performed with beamsearch coupled with a neural
language model.
To run this recipe, do the following:
> python train.py hparams/train_BPE1000.yaml
With the default hyperparameters, the system employs a CRDNN encoder.
The decoder is based on a standard  GRU. Beamsearch coupled with a RNN
language model is used  on the top of decoder probabilities.
The neural network is trained on both CTC and negative-log likelihood
targets and sub-word units estimated with Byte Pairwise Encoding (BPE)
are used as basic recognition tokens. Training is performed on the full
LibriSpeech dataset (960 h).
The experiment file is flexible enough to support a large variety of
different systems. By properly changing the parameter files, you can try
different encoders, decoders, tokens (e.g, characters instead of BPE),
training split (e.g, train-clean 100 rather than the full one), and many
other possible variations.
This recipe assumes that the tokenizer and the LM are already trained.
To avoid token mismatches, the tokenizer used for the acoustic model is
the same use for the LM.  The recipe downloads the pre-trained tokenizer
and LM.
If you would like to train a full system from scratch do the following:
1- Train a tokenizer (see ../../Tokenizer)
2- Train a language model (see ../../LM)
3- Train the acoustic model (with this code).
Authors
 * Ju-Chieh Chou 2020
 * Mirco Ravanelli 2020
 * Abdel Heba 2020
 * Peter Plantinga 2020
 * Samuele Cornell 2020
 * Andreas Nautsch 2021
"""

import sys
from pathlib import Path

import torch
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.utils.distributed import if_main_process, run_on_main
from speechbrain.utils.logger import get_logger

from enum import Enum, auto
import numpy as np
from torch.utils.data import DataLoader
from speechbrain.dataio.dataloader import LoopedLoader
from tqdm.contrib import tqdm
import torch.nn as nn
import torchaudio
import csv
import os
import pandas as pd
from typing import List, Optional, Tuple
from scipy.signal import argrelextrema
from speechbrain.utils.autocast import AMPConfig, TorchAutocast
import re

logger = get_logger(__name__)


class Stage(Enum):
    """Simple enum to track stage of experiments."""
    ATTACK = auto()

# Define training procedure
class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        # Variables affected by precision:
        # x = self.modules.enc(feats)
        # h, _ = self.modules.dec(e_in, x, wav_lens)
        # logits = self.modules.seq_lin(h)
        # logits = self.modules.ctc_lin(x)
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_bos, _ = batch.tokens_bos
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        # print("wavs: ", wavs, wavs.dtype, torch.is_autocast_enabled())
        # Add waveform augmentation if specified.
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            wavs, wav_lens = self.hparams.wav_augment(wavs, wav_lens)
            tokens_bos = self.hparams.wav_augment.replicate_labels(tokens_bos)

        # Forward pass
        feats = self.hparams.compute_features(wavs)
        # print("feats 1: ", feats.shape, feats, feats.dtype, torch.is_autocast_enabled())
        # with self.eval_precision:
        feats = self.modules.normalize(feats, wav_lens)
        '''
        print(
            "glob_mean:",
            self.modules.normalize.glob_mean,
            self.modules.normalize.glob_mean.dtype,
            self.modules.normalize.glob_mean.device,
        )
        '''
        # print("feats 2: ", feats.shape, feats, feats.dtype, torch.is_autocast_enabled())
        if(stage == Stage.ATTACK):
            x = self.modules.enc(feats)
        else:
            x = self.modules.enc(feats.detach())
        # print("x: ", x, x.dtype, torch.is_autocast_enabled())
        e_in = self.modules.emb(tokens_bos)  # y_in bos + tokens
        h, _ = self.modules.dec(e_in, x, wav_lens)
        # print("e_in: ", e_in, e_in.dtype, torch.is_autocast_enabled())
        # print("h: ", h, h.dtype, torch.is_autocast_enabled())
        # Output layer for seq2seq log-probabilities
        logits = self.modules.seq_lin(h)
        p_seq = self.hparams.log_softmax(logits)
        # print("logits: ", logits, logits.dtype, torch.is_autocast_enabled())
        # print("p_seq: ", p_seq, p_seq.dtype, torch.is_autocast_enabled())
        # Compute outputs
        p_ctc, p_tokens = None, None
        if stage == sb.Stage.TRAIN:
            current_epoch = self.hparams.epoch_counter.current
            if current_epoch <= self.hparams.number_of_ctc_epochs:
                # Output layer for ctc log-probabilities
                logits = self.modules.ctc_lin(x)
                p_ctc = self.hparams.log_softmax(logits)
                
        elif stage == Stage.ATTACK:
            logits = self.modules.ctc_lin(x)
            # print("logits: ", logits, logits.dtype, torch.is_autocast_enabled())
            p_ctc = self.hparams.log_softmax(logits)
            # print("p_ctc: ", p_ctc, p_ctc.dtype, torch.is_autocast_enabled())
        else:
            if stage == sb.Stage.VALID:
                # Get token strings from index prediction
                p_tokens, _, _, _ = self.hparams.valid_search(x, wav_lens)
            else:
                p_tokens, _, _, _ = self.hparams.test_search(x, wav_lens)
        return p_ctc, p_seq, wav_lens, p_tokens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        current_epoch = self.hparams.epoch_counter.current
        p_ctc, p_seq, wav_lens, predicted_tokens = predictions

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens

        # Labels must be extended if parallel augmentation or concatenated
        # augmentation was performed on the input (increasing the time dimension)
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            (
                tokens,
                tokens_lens,
                tokens_eos,
                tokens_eos_lens,
            ) = self.hparams.wav_augment.replicate_multiple_labels(
                tokens, tokens_lens, tokens_eos, tokens_eos_lens
            )

        loss_seq = self.hparams.seq_cost(
            p_seq, tokens_eos, length=tokens_eos_lens
        )
        # print("loss_seq: ", loss_seq)
        # Add ctc loss if necessary
        if (
            stage == sb.Stage.TRAIN
            and current_epoch <= self.hparams.number_of_ctc_epochs
        ):
            loss_ctc = self.hparams.ctc_cost(
                p_ctc, tokens, wav_lens, tokens_lens
            )
            loss = self.hparams.ctc_weight * loss_ctc
            loss += (1 - self.hparams.ctc_weight) * loss_seq
        elif stage == Stage.ATTACK:
            loss_ctc = self.hparams.ctc_cost(
                p_ctc, tokens, wav_lens, tokens_lens
            )
            # print("loss_ctc: ", loss_ctc)
            loss = self.hparams.ctc_weight * loss_ctc
            loss += (1 - self.hparams.ctc_weight) * loss_seq
        else:
            loss = loss_seq

        if stage != sb.Stage.TRAIN and stage !=Stage.ATTACK:
            # Decode token terms to words
            predicted_words = [
                self.tokenizer.decode_ids(utt_seq).split(" ")
                for utt_seq in predicted_tokens
            ]
            target_words = [wrd.split(" ") for wrd in batch.wrd]
            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)

        return loss

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(stage_stats["WER"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"]},
                min_keys=["WER"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            if if_main_process():
                with open(
                    self.hparams.test_wer_file, "w", encoding="utf-8"
                ) as w:
                    self.wer_metric.write_stats(w)

    def psy_initialize_vars(self, hparams):
        self.eps = 1  # 0.05
        self.max_iter_1 = 4000  # 4000 # 10
        self.learning_rate_1 = 0.000005 # 5e-4 
        self.global_max_length = 562480  # Need to check max length file!
        self.initial_rescale = 1.0
        self.decrease_factor_eps = 0.5  # 0.8
        self.num_iter_decrease_eps = 1  # 10  # In the code is 1, check it!, with one it checks every time
        self.clip_min = None
        self.clip_max = None
        self.const = 10  # 1.0
        self.targeted = True
        self.optimizer = None
        self.alpha = 1.2 # 2
        self._optimizer_arg_1 = None
        self.increase_factor_alpha = 2.0 # 1.2
        self.num_iter_increase_alpha = 20 
        self.decrease_factor_alpha = 0.5 # 0.8
        self.num_iter_decrease_alpha = 50 # 20
        self.win_length = hparams["win_length"]
        self.hop_length = hparams["hop_length"]
        self.n_fft_psy = hparams["n_fft_psy"]

    def attack_psy(
            self,
            train_set,
            max_key=None,
            min_key=None,
            hparams=None,
            progressbar=None,
            train_loader_kwargs={},
    ):

        if progressbar is None:
            progressbar = not self.noprogressbar

        if not (
                isinstance(train_set, DataLoader)
                or isinstance(train_set, LoopedLoader)
        ):
            train_loader_kwargs["ckpt_prefix"] = None
            train_set = self.make_dataloader(
                train_set, stage=sb.Stage.TEST, **train_loader_kwargs
            )
        self.on_evaluate_start(max_key=max_key, min_key=min_key)
        self.on_stage_start(sb.Stage.TEST, epoch=None)
        self.modules.eval()

        self.sample_rate = hparams["sample_rate"]

        for m in self.modules.modules():
            if m.__class__.__name__.startswith('LSTM'):
                m.train()
            if isinstance(m, nn.Dropout):
                m.p = 0
            elif isinstance(m, nn.LSTM):
                m.dropout = 0
            elif isinstance(m, nn.GRU):
                m.dropout = 0
    
        for batch in tqdm(train_set, dynamic_ncols=True, disable=not progressbar):
            self.psy_initialize_vars(hparams)
            batch = batch.to(self.device)
            # First reset delta
            global_optimal_delta = torch.zeros(batch.batchsize, self.global_max_length).to(self.device)
            self.global_optimal_delta = nn.Parameter(global_optimal_delta)
            # Next, reset optimizers
            if self._optimizer_arg_1 is None:
                self.optimizer_1 = torch.optim.Adam(
                    params=[self.global_optimal_delta], lr=self.learning_rate_1
                )
            else:
                self.optimizer_1 = self._optimizer_arg_1(  # type: ignore
                    params=[self.global_optimal_delta], lr=self.learning_rate_1
                )
            # Then calculate the adversarial sample
            original_input = torch.clone(batch.sig[0])
            theta_batch = []
            original_max_psd_batch = []
            # wav_init = batch.sig[0]
            lengths = (batch.sig[0].size(1) * batch.sig[1]).long()
            wavs = [batch.sig[0][i, : lengths[i]] for i in range(batch.batchsize)]
            for i, dirct in enumerate(batch.adv_path):
                root_path = hparams["path_adv"]
                os.makedirs(root_path, exist_ok=True)
                file_name = os.path.basename(dirct)
                save_dirct = os.path.join(root_path, file_name)
                save_dirct = save_dirct.replace(".flac", ".wav")
                # print("save_dirct: ", save_dirct)
                if (not os.path.exists(save_dirct)):
                    # Compute original masking threshold and maximum psd (power spectral density)
                    theta, original_max_psd = None, None
                    theta, original_max_psd = self.compute_masking_threshold(wavs[i])
                    theta = theta.transpose(1, 0)
                    theta_batch.append(theta)
                    theta = theta.to(self.device)
                    original_max_psd_batch.append(original_max_psd)
                    # Reset delta with new result
                    local_batch_shape = batch.sig[0].shape
                    self.global_optimal_delta.data = torch.zeros(batch.batchsize, self.global_max_length).to(
                        self.device)
                    self.global_optimal_delta.data[: local_batch_shape[0],
                    : local_batch_shape[1]] = batch.delta[i]
                    # Second stage of attack
                    self.attack_2nd_stage(batch, hparams, save_dirct, theta_batch=theta_batch, original_max_psd_batch=original_max_psd_batch)

    def attack_2nd_stage(self, batch, hparams, save_dirct, theta_batch: List[np.ndarray], original_max_psd_batch: List[np.ndarray]):
        # Compute local shape
        local_batch_size = batch.batchsize
        real_lengths = ((batch.sig[1] * batch.sig[0].size(1)).long().detach().cpu().numpy())
        local_max_length = np.max(real_lengths)
        rescale = (
                np.ones([local_batch_size, local_max_length], dtype=np.float32)
                * self.initial_rescale
        )
        # Reformat input
        input_mask = np.zeros([local_batch_size, local_max_length], dtype=np.float32)
        original_input = torch.clone(batch.sig[0])
        for local_batch_size_idx in range(local_batch_size):
            input_mask[local_batch_size_idx, : real_lengths[local_batch_size_idx]] = 1
        # Optimization loop
        almost_successful: List[Optional["torch.Tensor"]] = [None] * local_batch_size
        successful_adv_input_2: List[Optional["torch.Tensor"]] = [None] * local_batch_size
        first_hit: List[Optional["torch.Tensor"]] = [None] * local_batch_size
        best_hit: List[Optional["torch.Tensor"]] = [None] * local_batch_size
        count_succs: List[Optional["torch.Tensor"]] = [None] * local_batch_size
        token_lenghts = (batch.tokens[1] * batch.tokens[0].size(1)).long().detach().cpu().numpy()
        best_alpha: List[Optional["torch.Tensor"]] = [None] * local_batch_size
        # Optimization loop
        best_loss_2nd_stage = [np.inf] * local_batch_size
        # Initialize alpha and rescale
        alpha = np.array([self.alpha] * local_batch_size, dtype=np.float32)
        for iter_2nd_stage_idx in range(self.max_iter_1):
            # Zero the parameter gradients
            self.optimizer_1.zero_grad()
            # Call to forward pass of the first stage
            (
                loss_1st_stage,
                _,
                masked_adv_input,
                local_delta_rescale,
            ) = self.forward_1st_stage(
                original_input=original_input,
                batch=batch,
                local_batch_size=local_batch_size,
                local_max_length=local_max_length,
                rescale=rescale,
                input_mask=input_mask,
                real_lengths=real_lengths,
            )
            # Call to forward pass of the first stage
            loss_2nd_stage = self.forward_2nd_stage(
                local_delta_rescale=local_delta_rescale,
                theta_batch=theta_batch,
                original_max_psd_batch=original_max_psd_batch,
                real_lengths=real_lengths,
            )
            # Total loss
            loss = (
                    loss_1st_stage.type(torch.float32)
                    + torch.tensor(alpha).to(self.device) * loss_2nd_stage
            )
            loss = torch.mean(loss)
            loss.backward()
            # Do optimization
            self.optimizer_1.step()
            # Save the best adversarial example and adjust the alpha coefficient
            for local_batch_size_idx in range(local_batch_size):
                almost_successful[local_batch_size_idx] = masked_adv_input[local_batch_size_idx]
                torchaudio.save("adv_ex_0.wav", almost_successful[local_batch_size_idx][:real_lengths[local_batch_size_idx]].cpu()[None, :], self.sample_rate)
                data_adv, sample_rate = torchaudio.load("adv_ex_0.wav")
                batch.sig[0][local_batch_size_idx][:real_lengths[local_batch_size_idx]] = data_adv

            with torch.no_grad():
                # print("Test time! ")
                p_ctc, p_seq, wav_lens, p_tokens = self.compute_forward(batch, stage=sb.Stage.TEST)

            for local_batch_size_idx in range(local_batch_size):
                if p_tokens == batch.tokens.data.cpu().tolist():   
                    if (loss_2nd_stage[local_batch_size_idx] < best_loss_2nd_stage[local_batch_size_idx]):
                        # Update best loss at 2nd stage
                        best_loss_2nd_stage[local_batch_size_idx] = loss_2nd_stage[local_batch_size_idx]
                        # Save the best adversarial example
                        if successful_adv_input_2[local_batch_size_idx] is None:
                            first_hit[local_batch_size_idx] = iter_2nd_stage_idx
                        successful_adv_input_2[local_batch_size_idx] = masked_adv_input[local_batch_size_idx]
                        best_hit[local_batch_size_idx] = iter_2nd_stage_idx
                        best_alpha[local_batch_size_idx] = alpha[local_batch_size_idx]
                        if count_succs[local_batch_size_idx] is None:
                            count_succs[local_batch_size_idx] = 1
                        else:
                            count_succs[local_batch_size_idx] += 1
                        # Adjust to increase the alpha coefficient
                        if iter_2nd_stage_idx % self.num_iter_increase_alpha == 0:                 
                            alpha[local_batch_size_idx] *= self.increase_factor_alpha

                elif iter_2nd_stage_idx % self.num_iter_decrease_alpha == 0:
                    alpha[local_batch_size_idx] *= self.decrease_factor_alpha
                    alpha[local_batch_size_idx] = max(
                        alpha[local_batch_size_idx], 0.0005
                    )
            # If attack is unsuccessful
            if iter_2nd_stage_idx == self.max_iter_1 - 1 :
                for (local_batch_size_idx, dirct) in enumerate(batch.src_path):
                    if successful_adv_input_2[local_batch_size_idx] is None:
                        successful_adv_input_2[local_batch_size_idx] = masked_adv_input[local_batch_size_idx].detach()
                        with open(hparams["unsuccesfull_adv"], 'a') as myfile:
                            wr = csv.writer(myfile)                            
                            wr.writerow([[dirct], [first_hit[local_batch_size_idx]], [best_hit[local_batch_size_idx]], 
                                [alpha[local_batch_size_idx]], [count_succs[local_batch_size_idx]]])
                            myfile.close()
                    else:
                        with open(hparams["succesfull_adv"], 'a') as myfile:
                            wr = csv.writer(myfile)
                            wr.writerow([[dirct], [first_hit[local_batch_size_idx]], [best_hit[local_batch_size_idx]], 
                                [alpha[local_batch_size_idx]], [count_succs[local_batch_size_idx]]])
                            myfile.close()
                    torchaudio.save(save_dirct, successful_adv_input_2[local_batch_size_idx][ :real_lengths[local_batch_size_idx]].cpu()[None, :], self.sample_rate)

        pass 
 
    def forward_2nd_stage(
            self,
            local_delta_rescale: "torch.Tensor",
            theta_batch: List[np.ndarray],
            original_max_psd_batch: List[np.ndarray],
            real_lengths: np.ndarray,
    ):
        # Compute loss for masking threshold
        losses = []
        relu = torch.nn.ReLU()
        for i, _ in enumerate(theta_batch):
            psd_transform_delta = self.psd_transform(
                delta=local_delta_rescale[i, : real_lengths[i]],
                original_max_psd=original_max_psd_batch[i],
            )
            loss = torch.mean(
                relu(psd_transform_delta - theta_batch[i].to(self.device))
            )
            losses.append(loss)
        losses_stack = torch.stack(losses)
        return losses_stack

    def compute_masking_threshold(
            self, wav: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the masking threshold and the maximum psd of the original audio.
        :param wav: Samples of shape (seq_length,).
        :return: A tuple of the masking threshold and the maximum psd.
        """
        # First compute the psd matrix
        # Get window for the transformation
        # window = scipy.signal.get_window("hann", self.win_length, fftbins=True)
        window = torch.hann_window(self.win_length, periodic=True)
        # Do transformation
        # transformed_wav = librosa.core.stft(
        #    y=x, n_fft=self.n_fft, hop_length=self.hop_length,
        #    win_length=self.win_length, window=window, center=False
        # )
        transformed_wav = torch.stft(
            input=wav.detach().cpu(),
            n_fft=self.n_fft_psy,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=False,
            return_complex=True,
        ).numpy()
        transformed_wav *= np.sqrt(8.0 / 3.0)

        psd = abs(transformed_wav / self.win_length)
        original_max_psd = np.max(psd * psd)
        with np.errstate(divide="ignore"):
            psd = (10 * np.log10(psd * psd)).clip(min=-200)
        psd = 96 - np.max(psd) + psd
        # Compute freqs and barks
        # freqs = librosa.core.fft_frequencies(
        #    sr=self.asr_brain.hparams.sample_rate, n_fft=self.n_fft)
        freqs = torch.fft.rfftfreq(n=self.n_fft_psy, d=1.0 / 16000)
        barks = 13 * np.arctan(0.00076 * freqs) + 3.5 * np.arctan(
            pow(freqs / 7500.0, 2)
        )
        # Compute quiet threshold
        ath = np.zeros(len(barks), dtype=np.float32) - np.inf
        bark_idx = np.argmax(barks > 1)
        ath[bark_idx:] = (
                3.64 * pow(freqs[bark_idx:] * 0.001, -0.8)
                - 6.5 * np.exp(-0.6 * pow(0.001 * freqs[bark_idx:] - 3.3, 2))
                + 0.001 * pow(0.001 * freqs[bark_idx:], 4)
                - 12
        )
        # Compute the global masking threshold theta
        theta = []
        for i in range(psd.shape[1]):
            # Compute masker index
            masker_idx = argrelextrema(psd[:, i], np.greater)[0]
            if 0 in masker_idx:
                masker_idx = np.delete(masker_idx, 0)
            if len(psd[:, i]) - 1 in masker_idx:
                masker_idx = np.delete(masker_idx, len(psd[:, i]) - 1)
            barks_psd = np.zeros([len(masker_idx), 3], dtype=np.float32)
            barks_psd[:, 0] = barks[masker_idx]
            barks_psd[:, 1] = 10 * np.log10(
                pow(10, psd[:, i][masker_idx - 1] / 10.0)
                + pow(10, psd[:, i][masker_idx] / 10.0)
                + pow(10, psd[:, i][masker_idx + 1] / 10.0)
            )
            barks_psd[:, 2] = masker_idx
            for j in range(len(masker_idx)):
                if barks_psd.shape[0] <= j + 1:
                    break
                while barks_psd[j + 1, 0] - barks_psd[j, 0] < 0.5:
                    quiet_threshold = (
                            3.64 * pow(freqs[int(barks_psd[j, 2])] * 0.001, -0.8)
                            - 6.5
                            * np.exp(
                        -0.6 * pow(0.001 * freqs[int(barks_psd[j, 2])] - 3.3, 2)
                    )
                            + 0.001 * pow(0.001 * freqs[int(barks_psd[j, 2])], 4)
                            - 12
                    )
                    if barks_psd[j, 1] < quiet_threshold:
                        barks_psd = np.delete(barks_psd, j, axis=0)
                    if barks_psd.shape[0] == j + 1:
                        break
                    if barks_psd[j, 1] < barks_psd[j + 1, 1]:
                        barks_psd = np.delete(barks_psd, j, axis=0)
                    else:
                        barks_psd = np.delete(barks_psd, j + 1, axis=0)
                    if barks_psd.shape[0] == j + 1:
                        break
            # Compute the global masking threshold
            delta = 1 * (-6.025 - 0.275 * barks_psd[:, 0])
            t_s = []
            for psd_id in range(barks_psd.shape[0]):
                d_z = barks - barks_psd[psd_id, 0]
                zero_idx = np.argmax(d_z > 0)
                s_f = np.zeros(len(d_z), dtype=np.float32)
                s_f[:zero_idx] = 27 * d_z[:zero_idx]
                s_f[zero_idx:] = (-27 + 0.37 * max(barks_psd[psd_id, 1] - 40, 0)) * d_z[
                                                                                    zero_idx:
                                                                                    ]
                t_s.append(barks_psd[psd_id, 1] + delta[psd_id] + s_f)
            t_s_array = np.array(t_s)
            theta.append(
                np.sum(pow(10, t_s_array / 10.0), axis=0) + pow(10, ath / 10.0)
            )
        theta = np.array(theta)
        return torch.tensor(theta).to(self.device), original_max_psd

    def psd_transform(
            self, delta: "torch.Tensor", original_max_psd: "torch.Tensor"
    ) -> "torch.Tensor":
        """
        Compute the psd matrix of the perturbation.
        :param delta: The perturbation.
        :param original_max_psd: The maximum psd of the original audio.
        :return: The psd matrix.
        """
        # Get window for the transformation
        window_fn = torch.hann_window

        delta_stft = torch.stft(
            delta,
            n_fft=self.n_fft_psy,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=False,
            window=window_fn(self.win_length).to(self.device),
            return_complex=False
        ).to(self.device)
        # print(delta_stft, delta_stft.shape)
        delta_stft *= np.sqrt(8.0 / 3.0)
        psd = torch.sqrt(torch.sum(torch.square(delta_stft / self.win_length), -1))
        # Return STFT of delta
        # Take abs of complex STFT results
        # Compute the psd matrix
        psd = psd ** 2
        psd = (
                torch.pow(
                    torch.tensor(10.0).type(torch.float32),
                    torch.tensor(9.6).type(torch.float32),
                ).to(self.device)
                / torch.reshape(
            torch.tensor(original_max_psd).to(self.device), [-1, 1, 1]
        )
                * psd.type(torch.float32)
        )

        return psd

    def CW_initialize_vars(self):
        self.eps = 1  # 0.05
        self.max_iter_1 = 4000 # 4000  # 4000 # 10
        self.learning_rate_1 = 0.002  # 0.001
        self.global_max_length = 562480  # Need to check max length file!
        self.initial_rescale = 1.0
        self.decrease_factor_eps = 0.5  # 0.8
        self.num_iter_decrease_eps = 1  # 10  # In the code is 1, check it!, with one it checks every time
        self.clip_min = None
        self.clip_max = None
        self.const = 10  # 1.0
        self.targeted = True
        self.optimizer = None
        self.apha = 2
        self._optimizer_arg_1 = None

    def attack_CW(
            self,
            train_set,
            max_key=None,
            min_key=None,            
            hparams=None,
            progressbar=None,
            train_loader_kwargs={},
    ):
       
        if progressbar is None:
            progressbar = not self.noprogressbar

        if not (
                isinstance(train_set, DataLoader)
                or isinstance(train_set, LoopedLoader)
        ):
            train_loader_kwargs["ckpt_prefix"] = None
            train_set = self.make_dataloader(
                train_set, stage=sb.Stage.TEST, **train_loader_kwargs
            )
        self.on_evaluate_start(max_key=max_key, min_key=min_key)
        self.on_stage_start(sb.Stage.TEST, epoch=None)
        self.modules.eval()

        self.sample_rate = hparams["sample_rate"]

        for m in self.modules.modules():
            if m.__class__.__name__.startswith('LSTM'):
                m.train()
            if isinstance(m, nn.Dropout):
                m.p = 0
            elif isinstance(m, nn.LSTM):
                m.dropout = 0
            elif isinstance(m, nn.GRU):
                m.dropout = 0
    
        # cnt = 0
        for batch in tqdm(train_set, dynamic_ncols=True, disable=not progressbar):
            # for batch in train_set:
            self.CW_initialize_vars()
            batch = batch.to(self.device)
            # First reset delta
            global_optimal_delta = torch.zeros(batch.batchsize, self.global_max_length).to(self.device)
            self.global_optimal_delta = nn.Parameter(global_optimal_delta)
            # Next, reset optimizers
            if self._optimizer_arg_1 is None:
                self.optimizer_1 = torch.optim.Adam(
                    params=[self.global_optimal_delta], lr=self.learning_rate_1
                )
            else:
                self.optimizer_1 = self._optimizer_arg_1(  # type: ignore
                    params=[self.global_optimal_delta], lr=self.learning_rate_1
                )
            # Then calculate the adversarial sample
            original_input = torch.clone(batch.sig[0]) 
            
            for i in batch.path:
                root_path = hparams["path_adv"]
                os.makedirs(root_path, exist_ok=True)
                file_name = os.path.basename(i)
                save_dirct = os.path.join(root_path, file_name)
                # cnt += 1 
                save_dirct = save_dirct.replace(".flac", ".wav")

                if (not os.path.exists(save_dirct)):                    
                    # print(save_dirct)
                    # print(batch.path)
                    result = self.attack_1st_stage(batch, hparams, save_dirct)
                    # print(asfasf)
            # if cnt > 20:
            #   break
    
    def attack_1st_stage(self, batch, hparams, save_dirct):
        """
        The first stage of the attack.
        """
        # print("batch: ", batch.sig[0].shape, batch.sig[1].shape, batch.path)
        # Compute local shape
        local_batch_size = batch.batchsize
        real_lengths = (
            (batch.sig[1] * batch.sig[0].size(1)).long().detach().cpu().numpy()
        )
        local_max_length = np.max(real_lengths)
        # Initialize rescale
        rescale = (
                np.ones([local_batch_size, local_max_length], dtype=np.float32)
                * self.initial_rescale
        )
        # Reformat input
        input_mask = np.zeros([local_batch_size, local_max_length], dtype=np.float32)
        original_input = torch.clone(batch.sig[0])

        for local_batch_size_idx in range(local_batch_size):
            input_mask[local_batch_size_idx, : real_lengths[local_batch_size_idx]] = 1
        # Optimization loop
        almost_successful: List[Optional["torch.Tensor"]] = [None] * local_batch_size
        successful_adv_input_2: List[Optional["torch.Tensor"]] = [None] * local_batch_size
        first_hit: List[Optional["torch.Tensor"]] = [None] * local_batch_size
        best_hit: List[Optional["torch.Tensor"]] = [None] * local_batch_size
        best_eta: List[Optional["torch.Tensor"]] = [None] * local_batch_size
        trans = [None] * local_batch_size
        token_lenghts = (batch.tokens[1] * batch.tokens[0].size(1)).long().detach().cpu().numpy()
        count_succs: List[Optional["torch.Tensor"]] = [None] * local_batch_size
        for iter_1st_stage_idx in range(self.max_iter_1):
            # Zero the parameter gradients
            self.optimizer_1.zero_grad()
            # Call to forward pass
            (
                loss,
                local_delta,
                # decoded_output,
                masked_adv_input,
                _,
            ) = self.forward_1st_stage(
                original_input=original_input,
                batch=batch,
                local_batch_size=local_batch_size,
                local_max_length=local_max_length,
                rescale=rescale,
                input_mask=input_mask,
                real_lengths=real_lengths,
            )
            # print(loss)
            loss.backward()
            # Get sign of the gradients
            # print("global_optimal_delta: ", self.global_optimal_delta.grad)
            self.global_optimal_delta.grad = torch.sign(self.global_optimal_delta.grad)
            # Do optimization
            self.optimizer_1.step()

            for local_batch_size_idx in range(local_batch_size):
                almost_successful[local_batch_size_idx] = masked_adv_input[local_batch_size_idx].detach()
                # print("almost_successful :", almost_successful[local_batch_size_idx].shape, real_lengths[local_batch_size_idx])
                torchaudio.save("adv_ex_2.wav", almost_successful[local_batch_size_idx][ :real_lengths[local_batch_size_idx]].cpu()[None, :], self.sample_rate)
                data_adv, sample_rate = torchaudio.load("adv_ex_2.wav")
                # print(batch.sig[0].shape, batch.sig[1].shape, data_adv.shape)
                batch.sig[0][local_batch_size_idx][:real_lengths[local_batch_size_idx]] = data_adv
            # print("before forward ", batch.sig)
            # print("data_adv ", data_adv)
            with torch.no_grad():
                # print("Test time! ")
                p_ctc, p_seq, wav_lens, p_tokens = self.compute_forward(batch, stage=sb.Stage.TEST)
            # print("loss: ", loss)
            # print(best_hyps)
            for local_batch_size_idx in range(local_batch_size):
                if p_tokens == batch.tokens.data.cpu().tolist():   
                    # print(loss)
                    # print(p_tokens)
                    # print(batch.tokens.data.cpu().tolist())
                    best_eta[local_batch_size_idx] = rescale[local_batch_size_idx] * self.eps
                    # Adjust the rescale coefficient
                    max_local_delta = np.max(
                        np.abs(local_delta[local_batch_size_idx].detach().cpu().numpy())
                    )
                    if (rescale[local_batch_size_idx][0] * self.eps > max_local_delta):
                        rescale[local_batch_size_idx] = max_local_delta / self.eps
                    rescale[local_batch_size_idx] *= self.decrease_factor_eps
                    # Save the best adversarial example
                    if successful_adv_input_2[local_batch_size_idx] is None:
                        first_hit[local_batch_size_idx] = iter_1st_stage_idx
                    # masked_adv_input[local_batch_size_idx] = batch.sig[0][local_batch_size_idx][:real_lengths[local_batch_size_idx]]
                    successful_adv_input_2[local_batch_size_idx] = masked_adv_input[local_batch_size_idx].detach()
                    best_hit[local_batch_size_idx] = iter_1st_stage_idx
                    if count_succs[local_batch_size_idx] is None:
                        count_succs[local_batch_size_idx] = 1
                    else:
                        count_succs[local_batch_size_idx] += 1
            
            # If attack is unsuccessful
            if iter_1st_stage_idx == self.max_iter_1 - 1 :
                for (local_batch_size_idx, dirct) in enumerate(batch.path):
                    if successful_adv_input_2[local_batch_size_idx] is None:
                        successful_adv_input_2[local_batch_size_idx] = masked_adv_input[local_batch_size_idx].detach()
                        with open(hparams["unsuccesfull_adv"], 'a') as myfile:
                            wr = csv.writer(myfile)                            
                            wr.writerow([[dirct], [first_hit[local_batch_size_idx]], [best_hit[local_batch_size_idx]], 
                                [best_eta[local_batch_size_idx]], [count_succs[local_batch_size_idx]]])
                            myfile.close()
                    else:
                        with open(hparams["succesfull_adv"], 'a') as myfile:
                            wr = csv.writer(myfile)
                            wr.writerow([[dirct], [first_hit[local_batch_size_idx]], [best_hit[local_batch_size_idx]], 
                                [best_eta[local_batch_size_idx][0]], [count_succs[local_batch_size_idx]]])
                            myfile.close()
                    torchaudio.save(save_dirct, successful_adv_input_2[local_batch_size_idx][ :real_lengths[local_batch_size_idx]].cpu()[None, :], self.sample_rate)
            
        # return predictions
        result = torch.stack(successful_adv_input_2)  # type: ignore
        batch.sig = original_input, batch.sig[1]
        return result
 
    def forward_1st_stage(
            self,
            original_input: np.ndarray,
            batch: sb.dataio.batch.PaddedBatch,
            local_batch_size: int,
            local_max_length: int,
            rescale: np.ndarray,
            input_mask: np.ndarray,
            real_lengths: np.ndarray,
    ):

        # Compute perturbed inputs
        local_delta = self.global_optimal_delta[:local_batch_size, :local_max_length]
        # print("local_delta: ", local_delta)
        local_delta_rescale = torch.clamp(local_delta, -self.eps, self.eps).to(
            self.device
        )
        # print("local_delta_rescale: ", local_delta_rescale)
        local_delta_rescale *= torch.tensor(rescale).to(self.device)
        # print("local_delta_rescale: ", local_delta_rescale)
        adv_input = local_delta_rescale + torch.tensor(original_input).to(self.device)
        # print("adv_input: ", adv_input)
        masked_adv_input = adv_input * torch.tensor(input_mask).to(self.device)
        # print("masked_adv_input: ", masked_adv_input)
        # Compute loss and decoded output
        batch.sig = masked_adv_input, batch.sig[1]
        # print("batch.sig: ", batch.sig)
        predictions = self.compute_forward(batch, Stage.ATTACK)
        # print("predictions: ", predictions)
        loss = self.compute_objectives(predictions, batch, Stage.ATTACK)
        # print("loss: ", loss)
        loss = self.const * loss + torch.norm(local_delta_rescale)
        return loss, local_delta, masked_adv_input, local_delta_rescale

    def evaluate_edit(
        self,
        test_set,
        max_key=None,
        min_key=None,
        progressbar=None,
        test_loader_kwargs={},
    ):
        """Iterate test_set and evaluate brain performance. By default, loads
        the best-performing checkpoint (as recorded using the checkpointer).

        Arguments
        ---------
        test_set : Dataset, DataLoader
            If a DataLoader is given, it is iterated directly. Otherwise passed
            to ``self.make_dataloader()``.
        max_key : str
            Key to use for finding best checkpoint, passed to
            ``on_evaluate_start()``.
        min_key : str
            Key to use for finding best checkpoint, passed to
            ``on_evaluate_start()``.
        progressbar : bool
            Whether to display the progress in a progressbar.
        test_loader_kwargs : dict
            Kwargs passed to ``make_dataloader()`` if ``test_set`` is not a
            DataLoader. NOTE: ``loader_kwargs["ckpt_prefix"]`` gets
            automatically overwritten to ``None`` (so that the test DataLoader
            is not added to the checkpointer).

        Returns
        -------
        average test loss
        """
        if progressbar is None:
            progressbar = not self.noprogressbar

        # Only show progressbar if requested and main_process
        enable = progressbar and sb.utils.distributed.if_main_process()

        if not (
            isinstance(test_set, DataLoader)
            or isinstance(test_set, LoopedLoader)
        ):
            test_loader_kwargs["ckpt_prefix"] = None
            test_set = self.make_dataloader(
                test_set, sb.Stage.TEST, **test_loader_kwargs
            )
        self.on_evaluate_start(max_key=max_key, min_key=min_key)
        self.on_stage_start(sb.Stage.TEST, epoch=None)
        self.modules.eval()
        avg_test_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(
                test_set,
                dynamic_ncols=True,
                disable=not enable,
                colour=self.tqdm_barcolor["test"],
            ):
                self.step += 1
                loss = self.evaluate_batch_edit(batch, stage=sb.Stage.TEST)
                avg_test_loss = self.update_average(loss, avg_test_loss)

                # Debug mode only runs a few batches
                if self.debug and self.step == self.debug_batches:
                    break

            self.on_stage_end(sb.Stage.TEST, avg_test_loss, None)
        self.step = 0
        return avg_test_loss

    def evaluate_batch_edit(self, batch, stage):
        """Evaluate one batch, override for different procedure than train.

        The default implementation depends on two methods being defined
        with a particular behavior:

        * ``compute_forward()``
        * ``compute_objectives()``

        Arguments
        ---------
        batch : list of torch.Tensors
            Batch of data to use for evaluation. Default implementation assumes
            this batch has two elements: inputs and targets.
        stage : Stage
            The stage of the experiment: Stage.VALID, Stage.TEST

        Returns
        -------
        detached loss
        """
        precisions = ["fp32", "fp16", "bf16"]
        idx = torch.randint(0, 3, ()).item()
        rand_precision = precisions[idx]
        # print(rand_precision)
        # rand_precision = precisions[torch.randint(0, len(precisions), (1,)).item()]
        eval_dtype = AMPConfig.from_name(rand_precision).dtype
        self.evaluation_ctx = TorchAutocast(
            device_type=self.device, dtype=eval_dtype
        )
        with self.evaluation_ctx:
            out = self.compute_forward(batch, stage=stage)
            loss = self.compute_objectives(out, batch, stage=stage)
        return loss.detach().cpu()

def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"],
        replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"],
        replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    # test is separate
    test_datasets = {}
    for csv_file in hparams["test_csv"]:
        name = Path(csv_file).stem
        test_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=csv_file, replacements={"data_root": data_folder}
        )
        test_datasets[name] = test_datasets[name].filtered_sorted(
            sort_key="duration"
        )

    datasets = [train_data, valid_data] + [i for k, i in test_datasets.items()]

    # We get the tokenizer as we need it to encode the labels when creating
    # mini-batches.
    tokenizer = hparams["tokenizer"]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
        tokens_list = tokenizer.encode_as_ids(wrd)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", "wrd", "tokens_bos", "tokens_eos", "tokens"],
    )
    train_batch_sampler = None
    valid_batch_sampler = None
    if hparams["dynamic_batching"]:
        from speechbrain.dataio.batch import PaddedBatch  # noqa
        from speechbrain.dataio.dataloader import SaveableDataLoader  # noqa
        from speechbrain.dataio.sampler import DynamicBatchSampler  # noqa

        dynamic_hparams = hparams["dynamic_batch_sampler"]
        hop_size = hparams["feats_hop_size"]

        train_batch_sampler = DynamicBatchSampler(
            train_data,
            length_func=lambda x: x["duration"] * (1 / hop_size),
            **dynamic_hparams,
        )

        valid_batch_sampler = DynamicBatchSampler(
            valid_data,
            length_func=lambda x: x["duration"] * (1 / hop_size),
            **dynamic_hparams,
        )

    return (
        train_data,
        valid_data,
        test_datasets,
        train_batch_sampler,
        valid_batch_sampler,
    )

def dataio_prepare_2(hparams, file_path):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=file_path,
        replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    datasets = [train_data]

    # We get the tokenizer as we need it to encode the labels when creating
    # mini-batches.
    tokenizer = hparams["tokenizer"]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig", "path")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        yield sig
        yield wav

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
        tokens_list = tokenizer.encode_as_ids(wrd)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", "wrd", "tokens_bos", "tokens_eos", "tokens", "path"],
    )
    train_batch_sampler = None

    if hparams["dynamic_batching"]:
        from speechbrain.dataio.batch import PaddedBatch  # noqa
        from speechbrain.dataio.dataloader import SaveableDataLoader  # noqa
        from speechbrain.dataio.sampler import DynamicBatchSampler  # noqa

        dynamic_hparams = hparams["dynamic_batch_sampler"]
        hop_size = hparams["feats_hop_size"]

        train_batch_sampler = DynamicBatchSampler(
            train_data,
            length_func=lambda x: x["duration"] * (1 / hop_size),
            **dynamic_hparams,
        )

    return (
        train_data,
        train_batch_sampler,
    )

def dataio_prepare_psy(hparams, file_path):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=file_path,
        replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    datasets = [train_data]

    # We get the tokenizer as we need it to encode the labels when creating
    # mini-batches.
    tokenizer = hparams["tokenizer"]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav", "wav_benign")
    @sb.utils.data_pipeline.provides("adv", "adv_path", "sig", "src_path", "delta")
    def audio_pipeline(wav, wav_benign):
        # print(wav)
        # print(wav_benign)
        adver = sb.dataio.dataio.read_audio(wav)
        source = sb.dataio.dataio.read_audio(wav_benign)
        delta = adver - source
        # print(delta)
        # print(adver.shape, source.shape, delta.shape)
        yield adver
        yield wav        
        yield source
        yield wav_benign
        yield delta

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
        tokens_list = tokenizer.encode_as_ids(wrd)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "adv", "wrd", "tokens_bos", "tokens_eos", "tokens", "adv_path", "sig", "src_path", "delta"],
    )
    train_batch_sampler = None

    if hparams["dynamic_batching"]:
        from speechbrain.dataio.batch import PaddedBatch  # noqa
        from speechbrain.dataio.dataloader import SaveableDataLoader  # noqa
        from speechbrain.dataio.sampler import DynamicBatchSampler  # noqa

        dynamic_hparams = hparams["dynamic_batch_sampler"]
        hop_size = hparams["feats_hop_size"]

        train_batch_sampler = DynamicBatchSampler(
            train_data,
            length_func=lambda x: x["duration"] * (1 / hop_size),
            **dynamic_hparams,
        )

    return (
        train_data,
        train_batch_sampler,
    )

def merge_csv_by_row_and_save(csv1_path, csv2_path, output_path):
    df1 = pd.read_csv(csv1_path)
    df2 = pd.read_csv(csv2_path)

    if len(df1) != len(df2):
        raise ValueError("CSV files must have the same number of rows")
    df2 = df2.add_suffix("_benign")

    merged_df = pd.concat([df1, df2], axis=1)
    merged_df.to_csv(output_path, index=False)

    print(f"Merged CSV saved to: {output_path}")

def parse_wer_file(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()

    # Extract numbers like: %WER 11.93
    wer = float(re.search(r"%WER\s+([\d.]+)", lines[0]).group(1))
    ser = float(re.search(r"%SER\s+([\d.]+)", lines[1]).group(1))

    return wer, ser

if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    '''
    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Dataset prep (parsing Librispeech)
    from librispeech_prepare import prepare_librispeech  # noqa

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_librispeech,
        kwargs={
            "data_folder": hparams["data_folder"],
            "tr_splits": hparams["train_splits"],
            "dev_splits": hparams["dev_splits"],
            "te_splits": hparams["test_splits"],
            "save_folder": hparams["output_folder"],
            "merge_lst": hparams["train_splits"],
            "merge_name": "train.csv",
            "skip_prep": hparams["skip_prep"],
        },
    )
    run_on_main(hparams["prepare_noise_data"])
    
    # here we create the datasets objects as well as tokenization and encoding
    (
        train_data,
        valid_data,
        test_datasets,
        train_bsampler,
        valid_bsampler,
    ) = dataio_prepare(hparams)
    '''

    # We download the pretrained LM from HuggingFace (or elsewhere depending on
    # the path given in the YAML file). The tokenizer is loaded at the same time.
    hparams["pretrainer"].collect_files()
    hparams["pretrainer"].load_collected()

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # We dynamically add the tokenizer to our brain class.
    # NB: This tokenizer corresponds to the one used for the LM!!
    asr_brain.tokenizer = hparams["tokenizer"]
    
    train_dataloader_opts = hparams["train_dataloader_opts"]
    valid_dataloader_opts = hparams["valid_dataloader_opts"]
    '''
    if train_bsampler is not None:
        train_dataloader_opts = {"batch_sampler": train_bsampler}
    if valid_bsampler is not None:
        valid_dataloader_opts = {"batch_sampler": valid_bsampler}
    '''

    if hparams["train_adv_flg"]:
        with asr_brain.evaluation_ctx:
            print(f"PRECISION {hparams["precision"]}, EVAL PRECISION {hparams["eval_precision"]}")
            if hparams["adv_type"] == "cw":
                adv_data, adv_bsampler = dataio_prepare_2(hparams, hparams["clean_audio_adv_transcripts"])
                if adv_bsampler is not None:
                    train_dataloader_opts = {"batch_sampler": adv_bsampler}
                    
                asr_brain.attack_CW(
                    adv_data,
                    hparams=hparams,
                    train_loader_kwargs=train_dataloader_opts,
                    min_key="WER",
                    )
            else:
                # Testing
                if not os.path.exists(hparams["src_psy_folder"]): 
                    os.makedirs(hparams["src_psy_folder"])
                src_psy_transcripts = os.path.join(hparams["src_psy_folder"], "src_psy_transcripts.csv")
                merge_csv_by_row_and_save(
                    hparams["cw_audio_adv_transcripts"],
                    hparams["clean_audio_clean_transcripts"],
                    src_psy_transcripts
                )
                #'''
                adv_data, adv_bsampler = dataio_prepare_psy(hparams, src_psy_transcripts)
                if adv_bsampler is not None:
                    train_dataloader_opts = {"batch_sampler": adv_bsampler}

                asr_brain.attack_psy(
                    adv_data,
                    hparams=hparams,
                    train_loader_kwargs=train_dataloader_opts,
                    min_key="WER",
                    )
    import os

    # Testing
    if not os.path.exists(hparams["output_wer_folder"]):
        os.makedirs(hparams["output_wer_folder"])

    cw_test_data, label_encoder = dataio_prepare_2(hparams, hparams["adv_audio_adv_transcripts"])
    
    asr_brain.hparams.test_wer_file = os.path.join(
        hparams["output_wer_folder"], hparams["adv_WER"]
    )
    print(f"PRECISION {hparams["precision"]}, EVAL PRECISION {hparams["eval_precision"]}")
    asr_brain.evaluate(
        cw_test_data,
        test_loader_kwargs=hparams["test_dataloader_opts"],
        min_key="WER",
    )


    asr_brain.hparams.test_wer_file = os.path.join(
        hparams["output_wer_folder"], hparams["adv_WER_random"]
    )
    print(f"PRECISION {hparams["precision"]}, EVAL PRECISION {hparams["eval_precision"]}")
    wer_values = []
    ser_values = []
    for i in range(10):
        asr_brain.evaluate_edit(
            cw_test_data,
            test_loader_kwargs=hparams["test_dataloader_opts"],
            min_key="WER",
        )

        # Parse results after each run
        wer, ser = parse_wer_file(asr_brain.hparams.test_wer_file)
        wer_values.append(wer)
        ser_values.append(ser)
        # print(wer, ser)
    # Compute averages
    avg_wer = np.mean(wer_values)
    avg_ser = np.mean(ser_values)

    print("WER values:", wer_values)
    print("SER values:", ser_values)
    print(f"\nAverage WER: {avg_wer:.2f}")
    print(f"Average SER: {avg_ser:.2f}")

    '''

    clean_test_data, label_encoder = dataio_prepare_2(hparams, hparams["clean_audio_clean_transcripts"])
    
    asr_brain.hparams.test_wer_file = os.path.join(
        hparams["output_wer_folder"], hparams["clean_WER"]
    )
    
    asr_brain.evaluate(
        clean_test_data,
        test_loader_kwargs=hparams["test_dataloader_opts"],
        min_key="WER",
    )
    '''