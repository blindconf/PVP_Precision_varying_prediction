#!/usr/bin/env/python3
"""Recipe for training a wav2vec-based ctc ASR system with librispeech.
The system employs wav2vec as its encoder. Decoding is performed with
ctc greedy decoder during validation and a beam search with an optional
language model during test. The test searcher can be chosen from the following
options: CTCBeamSearcher, CTCPrefixBeamSearcher, TorchAudioCTCPrefixBeamSearcher.

To run this recipe, do the following:
> python train_with_wav2vec.py hparams/train_{hf,sb}_wav2vec.yaml
The neural network is trained on CTC likelihood target and character units
are used as basic recognition tokens.

Authors
 * Rudolf A Braun 2022
 * Titouan Parcollet 2022
 * Sung-Lin Yeh 2021
 * Ju-Chieh Chou 2020
 * Mirco Ravanelli 2020
 * Abdel Heba 2020
 * Peter Plantinga 2020
 * Samuele Cornell 2020
 * Adel Moumen 2023
"""
import os
import sys
from pathlib import Path

import torch
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.utils.distributed import if_main_process, run_on_main
from speechbrain.utils.logger import get_logger

# Added by me
from enum import Enum, auto
import numpy as np
from torch.utils.data import DataLoader
from speechbrain.dataio.dataloader import LoopedLoader
from tqdm.contrib import tqdm
import torch.nn as nn
import torchaudio
import csv
import pandas as pd
from typing import List, Optional, Tuple
from scipy.signal import argrelextrema
from speechbrain.utils.autocast import AMPConfig, TorchAutocast

logger = get_logger(__name__)

class Stage(Enum):
    """Simple enum to track stage of experiments."""
    ATTACK = auto()

# Define training procedure
class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)

        # print(f"precision {hparams["precision"]}, eval precision {hparams["eval_precision"]}")
        # print(f"[wavs] dtype: {wavs.dtype}, autocast: {torch.is_autocast_enabled()}")
        # print(next(self.modules.wav2vec2.parameters()).dtype, next(self.modules.enc.parameters()).dtype, next(self.modules.ctc_lin.parameters()).dtype)
        
        # Downsample the inputs if specified
        if hasattr(self.modules, "downsampler"):
            wavs = self.modules.downsampler(wavs)

        # Add waveform augmentation if specified.
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            wavs, wav_lens = self.hparams.wav_augment(wavs, wav_lens)
        # print(f"[wavs] dtype: {wavs.dtype}, autocast: {torch.is_autocast_enabled()}")
        # Forward pass

        # Handling SpeechBrain vs HuggingFace pretrained models
        if hasattr(self.modules, "extractor"):  # SpeechBrain pretrained model
            latents = self.modules.extractor(wavs)
            feats = self.modules.encoder_wrapper(latents, wav_lens=wav_lens)[
                "embeddings"
            ]
        else:  # HuggingFace pretrained model
            feats = self.modules.wav2vec2(wavs, wav_lens)
        # print(f"[feats] dtype: {feats.dtype}, autocast: {torch.is_autocast_enabled()}")
        ## x change precision and logits!!!!
        ## Autocast only changes the dtype of eligible ops, not tensors by themselves — and many speech front-end ops are explicitly kept in FP32
        ## enc is usually nn.Linear, nn.LSTM, or nn.Transformer
        ## These ops are on PyTorch’s autocast allowlist
        ## Autocast casts their outputs
        ## ctc_lin changes for the same reason, nn.Linear autocast
        x = self.modules.enc(feats)
        # print(f"[x] dtype: {x.dtype}, autocast: {torch.is_autocast_enabled()}")
        # Compute outputs
        p_tokens = None
        logits = self.modules.ctc_lin(x)
        # print(f"[logits] dtype: {logits.dtype}, autocast: {torch.is_autocast_enabled()}")
        # Upsample the inputs if they have been highly downsampled
        if hasattr(self.hparams, "upsampling") and self.hparams.upsampling:
            logits = logits.view(
                logits.shape[0], -1, self.hparams.output_neurons
            )

        p_ctc = self.hparams.log_softmax(logits)
        # print(f"[p_ctc] dtype: {p_ctc.dtype}, autocast: {torch.is_autocast_enabled()}")
        if stage == sb.Stage.VALID:
            p_tokens = sb.decoders.ctc_greedy_decode(
                p_ctc, wav_lens, blank_id=self.hparams.blank_index
            )
        
        elif stage == sb.Stage.TEST:
            p_tokens = test_searcher(p_ctc, wav_lens)
            # print(p_tokens)
            # print(f"[p_ctc] dtype: {p_ctc.dtype}, autocast: {torch.is_autocast_enabled()}")
            # print(f"[p_tokens] dtype: {p_tokens.dtype}, autocast: {torch.is_autocast_enabled()}")
            candidates = []
            scores = []

            for batch in p_tokens:
                candidates.append([hyp.text for hyp in batch])
                scores.append([hyp.score for hyp in batch])

            if hasattr(self.hparams, "rescorer"):
                p_tokens, _ = self.hparams.rescorer.rescore(candidates, scores)

        return p_ctc, wav_lens, p_tokens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        p_ctc, wav_lens, predicted_tokens = predictions

        ids = batch.id
        tokens, tokens_lens = batch.tokens

        # Labels must be extended if parallel augmentation or concatenated
        # augmentation was performed on the input (increasing the time dimension)
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            (
                tokens,
                tokens_lens,
            ) = self.hparams.wav_augment.replicate_multiple_labels(
                tokens, tokens_lens
            )

        loss_ctc = self.hparams.ctc_cost(p_ctc, tokens, wav_lens, tokens_lens)
        loss = loss_ctc

        if stage == sb.Stage.VALID:
            # Decode token terms to words
            predicted_words = [
                "".join(self.tokenizer.decode_ndim(utt_seq)).split(" ")
                for utt_seq in predicted_tokens
            ]
        elif stage == sb.Stage.TEST:
            if hasattr(self.hparams, "rescorer"):
                predicted_words = [
                    hyp[0].split(" ") for hyp in predicted_tokens
                ]
            else:
                predicted_words = [
                    hyp[0].text.split(" ") for hyp in predicted_tokens
                ]

        if stage != sb.Stage.TRAIN and stage !=Stage.ATTACK:
            target_words = [wrd.split(" ") for wrd in batch.wrd]
            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)

        return loss

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

        if stage == sb.Stage.TEST:
            if hasattr(self.hparams, "rescorer"):
                self.hparams.rescorer.move_rescorers_to_device()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr_model, new_lr_model = self.hparams.lr_annealing_model(
                stage_stats["loss"]
            )
            old_lr_wav2vec, new_lr_wav2vec = self.hparams.lr_annealing_wav2vec(
                stage_stats["loss"]
            )
            sb.nnet.schedulers.update_learning_rate(
                self.model_optimizer, new_lr_model
            )
            sb.nnet.schedulers.update_learning_rate(
                self.wav2vec_optimizer, new_lr_wav2vec
            )
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr_model": old_lr_model,
                    "lr_wav2vec": old_lr_wav2vec,
                },
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

    def init_optimizers(self):
        "Initializes the wav2vec2 optimizer and model optimizer"
        # Handling SpeechBrain vs HuggingFace pretrained models
        if hasattr(self.modules, "extractor"):  # SpeechBrain pretrained model
            self.wav2vec_optimizer = self.hparams.wav2vec_opt_class(
                self.modules.encoder_wrapper.parameters()
            )

        else:  # HuggingFace pretrained model
            self.wav2vec_optimizer = self.hparams.wav2vec_opt_class(
                self.modules.wav2vec2.parameters()
            )

        self.model_optimizer = self.hparams.model_opt_class(
            self.hparams.model.parameters()
        )

        # save the optimizers in a dictionary
        # the key will be used in `freeze_optimizers()`
        self.optimizers_dict = {
            "model_optimizer": self.model_optimizer,
        }
        if not self.hparams.freeze_wav2vec:
            self.optimizers_dict["wav2vec_optimizer"] = self.wav2vec_optimizer

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable(
                "wav2vec_opt", self.wav2vec_optimizer
            )
            self.checkpointer.add_recoverable("modelopt", self.model_optimizer)

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
                train_set, stage= sb.Stage.TEST, **train_loader_kwargs
            )
        self.on_evaluate_start(max_key=max_key, min_key=min_key)
        self.on_stage_start(sb.Stage.TEST, epoch=None)
        self.modules.eval()
        self.sample_rate = hparams["sample_rate"]
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
                torchaudio.save("adv_ex_2.wav", almost_successful[local_batch_size_idx][:real_lengths[local_batch_size_idx]].cpu()[None, :], self.sample_rate)
                data_adv, sample_rate = torchaudio.load("adv_ex_2.wav")
                batch.sig[0][local_batch_size_idx][:real_lengths[local_batch_size_idx]] = data_adv
            
            with torch.no_grad():
                p_seq, wav_lens, best_hyps = self.compute_forward(batch, stage=sb.Stage.TEST)
            for local_batch_size_idx in range(local_batch_size):
                # Decode token terms to words
                if hasattr(self.hparams, "rescorer"):
                    predicted_words = [
                        hyp[0].split(" ") for hyp in best_hyps
                    ]
                else:
                    predicted_words = [
                        hyp[0].text.split(" ") for hyp in best_hyps
                    ]

                target_words = [wrd.split(" ") for wrd in batch.wrd]
                
                if predicted_words == target_words:

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
                save_dirct = save_dirct.replace(".flac", ".wav")

                # cnt += 1 
                # print(save_dirct)
                if (not os.path.exists(save_dirct)):                    
                    result = self.attack_1st_stage(batch, hparams, save_dirct)
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
            self.global_optimal_delta.grad = torch.sign(self.global_optimal_delta.grad)
            # Do optimization
            self.optimizer_1.step()

            for local_batch_size_idx in range(local_batch_size):
                almost_successful[local_batch_size_idx] = masked_adv_input[local_batch_size_idx].detach()
                # print("almost_successful :", almost_successful)
                torchaudio.save("adv_ex_3.wav", almost_successful[local_batch_size_idx][ :real_lengths[local_batch_size_idx]].cpu()[None, :], self.sample_rate)
                data_adv, sample_rate = torchaudio.load("adv_ex_3.wav")
                # print(batch.sig[0].shape, batch.sig[1].shape, data_adv.shape)
                batch.sig[0][local_batch_size_idx][:real_lengths[local_batch_size_idx]] = data_adv
            # print("before forward ", batch.sig)
            # print("data_adv ", data_adv)
            with torch.no_grad():
                eval_dtype = AMPConfig.from_name("fp32").dtype
                self.evaluation_ctx = TorchAutocast(
                    device_type=self.device, dtype=eval_dtype
                )
                with self.evaluation_ctx:
                    p_seq, wav_lens, best_hyps = self.compute_forward(batch, sb.Stage.TEST)
                    if hasattr(self.hparams, "rescorer"):
                        predicted_words_1 = [
                            hyp[0].split(" ") for hyp in best_hyps
                        ]
                    else:
                        predicted_words_1 = [
                            hyp[0].text.split(" ") for hyp in best_hyps
                        ]

                eval_dtype = AMPConfig.from_name("fp16").dtype
                self.evaluation_ctx = TorchAutocast(
                    device_type=self.device, dtype=eval_dtype
                )
                with self.evaluation_ctx:
                    p_seq, wav_lens, best_hyps = self.compute_forward(batch, sb.Stage.TEST)
                    if hasattr(self.hparams, "rescorer"):
                        predicted_words_2 = [
                            hyp[0].split(" ") for hyp in best_hyps
                        ]
                    else:
                        predicted_words_2 = [
                            hyp[0].text.split(" ") for hyp in best_hyps
                        ]

                eval_dtype = AMPConfig.from_name("bf16").dtype
                self.evaluation_ctx = TorchAutocast(
                    device_type=self.device, dtype=eval_dtype
                )
                with self.evaluation_ctx:
                    p_seq, wav_lens, best_hyps = self.compute_forward(batch, sb.Stage.TEST)
                    if hasattr(self.hparams, "rescorer"):
                        predicted_words_3 = [
                            hyp[0].split(" ") for hyp in best_hyps
                        ]
                    else:
                        predicted_words_3 = [
                            hyp[0].text.split(" ") for hyp in best_hyps
                        ]
                # with self.evaluation_ctx:
                #     p_seq, wav_lens, best_hyps = self.compute_forward(batch, stage=sb.Stage.TEST)
            # print(best_hyps)
            for local_batch_size_idx in range(local_batch_size):
                # Decode token terms to words
                # if hasattr(self.hparams, "rescorer"):
                #     predicted_words = [
                #         hyp[0].split(" ") for hyp in best_hyps
                #     ]
                # else:
                #     predicted_words = [
                #         hyp[0].text.split(" ") for hyp in best_hyps
                #     ]

                target_words = [wrd.split(" ") for wrd in batch.wrd]
                # print("loss: ", loss)
                # print(predicted_words_1)  
                # print(predicted_words_2)
                # print(predicted_words_3)

                if predicted_words_1 == predicted_words_2 == predicted_words_3 == target_words:   
                    # print(loss)
                    # print(predicted_words)
                    # print(target_words)
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
        eval_dtype = AMPConfig.from_name("fp32").dtype
        self.evaluation_ctx = TorchAutocast(
            device_type=self.device, dtype=eval_dtype
        )
        with self.evaluation_ctx:
            predictions = self.compute_forward(batch, Stage.ATTACK)
            loss_1 = self.compute_objectives(predictions, batch, Stage.ATTACK)
        eval_dtype = AMPConfig.from_name("fp16").dtype
        self.evaluation_ctx = TorchAutocast(
            device_type=self.device, dtype=eval_dtype
        )
        with self.evaluation_ctx:
            predictions = self.compute_forward(batch, Stage.ATTACK)
            loss_2 = self.compute_objectives(predictions, batch, Stage.ATTACK)

        eval_dtype = AMPConfig.from_name("bf16").dtype
        self.evaluation_ctx = TorchAutocast(
            device_type=self.device, dtype=eval_dtype
        )
        with self.evaluation_ctx:
            predictions = self.compute_forward(batch, Stage.ATTACK)
            loss_3 = self.compute_objectives(predictions, batch, Stage.ATTACK)
        
        loss_all = 0.33 * loss_1 + 0.33 * loss_2 + 0.33 * loss_3
        # print("loss: ", loss_all, loss_1, loss_2, loss_3)
        loss = self.const * loss_all + torch.norm(local_delta_rescale)
        return loss, local_delta, masked_adv_input, local_delta_rescale

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

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)
    label_encoder = sb.dataio.encoder.CTCTextEncoder()

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "char_list", "tokens_list", "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
        char_list = list(wrd)
        yield char_list
        tokens_list = label_encoder.encode_sequence(char_list)
        yield tokens_list
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    special_labels = {
        "blank_label": hparams["blank_index"],
    }
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[train_data],
        output_key="char_list",
        special_labels=special_labels,
        sequence_input=True,
    )

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", "wrd", "char_list", "tokens"],
    )

    return train_data, valid_data, test_datasets, label_encoder

def dataio_prepare_2(hparams, file_path):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=file_path, replacements={"data_root": data_folder},
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

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig", "path")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        yield sig
        yield wav

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)
    label_encoder = sb.dataio.encoder.CTCTextEncoder()

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "char_list", "tokens_list", "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
        char_list = list(wrd)
        yield char_list
        tokens_list = label_encoder.encode_sequence(char_list)
        yield tokens_list
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    special_labels = {
        "blank_label": hparams["blank_index"],
    }
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[train_data],
        output_key="char_list",
        special_labels=special_labels,
        sequence_input=True,
    )

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "wrd", "char_list", "tokens", "path"],
    )

    return train_data, label_encoder

def dataio_prepare_psy(hparams, file_path):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=file_path, replacements={"data_root": data_folder},
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
    label_encoder = sb.dataio.encoder.CTCTextEncoder()

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "char_list", "tokens_list", "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
        char_list = list(wrd)
        yield char_list
        tokens_list = label_encoder.encode_sequence(char_list)
        yield tokens_list
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    special_labels = {
        "blank_label": hparams["blank_index"],
    }
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[train_data],
        output_key="char_list",
        special_labels=special_labels,
        sequence_input=True,
    )

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "adv", "wrd", "char_list", "tokens", "adv_path", "sig", "src_path", "delta"],
        # datasets, ["id", "sig", "wrd", "char_list", "tokens", "path"],
    )

    return train_data, label_encoder

def merge_csv_by_row_and_save(csv1_path, csv2_path, output_path):
    df1 = pd.read_csv(csv1_path)
    df2 = pd.read_csv(csv2_path)

    if len(df1) != len(df2):
        raise ValueError("CSV files must have the same number of rows")
    df2 = df2.add_suffix("_benign")

    merged_df = pd.concat([df1, df2], axis=1)
    merged_df.to_csv(output_path, index=False)

    print(f"Merged CSV saved to: {output_path}")

if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    # sb.create_experiment_directory(
    #     experiment_directory=hparams["output_folder"],
    #     hyperparams_to_save=hparams_file,
    #     overrides=overrides,
    # )

    # # Dataset prep (parsing Librispeech)
    # from librispeech_prepare import prepare_librispeech  # noqa

    # # multi-gpu (ddp) save data preparation
    # run_on_main(
    #     prepare_librispeech,
    #     kwargs={
    #         "data_folder": hparams["data_folder"],
    #         "tr_splits": hparams["train_splits"],
    #         "dev_splits": hparams["dev_splits"],
    #         "te_splits": hparams["test_splits"],
    #         "save_folder": hparams["output_folder"],
    #         "merge_lst": hparams["train_splits"],
    #         "merge_name": "train.csv",
    #         "skip_prep": hparams["skip_prep"],
    #     },
    # )

    # # here we create the datasets objects as well as tokenization and encoding
    train_data, valid_data, test_datasets, label_encoder = dataio_prepare(
        hparams
    )
    
    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # We load the pretrained wav2vec2 model
    if "pretrainer" in hparams.keys():
        hparams["pretrainer"].collect_files()
        hparams["pretrainer"].load_collected()

    # We dynamically add the tokenizer to our brain class.
    # NB: This tokenizer corresponds to the one used for the LM!!
    asr_brain.tokenizer = label_encoder

    ind2lab = label_encoder.ind2lab
    vocab_list = [ind2lab[x] for x in range(len(ind2lab))]

    from speechbrain.decoders.ctc import CTCBeamSearcher

    test_searcher = CTCBeamSearcher(
        **hparams["test_beam_search"],
        vocab_list=vocab_list,
    )

    if hparams["train_adv_flg"]:
        # with asr_brain.evaluation_ctx:
        print(f"PRECISION {hparams["precision"]}, EVAL PRECISION {hparams["eval_precision"]}")
        if hparams["adv_type"] == "cw_adapt":
            adv_data, label_encoder = dataio_prepare_2(hparams, hparams["clean_audio_adv_transcripts"])
            asr_brain.attack_CW(
                adv_data,
                hparams=hparams,
                train_loader_kwargs=hparams["train_dataloader_opts"],
                min_key="WER",
                )
        # else:
        #     # Testing
        #     if not os.path.exists(hparams["src_psy_folder"]): 
        #         os.makedirs(hparams["src_psy_folder"])
        #     src_psy_transcripts = os.path.join(hparams["src_psy_folder"], "src_psy_transcripts.csv")
        #     merge_csv_by_row_and_save(
        #         hparams["cw_audio_adv_transcripts"],
        #         hparams["clean_audio_clean_transcripts"],
        #         src_psy_transcripts
        #     )
        #     #'''
        #     adv_data, label_encoder = dataio_prepare_psy(hparams, src_psy_transcripts)
        #     asr_brain.attack_psy(
        #         adv_data,
        #         hparams=hparams,
        #         train_loader_kwargs=hparams["train_dataloader_opts"],
        #         min_key="WER",
        #         )
        #         # '''

        # '''
        # Testing
        if not os.path.exists(hparams["output_wer_folder"]): 
            os.makedirs(hparams["output_wer_folder"])

        adv_test_data, label_encoder = dataio_prepare_2(hparams, hparams["adv_audio_adv_transcripts"])
        
        asr_brain.hparams.test_wer_file = os.path.join(
            hparams["output_wer_folder"], hparams["adv_WER"]
        )
        print(f"PRECISION {hparams["precision"]}, EVAL PRECISION {hparams["eval_precision"]}")
        asr_brain.evaluate(
            adv_test_data,
            test_loader_kwargs=hparams["test_dataloader_opts"],
            min_key="WER",
        )
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