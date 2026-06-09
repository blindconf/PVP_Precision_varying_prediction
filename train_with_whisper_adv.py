#!/usr/bin/env python3
"""Recipe for training a whisper-based ASR system with librispeech.
The system employs whisper from OpenAI (https://cdn.openai.com/papers/whisper.pdf).
This recipe take the whisper encoder-decoder to fine-tune on the NLL.

If you want to only use the whisper encoder system, please refer to the recipe
speechbrain/recipes/LibriSpeech/ASR/CTC/train_with_whisper.py

To run this recipe, do the following:
> python train_with_whisper.py hparams/train_hf_whisper.yaml

To add adapters and train only a fraction of the parameters, do:
> python train_with_whisper.py hparams/train_whisper_lora.yaml

Authors
 * Peter Plantinga 2024
 * Adel Moumen 2022, 2024
 * Titouan Parcollet 2022
"""

import os
import sys
from pathlib import Path

import torch
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.utils.data_utils import undo_padding
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
import re

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
        bos_tokens, bos_tokens_lens = batch.tokens_bos

        # Add waveform augmentation if specified.
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            wavs, wav_lens = self.hparams.wav_augment(wavs, wav_lens)
            bos_tokens = self.hparams.wav_augment.replicate_labels(bos_tokens)
            bos_tokens_lens = self.hparams.wav_augment.replicate_labels(
                bos_tokens_lens
            )

        # We compute the padding mask and replace the values with the pad_token_id
        # that the Whisper decoder expect to see.
        abs_tokens_lens = torch.round(
            bos_tokens_lens * bos_tokens.shape[1]
        ).long()
        pad_mask = (
            torch.arange(abs_tokens_lens.max(), device=self.device)[None, :]
            < abs_tokens_lens[:, None]
        )
        bos_tokens[~pad_mask] = self.tokenizer.pad_token_id

        # Forward encoder + decoder
        # wavs = wavs.to(torch.bfloat16)
        # print(f"[wavs] dtype: {wavs.dtype}, autocast: {torch.is_autocast_enabled()}")
        # self.modules.whisper = self.modules.whisper.to(torch.bfloat16)
        enc_out, logits, _ = self.modules.whisper(wavs, bos_tokens)
        # print(f"[enc_out] dtype: {enc_out.dtype}")
        # print(f"[logits] dtype: {logits.dtype}")
        # print(f"[_] dtype: {_}")
        log_probs = self.hparams.log_softmax(logits)
        # print(f"[log_probs] dtype: {log_probs.dtype}")
        hyps = None
        if stage == sb.Stage.VALID:
            hyps, _, _, _ = self.hparams.valid_search(
                enc_out.detach(), wav_lens
            )
        elif stage == sb.Stage.TEST:
            hyps, _, _, _ = self.hparams.test_search(enc_out.detach(), wav_lens)
            # print(f"[_] dtype: {_.dtype}")
        return log_probs, hyps, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss NLL given predictions and targets."""

        (log_probs, hyps, wav_lens) = predictions
        batch = batch.to(self.device)
        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos

        # Label Augmentation
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            tokens_eos = self.hparams.wav_augment.replicate_labels(tokens_eos)
            tokens_eos_lens = self.hparams.wav_augment.replicate_labels(
                tokens_eos_lens
            )

        loss = self.hparams.nll_loss(
            log_probs, tokens_eos, length=tokens_eos_lens
        )

        if stage != sb.Stage.TRAIN and stage !=Stage.ATTACK:
            tokens, tokens_lens = batch.tokens

            # Decode token terms to words
            predicted_words = [
                self.tokenizer.decode(t, skip_special_tokens=True).strip()
                for t in hyps
            ]

            # Convert indices to words
            target_words = undo_padding(tokens, tokens_lens)
            target_words = self.tokenizer.batch_decode(
                target_words, skip_special_tokens=True
            )
            if hasattr(self.hparams, "normalized_transcripts"):
                if hasattr(self.tokenizer, "normalize"):
                    normalized_fn = self.tokenizer.normalize
                else:
                    normalized_fn = self.tokenizer._normalize

                predicted_words = [
                    normalized_fn(text).split(" ") for text in predicted_words
                ]

                target_words = [
                    normalized_fn(text).split(" ") for text in target_words
                ]
            else:
                predicted_words = [text.split(" ") for text in predicted_words]
                target_words = [text.split(" ") for text in target_words]

            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)

        return loss

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

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
            lr = self.hparams.lr_annealing_whisper.current_lr
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": lr},
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
                train_set, stage= sb.Stage.TEST, **train_loader_kwargs
            )
        self.on_evaluate_start(max_key=max_key, min_key=min_key)
        self.on_stage_start(sb.Stage.TEST, epoch=None)
        self.modules.eval()
        self.sample_rate = hparams["sampling_rate"]

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
            
            self.global_optimal_delta.grad = torch.sign(self.global_optimal_delta.grad)
            # Do optimization
            self.optimizer_1.step()
            # Save the best adversarial example and adjust the alpha coefficient
            for local_batch_size_idx in range(local_batch_size):
                almost_successful[local_batch_size_idx] = masked_adv_input[local_batch_size_idx]
                torchaudio.save("adv_ex_14.wav", almost_successful[local_batch_size_idx][:real_lengths[local_batch_size_idx]].cpu()[None, :], self.sample_rate)
                data_adv, sample_rate = torchaudio.load("adv_ex_14.wav")
                batch.sig[0][local_batch_size_idx][:real_lengths[local_batch_size_idx]] = data_adv

            with torch.no_grad():
                with self.evaluation_ctx:
                    log_probs, hyps, wav_lens = self.compute_forward(batch, stage=sb.Stage.TEST)
                
                    predicted_words = [
                        self.tokenizer.decode(t, skip_special_tokens=True).strip()
                        for t in hyps
                        ]
                    # Convert indices to words
                    tokens, tokens_lens = batch.tokens
                    target_words = undo_padding(tokens, tokens_lens)
                    target_words = self.tokenizer.batch_decode(
                        target_words, skip_special_tokens=True
                    )

            # print(best_hyps)
            for local_batch_size_idx in range(local_batch_size):
                # Decode token terms to words
                if predicted_words == target_words:   
                    if (loss_2nd_stage[local_batch_size_idx] < best_loss_2nd_stage[local_batch_size_idx]):
                        # Update best loss at 2nd stage
                        # print(loss)
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

        self.sample_rate = hparams["sampling_rate"]
        
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
                '''
                if file_name == "61-70970-0005.flac":
                    print(file_name)
                    print(save_dirct)
                    result = self.attack_1st_stage(batch, hparams, save_dirct)

                '''
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
                torchaudio.save("adv_ex_11.wav", almost_successful[local_batch_size_idx][ :real_lengths[local_batch_size_idx]].cpu()[None, :], self.sample_rate)
                data_adv, sample_rate = torchaudio.load("adv_ex_11.wav")
                # print(batch.sig[0].shape, batch.sig[1].shape, data_adv.shape)
                batch.sig[0][local_batch_size_idx][:real_lengths[local_batch_size_idx]] = data_adv
            # print("before forward ", batch.sig)
            # print("data_adv ", data_adv)
            with torch.no_grad():
                with self.evaluation_ctx:
                    log_probs, hyps, wav_lens = self.compute_forward(batch, stage=sb.Stage.TEST)
                
                    predicted_words = [
                        self.tokenizer.decode(t, skip_special_tokens=True).strip()
                        for t in hyps
                        ]
                    # Convert indices to words
                    tokens, tokens_lens = batch.tokens
                    target_words = undo_padding(tokens, tokens_lens)
                    target_words = self.tokenizer.batch_decode(
                        target_words, skip_special_tokens=True
                    )

            # print(best_hyps)
            for local_batch_size_idx in range(local_batch_size):
                # Decode token terms to words
                if predicted_words == target_words:   
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
        with self.evaluation_ctx:
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

def dataio_prepare(hparams, tokenizer):
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
        hparams["train_loader_kwargs"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_loader_kwargs"]["shuffle"] = False

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

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        if (
            "normalized_transcripts" in hparams
            and hparams["normalized_transcripts"]
        ):
            wrd = tokenizer.normalize(wrd)
        yield wrd
        tokens_list = tokenizer.encode(wrd, add_special_tokens=False)
        yield tokens_list
        tokens_list = tokenizer.build_inputs_with_special_tokens(tokens_list)
        tokens_bos = torch.LongTensor(tokens_list[:-1])
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list[1:])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", "tokens_list", "tokens_bos", "tokens_eos", "tokens"],
    )

    return train_data, valid_data, test_datasets

def dataio_prepare_2(hparams, tokenizer, file_path):
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
        hparams["train_loader_kwargs"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_loader_kwargs"]["shuffle"] = False

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

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        if (
            "normalized_transcripts" in hparams
            and hparams["normalized_transcripts"]
        ):
            wrd = tokenizer.normalize(wrd)
        yield wrd
        tokens_list = tokenizer.encode(wrd, add_special_tokens=False)
        yield tokens_list
        tokens_list = tokenizer.build_inputs_with_special_tokens(tokens_list)
        tokens_bos = torch.LongTensor(tokens_list[:-1])
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list[1:])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", "tokens_list", "tokens_bos", "tokens_eos", "tokens", "path"],
    )

    return train_data

def dataio_prepare_psy(hparams, tokenizer, file_path):
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
        hparams["train_loader_kwargs"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_loader_kwargs"]["shuffle"] = False

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

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        if (
            "normalized_transcripts" in hparams
            and hparams["normalized_transcripts"]
        ):
            wrd = tokenizer.normalize(wrd)
        yield wrd
        tokens_list = tokenizer.encode(wrd, add_special_tokens=False)
        yield tokens_list
        tokens_list = tokenizer.build_inputs_with_special_tokens(tokens_list)
        tokens_bos = torch.LongTensor(tokens_list[:-1])
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list[1:])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "adv", "tokens_list", "tokens_bos", "tokens_eos", "tokens", "adv_path", "sig", "src_path", "delta"] ,
    )

    return train_data

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
    '''
    # Defining tokenizer and loading it
    tokenizer = hparams["whisper"].tokenizer

    # here we create the datasets objects as well as tokenization and encoding
    # train_data, valid_data, test_datasets = dataio_prepare(hparams, tokenizer)

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
        opt_class=hparams["whisper_opt_class"],
    )

    # We load the pretrained whisper model
    if "pretrainer" in hparams.keys():
        hparams["pretrainer"].collect_files()
        hparams["pretrainer"].load_collected(asr_brain.device)

    # We dynamically add the tokenizer to our brain class.
    # NB: This tokenizer corresponds to the one used for Whisper.
    asr_brain.tokenizer = tokenizer
    '''
    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_loader_kwargs"],
        valid_loader_kwargs=hparams["valid_loader_kwargs"],
    )
    
    # Testing
    os.makedirs(hparams["output_wer_folder"], exist_ok=True)

    for k in test_datasets.keys():  # keys are test_clean, test_other etc
        asr_brain.hparams.test_wer_file = os.path.join(
            hparams["output_wer_folder"], f"wer_{k}_{hparams["eval_precision"]}.txt"
        )
        asr_brain.evaluate(
            test_datasets[k],
            test_loader_kwargs=hparams["test_loader_kwargs"],
            min_key="WER",
        )
    '''
    if hparams["train_adv_flg"]:
        # with asr_brain.evaluation_ctx:
        print(f"PRECISION {hparams["precision"]}, EVAL PRECISION {hparams["eval_precision"]}")
        if hparams["adv_type"] == "cw":
            adv_data = dataio_prepare_2(hparams, tokenizer, hparams["clean_audio_adv_transcripts"])
            asr_brain.attack_CW(
                adv_data,
                hparams=hparams,
                train_loader_kwargs=hparams["test_loader_kwargs"],
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
            adv_data = dataio_prepare_psy(hparams, tokenizer, src_psy_transcripts)
            asr_brain.attack_psy(
                adv_data,
                hparams=hparams,
                train_loader_kwargs=hparams["test_loader_kwargs"],
                min_key="WER",
                )

    
    import os

    # Testing
    if not os.path.exists(hparams["output_wer_folder"]):
        os.makedirs(hparams["output_wer_folder"])

    adv_test_data = dataio_prepare_2(hparams, tokenizer, hparams["adv_audio_adv_transcripts"])

    asr_brain.hparams.test_wer_file = os.path.join(
        hparams["output_wer_folder"], hparams["adv_WER"]
    )
    # with asr_brain.evaluation_ctx:
    print(f"PRECISION {hparams["precision"]}, EVAL PRECISION {hparams["eval_precision"]}")
    asr_brain.evaluate(
        adv_test_data,
        test_loader_kwargs=hparams["test_loader_kwargs"],
        min_key="WER",
    )
    
    asr_brain.hparams.test_wer_file = os.path.join(
        hparams["output_wer_folder"], hparams["adv_WER_random"]
    )
    # print(f"PRECISION {hparams["precision"]}, EVAL PRECISION {hparams["eval_precision"]}")
    # wer_values = []
    # ser_values = []
    # for i in range(10):
    #     asr_brain.evaluate_edit(
    #         adv_test_data,
    #         test_loader_kwargs=hparams["test_loader_kwargs"],
    #         min_key="WER",
    #     )

    #     # Parse results after each run
    #     wer, ser = parse_wer_file(asr_brain.hparams.test_wer_file)
    #     wer_values.append(wer)
    #     ser_values.append(ser)
    #     # print(wer, ser)
    # # Compute averages
    # avg_wer = np.mean(wer_values)
    # avg_ser = np.mean(ser_values)

    # print("WER values:", wer_values)
    # print("SER values:", ser_values)
    # print(f"\nAverage WER: {avg_wer:.2f}")
    # print(f"Average SER: {avg_ser:.2f}")
    
    '''
    clean_test_data = dataio_prepare_2(hparams, tokenizer, hparams["clean_audio_clean_transcripts"])

    asr_brain.hparams.test_wer_file = os.path.join(
        hparams["output_wer_folder"], hparams["clean_WER"]
    )
    
    asr_brain.evaluate(
        clean_test_data,
        test_loader_kwargs=hparams["test_loader_kwargs"],
        min_key="WER",
    )
    '''