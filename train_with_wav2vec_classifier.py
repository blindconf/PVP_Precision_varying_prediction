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
from itertools import combinations
from scipy.stats import norm 
from collections import defaultdict
import statistics
from sklearn.metrics import roc_curve, auc
from speechbrain.utils.edit_distance import (_str_equals, wer_details_for_batch)
from speechbrain.dataio.dataio import (split_word)
from scipy.stats import entropy
import pickle

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

    def compute_objectives_2(self, predictions, batch, stage):
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

        return ids, predicted_words

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

    def calculate_wer(
        self,
        test_set,
        precision_types=("fp32", "fp16", "bf16"),
        max_key=None,
        min_key=None,
        progressbar=None,
        test_loader_kwargs={},
    ):
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

        pred_words = {}

        for p in precision_types:
            eval_dtype = AMPConfig.from_name(p).dtype
            self.evaluation_ctx = TorchAutocast(
                device_type=self.device,
                dtype=eval_dtype,
            )
            pred_words_2 = []
            with torch.no_grad():
                for batch in tqdm(
                    test_set,
                    dynamic_ncols=True,
                    disable=enable, # not enable,
                    colour=self.tqdm_barcolor["test"],
                ):
                    # self.step += 1
                    # loss = self.evaluate_batch(batch, stage=Stage.TEST)
                    with self.evaluation_ctx:
                        out = self.compute_forward(batch, stage=sb.Stage.TEST)
                        ids, predicted_words = self.compute_objectives_2(out, batch, stage=sb.Stage.TEST)
                        pred_words_2.append((ids, predicted_words))
            pred_words[p] = pred_words_2
            # stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            # stage_stats["WER"] = self.wer_metric.summarize("error_rate")
            # self.on_stage_end(Stage.TEST, avg_test_loss, None)
        # self.step = 0
        return pred_words

    def characteristics(
        self,
        train_set,
        path_file, 
        max_key=None,
        min_key=None,            
        hparams=None,
        progressbar=None,
        train_loader_kwargs={},):
        """
        Function that calculates the 24 scores 
        (resulting from combing each of the 4 aggregation methods with the 6 characteristics).
        """
        
        # Characteristics
        measurements = {
            'Entropy mean': 0, 'Median mean': 0
            }
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

        entr_avg, med_avg = [], []

        with torch.no_grad():
            for batch in tqdm(train_set, dynamic_ncols=True, disable=not progressbar):
                with self.evaluation_ctx:
                    predictions = self.compute_forward(batch, stage=sb.Stage.TEST)
                p_ctc = torch.squeeze(predictions[0], dim=0)
                p_ctc_prob = torch.exp(p_ctc).detach().cpu()
                
                p_ctc_prob = np.array(p_ctc_prob)
                # Remove extreme cases which lead to undefined characteristic values
                p_ctc_prob = np.delete(p_ctc_prob, np.where((p_ctc_prob == 0))[0], axis=0)    
                p_ctc_prob = np.delete(p_ctc_prob, np.where((p_ctc_prob == 1))[0], axis=0)   
                # Entropy
                entropy_1 = entropy(p_ctc_prob, axis=1)
                entr_avg.append(np.mean(entropy_1))
                # Median
                median_prob = np.log(np.median(p_ctc_prob, axis=1))
                med_avg.append(np.mean(median_prob))
       
        # Saving the Characteristics
        measurements['Entropy mean'] = entr_avg
        measurements['Median mean'] = med_avg

        with open(path_file, 'wb') as file:
            pickle.dump(measurements, file, protocol=pickle.HIGHEST_PROTOCOL)
        pass

    def noise_flooding(
        self,
        train_set,
        path_file, 
        max_key=None,
        min_key=None,            
        hparams=None,
        progressbar=None,
        train_loader_kwargs={},):
        """
        Function that calculates the 24 scores 
        (resulting from combing each of the 4 aggregation methods with the 6 characteristics).
        """
        
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

        entr_avg, med_avg = [], []

        with torch.no_grad():
            scores = []
            for batch in tqdm(train_set, dynamic_ncols=True, disable=not progressbar):
                original_input = torch.clone(batch.sig[0])
                # rescale = 0
                rescale = 1e-4
                with self.evaluation_ctx:
                    predictions = self.compute_forward(batch, stage=sb.Stage.TEST)
                    if hasattr(self.hparams, "rescorer"):
                        predicted_words = [
                            hyp[0].split(" ") for hyp in predictions[2]
                        ]
                    else:
                        predicted_words = [
                            hyp[0].text.split(" ") for hyp in predictions[2]
                        ]
                target_words = predicted_words 
                while predicted_words == target_words and rescale < 1 :
                    # rescale += 0.0001
                    rescale *= 2   # exponential growth
                    eps = torch.FloatTensor(1, batch.sig[0].shape[1]).uniform_(-1, 1) * rescale
                    noisy_input = torch.clamp(original_input + eps, -1.0, 1.0).to(self.device)

                    batch.sig = noisy_input, batch.sig[1]
                    with self.evaluation_ctx:
                        predictions = self.compute_forward(batch, stage=sb.Stage.TEST)
                        if hasattr(self.hparams, "rescorer"):
                            predicted_words = [
                                hyp[0].split(" ") for hyp in predictions[2]
                            ]
                        else:
                            predicted_words = [
                                hyp[0].text.split(" ") for hyp in predictions[2]
                            ]
                    if predicted_words != target_words: # initial_tokens.shape != sequences.shape:
                        break
                scores.append(torch.max(torch.abs(eps)).item())
        measurements = {
                        "Scores": scores
                    }
        with open(path_file, 'wb') as file:
            pickle.dump(measurements, file, protocol=pickle.HIGHEST_PROTOCOL)
        pass

    def temporal_dependency(
        self,
        train_set,
        path_file, 
        max_key=None,
        min_key=None,            
        hparams=None,
        progressbar=None,
        train_loader_kwargs={},):
        """
        Function that calculates the 24 scores 
        (resulting from combing each of the 4 aggregation methods with the 6 characteristics).
        """
        
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

        entr_avg, med_avg = [], []

        with torch.no_grad():
            scores = []
            for batch in tqdm(train_set, dynamic_ncols=True, disable=not progressbar):
                original_input = torch.clone(batch.sig[0])

                with self.evaluation_ctx:
                    predictions = self.compute_forward(batch, stage=sb.Stage.TEST)
                    wavs_ratio =  int(batch.sig[0].shape[1] * 0.5)
                    batch.sig = original_input[:, :wavs_ratio], batch.sig[1]
                    half_predictions = self.compute_forward(batch, stage=sb.Stage.TEST)

                    if hasattr(self.hparams, "rescorer"):
                        predicted_words = [
                            hyp[0].split(" ") for hyp in predictions[2]
                        ]
                    else:
                        predicted_words = [
                            hyp[0].text.split(" ") for hyp in predictions[2]
                        ]
                    predicted_words = " ".join(predicted_words[0])
                    if hasattr(self.hparams, "rescorer"):
                        predicted_words_2 = [
                            hyp[0].split(" ") for hyp in half_predictions[2]
                        ]
                    else:
                        predicted_words_2 = [
                            hyp[0].text.split(" ") for hyp in half_predictions[2]
                        ]
                    predicted_words_2 = " ".join(predicted_words_2[0])
                    if (predicted_words == "" or predicted_words_2 == ""):
                        print("Empty String index: ")
                        print(predicted_words)
                        print(predicted_words_2)
                    else:
                        test = self.newWER(predicted_words, predicted_words_2)
                        scores.append(test)
        
        measurements = {
                        "Scores": scores
                    }
        with open(path_file, 'wb') as file:
            pickle.dump(measurements, file, protocol=pickle.HIGHEST_PROTOCOL)
        pass

    def newWER(self, x, y):
        x = x.split()
        y = y.split()
        n = len(x)
        m = len(y)
        k = min(n, m)
        d = np.zeros((k + 1) * (k + 1), dtype = np.uint8).reshape(k + 1, k + 1)
        for i in range(k + 1):
            for j in range(k + 1):
                if i == 0:
                    d[0][j] = j
                elif j == 0:
                    d[i][0] = i
        for i in range(1, k + 1):
            for j in range(1, k + 1):
                if (x[i - 1] == y[j - 1]):
                    d[i][j] = d[i - 1][j - 1]
                else:
                    S = d[i - 1][j - 1] + 1
                    I = d[i][j - 1] + 1
                    D = d[i - 1][j] + 1
                    d[i][j] = min(S, I, D)
        return d[k][k] * 1.0 / k

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

def to_utt_dict(results):
    utt_dict = {}
    for ids, words in results:
        utt_id = ids[0]
        utt_dict[utt_id] = words[0]  # list of tokens
    return utt_dict

# from itertools import combinations

def pairwise_wer_cer(preds, asr_brain, space="_"):
    results = {}
    results_scores = {}

    for ref_p, hyp_p in combinations(preds.keys(), 2):
        wer_metric = asr_brain.hparams.error_rate_computer()
        cer_metric = asr_brain.hparams.cer_computer()

        ref_dict = preds[ref_p]
        hyp_dict = preds[hyp_p]

        common_utts = ref_dict.keys() & hyp_dict.keys()

        wer_list = []
        cer_list = []

        for utt_id in common_utts:
            ref = ref_dict[utt_id]
            hyp = hyp_dict[utt_id]

            # WER (word-level)
            _, wer = error_score(
                [utt_id],
                [hyp],
                [ref],
                split_tokens=False,
            )

            # CER (char-level)
            _, cer = error_score(
                [utt_id],
                [hyp],
                [ref],
                split_tokens=True,
            )

            wer_list.append((utt_id, wer))
            cer_list.append((utt_id, cer))
            # Sort by error (descending = worst first)
            # wer_list.sort(key=lambda x: x[0], reverse=True)
            # cer_list.sort(key=lambda x: x[0], reverse=True)

            # error_score([utt_id], [hyp], [ref], False)            

            wer_metric.append(
                ids=[utt_id],
                predict=[hyp],
                target=[ref],
            )

            cer_metric.append(
                ids=[utt_id],
                predict=[hyp],
                target=[ref],
            )

        results[(ref_p, hyp_p)] = {
            "WER": wer_metric.summarize("error_rate"),
            "CER": cer_metric.summarize("error_rate"),
        }

        results_scores[(ref_p, hyp_p)] = {
            "WER": wer_list,
            "CER": cer_list,
        }

    return results, results_scores

def error_score(
    ids,
    predict,
    target,
    split_tokens=False,
    space="_"
    ):

    #if self.merge_tokens:
    #    predict = merge_char(predict, space=self.space_token)
    #    target = merge_char(target, space=self.space_token)

    if split_tokens:
        predict = split_word(predict, space="_")
        target = split_word(target, space="_")
    '''
    if self.extract_concepts_values:
        predict = extract_concepts_values(
            predict,
            self.keep_values,
            self.tag_in,
            self.tag_out,
            space=self.space_token,
        )
        target = extract_concepts_values(
            target,
            self.keep_values,
            self.tag_in,
            self.tag_out,
            space=self.space_token,
        )
    '''
    equality_comparator: Callable[[str, str], bool] = _str_equals

    scores = wer_details_for_batch(
        ids,
        target,
        predict,
        compute_alignments=True,
        equality_comparator=equality_comparator,
    )
    # print(scores)
    s = scores[0]
    return s["key"], s["WER"]
    # print(scores['WER'])

def build_char_dataset(values, benign=True, key_name="WER_max"):
    return {
        key_name: values,
        "benign_flg": [1 if benign else 0] * len(values)
    }

def merge(dict_1, dict_2, key):
    """
    Merge two dictionaries based on specific keys.

    :param dict_1: Dictionary variable.
    :param dict_2: Dictionary variable.
    :param key: Keys to use during the merge.
    :return: A dictionary merged based on specific keys.
    """
    dict_all = {x: dict_1[x] + dict_2[x] for x in key}
    return dict_all

def fit_gaussian(train_set, test_set, adv_set, key):
    """
    Gaussian distribution-based adversarial detector
    """

    char_key = [key, "benign_flg"]

    test_metrics = {x: test_set[x] for x in char_key}
    adv_metrics  = {x: adv_set[x]  for x in char_key}

    test_all = merge(test_metrics, adv_metrics, char_key)

    mean = np.mean(train_set[key])
    std  = np.std(train_set[key])
    std  = max(std, 1e-6)   # critical fix

    fitted_norm = norm.pdf(test_all[key], loc=mean, scale=std)

    # train_vals = np.array(train_set[key])
    # test_vals  = np.array(test_all[key])

    # # log transform
    # train_vals = np.log(train_vals + 1e-8)
    # test_vals  = np.log(test_vals + 1e-8)

    # mean = np.mean(train_vals)
    # std  = np.std(train_vals)
    # std  = max(std, 1e-6)

    # fitted_norm = norm.pdf(test_vals, loc=mean, scale=std)

    fpr, tpr, _ = roc_curve(test_all["benign_flg"], fitted_norm)
    
    roc_auc = auc(fpr, tpr)
    # print(roc_auc)
    # Ignorar primer punto (0,0) que siempre aparece
    fpr = fpr[1:]
    tpr = tpr[1:]
    # Remove (1,1) if present
    if len(tpr) > 0 and tpr[-1] == 1.0 and fpr[-1] == 1.0:
        fpr = fpr[:-1]
        tpr = tpr[:-1]

    fnr = 1 - tpr
    tnr = 1 - fpr
    worst_benign_idx = np.argmax(fnr)   # equivalente a np.argmin(tpr)
    worst_benign = {
        "tpr": float(tpr[worst_benign_idx]),
        "tnr": float(tnr[worst_benign_idx]),
        "fpr": float(fpr[worst_benign_idx]),
        "fnr": float(fnr[worst_benign_idx]),
    }

    # ============================
    # TPR @ FPR < 1% (or closest)
    # ============================
    target_fpr = 0.01  # FPR constraint
    target_tpr = 0.95  # desired minimum TPR

    mask = fpr < target_fpr

    if np.any(mask):
        # among FPR < 1%, choose the TPR closest to 0.95 but not smaller if possible
        candidate_tpr = tpr[mask]
        candidate_fpr = fpr[mask]
        # find the one closest to target_tpr
        idx = np.argmin(np.abs(candidate_tpr - target_tpr))
        selected_tpr = candidate_tpr[idx]
        selected_fpr = candidate_fpr[idx]
    else:
        # no TPR under FPR < 1%, choose the TPR closest to 0.95 globally
        idx = np.argmin(np.abs(tpr - target_tpr))
        selected_tpr = tpr[idx]
        selected_fpr = fpr[idx]

    tpr_at_fpr = float(selected_tpr)
    fpr_at_fpr = float(selected_fpr)
    fnr_at_fpr = 1 - tpr_at_fpr
    tnr_at_fpr = 1 - fpr_at_fpr
    
    return {
        "roc_auc": float(roc_auc),

        "worst_benign": worst_benign,

        "tpr_at_fpr": {
            "tpr": tpr_at_fpr,
            "fpr": fpr_at_fpr,
            "tnr": tnr_at_fpr,
            "fnr": fnr_at_fpr
        }
    }

def precision_robustness_stats(
    asr_brain,
    data,
    # precision_types=("fp32", "fp16", "bf16"),
    dataloader_opts=None,
):
    """
    Runs ASR in multiple precisions, computes pairwise WER/CER differences,
    and returns aggregated max/min/median/mean per-utterance statistics.
    """

    # 1) Run ASR + collect predictions
    # pred_words = {}

    # for p in precision_types:
    #     eval_dtype = AMPConfig.from_name(p).dtype
    #     asr_brain.evaluation_ctx = TorchAutocast(
    #         device_type=asr_brain.device,
    #         dtype=eval_dtype,
    #     )

    #     with asr_brain.evaluation_ctx:
    #         pred_words[p] = asr_brain.calculate_wer(
    #             data,
    #             test_loader_kwargs=dataloader_opts,
    #             min_key="WER",
    #         )

    pred_words = asr_brain.calculate_wer(
                    data,
                    test_loader_kwargs=dataloader_opts,
                    min_key="WER",
                )

    preds = {p: to_utt_dict(v) for p, v in pred_words.items()}

    # 2) Pairwise WER/CER
    _, pairwise_scores = pairwise_wer_cer(preds, asr_brain)

    # 3) Reorganize per utterance
    per_utt = defaultdict(lambda: {'WER': [], 'CER': []})

    for metrics in pairwise_scores.values():
        for metric in ['WER', 'CER']:
            for utt_id, score in metrics[metric]:
                per_utt[utt_id][metric].append(score)

    # 4) Aggregate stats
    aggregated = {}

    for utt_id, metrics in per_utt.items():
        aggregated[utt_id] = {
            metric: {
                'max': max(values),
                'min': min(values),
                'median': statistics.median(values),
                'mean': sum(values) / len(values),
            }
            for metric, values in metrics.items()
        }

    # 5) Extract lists
    stats = {
        'WER': {
            'max':    [v['WER']['max'] for v in aggregated.values()],
            'min':    [v['WER']['min'] for v in aggregated.values()],
            'median': [v['WER']['median'] for v in aggregated.values()],
            'mean':   [v['WER']['mean'] for v in aggregated.values()],
        },
        'CER': {
            'max':    [v['CER']['max'] for v in aggregated.values()],
            'min':    [v['CER']['min'] for v in aggregated.values()],
            'median': [v['CER']['median'] for v in aggregated.values()],
            'mean':   [v['CER']['mean'] for v in aggregated.values()],
        },
    }

    return stats, aggregated

def all_metrics(
    stats_gaussian,
    stats_test,
    stats_adv,
):
    """
    Compute AUROC for all WER/CER characteristics using
    Gaussian precision-instability modeling.
    """

    metrics = ['WER', 'CER']
    aggregations =  ['mean'] # ['max', 'min', 'median', 'mean']

    results = {}

    for metric in metrics:
        results[metric] = {}

        for agg in aggregations:
            key = f"{metric}_{agg}"

            train_set = build_char_dataset(
                stats_gaussian[metric][agg],
                benign=True,
                key_name=key
            )

            test_set = build_char_dataset(
                stats_test[metric][agg],
                benign=True,
                key_name=key
            )

            adv_set = build_char_dataset(
                stats_adv[metric][agg],
                benign=False,
                key_name=key
            )

            roc_auc_dict = fit_gaussian(
                train_set=train_set,
                test_set=test_set,
                adv_set=adv_set,
                key=key
            )

            results[metric][agg] = roc_auc_dict # auc

    return results

def load_csv(csv_path):
    return pd.read_csv(csv_path)

def sample_csv(df, n, seed=42):
    if len(df) < n:
        raise ValueError(f"Requested {n} samples but CSV has only {len(df)}")
    return df.sample(n=n, random_state=seed)

def combine_csvs(
    csv_paths,
    sample_sizes,
    output_csv,
    shuffle=True,
    seed=42,
):
    """
    csv_paths: list of paths to csv files
    sample_sizes: list of sample counts per csv (same length)
    output_csv: where to save the merged csv
    """

    assert len(csv_paths) == len(sample_sizes)

    dfs = []
    cnt = 0
    for csv_path, n in zip(csv_paths, sample_sizes):
        df = load_csv(csv_path)
        df = sample_csv(df, n, seed)

        # Rename ID column conditionally
        if Path(csv_path).name == "adv_audio_adv_transcripts.csv":
            if cnt == 0:
                cnt += 1
            else:
                df = df.reset_index(drop=True)
                df["ID"] = (
                    pd.Series(range(1, len(df) + 1))
                    .astype(str)
                    .str.zfill(3)
                )

        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    if shuffle:
        combined = combined.sample(frac=1, random_state=seed).reset_index(drop=True)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_csv, index=False)

    return str(output_csv)

def resolve_csv(spec):
    """
    spec can be:
    - string → already a csv path
    - dict → needs to be combined
    """
    if isinstance(spec, str):
        return spec

    return combine_csvs(
        csv_paths=spec["csvs"],
        sample_sizes=spec["sizes"],
        output_csv=spec["out"],
    )

def load_meas_data(file_path, benign_flg):
    """
    Load a file containing the Characteristics.

    :param file_path: *.pickle file path.
    :param benign_flg: Set 1 to Benign data and 0 to Adversarial data.
    :return: A dictionary containing the Characteristics.
    """
    with open(file_path, "rb") as file:
        measurements = pickle.load(file)
    # Get first key automatically
    first_key = list(measurements.keys())[0]
    total_length = len(measurements[first_key])
    measurements["benign_flg"] = [1 if benign_flg else 0] * total_length
    return measurements

def distriblock_gaussians(train_set, test_set, adv_set, key):
    """
    Fit a Gaussian distribution to each Characteristic score computed for the utterances from a training set of benign data. 
    If the probability of a new audio sample is below a chosen threshold under the Gaussian model, 
    this example is classified as adversarial.

    :param train_set: Training set of benign data.
    :param test_set: Testing set of benign data.
    :param adv_set: Testing set of adversarial data.
    :param key: Characteristic to fit the gaussian.
    :return: Classifier performance in terms of AUROC.
    """
    char_key = []
    char_key.append(key)
    char_key.append("benign_flg")
    test_metrics = {x: test_set[x] for x in char_key}      
    adv_metrics = {x: adv_set[x] for x in char_key}
    # print(np.mean(train_set[key]), np.mean(test_metrics[key]), np.mean(adv_metrics[key]))
    test_all = merge(test_metrics, adv_metrics, char_key)
    # mean, std = norm.fit(train_set[key])

    # train_vals = np.array(train_set[key])
    # test_vals  = np.array(test_all[key])

    # log transform
    # train_vals = np.log(train_vals + 1e-8)
    # test_vals  = np.log(test_vals + 1e-8)

    # mean = np.mean(train_vals)
    # std  = np.std(train_vals)
    # std  = max(std, 1e-6)
    # fitted_norm = norm.pdf(test_vals, loc=mean, scale=std)
    mean = np.mean(train_set[key])
    std  = np.std(train_set[key])
    std  = max(std, 1e-6)   # critical fix

    fitted_norm = norm.pdf(test_all[key], loc=mean, scale=std)

    # fitted_norm = norm.pdf(test_all[key], loc=mean, scale=std)
    fpr, tpr, threshold = roc_curve(test_all['benign_flg'], fitted_norm)
    roc_auc = auc(fpr, tpr)

    return roc_auc

def merge(dict_1, dict_2, key):
    """
    Merge two dictionaries based on specific keys.

    :param dict_1: Dictionary variable.
    :param dict_2: Dictionary variable.
    :param key: Keys to use during the merge.
    :return: A dictionary merged based on specific keys.
    """
    dict_all = {x: dict_1[x] + dict_2[x] for x in key}
    return dict_all

if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

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

    print(f"PRECISION {hparams["precision"]}, EVAL PRECISION {hparams["eval_precision"]}")
    if hparams["adv_type"] == "dist":
        EXPERIMENTS = [
            {
                "name": "full_union_test_adv_resampled",
                "benign": {
                    "csvs": [
                        hparams["benign_audio_test"],
                        hparams["benign_audio_test_other"],
                    ],
                    "sizes": [100, 100],
                    "out": "generated_csvs/benign_full.csv",
                },
                "adv": {
                    "csvs": [
                        hparams["cw_audio_adv_transcripts"],
                        hparams["psy_audio_adv_transcripts"],
                    ],
                    "sizes": [100, 100],
                    "out": "generated_csvs/adv_full.csv",
                },
                "test": {
                    "csvs": [
                        hparams["clean_audio_clean_transcripts"],
                        hparams["clean_audio_test_other_transcripts"],
                    ],
                    "sizes": [100, 100],
                    "out": "generated_csvs/test_full.csv",
                },
            },
        ]
        # adding objects to trainer:
        benign_csv = resolve_csv(EXPERIMENTS[0]["benign"])
        adv_csv    = resolve_csv(EXPERIMENTS[0]["adv"])
        # adv_adapt_csv    = resolve_csv(EXPERIMENTS[0]["adv_adapt"])
        test_csv   = resolve_csv(EXPERIMENTS[0]["test"])

        # # here we create the datasets objects as well as tokenization and encoding
        benign_data_test, label_encoder = dataio_prepare_2(hparams, benign_csv)
        adv_data, _ = dataio_prepare_2(hparams, adv_csv)
        # adv_adapt_data, _ = dataio_prepare_2(hparams, adv_adapt_csv)
        test_data, _ = dataio_prepare_2(hparams, test_csv)
        
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

        # file_names = ["train.pickle", "val.pickle", "test.pickle", "adv_train.pickle", "adv_test.pickle", ]
        # data_sets = [train_set, val_set, test_set, adv_val, adv_test]
        file_names = ["train.pickle", "test.pickle", "adv_test.pickle"]
        data_sets = [benign_data_test, test_data, adv_data]

        characteristic_folder = hparams["distriblock_folder"]
        if not os.path.exists(characteristic_folder):
            os.makedirs(characteristic_folder)

        for i, data in enumerate(data_sets):
            if(not os.path.exists(f"{characteristic_folder}/{file_names[i]}")):
                print(f"Saving characteristics in file: {file_names[i]}!")
                asr_brain.characteristics(
                    data, 
                    f"{characteristic_folder}/{file_names[i]}", 
                    train_loader_kwargs=hparams["test_dataloader_opts"],
                    min_key="WER",
                    )
        if os.path.exists(f"{characteristic_folder}/{file_names[0]}"):
            train_meas = load_meas_data(f"{characteristic_folder}/{file_names[0]}", benign_flg=True)
            keys = []
            for i in train_meas:
                keys.append(i)
        if os.path.exists(f"{characteristic_folder}/{file_names[1]}"):
            test_meas = load_meas_data(f"{characteristic_folder}/{file_names[1]}", benign_flg=True)
        if os.path.exists(f"{characteristic_folder}/{file_names[2]}"):
            adv_meas = load_meas_data(f"{characteristic_folder}/{file_names[2]}", benign_flg=False)
        if (keys == ['Entropy mean', 'Median mean', 'benign_flg']):
            print(" ")
            print("------------------------- Gaussian Classifiers results: ---------------------------")
            print(keys)
            auroc = distriblock_gaussians(train_meas, test_meas, adv_meas, keys[0])
            print("Characteristic: \"{}\". AUROC: {:.4f}".format(keys[0], auroc))
        else:
            sys.exit("-------------Error when Characteristics were calculated-------------")

        print("------------------------- Temporal Dependency calculation: ---------------------------")
        for i, data in enumerate(data_sets):
            if(not os.path.exists(f"{characteristic_folder}/{file_names[i]}_TD")):
                print(f"Saving Temporal Dependency in file: {file_names[i]}_TD!")
                asr_brain.temporal_dependency(
                    data, 
                    f"{characteristic_folder}/{file_names[i]}_TD", 
                    train_loader_kwargs=hparams["test_dataloader_opts"],
                    min_key="WER",
                    )
        if os.path.exists(f"{characteristic_folder}/{file_names[0]}_TD"):
            train_meas = load_meas_data(f"{characteristic_folder}/{file_names[0]}_TD", benign_flg=True)
            keys = []
            for i in train_meas:
                keys.append(i)
        if os.path.exists(f"{characteristic_folder}/{file_names[1]}_TD"):
            test_meas = load_meas_data(f"{characteristic_folder}/{file_names[1]}_TD", benign_flg=True)
        if os.path.exists(f"{characteristic_folder}/{file_names[2]}_TD"):
            adv_meas = load_meas_data(f"{characteristic_folder}/{file_names[2]}_TD", benign_flg=False)
        key = [k for k in train_meas.keys() if k != "benign_flg"][0]
        print(" ")
        print("------------------------- Temporal Dependency results: ---------------------------")
        print("Characteristic:", key)
        auroc = distriblock_gaussians(train_meas, test_meas, adv_meas, key)
        print("AUROC: {:.4f}".format(auroc))

        print("------------------------- Noise Flooding calculation: ---------------------------")
        for i, data in enumerate(data_sets):
            if(not os.path.exists(f"{characteristic_folder}/{file_names[i]}_NF")):
                print(f"Saving noise flooding in file: {file_names[i]}_NF!")
                asr_brain.noise_flooding(
                    data, 
                    f"{characteristic_folder}/{file_names[i]}_NF", 
                    train_loader_kwargs=hparams["test_dataloader_opts"],
                    min_key="WER",
                    )
        if os.path.exists(f"{characteristic_folder}/{file_names[0]}_NF"):
            train_meas = load_meas_data(f"{characteristic_folder}/{file_names[0]}_NF", benign_flg=True)
            keys = []
            for i in train_meas:
                keys.append(i)
        if os.path.exists(f"{characteristic_folder}/{file_names[1]}_NF"):
            test_meas = load_meas_data(f"{characteristic_folder}/{file_names[1]}_NF", benign_flg=True)
        if os.path.exists(f"{characteristic_folder}/{file_names[2]}_NF"):
            adv_meas = load_meas_data(f"{characteristic_folder}/{file_names[2]}_NF", benign_flg=False)
        key = [k for k in train_meas.keys() if k != "benign_flg"][0]
        print(" ")
        print("------------------------- Noise Flooding results: ---------------------------")
        print("Characteristic:", key)
        auroc = distriblock_gaussians(train_meas, test_meas, adv_meas, key)
        print("AUROC: {:.4f}".format(auroc))

    else:        
        EXPERIMENTS = [
            # {
            #     "name": "baseline_cw",
            #     "benign": hparams["benign_audio_test"],
            #     "adv": hparams["cw_audio_adv_transcripts"],
            #     "test": hparams["clean_audio_clean_transcripts"],
            # },
            # {
            #     "name": "baseline_psy",
            #     "benign": hparams["benign_audio_test"],
            #     "adv": hparams["psy_audio_adv_transcripts"],
            #     "test": hparams["clean_audio_clean_transcripts"],
            # },
            # {
            #     "name": "adaptive cw",
            #     "benign": {
            #         "csvs": [
            #             hparams["benign_audio_test"],
            #             hparams["benign_audio_test_other"],
            #         ],
            #         "sizes": [100, 100],
            #         "out": "generated_csvs/benign_50_50.csv",
            #     },
            #     "adv": hparams["cw_audio_adv_adapt_transcripts"],
            #     "test": {
            #         "csvs": [
            #             hparams["clean_audio_clean_transcripts"],
            #             hparams["clean_audio_test_other_transcripts"],
            #         ],
            #         "sizes": [50, 50],
            #         "out": "generated_csvs/test_50_50_cw.csv",
            #     },
            # },
            {
                "name": "benign_50_50_clean_50_50_cw",
                "benign": {
                    "csvs": [
                        hparams["benign_audio_test"],
                        hparams["benign_audio_test_other"],
                    ],
                    "sizes": [100, 100],
                    "out": "generated_csvs/benign_50_50.csv",
                },
                "adv": hparams["cw_audio_adv_transcripts"],
                "test": {
                    "csvs": [
                        hparams["clean_audio_clean_transcripts"],
                        hparams["clean_audio_test_other_transcripts"],
                    ],
                    "sizes": [50, 50],
                    "out": "generated_csvs/test_50_50_cw.csv",
                },
            },
            {
                "name": "benign_50_50_clean_50_50_psy",
                "benign": {
                    "csvs": [
                        hparams["benign_audio_test"],
                        hparams["benign_audio_test_other"],
                    ],
                    "sizes": [100, 100],
                    "out": "generated_csvs/benign_50_50.csv",
                },
                "adv": hparams["psy_audio_adv_transcripts"],
                "test": {
                    "csvs": [
                        hparams["clean_audio_clean_transcripts"],
                        hparams["clean_audio_test_other_transcripts"],
                    ],
                    "sizes": [50, 50],
                    "out": "generated_csvs/test_50_50_psy.csv",
                },
            },
            {
                "name": "full_union_test_adv_resampled",
                "benign": {
                    "csvs": [
                        hparams["benign_audio_test"],
                        hparams["benign_audio_test_other"],
                    ],
                    "sizes": [100, 100],
                    "out": "generated_csvs/benign_full.csv",
                },
                "adv": {
                    "csvs": [
                        hparams["cw_audio_adv_transcripts"],
                        hparams["psy_audio_adv_transcripts"],
                    ],
                    "sizes": [100, 100],
                    "out": "generated_csvs/adv_full.csv",
                },
                "test": {
                    "csvs": [
                        hparams["clean_audio_clean_transcripts"],
                        hparams["clean_audio_test_other_transcripts"],
                    ],
                    "sizes": [100, 100],
                    "out": "generated_csvs/test_full.csv",
                },
            },
        ]
        for exp in EXPERIMENTS:
            print(f"\n===== Running experiment: {exp['name']} =====")

            benign_csv = resolve_csv(exp["benign"])
            adv_csv    = resolve_csv(exp["adv"])
            test_csv   = resolve_csv(exp["test"])

            # # here we create the datasets objects as well as tokenization and encoding
            benign_data_test, label_encoder = dataio_prepare_2(hparams, benign_csv)
            adv_data, _ = dataio_prepare_2(hparams, adv_csv)
            test_data, _ = dataio_prepare_2(hparams, test_csv)
            
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

            stats_gaussian, aggregated_gaussian = precision_robustness_stats(
                                                                            asr_brain,
                                                                            benign_data_test,
                                                                            dataloader_opts=hparams["test_dataloader_opts"],
                                                                            )
            
            stats_test, aggregated_test = precision_robustness_stats(
                                                                            asr_brain,
                                                                            test_data,
                                                                            dataloader_opts=hparams["test_dataloader_opts"],
                                                                            )

            stats_adv, aggregated_adv = precision_robustness_stats(
                                                                            asr_brain,
                                                                            adv_data,
                                                                            dataloader_opts=hparams["test_dataloader_opts"],
                                                                            )

            auc_results = all_metrics(
                stats_gaussian=stats_gaussian,
                stats_test=stats_test,
                stats_adv=stats_adv
            )
            # roc_auc, min_fnr, corresponding_tpr, corresponding_fpr
            # print(auc_results)

            agg = "mean"

            wer_vals = auc_results["WER"][agg]
            cer_vals = auc_results["CER"][agg]

            # AUROC
            wer_auc = wer_vals["roc_auc"]
            cer_auc = cer_vals["roc_auc"]

            # Worst benign
            wer_wb = wer_vals["worst_benign"]
            cer_wb = cer_vals["worst_benign"]

            wer_at1 = wer_vals["tpr_at_fpr"]
            cer_at1 = cer_vals["tpr_at_fpr"]

            print(
                f"[{exp['name']}] "
                f"AUROC={wer_auc:.2f}/{cer_auc:.2f} | "
                f"WorstBenign("
                f"TPR={wer_wb['tpr']:.2f}/{cer_wb['tpr']:.2f}, "
                f"FPR={wer_wb['fpr']:.2f}/{cer_wb['fpr']:.2f}, "
                f"TNR={wer_wb['tnr']:.2f}/{cer_wb['tnr']:.2f}, "
                f"FNR={wer_wb['fnr']:.2f}/{cer_wb['fnr']:.2f}"
                f") | "
                f"TPR@FPR<1%("
                f"TPR={wer_at1['tpr']:.2f}/{cer_at1['tpr']:.2f}, "
                f"FPR={wer_at1['fpr']:.2f}/{cer_at1['fpr']:.2f}, "
                f"TNR={wer_at1['tnr']:.2f}/{cer_at1['tnr']:.2f}, "
                f"FNR={wer_at1['fnr']:.2f}/{cer_at1['fnr']:.2f}"
                f")"
            )
