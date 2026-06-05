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

import os
import sys
import torch
import logging
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from hyperpyyaml import load_hyperpyyaml
from pathlib import Path
from torch.utils.data import DataLoader
from speechbrain.dataio.dataloader import LoopedLoader
from enum import Enum, auto
import numpy as np
import torch.nn as nn
from tqdm.contrib import tqdm
import torchaudio
import csv
import linecache
from scipy.stats import entropy
import torch.nn.functional as F
from scipy.special import rel_entr

logger = logging.getLogger(__name__)

class Stage(Enum):
    """Simple enum to track stage of experiments."""
    ATTACK = auto()

# Define training procedure
class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_bos, _ = batch.tokens_bos
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)

        # Add augmentation if specified
        if stage == sb.Stage.TRAIN:
            if hasattr(self.modules, "env_corrupt"):
                wavs_noise = self.modules.env_corrupt(wavs, wav_lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                wav_lens = torch.cat([wav_lens, wav_lens])
                tokens_bos = torch.cat([tokens_bos, tokens_bos], dim=0)

            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wav_lens)

        # Forward pass
        feats = self.hparams.compute_features(wavs)
        feats = self.modules.normalize(feats, wav_lens)
        if(stage == Stage.ATTACK):
            x = self.modules.enc(feats)
        else:
            x = self.modules.enc(feats.detach())
        e_in = self.modules.emb(tokens_bos)  # y_in bos + tokens
        h, _ = self.modules.dec(e_in, x, wav_lens)

        # Output layer for seq2seq log-probabilities
        logits = self.modules.seq_lin(h)
        p_seq = self.hparams.log_softmax(logits)

        # Compute outputs
        if stage == sb.Stage.TRAIN or stage == Stage.ATTACK:
            current_epoch = self.hparams.epoch_counter.current
            if current_epoch <= self.hparams.number_of_ctc_epochs:
                # Output layer for ctc log-probabilities
                logits = self.modules.ctc_lin(x)
                p_ctc = self.hparams.log_softmax(logits)
                return p_ctc, p_seq, wav_lens
            else:
                return p_seq, wav_lens
        else:
            if stage == sb.Stage.VALID:
                p_tokens, scores = self.hparams.valid_search(x, wav_lens)
            else:
                p_tokens, scores = self.hparams.valid_search(x, wav_lens)
                # p_tokens, scores = self.hparams.test_search(x, wav_lens)
            return p_seq, wav_lens, p_tokens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        current_epoch = self.hparams.epoch_counter.current
        if stage == sb.Stage.TRAIN or stage == Stage.ATTACK:
            if current_epoch <= self.hparams.number_of_ctc_epochs:
                p_ctc, p_seq, wav_lens = predictions
            else:
                p_seq, wav_lens = predictions
        else:
            p_seq, wav_lens, predicted_tokens = predictions

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens

        if hasattr(self.modules, "env_corrupt") and stage == sb.Stage.TRAIN:
            tokens_eos = torch.cat([tokens_eos, tokens_eos], dim=0)
            tokens_eos_lens = torch.cat(
                [tokens_eos_lens, tokens_eos_lens], dim=0
            )
            tokens = torch.cat([tokens, tokens], dim=0)
            tokens_lens = torch.cat([tokens_lens, tokens_lens], dim=0)

        loss_seq = self.hparams.seq_cost(
            p_seq, tokens_eos, length=tokens_eos_lens
        )

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
        else:
            loss = loss_seq

        if stage not in [sb.Stage.TRAIN, Stage.ATTACK]:
            # Decode token terms to words
            predicted_words = [
                self.tokenizer.decode_ids(utt_seq).split(" ")
                for utt_seq in predicted_tokens
            ]
            target_words = [wrd.split(" ") for wrd in batch.wrd]
            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)

        return loss

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)
        loss.backward()
        if self.check_gradients(loss):
            self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.detach()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        predictions = self.compute_forward(batch, stage=stage)
        with torch.no_grad():
            loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

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
                meta={"WER": stage_stats["WER"]}, min_keys=["WER"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.wer_file, "w") as w:
                self.wer_metric.write_stats(w)

    def CW_initialize_vars(self):
        self.eps = 1  # 0.05
        self.max_iter_1 = 1000 # 4000  # 4000 # 10
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
        self.alpha = 0.3
        self._optimizer_arg_1 = None

    def attack_adapt_CW(
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
        # torch.backends.cudnn.enabled = False
        for m in self.modules.modules():
            if m.__class__.__name__.startswith('LSTM'):
                m.train()
            if isinstance(m, nn.Dropout):
                m.p = 0
            elif isinstance(m, nn.LSTM):
                m.dropout = 0
            elif isinstance(m, nn.GRU):
                m.dropout = 0
        for module in [self.modules.enc, self.modules.emb, self.modules.dec, self.modules.seq_lin,
                       self.modules.ctc_lin]:
            # , self.modules.normalize, self.modules.env_corrupt, self.modules.lm_model]:
            for p in module.parameters():
                p.requires_grad = False
        self.sample_rate = hparams["sample_rate"]

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
                if (not os.path.exists(i.replace(hparams["path_adv"], hparams["path_adapt"])) or True):
                    # print(i.replace(hparams["path_adv"], hparams["path_adapt"]))
                    result = self.attack_1st_stage(batch, hparams)
    
    def attack_1st_stage(self, batch, hparams):
        """
        The first stage of the attack.
        """
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
        best_loss_2nd_stage = [np.inf] * local_batch_size
        best_score: List[Optional["torch.Tensor"]] = [None] * local_batch_size

        for iter_1st_stage_idx in range(self.max_iter_1):
            # Zero the parameter gradients
            self.optimizer_1.zero_grad()
            # Call to forward pass
            (
                loss,
                loss_2, 
                characteristic,
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
            loss.backward()
            # Get sign of the gradients
            self.global_optimal_delta.grad = torch.sign(self.global_optimal_delta.grad)
            # Do optimization
            self.optimizer_1.step()

            for local_batch_size_idx in range(local_batch_size):
                almost_successful[local_batch_size_idx] = masked_adv_input[local_batch_size_idx]
                torchaudio.save("adv_ex_3.flac", almost_successful[local_batch_size_idx][ :real_lengths[local_batch_size_idx]].cpu()[None, :], self.sample_rate)
                data_adv, sample_rate = torchaudio.load("adv_ex_3.flac")
                batch.sig[0][local_batch_size_idx][:real_lengths[local_batch_size_idx]] = data_adv
            
            p_seq, wav_lens, best_hyps = self.compute_forward(batch, stage=sb.Stage.TEST)

            for local_batch_size_idx in range(local_batch_size):
                tokens = (
                    batch.tokens[0][local_batch_size_idx,  0:token_lenghts[local_batch_size_idx]]
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                )
                pred_test = np.array(best_hyps[local_batch_size_idx])
                if len(pred_test) == len(tokens) and (pred_test == tokens).all():  
                    if (loss_2.detach() < best_loss_2nd_stage[local_batch_size_idx]): 
                        best_loss_2nd_stage[local_batch_size_idx] = loss_2.detach()
                        self.alpha = min(self.alpha*1.2, 0.999999999)
                        best_eta[local_batch_size_idx] = rescale[local_batch_size_idx] * self.eps
                        # Adjust the rescale coefficient
                        # self.alpha = min(self.alpha*1.2, 0.999999999)
                        if iter_1st_stage_idx > 30:
                            max_local_delta = np.max(
                                np.abs(local_delta[local_batch_size_idx].detach().cpu().numpy())
                            )
                            if (rescale[local_batch_size_idx][0] * self.eps > max_local_delta):
                                rescale[local_batch_size_idx] = max_local_delta / self.eps
                            rescale[local_batch_size_idx] *= self.decrease_factor_eps
                        # print(rescale)
                        # Save the best adversarial example
                        if successful_adv_input_2[local_batch_size_idx] is None:
                            first_hit[local_batch_size_idx] = iter_1st_stage_idx
                        # masked_adv_input[local_batch_size_idx] = batch.sig[0][local_batch_size_idx][:real_lengths[local_batch_size_idx]]
                        successful_adv_input_2[local_batch_size_idx] = masked_adv_input[local_batch_size_idx]
                        best_hit[local_batch_size_idx] = iter_1st_stage_idx
                        if count_succs[local_batch_size_idx] is None:
                            count_succs[local_batch_size_idx] = 1
                        else:
                            count_succs[local_batch_size_idx] += 1
                        best_score[local_batch_size_idx] = characteristic
            # If attack is unsuccessful
            if iter_1st_stage_idx == self.max_iter_1 - 1:
                # print("Entro 2")
                for (local_batch_size_idx, dirct) in enumerate(batch.path):
                    dirct = dirct.replace(hparams["path_adv"], hparams["path_adapt"])
                # for local_batch_size_idx in range(local_batch_size):
                    if successful_adv_input_2[local_batch_size_idx] is None:
                        successful_adv_input_2[local_batch_size_idx] = masked_adv_input[local_batch_size_idx]
                        # trans[local_batch_size_idx] = decoded_output[local_batch_size_idx]
                        with open(hparams["unsuccesfull_adapt"], 'a') as myfile:
                            wr = csv.writer(myfile)                            
                            wr.writerow([[dirct], [first_hit[local_batch_size_idx]], [best_hit[local_batch_size_idx]], 
                                [best_eta[local_batch_size_idx]], [count_succs[local_batch_size_idx]], [self.alpha], [characteristic.cpu().detach().item()]])
                            myfile.close()
                    else:
                        with open(hparams["succesfull_adapt"], 'a') as myfile:
                            wr = csv.writer(myfile)
                            wr.writerow([[dirct], [first_hit[local_batch_size_idx]], [best_hit[local_batch_size_idx]], 
                                [best_eta[local_batch_size_idx][0]], [count_succs[local_batch_size_idx]], [self.alpha], [best_score[local_batch_size_idx].cpu().detach().item()]])
                            myfile.close()
                    # print(dirct)
                    torchaudio.save(dirct, successful_adv_input_2[local_batch_size_idx][ :real_lengths[local_batch_size_idx]].cpu()[None, :], self.sample_rate)
            # '''
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
        loss_cw = self.compute_objectives(predictions, batch, Stage.ATTACK)
        # print("loss: ", loss)
        loss_1 = self.const * loss_cw + torch.norm(local_delta_rescale)
        # Characteristic mean Entropy
        p_ctc = torch.squeeze(predictions[0], dim=0)
        p_ctc_prob = torch.exp(p_ctc)
        condition = p_ctc_prob != 0
        row_cond = condition.all(1)
        p_ctc_prob = p_ctc_prob[row_cond, :]
        condition = p_ctc_prob != 1
        row_cond = condition.all(1)
        p_ctc_prob = p_ctc_prob[row_cond, :]

        p_ctc_prob, _ = torch.median(p_ctc_prob, axis=1)
        p_ctc_prob = torch.log(p_ctc_prob)
        characteristic = torch.mean(p_ctc_prob)
        # print(characteristic)
        '''
        # p_ctc_prob = p_ctc_prob * torch.log(p_ctc_prob)
        # characteristic = -1 * torch.mean(torch.sum(p_ctc_prob, axis=1))
        d_sym = []
        for i in range(p_ctc_prob.shape[0] - 1):
            right = F.kl_div(torch.log(p_ctc_prob[i + 1, :]), p_ctc_prob[i, :], reduction = "sum")
            if torch.isinf(right) or right == 0:
                cnt += 1
                print(right)
            else:
                d_sym.append(right.unsqueeze(0))
        characteristic = torch.cat(d_sym)
        characteristic = torch.mean(characteristic)
        '''
        '''
        # print(characteristic)
        p_ctc_prob_2 = torch.exp(p_ctc).detach().cpu()
        p_ctc_prob_2 = np.array(p_ctc_prob_2)
        p_ctc_prob_2 = np.delete(p_ctc_prob_2, np.where((p_ctc_prob_2 == 0))[0], axis=0)    
        p_ctc_prob_2 = np.delete(p_ctc_prob_2, np.where((p_ctc_prob_2 == 1))[0], axis=0)   
       
        kl_sym = []
        for i in range(p_ctc_prob_2.shape[0] - 1):
            kl_sym.append(np.sum(rel_entr(p_ctc_prob_2[i, :], p_ctc_prob_2[i + 1, :])))       
        kl_sym = np.array(kl_sym)
        kld_avg = np.mean(kl_sym)
        print(characteristic, kld_avg)
        '''
        relu = torch.nn.ReLU()
        loss_2 = relu(torch.abs(torch.tensor(-23.0001) - characteristic))
        loss = (1-self.alpha) * loss_1 + self.alpha * loss_2

        return loss, loss_2, characteristic, local_delta, masked_adv_input, local_delta_rescale

def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder},
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
        csv_path=hparams["valid_csv"], replacements={"data_root": data_folder},
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
        datasets, ["id", "sig", "wrd", "tokens_bos", "tokens_eos", "tokens"],
    )
    train_batch_sampler = None
    valid_batch_sampler = None
    if hparams["dynamic_batching"]:
        from speechbrain.dataio.sampler import DynamicBatchSampler  # noqa
        from speechbrain.dataio.dataloader import SaveableDataLoader  # noqa
        from speechbrain.dataio.batch import PaddedBatch  # noqa

        dynamic_hparams = hparams["dynamic_batch_sampler"]
        hop_size = dynamic_hparams["feats_hop_size"]

        num_buckets = dynamic_hparams["num_buckets"]

        train_batch_sampler = DynamicBatchSampler(
            train_data,
            dynamic_hparams["max_batch_len"],
            num_buckets=num_buckets,
            length_func=lambda x: x["duration"] * (1 / hop_size),
            shuffle=dynamic_hparams["shuffle_ex"],
            batch_ordering=dynamic_hparams["batch_ordering"],
        )

        valid_batch_sampler = DynamicBatchSampler(
            valid_data,
            dynamic_hparams["max_batch_len"],
            num_buckets=num_buckets,
            length_func=lambda x: x["duration"] * (1 / hop_size),
            shuffle=dynamic_hparams["shuffle_ex"],
            batch_ordering=dynamic_hparams["batch_ordering"],
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
        datasets, ["id", "sig", "wrd", "tokens_bos", "tokens_eos", "tokens", "path"],
    )

    return train_data

if __name__ == "__main__":

    print("AA ", torch.cuda.device_count())
    use_cuda = torch.cuda.is_available()
    print(use_cuda)
    if use_cuda:
        print('__CUDNN VERSION:', torch.backends.cudnn.version())
        print('__Number CUDA Devices:', torch.cuda.device_count())
        print('__CUDA Device Name:',torch.cuda.get_device_name(0))
        print('__CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory)
    
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # If --distributed_launch then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

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

    # here we create the datasets objects as well as tokenization and encoding
    (
        train_data,
        valid_data,
        test_datasets,
        train_bsampler,
        valid_bsampler,
    ) = dataio_prepare(hparams)
   
    # We download the pretrained LM from HuggingFace (or elsewhere depending on
    # the path given in the YAML file). The tokenizer is loaded at the same time.
    run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected(device=run_opts["device"])

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # We dynamicaly add the tokenizer to our brain class.
    # NB: This tokenizer corresponds to the one used for the LM!!
    asr_brain.tokenizer = hparams["tokenizer"]
    train_dataloader_opts = hparams["train_dataloader_opts"]
    valid_dataloader_opts = hparams["valid_dataloader_opts"]

    if train_bsampler is not None:
        train_dataloader_opts = {"batch_sampler": train_bsampler}
    if valid_bsampler is not None:
        valid_dataloader_opts = {"batch_sampler": valid_bsampler}

    
    """    
    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=train_dataloader_opts,
        valid_loader_kwargs=valid_dataloader_opts,
    )
    """
    """
    # Testing
    for k in test_datasets.keys():  # keys are test_clean, test_other etc
        asr_brain.hparams.wer_file = os.path.join(
            hparams["output_folder"], "wer_{}.txt".format(k)
        )
        asr_brain.evaluate(
            test_datasets[k], test_loader_kwargs=hparams["test_dataloader_opts"]
        )
    """
    
    cw_data = dataio_prepare_2(hparams, hparams["adversarial_transcriptions_2"])
    asr_brain.attack_adapt_CW(
        cw_data,
        hparams=hparams,
        train_loader_kwargs=hparams["test_dataloader_opts"],
        )
    
