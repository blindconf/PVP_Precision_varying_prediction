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
import torch.nn.functional as F
import copy

logger = logging.getLogger(__name__)

ELITE_SIZE = 2
TEMPERATURE = 0.02
MUTATION_PROB_INIT = 0.0005
EPS_NUM_STRIDES = 4
ALPHA_MOMENTUM = 0.99
EPS_MOMENTUM = 0.001

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
            return p_seq, wav_lens, p_tokens

    def compute_objectives(self, predictions, batch, stage, reduction="mean"):
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
            p_seq, tokens_eos, length=tokens_eos_lens, reduction=reduction
        )

        # Add ctc loss if necessary
        if (
            stage == sb.Stage.TRAIN
            and current_epoch <= self.hparams.number_of_ctc_epochs
        ):
            loss_ctc = self.hparams.ctc_cost(
                p_ctc, tokens, wav_lens, tokens_lens, reduction=reduction
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
        self.sample_rate = hparams["sample_rate"]

        for batch in tqdm(train_set, dynamic_ncols=True, disable=not progressbar):
            batch = batch.to(self.device)
            for j in batch.path:
                if (not os.path.exists(j.replace(hparams["path_src"], hparams["path_kens"]).replace("flac", "wav"))):
                    self.snr = 15  
                    self.threshold = 10 ** (-self.snr / 10)
                    print("self.threshold: SNR not in decibels ", self.threshold)
                    wavs, rel_lengths = batch.sig
                    wavs = wavs.detach().clone()
                    batch_size = wavs.size(0)
                    wav_lengths = (rel_lengths.float() * wavs.size(1)).long()

                    for i in range(batch_size):
                        wav, len_wav = wavs[i, : wav_lengths[i]], wav_lengths[i]
                        wav_rfft = torch.fft.rfft(wav)
                        wav_psd = torch.abs(wav_rfft) ** 2
                        print("shape : ", wav_psd.shape)
                        if len(wav) % 2:  # odd: DC frequency
                            wav_psd[1:] *= 2
                        else:  # even: DC and Nyquist frequencies
                            wav_psd[1:-1] *= 2

                        # Scale the threshold based on the power of the signal
                        # Find frequencies in order with cumulative perturbation less than threshold
                        #     Sort frequencies by power density in ascending order
                        wav_psd_index = torch.argsort(wav_psd)
                        reordered = wav_psd[wav_psd_index]
                        cumulative = torch.cumsum(reordered, dim=0)
                        norm_threshold = self.threshold * cumulative[-1]
                        j_dec = torch.searchsorted(cumulative, norm_threshold, right=True)
                        print("a", a)
                        # Zero out low power frequencies and invert to time domain
                        wav_rfft[wav_psd_index[:j_dec]] = 0
                        wav = torch.fft.irfft(wav_rfft, len(wav)).type(wav.dtype)
                        wavs[i, :len_wav] = wav
                    
                    result = wavs
                    root_path = hparams["path_kens"]
                    file_name = os.path.basename(j)
                    dirct = os.path.join(root_path, file_name).replace("flac", "wav")
                    with open(hparams["succesfull_kens"], 'a') as myfile:
                            wr = csv.writer(myfile)
                            wr.writerow([[dirct]])
                            myfile.close()
                    torchaudio.save(dirct, result.cpu(), self.sample_rate)
                    # torchaudio.save(dirct, result.cpu()[None, :], self.sample_rate)
                    self.eps = 1.0

    def fft_compression(self, path,audio_image,factor,fs):
        '''
        # DFT Attack
        # path: path to audio file
        # Audio_image: audio file as an np.array object
        # factor: the intensity below which you want to zero out
        # fs: sample rate
        '''
        # Take FFT
        fft_image = torch.fft.rfft(audio_image) 
        # Zero out values below threshold
        fft_image[torch.abs(fft_image) < factor] = 0
        
        # inverse fft
        ifft_audio = torch.fft.irfft(fft_image, len(audio_image[0])).type(audio_image.detach().dtype) 
        # New file name
        new_audio_path = path[0:-4]+'_'+str(fs)+'_FFT_'+str(factor)+'.wav'
        return new_audio_path, ifft_audio
        
    def perturb(self,
                og_audio_path,
                audio,
                atk_name,
                fs, 
                factor,
            ):
        frame = audio  
        path, perturbed_frame= self.fft_compression(og_audio_path,frame,factor,fs)

        return path, perturbed_frame.ravel()

    def bst_atk_factor(self, min_atk, max_atk, val_atk, succ):
        '''
        # For searching the best attack factor using binary search
        # For DCT, decrease factor if evasion success, increase other wise
        # For SSA, SVD and PCA, increase factor if evasion success, decrease other wise
        '''
        init_val_atk = val_atk
        if(succ):
            max_atk = val_atk
            val_atk = abs(min_atk + max_atk) / 2
        else:
            min_atk = val_atk
            val_atk = abs(min_atk + max_atk) / 2
        # return int(min_atk),int(max_atk),int(val_atk),(init_val_atk==int(val_atk)) 
        return int(min_atk), int(max_atk), int(val_atk), (int(init_val_atk)==int(val_atk)) 

    def normalize(self, data):
        normalized = data*1.0/torch.max(torch.abs(data))
        magnitude = torch.abs(normalized)
        return magnitude

    # MSE between audio samples
    def diff_avg(self, audio1, audio2):
        # Normalize
        n_audio1 = self.normalize(audio1)
        n_audio2 = self.normalize(audio2)
        # Diff
        diff = n_audio1 - n_audio2
        abs_diff = torch.abs(diff)
        overall_change = torch.sum(abs_diff)
        average_change = overall_change/len(audio1)
        return average_change

    def attack_adapt_kn(
            self,
            test_set,
            hparams,
            max_key=None,
            min_key=None,
            progressbar=None,
            test_loader_kwargs={},
    ):
        if progressbar is None:
            progressbar = not self.noprogressbar

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

        self.sample_rate = hparams["sample_rate"]
        cnt = 0
        for batch in tqdm(test_set, dynamic_ncols=True, disable=not progressbar):
            batch = batch.to(self.device)
            root_path = hparams["path_kens"]
            file_name = os.path.basename(batch.path[0])
            dirct = os.path.join(root_path, file_name).replace("flac", "wav")
            if (not os.path.exists(dirct) or True):
                # Initialize iteration counter
                itr = 0
                max_allowed_iterations = 15
                og_audio_path = batch.path[0]
                # Need the min var for BST
                min_attack_factor = torch.tensor(0)
                # Read file to attack
                frame_to_perturb = batch.sig[0]
                perturbed_audio = torch.clone(batch.sig[0])
                _attack_name = 'fft'
                # max_attack_factor = max(abs(sc.fftpack.fft(frame_to_perturb)))
                max_attack_factor = torch.max(torch.abs(torch.fft.rfft(frame_to_perturb))) 
                _attack_factor= max_attack_factor/2
                _raster_width = 100
                # '''
                predictions = self.compute_forward(batch, sb.Stage.TEST)
                labeling = predictions[2]

                target_words = [wrd.split(" ") for wrd in batch.wrd]

                og_label = (
                    batch.tokens[0]
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                )
                
                if (labeling[0] != batch.tokens[0].detach().cpu().reshape(-1).tolist()):
                    print(labeling[0])
                    print(batch.tokens[0].detach().cpu().reshape(-1).tolist())
                    
                    # print(batch.path)
                    
                    # print(labeling)
                    # print(batch.tokens[0].detach().cpu().tolist())
                    predicted_words = [
                                        self.tokenizer.decode_ids(utt_seq).split(" ") for utt_seq in labeling
                                    ]
                    # predicted_words = ["".join(p) for p in predicted_words]
                    # target_words = ["".join(t) for t in target_words]
                    print(predicted_words, type(predicted_words))
                    print(target_words, type(target_words), predicted_words != target_words)
                    if predicted_words != target_words :
                        og_label = np.array(labeling[0])                    
                        cnt += 1
                
                flg = False
                # print("_attack_factor: ", _attack_factor)
                while(itr < max_allowed_iterations):
                    # This variable is written to the dataframe to show the last iteration
                    bst = False
                    # Attack!!
                    perturbed_audio_path, perturbed_audio = self.perturb(og_audio_path,
                        audio = frame_to_perturb,
                        atk_name = _attack_name,
                        fs = self.sample_rate, 
                        factor = _attack_factor,
                    )

                    perturbed_audio_path = perturbed_audio_path[:-4]+'_BST.wav'
                    torchaudio.save("adv_ex.wav", perturbed_audio.cpu()[None, :], self.sample_rate)
                    perturbed_audio, _ = torchaudio.load("adv_ex.wav")
                    batch.sig = perturbed_audio, batch.sig[1]
                     # Transcribe
                    transcribed_perturbation = self.compute_forward(batch, sb.Stage.TEST)
                    trans_label = transcribed_perturbation[2]
                    perturbed_audio = perturbed_audio.to(self.device)
                    # trans_label_2 = trans_label
                    trans_label = np.array(trans_label[0])

                    if(len(og_label) != len(trans_label) or (og_label == trans_label).all() == False):
                        predicted_words = [
                            self.tokenizer.decode_ids(utt_seq).split(" ") for utt_seq in transcribed_perturbation[2]
                        ]
                        
                        if target_words != predicted_words:   
                            mistranscribed_audio = perturbed_audio
                            flg = True
                            succ = True
                    else:
                        succ = False
                
                    # Adjust max and min factor varaibles
                    new_min_attack_factor,new_max_attack_factor,new_attack_factor,complete = \
                    self.bst_atk_factor(min_atk = min_attack_factor ,max_atk = \
                                max_attack_factor ,val_atk = _attack_factor, succ= succ)


                    min_attack_factor,max_attack_factor,_attack_factor = \
                        new_min_attack_factor,new_max_attack_factor,new_attack_factor 
                    # print("_attack_factor: ", _attack_factor)
                    
                    # Distances between original and perturbe audio file
                    avg = self.diff_avg(frame_to_perturb,perturbed_audio)
                    if(complete): break
                    
                    itr = itr + 1

                # torch.cuda.empty_cache()
                if flg != True:
                    print(batch.path)
                result = mistranscribed_audio                 
                with open(hparams["succesfull_kens"], 'a') as myfile:
                        wr = csv.writer(myfile)
                        wr.writerow([[dirct], [itr], [avg.cpu().detach().item()], [_attack_factor]])
                        myfile.close()
                torchaudio.save(dirct, mistranscribed_audio.cpu(), self.sample_rate)
            # break
        print("cnt: ", cnt)

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
    
    
    # Testing
    for k in test_datasets.keys():  # keys are test_clean, test_other etc
        asr_brain.hparams.wer_file = os.path.join(
            hparams["output_folder"], "wer_{}.txt".format(k)
        )
        asr_brain.evaluate(
            test_datasets[k], test_loader_kwargs=hparams["test_dataloader_opts"]
        )
    """
    
    cw_data = dataio_prepare_2(hparams, hparams["source_transcriptions"])
    asr_brain.attack_adapt_kn(
        cw_data,
        hparams=hparams,
        test_loader_kwargs=hparams["test_dataloader_opts"],
        )
    
    