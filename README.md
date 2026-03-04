# Precision-Varying Prediction (PVP)
# [INTERSPEECH 2026] Precision-Varying Prediction (PVP): Robustifying ASR systems against adversarial attacks



This paper [Precision-Varying Prediction (PVP): Robustifying ASR systems against adversarial attacks](https://arxiv.org/abs/2305.17000) is submitted to Interspeech 2026. A demo with a selection of benign, adversarial, and noisy data employed in our experiments is available online: [PVP Demo](Link: ).
>  With the increasing deployment of automated and agentic systems, ensuring the adversarial robustness of Automatic Speech Recognition (ASR) models has become critical. We observe that changing the precision of an ASR model during inference reduces the likelihood of adversarial attacks to succeed. We take advantage of this fact to make the models more robust by simple random sampling of the precision during prediction. Moreover, the insight can be turned into an adversarial example detection strategy by comparing outputs resulting from different precisions and leveraging a simple Gaussian classifier.
     An experimental analysis demonstrates a significant increase in robustness and competitive detection performance for various ASR models and attack types. 
## Prerequisites
Before running the PVP scripts, the following tasks need to be completed:
1. SpeechBrain installation
2. Download datasets
3. Download pre-trained ASR models
4. Generate adversarial examples
   
#### 1. SpeechBrain installation
We analyze a variety of fully integrated PyTorch-based deep learning E2E speech engines using [SpeechBrain](https://github.com/speechbrain/speechbrain). 
Please refer to their website for instructions on how to install it.
We perform evaluations of our detectors using an NVIDIA A40 GPU with 48 GB of memory, along with ASR recipes from SpeechBrain version 0.5.14.

#### 2. Download datasets
We use the following large-scale speech corpus:
* [LibriSpeech (English)](https://www.openslr.org/12)

#### 3. Download pre-trained ASR models
We provide our [pre-trained models](https://ruhr-uni-bochum.sciebo.de/s/lpjW0vxFikG2WqD) that can be used to generate adversarial examples and to test our PVP defense strategy.
In addition, Speechbrain contains pre-trained models that can also be used:

* [CRDNN with CTC/Attention trained on LibriSpeech](https://huggingface.co/speechbrain/asr-crdnn-rnnlm-librispeech)

* [wav2vec 2.0 with CTC trained on LibriSpeech]((https://huggingface.co/speechbrain/asr-wav2vec2-librispeech))

* [Transformer trained on LibriSpeech](https://huggingface.co/speechbrain/asr-transformer-transformerlm-librispeech)
* Wh
  
#### 4. Adversarial attacks
To generate Adversarial Examples in different precisions,

As an example, csv files for Librispeech corpus and its corresponding CW adversarial examples, along with the target transcription, are provided in the results folder. These files can also be used to generate the adversarial examples. 
The data should be stored using the following folder structure:
```
data_set
└──librispeech
    ├── test-clean
    ├── cw
    ├── ...
└──aishell
    ├── test-clean
    ├── cw
    ├── ...
└──cv-italian
    ├── test-clean
    ├── cw
    ├── ...
└──cv-german
    ├── test-clean
    ├── cw
    ├── ...
```
## Running the experiments
The scripts below have been tested on a Transformer ASR model trained on LibriSpeech. You can download the pre-trained model from [here](https://ruhr-uni-bochum.sciebo.de/s/lpjW0vxFikG2WqD). These scripts can be easily adjusted for any SpeechBrain recipe. 

### Computing Characteristics
To compute the characteristics of the output distribution run the script as follows:
```
python distriblock_characteristics.py hparams/transformer.yaml
```
We use the hyperparameter file just like in any SpeechBrain recipe. The default `attack_type` hyperparameter refers to the `CW` attack. Change this hyperparameter and csv files if testing another type of adversarial attack.

### Building and evaluating binary classifiers
The extracted characteristics of the output distribution are used as features for different binary classifiers. To train and evaluate the classifiers use:
```
python distriblock_classifiers.py -h

usage: distriblock_classifiers.py [-h] FOLDER_ORIG FOLDER_ADV

positional arguments:
  FOLDER_ORIG  Folder path that contains the characteristics of the benign examples.
  FOLDER_ADV   Folder path that contains the characteristics of the adversarial examples.```

options:
  -h, --help   show this help message and exit
```
Example for evaluating Distriblock defense against CW attack:
```
python distriblock_classifiers.py DistriBlock_data DistriBlock_data/CW/
```
### Using filtering to preserve model robustness
To evaluate Low-pass and Spectral-Gating filters to preserve system robustness, run the following script:
```
python distriblock_filtering.py hparams/transformer.yaml
```
Again, change the hyperparameters in the YAML file and CSV files if testing another type of adversarial attack.

