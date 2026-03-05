# Precision-Varying Prediction (PVP)
# [INTERSPEECH 2026] Precision-Varying Prediction (PVP): Robustifying ASR systems against adversarial attacks

This repository contains the implementation for the paper
Precision-Varying Prediction (PVP): Robustifying ASR systems against adversarial attacks
submitted to Interspeech 2026.

A demo with a selection of benign, adversarial, and noisy audio samples used in our experiments is available here:
PVP Demo: (link coming soon)

Overview

With the increasing deployment of automated and agentic systems, ensuring the adversarial robustness of Automatic Speech Recognition (ASR) models has become critical.

We observe that changing the numerical precision of an ASR model during inference reduces the success rate of adversarial attacks. Based on this observation, we introduce Precision-Varying Prediction (PVP), a lightweight strategy that improves robustness by randomly sampling different precision settings during inference.

Additionally, the same principle can be used for adversarial example detection by comparing outputs produced at different precisions and modeling their variability with a Gaussian classifier.

Experimental results show:

Improved robustness against multiple attack types

Competitive adversarial detection performance

Applicability across multiple ASR architectures

Prerequisites

Before running the PVP experiments, complete the following steps:

1. Install SpeechBrain

2. Download the datasets

3. Download pre-trained ASR models

4. Generate or obtain adversarial examples

# 1. Install SpeechBrain

We use SpeechBrain
, a PyTorch-based toolkit for speech processing.

Follow the official installation instructions:

https://speechbrain.github.io/

Example environment setup:

```
pip install speechbrain

```

Our experiments were conducted using:

SpeechBrain version: 0.5.14

GPU: NVIDIA A40 (48 GB memory)

Framework: PyTorch

# 2. Download Datasets

We use the following speech corpus:
https://www.openslr.org/12
LibriSpeech (English)
After downloading, extract the dataset and place it inside the data_set directory.

# 3. Download Pre-trained ASR Models
You can use pre-trained models from SpeechBrain / HuggingFace:

Seq2Seq (CRDNN + CTC/Attention)
https://huggingface.co/speechbrain/asr-crdnn-rnnlm-librispeech

wav2vec 2.0
https://huggingface.co/speechbrain/asr-wav2vec2-librispeech

Transformer ASR
https://huggingface.co/speechbrain/asr-transformer-transformerlm-librispeech

Whisper
https://huggingface.co/openai/whisper-large-v3
Download the models and store them in the pretrained_models/ directory#

# 4. Adversarial Attacks

Adversarial examples (e.g., Carlini & Wagner (CW) or psychoacoustic attacks) can be generated using the provided attack scripts.

Steps: 

1. Set the desired precision configuration in the model hyperparameter files.

2. Run the attack to generate adversarial audio samples.

3. Store generated samples in the corresponding dataset attack folders.

4. For cross-precision evaluation, modify the inference precision when running the ASR model.

Example .csv files containing:

- LibriSpeech samples

- CW adversarial examples

- target transcriptions

are provided in the results/ directory.

These CSV files can also be used to reproduce adversarial examples.

# 5. Precision-Varying Prediction (PVP)

During inference, the ASR model is evaluated using multiple numerical precision settings.

Computing Precision-Diversity Characteristics: 

1. Run the ASR model on the same input using multiple precision settings.

2. Collect the transcriptions generated at each precision.

3. Compute pairwise transcription differences using Word Error Rate (WER).

4. Average these differences to obtain the precision-diversity score.

5. Use this score to measure the sensitivity of predictions to precision changes.

# 6. Building and Evaluating Gaussian Classifiers

Adversarial detection is performed by modeling the distribution of precision-diversity scores.

Steps:

1. Run the ASR model on an input x using multiple precision settings {p1,...,pK}.

2. Collect the resulting transcriptions.

3. Compute pairwise WER differences between them.

4.  the values to obtain the precision-diversity score D(x).

5. Estimate the mean μ and variance σ² of D(x) using benign samples.

6. Classify an input as adversarial if its score deviates significantly from the benign distribution.

To run the classifier:
 ```
python distriblock_classifiers.py -h

```

This command shows the available configuration options.




