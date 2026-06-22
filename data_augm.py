import os
import csv
import torch
import torchaudio
import zipfile
import urllib.request
from speechbrain.augment.time_domain import SpeedPerturb, DropFreq, DropChunk, AddNoise
from speechbrain.augment.augmenter import Augmenter

# 1. Define hyperparameters and explicit paths
SAMPLE_RATE = 16000
NUM_WORKERS = 2 
DATA_FOLDER = "./augm_data/data"
SAVE_FOLDER = "./augm_data/save"

DATA_FOLDER_NOISE = os.path.join(DATA_FOLDER, "noise")
NOISE_ANNOTATION = os.path.join(SAVE_FOLDER, "noise.csv")
NOISE_DATASET_URL = "https://www.dropbox.com/scl/fi/a09pj97s5ifan81dqhi4n/noises.zip?rlkey=j8b0n9kdjdr32o1f06t0cw5b7&dl=1"

# Create directories
os.makedirs(DATA_FOLDER_NOISE, exist_ok=True)
os.makedirs(SAVE_FOLDER, exist_ok=True)

# 2. Handle Custom Download and Unzip
zip_filepath = os.path.join(DATA_FOLDER_NOISE, "data.zip")
extracted_noise_dir = os.path.join(DATA_FOLDER_NOISE, "pointsource_noises")

if not os.path.exists(extracted_noise_dir):
    print("Downloading noise dataset via Python...")
    urllib.request.urlretrieve(NOISE_DATASET_URL, zip_filepath)
    
    print("Extracting noise files manually...")
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        zip_ref.extractall(DATA_FOLDER_NOISE)
        
    if os.path.exists(zip_filepath):
        os.remove(zip_filepath)
else:
    print("Noise directory already exists. Skipping download.")

# 3. Custom native Python CSV Generation (Bypassing SpeechBrain helpers)
print(f"Generating noise.csv by scanning: {extracted_noise_dir}")
with open(NOISE_ANNOTATION, mode="w", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)
    # SpeechBrain CSV header requires 'ID' and the asset details
    writer.writerow(["ID", "duration", "wav", "wav_format", "wav_opts"])
    
    count = 0
    for root, _, files in os.walk(extracted_noise_dir):
        for file in files:
            if file.lower().endswith(".wav"):
                file_path = os.path.abspath(os.path.join(root, file))
                try:
                    # Get the duration of the audio file dynamically
                    info = torchaudio.info(file_path)
                    duration = info.num_frames / info.sample_rate
                    
                    # Create a unique ID for this sample
                    file_id = os.path.splitext(file)[0] + f"_{count}"
                    writer.writerow([file_id, duration, file_path, "wav", ""])
                    count += 1
                except Exception as e:
                    print(f"Skipping broken file {file}: {e}")

print(f"CSV Generation Complete! Added {count} noise files.")

# 4. Instantiate individual augmentations
speed_perturb = SpeedPerturb(
    orig_freq=SAMPLE_RATE,
    speeds=[95, 100, 105]
)

drop_freq = DropFreq(
    drop_freq_low=0.0,
    drop_freq_high=1.0,
    drop_freq_count_low=1,
    drop_freq_count_high=3,
    drop_freq_width=0.05
)

drop_chunk = DropChunk(
    drop_length_low=1000,
    drop_length_high=2000,
    drop_count_low=1,
    drop_count_high=5
)

add_noise = AddNoise(
    csv_file=NOISE_ANNOTATION,
    snr_low=0,
    snr_high=15,
    noise_sample_rate=SAMPLE_RATE,
    clean_sample_rate=SAMPLE_RATE,
    num_workers=NUM_WORKERS
)

# 5. Combine into pipeline
wav_augment = Augmenter(
    min_augmentations=3,
    max_augmentations=3,
    augment_prob=1.0,
    augmentations=[speed_perturb, drop_freq, drop_chunk]
)
"""
wav_augment = Augmenter(
    min_augmentations=4,
    max_augmentations=4,
    augment_prob=1.0,
    augmentations=[speed_perturb, drop_freq, drop_chunk, add_noise]
)
"""

# 6. Load your audio file
waveform, sr = torchaudio.load("3081-166546-0063.flac")

# Resample if necessary
if sr != SAMPLE_RATE:
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
    waveform = resampler(waveform)

# Format shape to [batch, time]
if waveform.shape[0] > 1:
    waveform = torch.mean(waveform, dim=0, keepdim=True)
elif waveform.ndim == 1:
    waveform = waveform.unsqueeze(0)

# 7. Apply the pipeline
lengths = torch.tensor([1.0]) 
augmented_waveform, out_lengths = wav_augment(waveform, lengths)

# 8. Save output
torchaudio.save("augmented_audio.wav", augmented_waveform, SAMPLE_RATE)
print("Augmentation completed successfully and saved as 'augmented_audio.wav'")