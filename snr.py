import sys
import csv
import torchaudio
import numpy as np
import os 
from pathlib import Path

def numel(array):
	# Number of elements in an array
    s = array.shape
    n = 1
    for i in range(len(s)):
        n *= s[i]
    return n
    
def snrseg(noisy, clean, fs, tf=0.05):
    '''
    Segmental SNR computation. Does NOT support VAD (voice activity dection) or Interpolation (at the moment). Corresponds to the mode 'wz' in
    the original Matlab implementation.

    SEG = mean(10*log10(sum(Ri^2)/sum((Si-Ri)^2))

    '''
    snmax = 100
    noisy = noisy.squeeze()
    clean = clean.squeeze()
    if clean.shape[0] != noisy.shape[0]:
        print("ERROR SHAPE! ")
    nr = min(clean.shape[0], noisy.shape[0])
    kf = round(tf * fs)
    ifr = np.arange(kf, nr, kf)
    ifl = int(ifr[len(ifr)-1])
    nf = numel(ifr)
    ef = np.sum(np.reshape(np.square((noisy[:ifl] - clean[:ifl]), dtype='float32'), (kf, nf), order='F'), 0)
    rf = np.sum(np.reshape(np.square(clean[:ifl], dtype='float32'), (kf, nf), order='F'), 0)
    em = ef == 0
    rm = rf == 0
    snf = 10 * np.log10((rf + rm) / (ef + em))
    snf[rm] = -snmax
    snf[em] = snmax
    temp = np.ones(nf)
    vf = temp == 1
    seg = np.mean(snf[vf])
    return seg

def snrorig(noisy, clean):
    
    noisy = noisy.squeeze()
    clean = clean.squeeze()
    if clean.shape[0] != noisy.shape[0]:
        print("ERROR SHAPE! ")
    ef = np.sum(np.square((noisy - clean)))
    rf = np.sum(np.square(clean))
    snf = 10 * np.log10(rf / ef)
    if ef == 0:
        print("Here is happening! ")
        return np.inf
    return snf


def Seg_SNR(source, target, filenames):
    rdr_src = csv.DictReader(open(source,"r",encoding="utf-8"))
    rdr_tgt = csv.DictReader(open(target,"r",encoding="utf-8"))
    count = 0
    db_neg = 0
    snr_seg, snr_orig = [], []
    failed_filenames = set(filenames)
    # with open("output.txt", "w") as f:
    for src, tgt in zip(rdr_src, rdr_tgt):
        if src["ID"] == tgt["ID"]:         
            name_file = os.path.splitext(os.path.basename(tgt["wav"]))[0]
            
            if name_file not in failed_filenames:  
                print(tgt["wav"])
                print(src["wav"])                  
                data_adv, _ = torchaudio.load(tgt["wav"])
                filename, file_extension = os.path.splitext(src["wav"])

                data_src, _ = torchaudio.load(src["wav"])
                
                db = snrseg(data_adv[0].numpy(), data_src[0].numpy(), _, tf=0.05)
                db_orig = snrorig(data_adv[0].numpy(), data_src[0].numpy())
                if db_orig == np.inf:
                    print(tgt["wav"])
                    print(src["wav"])
                else:
                    snr_orig.append(db_orig)

                if (db < 0):
                    db_neg += 1 
                snr_seg.append(db)
                count += 1
            else:
                print("Unsuccesfull AE: ", name_file)
        else:
            print("ERROR")
        
        
    return np.mean(snr_seg), np.mean(snr_orig), db_neg, count

if __name__ == "__main__":    
    tgt_path = "/home/pizarm5k/speechbrain/adversarial_examples"
    tgt_corpus = "librispeech" # "aishell" # "librispeech"
    tgt_model = "Transformer" # "Transformer" # "Transformer/whisper" # seq2seq" # "CTC"
    tgt_attack = ["psy"] # ["cw", "psy"]
    precision = ["fp16"]  # ["fp32", "fp16", "bf16"]    

    model_0 = 'LibriSpeech'
    model_1 = 'Transformer'
    model_2 = 'transformer' # 'transformer' # 'train_wav2vec2_char' # 'CRDNN_BPE_960h_5k_LM'
    id_ = '7444' # '7444' # '1986'
    src_csv = f'{tgt_path}/{tgt_corpus}/clean_audio_clean_transcripts.csv'
    
    for attack_type in tgt_attack:
        for precision_type in precision:
            adv_csv = Path(tgt_path) / tgt_corpus / tgt_model / attack_type / precision_type / precision_type / f"adv_audio_adv_transcripts.csv"
            unsuccesfull_data = Path(model_0) / model_1 / f'results' / model_2 / id_ / precision_type / f"unsuccesfull_{attack_type}_{precision_type}_{precision_type}.csv"
            print(adv_csv)
            print(unsuccesfull_data)
            filenames = []

            if os.path.exists(unsuccesfull_data):
                with open(unsuccesfull_data, "r") as f:
                    for line in f:
                        # extract the path between quotes
                        start = line.find("'") + 1
                        end = line.find("'", start)
                        path = line[start:end]

                        # get filename without extension
                        name = os.path.splitext(os.path.basename(path))[0]
                        filenames.append(name)
            print(filenames)
            # adv_csv = "/home/pizarm5k/speechbrain/adversarial_examples/librispeech/CTC/cw/bf16/bf16/adv_audio_adv_transcripts.csv"
            SNR_SEG, SNR_ORG, db_neg, total  = Seg_SNR(src_csv, adv_csv, filenames)
            print(f'{tgt_corpus} - {tgt_model} - {attack_type} - {precision_type}')
            print("SNR_seg {:.2f} SNR {:.2f} db_neg {} total samples {}".format(SNR_SEG, SNR_ORG, db_neg, total))