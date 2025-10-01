import requests
from io import BytesIO
import os
import pickle
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

import soundfile as sf
import librosa
import numpy as np
from scipy.stats import skew, kurtosis

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def calc_features(y, sr, n_bands, cut_len, n_fft):
    S = librosa.stft(y, n_fft=n_fft)
    power = np.abs(S) ** 2
    psd = power / (sr * n_fft)

    stats = []
    cuts = [psd[:, i * cut_len: (i+1) * cut_len] for i in range(psd.shape[1] // cut_len)]
    if not cuts:
        cuts = [psd]
    for cut in cuts:
        edges = librosa.mel_frequencies(n_mels=n_bands, fmin=0, fmax=sr // 2)
        bands = [(edges[i], edges[i+1]) for i in range(n_bands - 1)]
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        cut_stats = {}

        cut_stats['MIN'] = np.min(cut.flatten())
        cut_stats['MAX'] = np.max(cut.flatten())
        cut_stats['MEAN'] = np.mean(cut.flatten())
        cut_stats['MEDIAN'] = np.median(cut.flatten())
        cut_stats['VAR'] = np.var(cut.flatten())
        cut_stats['SKEW'] = skew(cut.flatten())
        cut_stats['KURT'] = kurtosis(cut.flatten())

        for j, (low, high) in enumerate(bands):
            mask = (freqs >= low) & (freqs < high)
            data = cut[mask].flatten()

            cut_stats[f'BANDS{j}_MIN'] = np.min(data)
            cut_stats[f'BANDS{j}_MAX'] = np.max(data)
            cut_stats[f'BANDS{j}_MEAN'] = np.mean(data)
            cut_stats[f'BANDS{j}_MEDIAN'] = np.median(data)
            cut_stats[f'BANDS{j}_VAR'] = np.var(data)
            cut_stats[f'BANDS{j}_SKEW'] = skew(data)
            cut_stats[f'BANDS{j}_KURT'] = kurtosis(data)
        
        stats.append(cut_stats)

    return stats

from scipy.ndimage import gaussian_filter1d

def smooth_probs_gaussian(probs, sigma=2):
    return gaussian_filter1d(probs, sigma=sigma)

def plot(y):
    S = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(2, 2, width_ratios=[20, 1], height_ratios=[3, 1], wspace=0.05, hspace=0.4)

    ax1 = fig.add_subplot(gs[0, 0])
    img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', cmap='magma', ax=ax1)
    ax1.set(title="Spectrogram")

    cax = fig.add_subplot(gs[0, 1])
    fig.colorbar(img, cax=cax, format='%+2.0f dB')

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(cls)
    ax2.set(title="Clean Probability", ylabel="p", xlabel="Time (s)")
    # ax2.set_ylim(0, 1)

    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax1.get_xticklabels(), visible=False)

    plt.suptitle('{} by {}'.format(card['TITLE'], card['ARTIST']))
    plt.tight_layout()
    plt.show()
    plt.close('all')

rate = 16000
n_bands = 40
n_fft = 2048
cut_len = (rate // 512) * 5

out_dir = f'/Users/dylan.d/Documents/research/music/jazz_data_{rate}_full_clean'
os.makedirs(out_dir, exist_ok=True)

n_samples = 5000
counter = 0
i = 0
cards = pickle.load(open('/Users/dylan.d/Documents/research/music/JazzSet.0.9.pkl', "rb"))[6:]
model = pickle.load(open('/Users/dylan.d/Documents/research/music/jazz_data_16000_noise/model.pkl', 'rb'))

# cards = np.random.choice(cards, n_samples, replace=False)

def process_card(card):
    if card == False:
        return False

    mp3_url = card['URLS'][0]['FILE']
    out_url = '-'.join(mp3_url.split('/')[-2:])
    out_url = out_url.replace('.mp3', '.wav')
    out_url = os.path.join(out_dir, out_url)

    try:
        response = requests.get(mp3_url)
        mp3_audio = BytesIO(response.content)

        y, sr = librosa.load(mp3_audio, sr=None)
        y = librosa.resample(y, orig_sr=sr, target_sr=rate)

        feats = calc_features(y, rate, n_bands, cut_len, n_fft)
        feats = np.asarray([list(feat.values()) for feat in feats])
        feats = np.nan_to_num(feats)
        
        cls = smooth_probs_gaussian(model.predict_proba(feats)[:, 1])
        if np.mean(cls) < 0.55:
            return False

        sf.write(out_url, y, rate)
        return True
    except Exception as e:
        return str(e)

def run_parallel_processing(max_workers=4):
    success_count = 0
    processed_count = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_card, card)
            for card in cards
        ]

        with tqdm(total=len(futures)) as pbar:
            for future in as_completed(futures):
                result = future.result()
                processed_count += 1

                if result is True:
                    success_count += 1
                elif result is not False:
                    print(result)

                pbar.update(1)
                pbar.set_postfix(pass_fraction=success_count / processed_count if processed_count else 0)

if __name__ == '__main__':
    run_parallel_processing(os.cpu_count() // 2)
