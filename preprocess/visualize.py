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
        cut_stats['ENERGY'] = np.sum(cut.flatten())
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
            cut_stats[f'BANDS{j}_ENERGY'] = np.sum(data)
            cut_stats[f'BANDS{j}_SKEW'] = skew(data)
            cut_stats[f'BANDS{j}_KURT'] = kurtosis(data)
        
        stats.append(cut_stats)

    return stats

from scipy.ndimage import gaussian_filter1d

def smooth_probs_gaussian(probs, sigma=2):
    return gaussian_filter1d(probs, sigma=sigma)

def plot(y, sr, cls):
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

    # plt.suptitle('{} by {}'.format(card['TITLE'], card['ARTIST']))
    plt.tight_layout()
    plt.show()
    plt.close('all')

def plot_spectrogram(y, sr, cls):
    S = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

    plt.figure(figsize=(8, 4))
    # librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', cmap='magma')
    librosa.display.specshow(librosa.amplitude_to_db(librosa.feature.melspectrogram(y=y, sr=sr), ref=np.max), sr=sr, x_axis='time', y_axis='hz', cmap='magma')
    plt.title(f"P(Good) = {int(100 * cls)}%")
    plt.colorbar(format='%+2.0f dB')

    plt.tight_layout()
    plt.show()
    plt.close('all')

rate = 16000
n_bands = 40
n_fft = 2048
cut_len = (rate // 512) * 5
model = pickle.load(open('/Users/dylan.d/Documents/research/music/jazz_data_16000_train/model.pkl', 'rb'))

bad = '/Users/dylan.d/Documents/research/music/jazz_data_16000/JV-12-1916-QmYZJNDBn5WPRNakbBi9UujN8dMrJo3n8vpaHo8RNon4xt.wav-JV-12-1916-QmYZJNDBn5WPRNakbBi9UujN8dMrJo3n8vpaHo8RNon4xt.wav' 
good = '/Users/dylan.d/Documents/research/music/jazz_data_16000_random/JV-36144-1957-QmbSPzr8VX8LUrnatKVGRK9G9wZuVxah5VdMBgXVTpNBDn.wav-TS485813.wav'

bad, sr = librosa.load(bad, sr=None)
good, sr = librosa.load(good, sr=None)
print(sr)
bad = bad[:rate * 5]
good = good[:rate * 5]

feats = calc_features(bad, rate, n_bands, cut_len, n_fft)
feats = np.asarray([list(feat.values()) for feat in feats])
feats = np.nan_to_num(feats)
cls = smooth_probs_gaussian(model.predict_proba(feats)[:, 1])[0]
print(cls)
plot_spectrogram(bad, sr, cls)

feats = calc_features(good, rate, n_bands, cut_len, n_fft)
feats = np.asarray([list(feat.values()) for feat in feats])
feats = np.nan_to_num(feats)
cls = smooth_probs_gaussian(model.predict_proba(feats)[:, 1])[0]
plot_spectrogram(good, sr, cls)

