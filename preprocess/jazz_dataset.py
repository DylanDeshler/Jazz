from pydub import AudioSegment
import requests
from io import BytesIO
import os
import pickle
from tqdm import tqdm
import numpy as np
from scipy.io import wavfile

import glob
import librosa
import matplotlib.pyplot as plt

from scipy.signal import butter, lfilter

def butter_lowpass(cutoff, sr, order=5):
    nyquist = 0.5 * sr
    norm_cutoff = cutoff / nyquist
    b, a = butter(order, norm_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, sr, order=5):
    b, a = butter_lowpass(cutoff, sr, order=order)
    return lfilter(b, a, data)

from pydub import AudioSegment
import numpy as np
from scipy.signal import butter, filtfilt
import soundfile as sf

def mp3_to_lowpass_wav(
    input_mp3,
    output_wav,
    cutoff=4000,
    order=5
):
    # Step 1: Load MP3
    audio = AudioSegment.from_mp3(input_mp3)

    # Step 2: Convert to NumPy
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)

    # Handle stereo
    if audio.channels == 2:
        samples = samples.reshape((-1, 2)).T  # (2, num_samples)
    else:
        samples = samples.reshape((1, -1))  # (1, num_samples)

    # Step 3: Low-pass filter
    def butter_lowpass(cutoff, sr, order):
        nyq = 0.5 * sr
        norm = cutoff / nyq
        return butter(order, norm, btype="low")

    b, a = butter_lowpass(cutoff, audio.frame_rate, order)
    filtered = np.array([filtfilt(b, a, ch) for ch in samples])

    # Step 4: Back to int16 + reshape for AudioSegment
    filtered = np.clip(filtered, -32768, 32767).astype(np.int16)
    filtered = filtered.T.reshape(-1)

    # Step 5: Create and save WAV
    filtered_audio = AudioSegment(
        filtered.tobytes(),
        frame_rate=audio.frame_rate,
        sample_width=2,
        channels=audio.channels
    )
    filtered_audio.export(output_wav, format="wav")

# paths = sorted(glob.glob('/Users/dylan.d/Documents/research/music/jazz_data44100/*.wav'))
# for path in paths:
#     y, sr = librosa.load(path, sr=None)
#     flatness = np.mean(librosa.feature.spectral_flatness(y=y)[0])
#     S = librosa.stft(y)
#     S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max);print(np.min(S_db), np.mean(S_db), np.std(S_db), np.median(S_db), np.max(S_db))
#     librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', cmap='magma')
#     plt.colorbar(format='%+2.0f dB')
#     plt.title(path.split('-')[-1] + f', {flatness:.3f}')
#     plt.show()
#     plt.close('all')

    # y = lowpass_filter(y, cutoff=4000, sr=sr)
    # flatness = np.mean(librosa.feature.spectral_flatness(y=y)[0])
    # S = librosa.stft(y)
    # S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    # librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', cmap='magma')
    # plt.colorbar(format='%+2.0f dB')
    # plt.title(path.split('-')[-1] + f', {flatness:.3f} + filtered')
    # plt.show()
    # plt.close('all')

#     wavfile.write(path + 'lowpass.wav', sr, y)

def compute_snr(audio: AudioSegment):
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    power = np.mean(samples ** 2)
    noise_floor = np.percentile(np.abs(samples), 5)
    snr = 10 * np.log10(power / (noise_floor**2 + 1e-9))
    return snr

def hf_energy(y, sr):
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    hf_mask = freqs > 8000
    hf_energy = np.sum(S[hf_mask])
    total_energy = np.sum(S)
    hf_ratio = hf_energy / (total_energy + 1e-10)
    return hf_ratio

def calc_features(y, sr, n_bands, n_fft=2048):
    from scipy.stats import skew, kurtosis

    S = librosa.stft(y, n_fft=n_fft)
    power = np.abs(S) ** 2
    psd = power / (sr * n_fft)
    print(psd.shape)

    edges = librosa.mel_frequencies(n_mels=n_bands, fmin=0, fmax=sr // 2)
    bands = [(edges[i], edges[i+1]) for i in range(n_bands - 1)]
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    stats = {}

    stats['MIN'] = np.min(psd.flatten())
    stats['MAX'] = np.max(psd.flatten())
    stats['MEAN'] = np.mean(psd.flatten())
    stats['MEDIAN'] = np.median(psd.flatten())
    stats['VAR'] = np.var(psd.flatten())
    stats['SKEW'] = skew(psd.flatten())
    stats['KURT'] = kurtosis(psd.flatten())

    for j, (low, high) in enumerate(bands):
        mask = (freqs >= low) & (freqs < high)
        data = psd[mask].flatten()

        stats[f'BANDS{j}_MIN'] = np.min(data)
        stats[f'BANDS{j}_MAX'] = np.max(data)
        stats[f'BANDS{j}_MEAN'] = np.mean(data)
        stats[f'BANDS{j}_MEDIAN'] = np.median(data)
        stats[f'BANDS{j}_VAR'] = np.var(data)
        stats[f'BANDS{j}_SKEW'] = skew(data)
        stats[f'BANDS{j}_KURT'] = kurtosis(data)

    return stats


def plot(y, sr):
    S = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max);print(np.min(S_db), np.mean(S_db), np.std(S_db), np.median(S_db), np.max(S_db))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title(path.split('-')[-1] + f', {flatness:.3f}')
    plt.show()
    plt.close('all')

rate = 16000
out_dir = f'/Users/dylan.d/Documents/research/music/jazz_data_{rate}_random'
os.makedirs(out_dir, exist_ok=True)

n_samples = 2000
counter = 0
i = 0
cards = pickle.load(open('/Users/dylan.d/Documents/research/music/JazzSet.0.9.pkl', "rb"))

all_data = []

cards = np.random.choice(cards[6:], n_samples, replace=False)

with tqdm(total=n_samples) as pbar:
    for card in cards:
        if card == False:
            continue
        i += 1

        mp3_url = card['URLS'][0]['FILE']
        out_url = '-'.join(mp3_url.split('/')[-2:])
        out_url = out_url.replace('.mp3', '.wav')
        out_url = os.path.join(out_dir, out_url)

        try:
            response = requests.get(mp3_url)
            mp3_audio = BytesIO(response.content)

            y, sr = librosa.load(mp3_audio, sr=None)
            y = librosa.resample(y, orig_sr=sr, target_sr=rate)

            sf.write(out_url, y, rate)

            feats = calc_features(y, rate, 40)
            cls = int(input(f'{out_url} Is good quality?').strip())

            all_data.append({'url': out_url, 'x': feats, 'y': cls})

            with open(os.path.join(out_dir, 'data.pkl'), 'wb') as f:
                pickle.dump(all_data, f)

            counter += 1

            pbar.update(1)
            pbar.set_postfix(pass_fraction=counter/i)
        except Exception as e:
            print(e)
            continue
