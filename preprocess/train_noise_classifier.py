import os
import pickle
from tqdm import tqdm
import numpy as np

import librosa
import matplotlib.pyplot as plt
from collections import defaultdict

def calc_features(y, sr, n_bands, cut_len, n_fft):
    from scipy.stats import skew, kurtosis

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

rate = 16000
n_bands = 40
n_fft = 2048
cut_len = (rate // 512) * 5

out_dir = '/Users/dylan.d/Documents/research/music/jazz_data_16000_train'

## Generate training data
if not os.path.exists(out_dir):
    os.makedirs(out_dir, exist_ok=True)

    with open('/Users/dylan.d/Documents/research/music/jazz_data_16000_random/data.pkl', 'rb') as f:
        data = pickle.load(f)

    with open('/Users/dylan.d/Documents/research/music/jazz_data_16000/data.pkl', 'rb') as f:
        data.extend(pickle.load(f))

    X = []
    Y = []
    for sample in tqdm(data):
        wav = sample['url']
        cls = sample['y']

        y, sr = librosa.load(wav, sr=None)[:rate * 60 * 5]
        assert sr == rate

        feats = calc_features(y, rate, n_bands, cut_len, n_fft)

        X.append([list(feat.values()) for feat in feats])
        Y.append([cls] * len(feats))

        assert len(X) == len(Y), f'{len(X)} != {len(Y)}'

    with open(os.path.join(out_dir, 'X.pkl'), 'wb') as f:
        pickle.dump(X, f)

    with open(os.path.join(out_dir, 'Y.pkl'), 'wb') as f:
        pickle.dump(Y, f)

## Load training data
with open(os.path.join(out_dir, 'X.pkl'), 'rb') as f:
    X = pickle.load(f)

with open(os.path.join(out_dir, 'Y.pkl'), 'rb') as f:
    y = pickle.load(f)

X = [np.nan_to_num(np.asarray(x)) for x in X]
y = [np.asarray(y_) for y_ in y]

from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier

# Crude but sufficient for this
edges = librosa.mel_frequencies(n_mels=n_bands, fmin=0, fmax=rate // 2)
bands = [(edges[i], edges[i+1]) for i in range(n_bands - 1)]
freqs = librosa.fft_frequencies(sr=rate, n_fft=n_fft)

feature_names = []

feature_names.append('MIN')
feature_names.append('MAX')
feature_names.append('MEAN')
feature_names.append('MEDIAN')
feature_names.append('VAR')
feature_names.append('ENERGY')
feature_names.append('SKEW')
feature_names.append('KURT')

for j, (low, high) in enumerate(bands):
    feature_names.append(f'BANDS{j}_MIN')
    feature_names.append(f'BANDS{j}_MAX')
    feature_names.append(f'BANDS{j}_MEAN')
    feature_names.append(f'BANDS{j}_MEDIAN')
    feature_names.append(f'BANDS{j}_VAR')
    feature_names.append(f'BANDS{j}_ENERGY')
    feature_names.append(f'BANDS{j}_SKEW')
    feature_names.append(f'BANDS{j}_KURT')

from itertools import product

hyper_params = {
    'n_estimators': [50, 100, 200, 300, 400, 500, 1000],
    'max_depth': [3, 5, 8, 12],
    'bootstrap': [False, True],
}

keys = list(hyper_params.keys())
combinations = list(product(*hyper_params.values()))

stats = defaultdict(list)
for split in tqdm(range(20)):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)
    X_train, X_test, y_train, y_test = np.concatenate(X_train), np.concatenate(X_test), np.concatenate(y_train), np.concatenate(y_test)

    for combs in combinations:
        params = dict(zip(keys, combs))
        model = ExtraTreesClassifier(**params, random_state=None, n_jobs=-1, class_weight='balanced_subsample')
        model.fit(X_train, y_train)

        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)

        name = '_'.join(f'{k}{v}' for k, v in params.items())
        stats[name].append(test_acc)

for k, v in stats.items():
    print(k, np.mean(v), np.std(v))

stats = {k: np.mean(v) for k, v in stats.items()}
top_keys = sorted(stats, key=stats.get, reverse=True)[-5:]
for top_key in top_keys:
    print(top_key)

# model = ExtraTreesClassifier(n_estimators=100, max_depth=12, bootstrap=False, random_state=0, n_jobs=-1, class_weight='balanced_subsample')
# model.fit(np.concatenate(X), np.concatenate(y))

# with open(os.path.join(out_dir, 'model.pkl'), 'wb') as f:
#     pickle.dump(model, f)

# importances = model.feature_importances_
# sorted_indices = np.argsort(importances)[-20:]

# plt.figure(figsize=(10, 6))
# plt.barh(range(len(sorted_indices)), importances[sorted_indices], align='center')
# plt.yticks(range(len(sorted_indices)), np.array(feature_names)[sorted_indices])
# plt.xlabel("Feature Importance")
# plt.title("Feature Importance from Random Forest")
# plt.show()