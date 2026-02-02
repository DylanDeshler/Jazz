from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

import numpy as np

spatial_window = 48
vae_embed_dim = 16
N = 4403211

print('Loading data...')
X = np.memmap('/home/ubuntu/Data/low_measures_large.bin', dtype=np.float16, mode='r', shape=(N, spatial_window, vae_embed_dim)).copy()
# X = X.reshape(N, -1)
X = X.mean(1)
# X = np.concatenate([X.mean(1), X.std(1)], -1)

print('Normalizing...')
X = normalize(X, norm='l2', axis=1)

print('Fitting')
np.random.seed(0)
kmeans = KMeans(n_clusters=256, n_init=100, max_iter=1000, verbose=2, random_state=0)
kmeans.fit(X)

print('Renormalizing...')
centroids = kmeans.cluster_centers_
style_bank = normalize(centroids, norm='l2', axis=1)

print('Writing...')
with open('/home/ubuntu/Data/low_measures_large_clusters.npy', 'w+') as f:
    np.save(f, style_bank)