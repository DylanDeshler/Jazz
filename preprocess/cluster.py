from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

import numpy as np

spatial_window = 48
vae_embed_dim = 16
N = 4403211
X = np.memmap('/home/ubuntu/Data/low_measures_large.bin', dtype=np.float16, mode='r', shape=(N, spatial_window, vae_embed_dim)).reshape(N, -1)
X = normalize(X, norm='l2', axis=1)

kmeans = KMeans(n_clusters=256, n_init=100, max_iter=1000, verbose=2)
kmeans.fit(X)

centroids = kmeans.cluster_centers_
style_bank = normalize(centroids, norm='l2', axis=1)

# Save 'style_bank' to be the weights of your new embedding layer
with open('/home/ubuntu/Data/low_measures_large_clusters.npy', 'w+') as f:
    np.save(f, style_bank)