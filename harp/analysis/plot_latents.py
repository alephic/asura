import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from matplotlib.colors import hsv_to_rgb
import numpy as np
import os
from scipy.stats import gaussian_kde
import torch
from harp.modeling.util import log_mel_spectrogram

def main(mode='umap'):
    np.random.seed(1337)
    X = np.load('embeddings.npy')
    if mode == 'umap':
        if not os.path.exists('embeddings_umap.npy'):
            print('Computing UMAP')
            from umap import UMAP
            X_t = UMAP(n_components=2, n_neighbors=60).fit_transform(X)
            np.save('embeddings_umap.npy', X_t)
        else:
            print('Loading UMAP')
            X_t = np.load('embeddings_umap.npy')
    else:
        if not os.path.exists('embeddings_tsne.npy'):
            print('Computing TSNE')
            from sklearn.manifold import TSNE
            X_t = TSNE(n_components=2, perplexity=50).fit_transform(X)
            np.save('embeddings_tsne.npy', X_t)
        else:
            print('Loading TSNE')
            X_t = np.load('embeddings_tsne.npy')
    # if not os.path.exists('embeddings_densities.npy'):
    #     print('Computing KDE')
    #     z = gaussian_kde(X_t.T, bw_method=0.025)(X_t.T)
    #     np.save('embeddings_densities.npy', z)
    # else:
    #     print('Loading KDE')
    #     z = np.load('embeddings_densities.npy')

    X_std = X - np.mean(X, axis=0)
    X_std /= X_std.std(axis=0)
    dim_hue_vectors = np.exp(2.0j*np.pi*np.random.rand(X.shape[1]))
    X_hue_vectors = np.linalg.vecdot(X_std, dim_hue_vectors[np.newaxis, :], axis=-1)
    X_hue_angles = np.angle(X_hue_vectors) / (2.0 * np.pi) + 0.5
    X_sats = np.abs(X_hue_vectors)
    X_sats /= np.mean(X_sats) + np.std(X_sats)
    X_sats = np.clip(X_sats, a_min=0.0, a_max=1.0)
    c = np.stack([X_hue_angles, X_sats, np.full_like(X_hue_angles, 0.8)], axis=1)
    c = hsv_to_rgb(c)

    # select_idxs = [43, 75, 9, 13, 51, 52]
    # audio = np.load('embed_audio.npy')[select_idxs, :132608, 0]
    # specs = log_mel_spectrogram(torch.from_numpy(audio), 44100, 2048, 256).numpy()
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.axis('off')
    ax.set_aspect('equal', 'datalim')
    ax.scatter(X_t[:, 0], X_t[:, 1], c=c, marker='.', s=1.0, edgecolors='none')
    # ax.scatter(X_t[select_idxs, 0], X_t[select_idxs, 1], c='green', marker='o', s=10.0, edgecolors='none')
    # for i in range(6):
    #     iax_x = (i//3)*(1-0.16)
    #     iax_y = (i % 3)/3
    #     iax = ax.inset_axes((iax_x, iax_y, 0.16, 0.3))
    #     iax.set_xticks([])
    #     iax.set_yticks([])
    #     iax.set_aspect('equal')
    #     iax.imshow(specs[i])
    #     conn = ConnectionPatch((1-i//3, 0.0), (X_t[select_idxs[i], 0], X_t[select_idxs[i], 1]), coordsA=iax.transAxes, coordsB=ax.transData)
    #     ax.add_artist(conn)
    fig.tight_layout()
    plt.savefig('latents.png', dpi=1200, transparent=False)

if __name__ == "__main__":
    main()