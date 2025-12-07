import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb
import torch

__all__ = (
    'save_image',
    'apply_colormap',
    'sizeof_fmt',
    'thou_fmt',
    'unitary_log_np',
    'rescale_complex_np',
    'complex_to_rgb'
)

def save_image(arr: np.ndarray, path: str, cmap='magma'):
    fig = plt.figure()
    fig.set_size_inches(arr.shape[1]/arr.shape[0], 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(arr, cmap=cmap)
    plt.savefig(path, dpi=arr.shape[0])
    plt.close(fig)

def apply_colormap(arr, cmap='magma'):
    cmap = plt.get_cmap(cmap)
    return cmap(arr)[..., :3]

def sizeof_fmt(num, suffix="B"):
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"

def thou_fmt(num, suffix=""):
    for unit in ("", "k", "m", "b"):
        if abs(num) < 1000.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1000.0
    return f"{num:.1f}t{suffix}"

def unitary_log_np(x, eps=1e-3):
    log_eps = np.log(eps)
    f = 1 - np.log(x + eps) / log_eps
    return f/(1 - np.log(1 + eps)/log_eps)

def rescale_complex_np(x, new_mag):
    x_mag = np.abs(x)
    nonzero = x_mag != 0.0
    scale = new_mag / np.where(nonzero, x_mag, 1.0)
    return x * np.where(nonzero, scale, 1.0)

def complex_to_rgb(x):
    theta = np.angle(x)
    mag = np.abs(x)
    h = theta * (0.5/np.pi) + 0.5
    s = np.ones_like(h)
    l = mag
    return hsv_to_rgb(np.stack((h, s, l), axis=-1))
