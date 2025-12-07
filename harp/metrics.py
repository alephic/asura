import torch

def log_spectral_distance(x, y, n_fft=1024):
    bsz, length, ch = x.shape
    x = x.transpose(2, 1).reshape(-1, length)
    y = y.transpose(2, 1).reshape(-1, length)
    device = x.device
    w = torch.hann_window(n_fft, device=device)
    norm_factor = 1.0 / torch.sqrt(torch.sum(torch.square(w)))
    s_x = torch.stft(x, n_fft=n_fft, window=w, center=True, pad_mode='constant', return_complex=True).abs() * norm_factor
    s_y = torch.stft(y, n_fft=n_fft, window=w, center=True, pad_mode='constant', return_complex=True).abs() * norm_factor

    frames = s_x.shape[2]

    eps = 1e-12
    lsd = torch.mean(
        torch.sqrt(
            torch.mean(torch.square(2.0 * (torch.log10(s_y + eps) - torch.log10(s_x + eps))), dim=1)
        ).reshape(bsz, ch, frames),
        dim=(1,2)
    )
    return lsd

def psnr(x, y):
    return -10.0 * torch.log10(torch.mean(torch.square(x - y), dim=(1,2)))