import torch
import math
import torch.distributed as dist
from scipy.optimize import linear_sum_assignment
import numpy as np

def autoscaled_mse_loss(x, y, c=1.0):
    mse = torch.mean(torch.square(x - y), dim=tuple(range(1, x.dim())))
    return torch.mean(mse/(mse + c).detach())

def adaptive_mse_loss(x, y, w):
    mse = torch.mean(torch.square(x - y), dim=tuple(range(1, x.dim())))
    return torch.mean(mse*torch.exp(-w))

def pseudo_huber_loss(a, b, c=1.0):
    sq_err = torch.square(a-b).flatten(1)
    return torch.mean(torch.sqrt(sq_err.sum(1) + c*c*sq_err.size(1)))/(sq_err.size(1)**0.5) - c

def dispersion_loss(x):
    return torch.mean(torch.exp(-torch.square(torch.cdist(x, x))))

def normal_energy_distance_loss(x):
    scale = x.size(-1)**-0.5
    return scale * (torch.mean(torch.cdist(x, torch.randn_like(x))) - torch.mean(torch.cdist(x, x)))

def warmup_weight(step, start, end):
    step = min(end, max(start, step))
    return (step-start) / (end-start) if step < end else 1.0

# inverse of torch.unfold (overlap-add)
def fold(x_chunks, fold_dim, step):
    n_chunks = x_chunks.size(fold_dim)
    chunk_size = x_chunks.size(-1)
    output_length = (n_chunks-1)*step + chunk_size
    x_chunks_transposed = x_chunks.transpose(fold_dim, -2).transpose(-1, -2)
    x_chunks_unflatten_shape = x_chunks_transposed.shape[:-2]
    x_chunks_reshaped = x_chunks_transposed.flatten(0, x_chunks.dim()-3) if len(x_chunks_unflatten_shape) > 0 else x_chunks_transposed
    folded = torch.nn.functional.fold(x_chunks_reshaped, (1, output_length), (1, chunk_size), stride=(1, step)).squeeze(-2).squeeze(-2)
    folded = folded.view(*x_chunks_unflatten_shape, folded.size(-1)).transpose(-1, fold_dim)
    return folded

def smoothstep(x):
    x = torch.clamp(x, min=0.0, max=1.0)
    return x * x * (3.0 - 2.0*x)

def smooth_fold_chunks(x_chunks: torch.Tensor, overlap: int):
    if overlap == 0:
        return x_chunks.flatten(1, 2)
    token_samples = x_chunks.size(2)
    edge_window = smoothstep(torch.linspace(1.0/(overlap+1), 1.0 - 1.0/(overlap+1), overlap, device=x_chunks.device, dtype=x_chunks.dtype))
    left_window = torch.ones(token_samples-overlap, device=x_chunks.device, dtype=x_chunks.dtype)
    left_window[-overlap:] = edge_window.flip(0)
    overlaps = torch.cat((x_chunks[:, 1:, :overlap], x_chunks[:, -1:, -overlap:]), dim=1)
    overlaps = torch.cat((
        torch.zeros(x_chunks.size(0), x_chunks.size(1), token_samples - 2*overlap, x_chunks.size(3), device=x_chunks.device, dtype=x_chunks.dtype),
        overlaps * edge_window[None, None, :, None]
    ), dim=2)
    overlapped_chunks = x_chunks[:, :, overlap:] * left_window[None, None, :, None] + overlaps
    return torch.cat((x_chunks[:, 0, :overlap], overlapped_chunks.flatten(1, 2)), dim=1)

def smooth_chunk_overlap(x_chunks: torch.Tensor, overlap: int, right: bool=True, left: bool=True):
    if overlap == 0:
        return x_chunks
    edge_window = smoothstep(torch.linspace(1.0/(overlap+1), 1.0 - 1.0/(overlap+1), overlap, device=x_chunks.device, dtype=x_chunks.dtype))
    windows = torch.ones_like(x_chunks)
    right_side = edge_window[None, None, :, None].expand(x_chunks.size(0), x_chunks.size(1)-1, -1, x_chunks.size(3)).flip(2)
    left_side = edge_window[None, None, :, None].expand(x_chunks.size(0), x_chunks.size(1)-1, -1, x_chunks.size(3))
    if right:
        windows[:, :-1, -overlap:] = right_side
    if left:
        windows[:, 1:, :overlap] = left_side
    windowed = x_chunks * windows
    overlap_padding = torch.zeros(x_chunks.size(0), x_chunks.size(1)-1, x_chunks.size(2)-overlap, x_chunks.size(3), device=x_chunks.device, dtype=x_chunks.dtype)
    chunk_padding = torch.zeros(x_chunks.size(0), 1, x_chunks.size(2), x_chunks.size(3), device=x_chunks.device, dtype=x_chunks.dtype)
    out = windowed
    if right:
        out = out + torch.cat((
            torch.cat((
                overlap_padding,
                x_chunks[:, 1:, :overlap] * left_side
            ), dim=2),
            chunk_padding
        ), dim=1)
    if left:
        out = out + torch.cat((
            chunk_padding,
            torch.cat((
                windowed[:, :-1, -overlap:] * right_side,
                overlap_padding
            ), dim=2)
        ), dim=1)
    return out

# encodes x into d/2 sine-cosine pairs with exponentially increasing periods starting from 2
# and ending at 2*max_range if specified, otherwise 2**d
# the minimum distinguishable difference in x value is 1, the maximum is max_range or 2**(d-1)
def sinusoid_encoding(x, d, max_range=None):
    assert (d//2)*2 == d
    phases = x.unsqueeze(-1).expand(*x.shape, d//2)
    if max_range is None:
        half_freqs = torch.exp2(-torch.arange(0, d//2, device=x.device, dtype=torch.float32))
    else:
        half_freqs = torch.exp2(-torch.linspace(0, math.log2(max_range), steps=d//2, device=x.device))
    phases = phases * (torch.pi * half_freqs.view(*(1 for _ in x.shape), half_freqs.size(0)))
    return torch.cat((torch.cos(phases), torch.sin(phases)), dim=-1)

def apply_rope(x, pos_encs):
    c, s = torch.chunk(pos_encs, 2, dim=-1)
    affected_dims = c.size(-1)
    if x.size(-1) < affected_dims:
        affected_dims = (x.size(-1)//2)*2
        c = c[..., :affected_dims]
        s = s[..., :affected_dims]

    if x.size(-1) == affected_dims:
        unaffected = None
    else:
        unaffected = x[..., affected_dims:]
        x = x[..., :affected_dims]

    x1, x2 = torch.chunk(x, 2, dim=-1)
    half_rot = torch.cat((-x2, x1), dim=-1)
    rotated = x * c + half_rot * s
    return rotated if unaffected is None else torch.cat((rotated, unaffected), dim=-1)

def cubic_hermite(t, out=None):
    if out is None:
        out = torch.clone(t)
    elif out is not t:
        out[:] = t
    g = -2.0*t
    g += 3.0
    out *= t
    out *= g
    return out # t^2 * (3 - 2*t)

def hz_to_mel(freqs):
    return 2595.0 * torch.log10(1.0 + freqs / 700.0)

def mel_to_hz(mels):
    return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

def make_mel_filterbank(sr, n_fft, n_bins, dtype=torch.float32):
    fft_freqs = torch.fft.rfftfreq(n=n_fft, d=1.0 / sr, dtype=dtype)
    mel_fft_freqs = hz_to_mel(fft_freqs)
    mel_bin_center_freqs = torch.linspace(mel_fft_freqs[0], mel_fft_freqs[-1], n_bins, dtype=dtype)
    mel_bin_center_diff = mel_bin_center_freqs[1] - mel_bin_center_freqs[0]
    mel_fft_freq_diff = (fft_freqs[1] - fft_freqs[0])*2595.0/((fft_freqs+700.0)*math.log(10))
    mel_center_dist = np.abs(mel_fft_freqs[None, :] - mel_bin_center_freqs[:, None])
    mel_center_dist *= 1.0 / torch.maximum(mel_fft_freq_diff[None, :], mel_bin_center_diff)
    m = 1.0 - mel_center_dist
    torch.clip(m, 0.0, 1.0, out=m)
    cubic_hermite(m, out=m)
    m *= 1.0/m.sum(dim=1, keepdim=True)
    return m

def spectrogram(
        x: torch.Tensor,
        n_fft: int,
        log_magnitude: bool=False
    ):
    window = torch.hann_window(n_fft+1, device=x.device)[1:]
    norm_factor = 2.0 / torch.sum(window)
    s = norm_factor * torch.stft(x, n_fft, window=window, center=True, pad_mode='constant', return_complex=True)
    if log_magnitude:
        s = rescale_complex(s, unitary_log(torch.abs(s)))
    return s

def log_mel_spectrogram(x, sr, n_fft, n_bins):
    fb = make_mel_filterbank(sr, n_fft, n_bins).to(device=x.device)
    return torch.matmul(fb, unitary_log(torch.abs(spectrogram(x, n_fft)))).flip(1)

def log_mel_spectrogram_diff(x, y, sr, n_fft, n_bins):
    fb = make_mel_filterbank(sr, n_fft, n_bins).to(device=x.device)
    m_x = unitary_log(torch.abs(spectrogram(x, n_fft)))
    m_y = unitary_log(torch.abs(spectrogram(y, n_fft)))
    d = m_x - m_y
    ulog_diff = (torch.abs(d) * torch.sign(d) + 1.0) * 0.5
    return torch.matmul(fb, ulog_diff).flip(1)

def log_mel_spectrum(
        x: torch.Tensor,
        sr: int = 44100,
        n_bins: int = 256,
        dim: int=-1
    ):
    n_fft = x.size(dim)
    x_perm = x.transpose(dim, -1)
    window = torch.hann_window(n_fft+1, device=x.device)[1:]
    norm_factor = 2.0 / torch.sum(window)
    x_windowed = x_perm.reshape(-1, n_fft) * window[None]
    fft_mags = unitary_log(torch.abs(torch.fft.rfft(x_windowed)) * norm_factor)
    fb = make_mel_filterbank(sr, n_fft, n_bins).to(device=x.device)
    mel_mags = torch.matmul(fft_mags, fb.transpose(0, 1))
    out = mel_mags.reshape(*x_perm.shape[:-1], n_bins)
    return out.transpose(dim, -1)

def get_unwrapped_phase_diff(
        s: torch.Tensor,
        dim: int
    ) -> torch.Tensor:
    slice_l = tuple(slice(None, -1) if i == dim else slice(None, None) for i in range(s.dim()))
    slice_r = tuple(slice(1, None) if i == dim else slice(None, None) for i in range(s.dim()))
    theta = torch.angle(s)
    d_theta = theta[slice_r] - theta[slice_l]
    d_theta = d_theta - (2.0*torch.pi)*torch.round(d_theta / (2.0*torch.pi))
    return d_theta

def get_stabilized_batch_dropout_mask(p, local_bsz, device):
    global_bsz = local_bsz
    if dist.is_initialized():
        global_bsz *= dist.get_world_size()
    
    mask = torch.empty((global_bsz,), device=device)
    mask.bernoulli_(1.0 - p)
    if mask.sum().item() == 0.0 and p < 1.0:
        mask[torch.randint(global_bsz, (1,)).item()] = 1.0

    if dist.is_initialized():
        dist.broadcast(mask, 0)
        offset = dist.get_rank() * local_bsz
        mask = mask[offset:offset+local_bsz]
    return mask

def unitary_log(x, eps=1e-3):
    log_eps = math.log(eps)
    return (torch.log(x + eps) - log_eps) * (1.0/(math.log(1 + eps) - log_eps))

def unitary_exp(x, eps=1e-3):
    return eps * (torch.pow(1/eps + 1, x) - 1)

def rescale_complex(x, new_mag):
    x_mag = torch.abs(x)
    nonzero = x_mag != 0.0
    scale = new_mag / x_mag.where(nonzero, 1.0)
    return x * scale.where(nonzero, 1.0)

# assumes x is BND or BNHD
def interleave_padding(x: torch.Tensor, period: int) -> torch.Tensor:
    padding = torch.zeros(
        (1,)*x.dim(), device=x.device, dtype=x.dtype
    ).expand(x.shape[0], 1, *x.shape[2:])
    if period == -1:
        return torch.cat((padding, x), dim=1)
    else:
        pad_for_view = (period - 1) - x.size(1) % (period - 1)
        x_p = torch.cat((x, padding.expand(-1, pad_for_view, *x.shape[2:])), dim=1)
        x_p = x_p.view(x_p.size(0), x_p.size(1)//(period-1), period-1, *x_p.shape[2:])
        x_p = torch.cat((padding.unsqueeze(2).expand(-1, x_p.size(1), 1, *x_p.shape[3:]), x_p), dim=2)
        x_p = x_p.flatten(1, 2)
        x_p = x_p[:, :x_p.size(1)-pad_for_view]
        return x_p

def strip_interleaved_padding(x: torch.Tensor, period: int) -> torch.Tensor:
    if period == -1:
        return x[:, 1:]
    else:
        padding = torch.zeros((1,)*x.dim(), device=x.device, dtype=x.dtype)
        pad_for_view = period - x.size(1) % period
        x_p = torch.cat((x, padding.expand(x.size(0), pad_for_view, *x.shape[2:])), dim=1)
        x_p = x_p.view(x_p.size(0), x_p.size(1)//period, period, *x_p.shape[2:])
        x_p = x_p[:, :, 1:].flatten(1, 2)
        x_p = x_p[:, :x_p.size(1)-pad_for_view]
        return x_p

def gaussian_kernel(a: torch.Tensor, b: torch.Tensor, scale, dim=0):
    a = a.unsqueeze(dim)
    b = b.unsqueeze(dim+1)
    return torch.exp(-torch.mean(torch.square(a - b).flatten(dim+2), dim=dim+2) * scale)

# x: (b, d)
def linear_match(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.dim() == 2 and y.dim() == 2 and x.shape[0] <= y.shape[0] and x.shape[1] == y.shape[1]
    dists = torch.cdist(x, y).cpu().numpy()
    row_idxs, col_idxs = linear_sum_assignment(dists)
    assmt = torch.from_numpy(col_idxs).to(device=y.device)
    return torch.gather(y, 0, assmt[:, None].expand_as(x))

def matched_normal_variates(x: torch.Tensor, gather: bool = True) -> torch.Tensor:
    if gather and dist.is_initialized():
        x_gathered = torch.zeros((x.size(0)*dist.get_world_size(), x.size(1)), device=x.device, dtype=x.dtype)
        dist.all_gather_into_tensor(x_gathered, x.contiguous())
        scatter_list = None
        scattered = torch.zeros_like(x)
        if dist.get_rank() == 0:
            matched = linear_match(x_gathered, torch.randn_like(x_gathered))
            scatter_list = list(matched.chunk(dist.get_world_size(), dim=0))
        dist.scatter(scattered, scatter_list=scatter_list, src=0)
        return scattered
    else:
        return linear_match(x, torch.randn_like(x))

class GLU(torch.nn.Module):
    def __init__(self, inner_act: torch.nn.Module, dim: int=-1):
        super().__init__()
        self.inner_act = inner_act
        self.dim = dim
    def forward(self, x):
        x, g = torch.chunk(x, 2, dim=self.dim)
        return x * self.inner_act(g)

class Snake(torch.nn.Module):
    def __init__(self, d: int, min_a: float=1.0, max_a: float=100.0, dim: int=-1, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.log_a = torch.nn.Parameter(
            torch.linspace(math.log(min_a), math.log(max_a), steps=d, device=device, dtype=dtype)
        )
        self.dim = dim
    def forward(self, x):
        shape = [1 for _ in x.shape]
        shape[self.dim] = self.log_a.size(0)
        a = torch.exp(self.log_a).view(*shape)
        return x - (torch.cos(a * x) - 1.0) / a
    
def get_act_fn(act_fn: str, d: int, dim: int=-1, device: torch.device | None = None, dtype: torch.dtype | None = None):
    if act_fn.endswith('glu'):
        return GLU(get_act_fn({
            'swi': 'swish',
            'ge': 'gelu',
            're': 'relu',
            'sna': 'snake',
            '': None
        }[act_fn[:-3]], d//2, dim=dim, device=device, dtype=dtype), dim=dim)
    else:
        return {
            'swish': torch.nn.SiLU,
            'silu': torch.nn.SiLU,
            'gelu': torch.nn.GELU,
            'relu': torch.nn.ReLU,
            'snake': lambda: Snake(d, dim=dim, device=device, dtype=dtype),
            None: torch.nn.Identity
        }[act_fn]()

class TransposeWrapper(torch.nn.Module):
    def __init__(self, inner_module, d1, d2):
        super().__init__()
        self.inner_module = inner_module
        self.d1 = d1
        self.d2 = d2

    def forward(self, x):
        return self.inner_module(x.transpose(self.d1, self.d2)).transpose(self.d1, self.d2)
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.inner_module, name)
        
class PermuteWrapper(torch.nn.Module):
    def __init__(self, inner_module, permuted_dims):
        super().__init__()
        self.inner_module = inner_module
        self.permute_dims = permuted_dims
        unpermute_dims = [0 for _ in permuted_dims]
        for i, d in enumerate(permuted_dims):
            unpermute_dims[d] = i
        self.unpermute_dims = tuple(unpermute_dims)

    def forward(self, x):
        return self.inner_module(x.permute(self.permute_dims)).permute(self.unpermute_dims)
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.inner_module, name)

class noop_ctx:
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass

def clone_and_freeze_module(module):
    """
    Adapted from https://github.com/learnables/learn2learn/blob/master/learn2learn/utils.py

    Creates a copy of a module, whose parameters/buffers/submodules
    are detached using PyTorch's torch.detach().
    """

    # First, create a copy of the module.
    # Adapted from:
    # https://github.com/pytorch/pytorch/blob/65bad41cbec096aa767b3752843eddebf845726f/torch/nn/modules/module.py#L1171
    if not isinstance(module, torch.nn.Module):
        return module
    clone = module.__new__(type(module))
    clone.__dict__ = module.__dict__.copy()
    clone._parameters = clone._parameters.copy()
    clone._buffers = clone._buffers.copy()
    clone._modules = clone._modules.copy()

    # Second, re-write all parameters
    if hasattr(clone, '_parameters'):
        for param_key in module._parameters:
            if module._parameters[param_key] is not None:
                clone._parameters[param_key] = module._parameters[param_key].detach()

    # Third, handle the buffers if necessary
    if hasattr(clone, '_buffers'):
        for buffer_key in module._buffers:
            if clone._buffers[buffer_key] is not None and \
                    clone._buffers[buffer_key].requires_grad:
                clone._buffers[buffer_key] = module._buffers[buffer_key].detach()

    # Then, recurse for each submodule
    if hasattr(clone, '_modules'):
        for module_key in clone._modules:
            clone._modules[module_key] = clone_and_freeze_module(
                module._modules[module_key]
            )
    
    return clone

@torch.no_grad()
def update_ema(target_module, source_module, target_coef):
    for param_key in target_module._parameters:
        if target_module._parameters[param_key] is not None:
            source = source_module._parameters[param_key]
            if target_module._parameters[param_key].data_ptr() == source.data_ptr():
                target_module._parameters[param_key] = source.detach().clone()
            target_module._parameters[param_key] *= target_coef
            target_module._parameters[param_key] += (1.0 - target_coef) * source.detach()
    if hasattr(target_module, '_modules'):
        for module_key in target_module._modules:
            update_ema(target_module._modules[module_key], source_module._modules[module_key], target_coef)
