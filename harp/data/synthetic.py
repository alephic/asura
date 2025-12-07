from torch.utils.data.dataset import Dataset
import torch

class NoiseDataset(Dataset):
    def __init__(self, size: int, example_length: int, channels=2):
        self.size = size
        self.example_length = example_length
        self.channels = channels
    def __len__(self):
        return self.size
    def __getitem__(self, i):
        return torch.rand(self.example_length, self.channels) * 2.0 - 1.0

class ClickDataset(Dataset):
    def __init__(self, size: int, example_length: int, click_interval: int, channels: int = 2):
        self.size = size
        self.example_length = example_length
        self.channels = channels
        self.click_interval = click_interval
    def __len__(self):
        return self.size
    def __getitem__(self, i):
        t = torch.zeros(self.example_length, self.channels)
        t[torch.randint(self.click_interval, tuple()).item()::self.click_interval] = 1.0
        return t

class NoiseBurstDataset(Dataset):
    def __init__(self, size: int, example_length: int, burst_interval_min: int, burst_interval_max: int, channels: int = 2):
        self.size = size
        self.example_length = example_length
        self.channels = channels
        self.burst_interval_min = burst_interval_min
        self.burst_interval_max = burst_interval_max
    def __len__(self):
        return self.size
    def __getitem__(self, i):
        burst_interval = torch.randint(self.burst_interval_min, self.burst_interval_max, tuple()).item()
        t = torch.zeros(self.example_length, self.channels)
        i = torch.arange(self.example_length)[:, None]
        i += torch.randint(burst_interval, tuple())
        i %= burst_interval
        return torch.where(i >= burst_interval//2, torch.rand_like(t)*2.0 - 1.0, t)

class SinWobbleDataset(Dataset):
    def __init__(self, size: int, example_length: int, freq_min: float, freq_max: float, lfo_min: float, lfo_max: float, channels: int=2, sr=44100):
        self.size = size
        self.example_length = example_length
        self.channels = channels
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.lfo_min = lfo_min
        self.lfo_max = lfo_max
        self.sr = sr
    def __len__(self):
        return self.size
    def __getitem__(self, i):
        example_length_s = self.example_length/self.sr
        f_lo, f_hi = (torch.rand(2)*(self.freq_max - self.freq_min) + self.freq_min).tolist()
        f_lfo = (torch.rand(tuple())*(self.lfo_max - self.lfo_min) + self.lfo_min).item()
        f_env = 0.5*(1.0 + torch.sin(torch.linspace(0, example_length_s*f_lfo*2.0*torch.pi, self.example_length) + torch.rand(tuple())*2.0*torch.pi)) \
            * (f_hi - f_lo) + f_lo
        inc = f_env * (2.0*torch.pi/self.sr)
        return torch.sin(torch.cumsum(inc, dim=0))[:, None].expand(-1, self.channels)

class MultiWobbleDataset(SinWobbleDataset):
    def __init__(self, size: int, example_length: int, max_voices: int, freq_min: float, freq_max: float, lfo_min: float, lfo_max: float, channels: int=2, sr=44100):
        super(MultiWobbleDataset, self).__init__(size, example_length, freq_min, freq_max, lfo_min, lfo_max, channels=channels, sr=sr)
        self.max_voices = max_voices

    def __getitem__(self, i):
        n_voices = torch.randint(0, self.max_voices, tuple()).item() + 1
        return torch.mean(torch.stack([super(MultiWobbleDataset, self).__getitem__(i) for _ in range(n_voices)]), dim=0)