import torch

def dist_from_config(cfg: dict | str, device=None):
    if isinstance(cfg, str):
        cfg = {'type': cfg}
    return {
        'beta': lambda: torch.distributions.Beta(
            torch.tensor(cfg['alpha'], device=device),
            torch.tensor(cfg['beta'], device=device)
        ),
        'uniform': lambda: torch.distributions.Uniform(
            torch.tensor(cfg.get('low', 0.0), device=device),
            torch.tensor(cfg.get('high', 1.0), device=device)
        ),
        'logit_normal': lambda: LogitNormal(
            torch.tensor(cfg.get('mean', 0.0), device=device),
            torch.tensor(cfg.get('scale', 1.0), device=device)
        ),
        'mixture': lambda: Mixture(
            torch.tensor(cfg['p'], device=device),
            dist_from_config(cfg['a'], device=device),
            dist_from_config(cfg['b'], device=device)
        ),
        'tri': lambda: Triangular(cfg.get('scale', 1.0), device),
        'trilo': lambda: TriangularLow(cfg.get('scale', 1.0), device)
    }[cfg['type']]()

class LogitNormal:
    def __init__(self, logit_mean, logit_scale):
        self.logit_mean = logit_mean
        self.logit_scale = logit_scale

    def sample(self, size):
        return torch.sigmoid(torch.randn(size, device=self.logit_mean.device) * self.logit_scale + self.logit_mean)

class Mixture:
    def __init__(self, proportion, dist_a, dist_b):
        self.proportion = proportion
        self.dist_a = dist_a
        self.dist_b = dist_b
    def sample(self, size):
        return torch.where(torch.rand(size, device=self.proportion.device) < self.proportion,
                           self.dist_a.sample(size), self.dist_b.sample(size))

class Triangular:
    def __init__(self, scale, device):
        self.scale = scale
        self.device = device
    def sample(self, size):
        return self.scale * torch.sqrt(torch.rand(size, device=self.device))
    
class TriangularLow:
    def __init__(self, scale, device):
        self.scale = scale
        self.device = device
    def sample(self, size):
        return self.scale * (1.0 - torch.sqrt(torch.rand(size, device=self.device)))