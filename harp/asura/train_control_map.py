import torch
import numpy as np
from copy import deepcopy
from .modules import MLP
from harp.modeling.distributions import dist_from_config
from harp.modeling.util import sinusoid_encoding
import math
import json
import os
import sys
import hashlib
import base64
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import TensorDataset, DataLoader
from itertools import cycle

def save_config(cfg, path):
    with open(path, mode='w') as f:
        json.dump(cfg, f, indent=4)

def config_hash(cfg):
    return base64.b16encode(hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).digest())[:6].decode().lower()

def override_config(cfg, overrides):
    for k, v in overrides.items():
        if k in cfg and isinstance(cfg[k], dict) and isinstance(v, dict):
            override_config(cfg[k], v)
        else:
            cfg[k] = v

def parse_bool(s):
    s = s.lower()
    if s == 'true':
        return True
    elif s == 'false':
        return False
    else:
        raise ValueError('Not a bool!')

def load_config(base_config=None):
    args = sys.argv[1:]
    config = {} if base_config is None else base_config
    if len(args) > 0 and ('=' not in args[0]) and os.path.isfile(args[0]):
        with open(args.pop(0)) as config_file:
            overrides = json.load(config_file)
        override_config(config, overrides)
    for arg in args:
        arg, value_str = arg.split('=', maxsplit=1)
        key_path = arg.split('.')
        value = None
        for parser in (int, float, parse_bool, lambda v: v):
            try:
                value = parser(value_str)
            except ValueError:
                pass
            else:
                break
        curr = config
        for path_entry in key_path[:-1]:
            if path_entry not in curr:
                curr[path_entry] = {}
            curr = curr[path_entry]
        curr[key_path[-1]] = value
    return config

class BAF(torch.nn.Module): # Bidirectional Amortized Flow
    def __init__(self, *,
            d_x, d_h, layers=2, act_fn='silu',
            d_c, t_scale,
            p_t,
            autoguidance=2.0,
            ema,
            adaptive_loss_cap=1000.0,
            use_bwd_u_loss=False,
            y_anchor,
            source='normal',
            device=None
        ):
        super().__init__()
        self.v_net = MLP(d_x+d_c, d_h=d_h, d_out=d_x, layers=layers, act_fn=act_fn, device=device)
        self.v_net_ema = MLP(d_x+d_c, d_h=d_h, d_out=d_x, layers=layers, act_fn=act_fn, device=device)
        self.u_net = MLP(d_x+d_c, d_h=d_h, d_out=d_x, layers=layers, act_fn=act_fn, device=device)
        self.u_net_ema = MLP(d_x+d_c, d_h=d_h, d_out=d_x, layers=layers, act_fn=act_fn, device=device)
        self.y_net = MLP(d_x, d_h=d_h, layers=layers, act_fn=act_fn, device=device)
        self.y_net_ema = MLP(d_x, d_h=d_h, layers=layers, act_fn=act_fn, device=device)
        for p, p_ema in zip(self.v_net.parameters(), self.v_net_ema.parameters()):
            p_ema.data = p.data.clone().detach()
        for p, p_ema in zip(self.u_net.parameters(), self.u_net_ema.parameters()):
            p_ema.data = p.data.clone().detach()
        for p, p_ema in zip(self.y_net.parameters(), self.y_net_ema.parameters()):
            p_ema.data = p.data.clone().detach()
        self.ema = ema
        self.w_v_net = MLP(d_c, d_h=d_c*2, layers=2, d_out=1, device=device)
        self.d_c = d_c
        self.t_scale = t_scale
        self.p_t = dist_from_config(p_t, device=device)
        self.autoguidance = autoguidance
        self.adaptive_loss_min_w = -math.log(adaptive_loss_cap)
        self.use_bwd_u_loss = use_bwd_u_loss
        self.y_anchor = y_anchor
        self.source = source
        assert y_anchor in {'midpoint', 'source', 'target', 'source_target'}

    def get_c_v(self, t):
        return sinusoid_encoding(t * self.t_scale, self.d_c, max_range=self.t_scale)

    def get_c_u(self, t, r):
        return torch.cat((
            sinusoid_encoding(t * self.t_scale, self.d_c//2, max_range=self.t_scale),
            sinusoid_encoding((r - t) * self.t_scale, self.d_c//2, max_range=2.0 * self.t_scale)
        ), dim=-1)

    def sample_source(self, size, device):
        if self.source == 'uniform':
            return torch.rand(size, device=device) * 2.0 - 1.0
        else:
            return torch.randn(size, device=device)
    
    def sample_source_like(self, t):
        return self.sample_source(t.size(), t.device)

    def forward(self, x):
        bsz = x.size(0)
        t_v = self.p_t.sample((bsz,))
        t_a = self.p_t.sample((bsz,))
        t_b = self.p_t.sample((bsz,))
        s = torch.sigmoid(torch.randn_like(t_a))
        t_m = t_a + s * (t_b - t_a)
        e_v = self.sample_source_like(x)
        e_u = self.sample_source_like(x)
        e_y = self.sample_source_like(x)
        t_v_like_x = t_v.view(t_v.size(0), *(1 for _ in x.shape[1:]))
        t_a_like_x = t_a.view(t_a.size(0), *(1 for _ in x.shape[1:]))
        s_like_x = s.view(s.size(0), *(1 for _ in x.shape[1:]))
        z_v = (1.0-t_v_like_x) * e_v + t_v_like_x * x
        z_a = (1.0-t_a_like_x) * e_u + t_a_like_x * x
        v = x - e_v
        with torch.no_grad():
            v_a = self.v_net(torch.cat((z_a, self.get_c_v(t_a)), dim=-1))
            u_aa, u_am = torch.chunk(self.u_net(
                torch.cat((
                    torch.cat((z_a, z_a), dim=0),
                    self.get_c_u(
                        torch.cat((t_a, t_a), dim=0),
                        torch.cat((t_a, t_m), dim=0)
                    )
                ), dim=-1)
            ), 2, dim=0)
            z_m = z_a + (t_m - t_a).view(bsz, *(1 for _ in x.shape[1:]))*u_am
            u_mb = self.u_net(torch.cat((z_m, self.get_c_u(t_m, t_b)), dim=-1))
            u_avg_ab = s_like_x * u_am + (1.0 - s_like_x) * u_mb
            if self.use_bwd_u_loss:
                z_b = z_a + (t_b - t_a).view(bsz, *(1 for _ in x.shape[1:]))*u_avg_ab

        if self.use_bwd_u_loss:
            u_hat_ab, u_hat_ba = torch.chunk(self.u_net(
                torch.cat((
                    torch.cat((z_a, z_b), dim=0),
                    self.get_c_u(
                        torch.cat((t_a, t_b), dim=0),
                        torch.cat((t_b, t_a), dim=0)
                    )
                ), dim=-1)
            ), 2, dim=0)
        else:
            u_hat_ab = self.u_net(torch.cat((z_a, self.get_c_u(t_a, t_b)), dim=-1))

        v_hat = self.v_net(torch.cat((z_v, self.get_c_v(t_v)), dim=-1))

        with torch.no_grad():
            u_tgt = self.autoguidance * u_avg_ab + (1.0 - self.autoguidance) * u_hat_ab + (v_a - u_aa)
            x_y = x
            if self.y_anchor == 'source':
                t_y_a = torch.zeros_like(t_a)
                t_y_b = torch.ones_like(t_a)
            elif self.y_anchor == 'target':
                t_y_a = torch.ones_like(t_a)
                t_y_b = torch.zeros_like(t_a)
            elif self.y_anchor == 'source_target':
                t_y_a = torch.zeros_like(t_a)
                t_y_a[bsz//2:] = 1.0
                t_y_b = torch.zeros_like(t_a)
                t_y_b[:bsz//2] = 1.0
            s_y = torch.sigmoid(torch.randn_like(t_y_a))
            s_y_like_x = s_y.view(t_y_a.size(0), *(1 for _ in x.shape[1:]))
            t_y_m = t_y_a + s_y * (t_y_b - t_y_a)
            t_y_a_like_x = t_y_a.view(t_y_a.size(0), *(1 for _ in x.shape[1:]))
            z_y_a = (1.0 - t_y_a_like_x) * e_y + t_y_a_like_x * x_y
            y_am = self.u_net(torch.cat((z_y_a, self.get_c_u(t_y_a, t_y_m)), dim=-1))
            z_y_m = z_y_a + (t_y_m - t_y_a).view(t_y_a.size(0), *(1 for _ in x.shape[1:]))*y_am
            y_mb = self.u_net(torch.cat((z_y_m, self.get_c_u(t_y_m, t_y_b)), dim=-1))
            y_tgt = s_y_like_x * y_am + (1.0 - s_y_like_x) * y_mb
            x_y = z_y_a[:bsz] - t_y_a_like_x[:bsz] * y_tgt[:bsz]

        y_hat = self.y_net(x_y)
            
        w_v = self.w_v_net(self.get_c_v(t_v))
        w_v = torch.nn.functional.softplus(w_v - self.adaptive_loss_min_w) + self.adaptive_loss_min_w
        loss_v = torch.mean(w_v + torch.exp(-w_v)*torch.mean(torch.square(v_hat - v), dim=tuple(range(1, v.dim()))))
        # loss_v = torch.mean(torch.square(v_hat - v))
        loss_u_fwd = torch.mean(torch.square(u_hat_ab - u_tgt))
        loss = loss_v + loss_u_fwd
        if self.use_bwd_u_loss:
            loss = loss + torch.mean(torch.square(u_hat_ba - u_avg_ab))
        loss = loss + torch.mean(torch.square(y_hat - y_tgt))
        return loss

    @torch.no_grad
    def sample(self, noise, steps=1, return_all=False):
        y = noise + self.y_net_ema(noise)
        if return_all:
            return [y] * steps
        else:
            return y

    @torch.no_grad
    def update_ema(self):
        for p, p_ema in zip(self.v_net.parameters(), self.v_net_ema.parameters()):
            p_ema.data = self.ema * p_ema.data + (1-self.ema) * p.data
        for p, p_ema in zip(self.u_net.parameters(), self.u_net_ema.parameters()):
            p_ema.data = self.ema * p_ema.data + (1-self.ema) * p.data
        for p, p_ema in zip(self.y_net.parameters(), self.y_net_ema.parameters()):
            p_ema.data = self.ema * p_ema.data + (1-self.ema) * p.data


def train(cfg: dict):
    run_name = 'control_map_'+datetime.now().strftime('%y%m%d-%H%M')
    ckpt_dir = os.path.join('checkpoints', run_name)
    log_dir = os.path.join('logs', run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    save_config(cfg, os.path.join(ckpt_dir, 'config.json'))

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')

    data = torch.from_numpy(np.load(cfg.pop('data')))
    d_x = data.size(-1)
    override_config(cfg['model'], {'d_x': d_x})

    model = BAF(**cfg.pop('model'), device=device)

    batch_size = cfg.pop('batch_size')
    warmup = cfg.pop('warmup')
    train_steps = cfg.pop('train_steps')

    optim = torch.optim.Adam(model.parameters(), lr=cfg.pop('lr'))
    if cfg.pop('lr_decay'):
        lr_fn = lambda step: (step / warmup if step < warmup else 1.0 - (step - warmup) / max(1, train_steps - warmup))
    else:
        lr_fn = lambda step: (step / warmup if step < warmup else 1.0)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_fn)
    step_iter = tqdm(range(train_steps))

    log_every = cfg.pop('log_every')
    if isinstance(log_every, float):
        log_every = round(log_every * train_steps)
    sample_batch_size = cfg.pop('sample_batch_size')

    dataloader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True)
    batch_iter = cycle(dataloader)

    if len(cfg) > 0:
        print("Warning! Unused config entries:", ', '.join(cfg.keys()))

    fixed_sample_sources = model.sample_source((sample_batch_size, d_x), device=device)
    
    for step in step_iter:
        x, = next(batch_iter)
        x = x.to(device=device)
        loss = model(x)
        step_iter.set_description(f'Loss = {loss.item():.3g}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        lr_scheduler.step()
        optim.zero_grad()
        if hasattr(model, 'update_ema'):
            model.update_ema()
        if log_every > 0 and (step + 1) % log_every == 0:
            np.save(os.path.join(log_dir, f'samples_step{step+1}.npy'), model.sample(fixed_sample_sources).cpu().numpy())
            torch.save(model.state_dict(), os.path.join(ckpt_dir, 'control_map.pt'))
    torch.save(model.state_dict(), 'control_map.pt')

if __name__ == "__main__":
    torch.manual_seed(42)
    config = {
        "data": "embeddings.npy",
        "model": {
            "d_h": 2048,
            "d_c": 64,
            "ema": 0.9999,
            "t_scale": 1000,
            "p_t": "uniform",
            "autoguidance": 2.0,
            "y_anchor": "source_target",
            "use_bwd_u_loss": True,
            "source": "normal"
        },
        "train_steps": 2000000,
        "batch_size": 256,
        "lr": 1e-4,
        "lr_decay": False,
        "warmup": 10000,
        "log_every": 0.01,
        "sample_batch_size": 1024
    }
    train(load_config(config))
