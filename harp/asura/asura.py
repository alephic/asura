
import torch
from harp.modeling.util import sinusoid_encoding, smooth_fold_chunks, unitary_exp, unitary_log, rescale_complex, fold, warmup_weight
from .modules import UNet, Transformer, FFTNet, MLP
from harp.modeling.util import (
    clone_and_freeze_module,
    update_ema,
    smoothstep
)
from harp.modeling.distributions import dist_from_config
import json
import os
import torch.distributed as dist
from tqdm import trange

class ASURA(torch.nn.Module):
    def __init__(self,
            *,
            token_encoder: dict,
            decoder_m: dict,
            decoder_phi: dict,
            seq_predictor: dict,
            seq_encoder: dict | None = None,
            d_token: int,
            d_signal: int,
            d_t_enc: int = 64,
            decoder_lookback_frames: int,
            t_scale: float = 1000.0,
            frame_samples: int | None = None,
            frame_overlap_samples: int = 0,
            frames_per_token: int = 1,
            seq_pred_extra_future: int = 0,
            length_tokens: int,
            seq_enc_length_tokens: int | None = None,
            pred_grad_frames: int = -1,
            p_eps_ctx_global: str | dict,
            p_eps_ctx_frame: str | dict,
            p_t: str | dict,
            p_s: str | dict,
            substep_autoguidance: float = 1.0,
            c_enc_drop: float = 0.0,
            prior: str = 'uniform',
            sample_eps_ctx: float = 0.0,
            ema: float = 0.9999,
            adapt_w_max: float = 7.0,
            train_v: bool = True,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None
        ):
        super().__init__()
        self.d_signal = d_signal
        self.d_token = d_token
        self.decoder_lookback_frames = decoder_lookback_frames
        self.d_signal_enc = self.d_signal * 2
        self.seq_pred_extra_future = seq_pred_extra_future
        self.frame_samples = frame_samples
        self.frame_bins = frame_samples//2
        self.frames_per_token = frames_per_token

        seq_predictor = dict(seq_predictor)
        seq_predictor['d_in'] = self.d_token
        seq_predictor['d_out'] = self.d_token * (1 + seq_pred_extra_future) * frames_per_token
        seq_predictor['causal'] = True
        seq_predictor['num_scratch_states'] = 1
        seq_predictor['pos_enc_length'] = length_tokens
        self.seq_predictor = Transformer(**seq_predictor, device=device, dtype=dtype)
        self.seq_predictor_ema = clone_and_freeze_module(self.seq_predictor)

        seq_encoder = dict(seq_encoder)
        seq_encoder['d_in'] = self.d_token
        seq_encoder['d_out'] = self.d_token
        seq_encoder['causal'] = False
        seq_encoder['num_scratch_states'] = 1
        seq_encoder['pos_enc_length'] = seq_enc_length_tokens
        self.seq_encoder = Transformer(**seq_encoder, device=device, dtype=dtype)
        self.seq_encoder_ema = clone_and_freeze_module(self.seq_encoder)

        decoder_m = dict(decoder_m)
        decoder_m['d_in'] = (1 + self.decoder_lookback_frames) * self.d_signal
        decoder_m['d_aux'] = self.d_token
        decoder_m['d_out'] = self.d_signal
        decoder_m['bins_in'] = self.frame_bins
        self.decoder_v_m = UNet(**decoder_m, device=device, dtype=dtype)
        self.decoder_u_m = UNet(**decoder_m, device=device, dtype=dtype)
        self.decoder_y_m = UNet(**decoder_m, device=device, dtype=dtype)
        self.decoder_v_m_ema = clone_and_freeze_module(self.decoder_v_m)
        self.decoder_u_m_ema = clone_and_freeze_module(self.decoder_u_m)
        self.decoder_y_m_ema = clone_and_freeze_module(self.decoder_y_m)

        decoder_phi = dict(decoder_phi)
        decoder_phi['d_in'] = (1 + self.decoder_lookback_frames) * (self.d_signal_enc + self.d_signal)
        decoder_phi['d_aux'] = self.d_token
        decoder_phi['d_out'] = self.d_signal_enc
        decoder_phi['bins_in'] = self.frame_bins
        self.decoder_v_phi = UNet(**decoder_phi, device=device, dtype=dtype)
        self.decoder_u_phi = UNet(**decoder_phi, device=device, dtype=dtype)
        self.decoder_y_phi = UNet(**decoder_phi, device=device, dtype=dtype)
        self.decoder_v_phi_ema = clone_and_freeze_module(self.decoder_v_phi)
        self.decoder_u_phi_ema = clone_and_freeze_module(self.decoder_u_phi)
        self.decoder_y_phi_ema = clone_and_freeze_module(self.decoder_y_phi)

        self.d_t_enc = d_t_enc
        self.t_scale = t_scale
        self.c_v_net = MLP(d_t_enc + 2*d_token, d_h=2*d_token, layers=2, d_out=d_token, act_fn='swiglu', device=device)
        self.c_u_net = MLP(2*d_t_enc + 2*d_token, d_h=2*d_token, layers=2, d_out=d_token, act_fn='swiglu', device=device)
        self.c_y_net = MLP(2*d_token, d_h=2*d_token, layers=2, d_out=d_token, act_fn='swiglu', device=device)
        self.c_v_net_ema = clone_and_freeze_module(self.c_v_net)
        self.c_u_net_ema = clone_and_freeze_module(self.c_u_net)
        self.c_y_net_ema = clone_and_freeze_module(self.c_y_net)

        self.w_net = MLP(d_t_enc, d_h=d_t_enc, layers=2, d_out=1, device=device)
        # self.w_v = torch.nn.Parameter(torch.zeros(tuple(), device=device))

        self.ema_weight = ema

        token_encoder = dict(token_encoder)
        token_encoder['d_in'] = self.d_signal * self.frames_per_token
        cumulative_k = 1
        for k in token_encoder['conv_k_stages']:
            cumulative_k *= k
        bottleneck_size = self.frame_bins // cumulative_k
        token_encoder['down_only'] = True
        token_encoder['d_bottleneck'] = self.d_token // bottleneck_size
        self.token_encoder = UNet(**token_encoder, device=device, dtype=dtype)
        self.token_encoder_ema = clone_and_freeze_module(self.token_encoder)
        
        self.length_tokens = self.example_length_tokens = length_tokens
        self.length_frames = self.length_tokens * frames_per_token
        self.frame_overlap_samples = frame_overlap_samples
        self.frame_hop = self.frame_samples - self.frame_overlap_samples
        self.length_samples = (self.length_frames - 1) * self.frame_hop + self.frame_samples
        self.pred_grad_frames = pred_grad_frames
        self.seq_enc_length_tokens = seq_enc_length_tokens
        self.example_length_samples = self.length_samples
        self.seq_enc_length_frames = self.seq_enc_length_tokens * frames_per_token
        self.seq_enc_length_samples = (self.seq_enc_length_frames - 1) * self.frame_hop + self.frame_samples
        self.example_length_tokens += self.seq_enc_length_tokens
        self.example_length_samples += self.seq_enc_length_samples
        
        self.p_eps_ctx_global = dist_from_config(p_eps_ctx_global, device=device)
        self.p_eps_ctx_frame = dist_from_config(p_eps_ctx_frame, device=device)
        self.p_t = dist_from_config(p_t, device=device)
        self.p_s = dist_from_config(p_s, device=device)
        self.prior = prior
        self.sample_eps_ctx = sample_eps_ctx

        self.c_enc_drop = c_enc_drop
        self.substep_autoguidance = substep_autoguidance
        self.adapt_w_max = adapt_w_max

        self.train_v = train_v
        if not train_v:
            self.token_encoder.requires_grad_(False)
            self.seq_predictor.requires_grad_(False)
            self.seq_encoder.requires_grad_(False)
            self.c_v_net.requires_grad_(False)
            self.decoder_v_m.requires_grad_(False)
            self.decoder_v_phi.requires_grad_(False)
            self.w_net.requires_grad_(False)

    @classmethod
    def from_checkpoint(cls, ckpt_path, device: torch.device | None = None):
        with open(os.path.join(ckpt_path, 'config.json'), encoding='utf-8') as config_file:
            config = json.load(config_file)
            model_config = config['model']
        model = cls(**model_config, device=device)
        state_dict = torch.load(os.path.join(ckpt_path, 'model.pt'), map_location=device)
        # unwrap DDP param paths
        state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        return model

    def get_c_y(self, c_enc, c_ctx, ema=False):
        c_net = self.c_y_net_ema if ema else self.c_y_net
        return c_net(torch.cat((c_enc, c_ctx), dim=-1))

    def get_c_v(self, t, c_enc, c_ctx, ema=False):
        c_net = self.c_v_net_ema if ema else self.c_v_net
        return c_net(torch.cat((
            sinusoid_encoding(t*self.t_scale, self.d_t_enc, max_range=self.t_scale).to(dtype=c_enc.dtype),
            c_enc, c_ctx
        ), dim=-1))
    
    def get_c_u(self, t, r, c_enc, c_ctx, ema=False):
        c_net = self.c_u_net_ema if ema else self.c_u_net
        return c_net(torch.cat((
            sinusoid_encoding(t*self.t_scale, self.d_t_enc, max_range=self.t_scale).to(dtype=c_enc.dtype),
            sinusoid_encoding((r - t)*self.t_scale, self.d_t_enc, max_range=self.t_scale*2.0).to(dtype=c_enc.dtype),
            c_enc, c_ctx
        ), dim=-1))
    
    def get_w(self, t):
        w = self.w_net(
            sinusoid_encoding(t*1000.0, self.d_t_enc, max_range=1000.0)
        ).squeeze(1)
        return torch.clamp(w, max=self.adapt_w_max)
    
    def signal_to_frames(self, x: torch.Tensor):
        frames = x.unfold(1, self.frame_samples, self.frame_hop).transpose(2, 3)
        # bsz, n_chunks, chunk_size, d
        window = torch.hann_window(self.frame_samples+1, device=x.device)[1:]
        norm_factor = 2.0 / torch.sum(window)
        s = torch.fft.rfft(frames.transpose(2, 3) * window[None, None, None, :])[..., :self.frame_bins] * norm_factor
        # conjugate odd frequencies to make signal friendlier (1-shift-invariant) for unet
        # TODO this is specific to hop_length*4 = window_length
        s = s * (((torch.arange(self.frame_bins, device=x.device) + 1) % 2) * 2 - 1).to(dtype=s.dtype)[None, None, None, :]
        s_m = unitary_log(torch.abs(s))
        s_phi = rescale_complex(s, s_m)
        frames_m = s_m.transpose(2, 3)
        frames_phi = torch.view_as_real(s_phi).transpose(2, 3).flatten(3)
        return frames_m, frames_phi

    def frames_to_signal(self, frames: torch.Tensor):
        window = torch.hann_window(self.frame_samples+1, device=frames.device)[1:]
        norm_factor = 2.0 / torch.sum(window)
        s = torch.view_as_complex(frames.view(*frames.shape[:-1], frames.shape[-1]//2, 2)).transpose(2, 3)
        # conjugate odd frequencies, inverse of conjugate in signal_to_chunks
        s = s * (((torch.arange(self.frame_bins, device=frames.device) + 1) % 2) * 2 - 1).to(dtype=s.dtype)[None, None, None, :]
        s = torch.cat((rescale_complex(s, unitary_exp(torch.abs(s))), torch.zeros((*s.shape[:-1], 1), dtype=s.dtype, device=s.device)), dim=-1) / norm_factor
        frames = torch.fft.irfft(s).transpose(2, 3) * window[None, None, : , None]
        pad_frame = torch.zeros_like(frames[:, :1])
        frames = torch.cat((pad_frame, frames, pad_frame), dim=1)
        # bsz, n_chunks, chunk_size, d
        window_folded = fold(torch.square(window)[None].expand(frames.size(1), -1), 0, self.frame_hop)
        frames_folded = fold(frames.transpose(2, 3), 1, self.frame_hop)
        frames_folded = frames_folded / window_folded[None, :, None]
        # bsz, l, d
        return frames_folded[:, self.frame_hop:-self.frame_hop]
    
    def clip_frames(self, frames: torch.Tensor):
        channels = frames.view(*frames.shape[:-1], frames.shape[-1]//2, 2)
        a = torch.linalg.norm(channels, dim=-1)
        return torch.flatten(channels / torch.clamp(a, min=1.0)[..., None], -2)
    
    def sample_prior(self, size: tuple, device: torch.device = None, dtype: torch.dtype = torch.float32):
        if self.prior == 'normal':
            return torch.randn(size, device=device, dtype=dtype)
        elif self.prior == 'uniform':
            if size[-1] == self.d_signal:
                return torch.rand(size, device=device, dtype=dtype)
            theta = torch.rand((*size[:-1], size[-1]//2), device=device, dtype=dtype) * (2.0*torch.pi)
            r = torch.sqrt(torch.rand_like(theta))
            e = torch.polar(r, theta) * r
            return torch.view_as_real(e).flatten(-2)

    def sample_prior_like(self, x: torch.Tensor):
        return self.sample_prior(x.shape, device=x.device, dtype=x.dtype)

    def frames_to_tokens(self, frames: torch.Tensor, ema=False):
        assert frames.size(1) % self.frames_per_token == 0, f'encoding {frames.size(1)} frames'
        frames_grouped = frames.reshape(
            frames.size(0), frames.size(1)//self.frames_per_token, self.frames_per_token, frames.size(2), frames.size(3)
        ).transpose(2, 3).flatten(3)
        token_encoder = self.token_encoder_ema if ema else self.token_encoder
        return token_encoder(frames_grouped.flatten(0, 1)).view(frames_grouped.size(0), frames_grouped.size(1), self.d_token)

    # x_clean: (batch_size, self.length_samples, d_signal)
    def forward(self, x: torch.Tensor):
        device = x.device
        bsz = x.size(0)

        x_m_frames, x_phi_frames = self.signal_to_frames(x[:, :self.length_samples])
        # bsz, self.length_tokens, self.token_samples, self.d_signal
        y_m_frames = x_m_frames
        y_phi_frames = x_phi_frames

        eps_ctx = self.p_eps_ctx_global.sample((bsz, 1, 1, 1)) * self.p_eps_ctx_frame.sample((bsz, x_m_frames.size(1), 1, 1))

        x_m_frames = (1.0 - eps_ctx) * x_m_frames + eps_ctx * self.sample_prior_like(x_m_frames)
        x_phi_frames = (1.0 - eps_ctx) * x_phi_frames + eps_ctx * self.sample_prior_like(x_phi_frames)
        all_prev_x_m_frames = []
        all_prev_x_phi_frames = []
        prev_x_m_frames = x_m_frames
        prev_x_phi_frames = x_phi_frames
        for _ in range(self.decoder_lookback_frames):
            prev_x_m_frames = prev_x_m_frames.roll(1, 1)
            prev_x_phi_frames = prev_x_phi_frames.roll(1, 1)
            prev_x_m_frames[:, 0] = self.sample_prior_like(prev_x_m_frames[:, 0])
            prev_x_phi_frames[:, 0] = self.sample_prior_like(prev_x_phi_frames[:, 0])
            all_prev_x_m_frames.append(prev_x_m_frames)
            all_prev_x_phi_frames.append(prev_x_phi_frames)
        all_prev_x_m_frames.reverse()
        all_prev_x_phi_frames.reverse()
        prev_x_m_frames = torch.cat(all_prev_x_m_frames, dim=-1)
        prev_x_phi_frames = torch.cat(all_prev_x_phi_frames, dim=-1)
        del all_prev_x_m_frames
        del all_prev_x_phi_frames

        seq_enc_offsets = torch.randint(0, self.length_samples, (bsz, 1), device=device)
        seq_enc_idxs = torch.arange(self.seq_enc_length_samples, device=device)[None] + seq_enc_offsets
        seq_enc_samples = torch.gather(x, 1, seq_enc_idxs[:, :, None].expand(-1, -1, *x.shape[-1:]))
        seq_enc_frames, _ = self.signal_to_frames(seq_enc_samples)

        all_enc_frames = torch.cat((x_m_frames[:, :-self.frames_per_token], seq_enc_frames), dim=1)

        all_encs: torch.Tensor = self.frames_to_tokens(all_enc_frames)

        seq_enc_in = all_encs[:, self.length_tokens-1:]
        c_enc = self.seq_encoder(seq_enc_in, include_scratch_outputs=1, do_checkpoint=True)[:, 0]
        
        ctx_encs = all_encs[:, :self.length_tokens-1]
        c_ctx = self.seq_predictor(ctx_encs, include_scratch_outputs=1, do_checkpoint=True)
        if self.seq_pred_extra_future > 0:
            c_ctx_alts = torch.chunk(c_ctx, 1+self.seq_pred_extra_future, dim=-1)
            c_ctx_alts_shifted = [c_ctx_alts[0]]
            for i in range(1, len(c_ctx_alts)):
                c_ctx_alts_shifted.append(torch.cat((c_ctx_alts_shifted[-1][:, :i], c_ctx_alts[i][:, :-i]), dim=1))
            c_ctx_alts_shifted = torch.stack(c_ctx_alts_shifted)
            alt_idxs = torch.randint(0, len(c_ctx_alts), (1, *c_ctx.shape[:2], 1), device=device)
            c_ctx = torch.gather(c_ctx_alts_shifted, 0, alt_idxs.expand(-1, -1, -1, self.frames_per_token*self.d_token)).squeeze(0)
        
        c_ctx = c_ctx.view(bsz, self.length_tokens, self.frames_per_token, self.d_token).flatten(1, 2)
        assert c_ctx.size(1) == self.length_frames, f'{c_ctx.size(1)=}, {self.length_frames=}'

        if self.pred_grad_frames != -1:
            if self.training:
                y_idxs = torch.multinomial(torch.ones((bsz, self.length_frames), device=device), num_samples = self.pred_grad_frames)
            else:
                y_idxs = torch.arange(self.length_frames - self.pred_grad_frames, self.length_frames, device=device)[None, :].expand(bsz, -1)
            x_m_frames = torch.gather(x_m_frames, 1, y_idxs[:, :, None, None].expand(-1, -1, *x_m_frames.shape[-2:]))
            y_m_frames = torch.gather(y_m_frames, 1, y_idxs[:, :, None, None].expand(-1, -1, *y_m_frames.shape[-2:]))
            y_phi_frames = torch.gather(y_phi_frames, 1, y_idxs[:, :, None, None].expand(-1, -1, *y_phi_frames.shape[-2:]))
            prev_x_m_frames = torch.gather(prev_x_m_frames, 1, y_idxs[:, :, None, None].expand(-1, -1, *prev_x_m_frames.shape[-2:]))
            prev_x_phi_frames = torch.gather(prev_x_phi_frames, 1, y_idxs[:, :, None, None].expand(-1, -1, *prev_x_phi_frames.shape[-2:]))
            c_ctx = torch.gather(c_ctx, 1, y_idxs[:, :, None].expand(-1, -1, *c_ctx.shape[-1:]))

        if c_enc is not None:
            c_enc = c_enc[:, None].expand(-1, y_m_frames.shape[1], -1).flatten(0, 1)
        x_m_frames = x_m_frames.flatten(0, 1)
        y_m_frames = y_m_frames.flatten(0, 1)
        y_phi_frames = y_phi_frames.flatten(0, 1)
        prev_x_m_frames = prev_x_m_frames.flatten(0, 1)
        prev_x_phi_frames = prev_x_phi_frames.flatten(0, 1)
        c_ctx = c_ctx.flatten(0, 1)
        dn_bsz = y_m_frames.size(0)

        t_a = self.p_t.sample((dn_bsz,))
        t_b = self.p_t.sample((dn_bsz,))
        s = self.p_s.sample((dn_bsz,))
        t_s = t_a + s * (t_b - t_a)
        
        t_a_like_y = t_a[:, None, None]

        s_like_y = s[:, None, None]

        e_m = self.sample_prior_like(y_m_frames)
        e_phi = self.sample_prior_like(y_phi_frames)
        z_m_a = (1 - t_a_like_y) * e_m + t_a_like_y * y_m_frames
        z_phi_a = (1 - t_a_like_y) * e_phi + t_a_like_y * y_phi_frames
        v_m = y_m_frames - e_m
        v_phi = y_phi_frames - e_phi

        if self.training and self.c_enc_drop > 0.0:
            c_enc_mask = torch.bernoulli(torch.full((dn_bsz, 1), 1.0-self.c_enc_drop, device=device)).to(torch.float32)
            c_enc = c_enc * c_enc_mask
        
        vu_m_a_in = torch.cat((prev_x_m_frames, z_m_a), dim=-1)
        vu_phi_a_in = torch.cat((prev_x_m_frames, x_m_frames, prev_x_phi_frames, z_phi_a), dim=-1)
        with torch.no_grad():
            c_aa_as = self.get_c_u(t_a.repeat(2), torch.cat((t_a, t_s), dim=0), c_enc.repeat(2, 1), c_ctx.repeat(2, 1))
            u_m_aa, u_m_as = torch.chunk(self.decoder_u_m(
                vu_m_a_in.repeat(2, 1, 1),
                aux=c_aa_as
            ), 2, dim=0)
            u_phi_aa, u_phi_as = torch.chunk(self.decoder_u_phi(
                vu_phi_a_in.repeat(2, 1, 1),
                aux=c_aa_as
            ), 2, dim=0)
            z_m_s = z_m_a + (t_s - t_a)[:, None, None] * u_m_as
            z_phi_s = z_phi_a + (t_s - t_a)[:, None, None] * u_phi_as
            c_sb = self.get_c_u(t_s, t_b, c_enc, c_ctx)
            u_m_sb = self.decoder_u_m(
                torch.cat((prev_x_m_frames, z_m_s), dim=-1),
                aux=c_sb
            )
            u_phi_sb = self.decoder_u_phi(
                torch.cat((prev_x_m_frames, x_m_frames, prev_x_phi_frames, z_phi_s), dim=-1),
                aux=c_sb
            )
            u_m_avg_ab = s_like_y * u_m_as + (1.0 - s_like_y) * u_m_sb
            u_phi_avg_ab = s_like_y * u_phi_as + (1.0 - s_like_y) * u_phi_sb
            z_m_b = z_m_a + (t_b - t_a)[:, None, None] * u_m_avg_ab
            z_phi_b = z_phi_a + (t_b - t_a)[:, None, None] * u_phi_avg_ab

        c_ab_ba = self.get_c_u(
            torch.cat((t_a, t_b), dim=0),
            torch.cat((t_b, t_a), dim=0),
            c_enc.repeat(2, 1), c_ctx.repeat(2, 1)
        )
        
        u_m_hat_ab, u_m_hat_ba = torch.chunk(self.decoder_u_m(
            torch.cat((vu_m_a_in, torch.cat((prev_x_m_frames, z_m_b), dim=-1)), dim=0),
            aux=c_ab_ba
        ), 2, dim=0)

        u_phi_hat_ab, u_phi_hat_ba = torch.chunk(self.decoder_u_phi(
            torch.cat((vu_phi_a_in, torch.cat((prev_x_m_frames, x_m_frames, prev_x_phi_frames, z_phi_b), dim=-1)), dim=0),
            aux=c_ab_ba
        ), 2, dim=0)

        c_a = self.get_c_v(t_a, c_enc, c_ctx)

        v_m_hat = self.decoder_v_m(
            vu_m_a_in, aux=c_a
        )
        v_phi_hat = self.decoder_v_phi(
            vu_phi_a_in, aux=c_a
        )

        with torch.no_grad():
            u_m_target = self.substep_autoguidance * u_m_avg_ab + (1.0 - self.substep_autoguidance) * u_m_hat_ab + (v_m_hat - u_m_aa)
            u_phi_target = self.substep_autoguidance * u_phi_avg_ab + (1.0 - self.substep_autoguidance) * u_phi_hat_ab + (v_phi_hat - u_phi_aa)
            
            t_y_a = torch.zeros_like(t_a)
            t_y_b = torch.ones_like(t_a)
            if self.training:
                t_y_a[dn_bsz//2:] = 1.0
                t_y_b[dn_bsz//2:] = 0.0
            e_y_m = self.sample_prior_like(y_m_frames)
            e_y_phi = self.sample_prior_like(y_phi_frames)
            s_y = self.p_s.sample((dn_bsz,))
            s_y_like_y = s_y[:, None, None]
            t_y_s = t_y_a + s_y * (t_y_b - t_y_a)
            t_y_a_like_y = t_y_a[:, None, None]
            z_y_m_a = (1.0 - t_y_a_like_y) * e_y_m + t_y_a_like_y * y_m_frames
            z_y_phi_a = (1.0 - t_y_a_like_y) * e_y_phi + t_y_a_like_y * y_phi_frames
            c_u_y_as = self.get_c_u(t_y_a, t_y_s, c_enc, c_ctx)
            u_m_y_as = self.decoder_u_m(
                torch.cat((prev_x_m_frames, z_y_m_a), dim=-1),
                aux=c_u_y_as
            )
            u_phi_y_as = self.decoder_u_phi(
                torch.cat((prev_x_m_frames, x_m_frames, prev_x_phi_frames, z_y_phi_a), dim=-1),
                aux=c_u_y_as
            )
            z_y_m_m = z_y_m_a + (t_y_s - t_y_a)[:, None, None] * u_m_y_as
            z_y_phi_m = z_y_phi_a + (t_y_s - t_y_a)[:, None, None] * u_phi_y_as
            c_u_y_sb = self.get_c_u(t_y_s, t_y_b, c_enc, c_ctx)
            u_m_y_sb = self.decoder_u_m(
                torch.cat((prev_x_m_frames, z_y_m_m), dim=-1),
                aux=c_u_y_sb
            )
            u_phi_y_sb = self.decoder_u_phi(
                torch.cat((prev_x_m_frames, x_m_frames, prev_x_phi_frames, z_y_phi_m), dim=-1),
                aux=c_u_y_sb
            )
            y_m_target = s_y_like_y * u_m_y_as + (1.0 - s_y_like_y) * u_m_y_sb
            y_phi_target = s_y_like_y * u_phi_y_as + (1.0 - s_y_like_y) * u_phi_y_sb
            x_m_y = z_y_m_a - t_y_a_like_y * y_m_target
            x_phi_y = z_y_phi_a - t_y_a_like_y * y_phi_target

        c_y = self.get_c_y(c_enc, c_ctx)

        y_m_hat = self.decoder_y_m(
            torch.cat((prev_x_m_frames, x_m_y), dim=-1),
            aux=c_y
        )
        y_phi_hat = self.decoder_y_phi(
            torch.cat((prev_x_m_frames, x_m_frames, prev_x_phi_frames, x_phi_y), dim=-1),
            aux=c_y
        )
        
        out = {
        }

        if not self.training:
            out['c_enc'] = c_enc
            out['y_m_pred'] = (x_m_y + y_m_hat.detach()).view(bsz, -1, self.frame_bins, self.d_signal)
            out['y_pred'] = self.frames_to_signal((x_phi_y + y_phi_hat.detach()).view(bsz, -1, self.frame_bins, self.d_signal_enc))
            out['y'] = self.frames_to_signal(y_phi_frames.view(bsz, -1, self.frame_bins, self.d_signal_enc))

        w = self.get_w(t_a)
        out['loss_w'] = -torch.mean(w)

        frame_mse_v_m = torch.mean(torch.square(v_m_hat - v_m), dim=(1, 2))
        frame_mse_v_phi = torch.mean(torch.square(v_phi_hat - v_phi), dim=(1, 2))
        out['loss_v_m'] = torch.mean(torch.exp(w) * frame_mse_v_m)
        out['loss_v_m_raw'] = torch.mean(frame_mse_v_m)
        out['loss_v_phi'] = torch.mean(torch.exp(w) * frame_mse_v_phi)
        out['loss_v_phi_raw'] = torch.mean(frame_mse_v_phi)
        out['loss_u_m'] = torch.nn.functional.mse_loss(u_m_hat_ab, u_m_target) + torch.nn.functional.mse_loss(u_m_hat_ba, u_m_avg_ab)
        out['loss_u_phi'] = torch.nn.functional.mse_loss(u_phi_hat_ab, u_phi_target) + torch.nn.functional.mse_loss(u_phi_hat_ba, u_phi_avg_ab)
        out['loss_y_m'] = torch.nn.functional.mse_loss(y_m_hat, y_m_target)
        out['loss_y_phi'] = torch.nn.functional.mse_loss(y_phi_hat, y_phi_target)
        out['loss_all'] = out['loss_w'] + out['loss_v_m'] + out['loss_v_phi'] \
            + out['loss_u_m'] + out['loss_u_phi'] + out['loss_y_m'] + out['loss_y_phi']
        return out

    def encode(self, samples: torch.Tensor, ema=False):
        token_encs = self.frames_to_tokens(self.signal_to_frames(samples)[0][:, :self.seq_enc_length_tokens*self.frames_per_token], ema=ema)
        encoder = self.seq_encoder_ema if ema else self.seq_encoder
        return encoder(token_encs[:, :self.seq_enc_length_tokens], include_scratch_outputs=1)[:, 0]

    @torch.no_grad()
    def sample(self,
            batch_size: int,
            length_tokens: int,
            c_enc: torch.Tensor | None = None,
            c_encs: list[torch.Tensor] | None = None,
            c_samples: torch.Tensor | None = None,
            prefill: int = 0,
            steps: int = 1,
            eps_ctx: float | None = None,
            ema: bool = True,
            show_progress: bool = False,
            device: torch.device | None = None
        ):
        eps_ctx = self.sample_eps_ctx if eps_ctx is None else eps_ctx
        if device is None:
            device = next(self.parameters()).device

        if steps == 1:
            decoder_m = self.decoder_y_m_ema if ema else self.decoder_y_m
            decoder_phi = self.decoder_y_phi_ema if ema else self.decoder_y_phi
        else:
            decoder_m = self.decoder_v_m_ema if ema else self.decoder_v_m
            decoder_phi = self.decoder_v_phi_ema if ema else self.decoder_v_phi

        seq_predictor = self.seq_predictor_ema if ema else self.seq_predictor

        length_frames_padded = length_tokens*self.frames_per_token + 1
        
        sample_encs = torch.empty((batch_size, length_tokens, self.d_token), device=device, dtype=torch.float32)
        x_m_frames = self.sample_prior((batch_size, length_frames_padded, self.frame_bins, self.d_signal), device=device, dtype=torch.float32)
        x_phi_frames = self.sample_prior((batch_size, length_frames_padded, self.frame_bins, self.d_signal_enc), device=device, dtype=torch.float32)
        y_m_frames = x_m_frames.clone()
        y_phi_frames = x_phi_frames.clone()

        prev_m_frames = self.sample_prior((batch_size, self.decoder_lookback_frames, self.frame_bins, self.d_signal), device=device)
        prev_phi_frames = self.sample_prior((batch_size, self.decoder_lookback_frames, self.frame_bins, self.d_signal_enc), device=device)

        if c_samples is not None:
            c_m_frames, c_phi_frames = self.signal_to_frames(c_samples)
            c_token_encs = self.frames_to_tokens(c_m_frames, ema=ema)
            if c_enc is None:
                c_enc = self.seq_encoder_ema(c_token_encs[:, :self.seq_enc_length_tokens], include_scratch_outputs=1)[:, 0]
            if prefill > 0:
                x_m_frames[:, :prefill] = y_m_frames[:, :prefill] = c_m_frames[:, :prefill]
                x_phi_frames[:, :prefill] = y_phi_frames[:, :prefill] = c_phi_frames[:, :prefill]
                sample_encs[:, :prefill//self.frames_per_token] = c_token_encs[:, :prefill//self.frames_per_token]
                prev_m_frames[:, -prefill:] = c_m_frames[:, -self.decoder_lookback_frames:]
                prev_phi_frames[:, -prefill:] = c_phi_frames[:, -self.decoder_lookback_frames:]
        if c_enc is None:
            c_enc = torch.zeros((batch_size, self.d_token), device=device)

        prev_m_frames = list(prev_m_frames.unbind(dim=1))
        prev_phi_frames = list(prev_phi_frames.unbind(dim=1))

        if steps > 1:
            t = torch.linspace(0.0, 1.0 - 1.0/steps, steps, device=device)[None].expand(batch_size, -1)
            # r = torch.linspace(1.0 - 1.0/steps, 0.0, steps, device=device)
        else:
            t = None

        sample_encs_dirty = True

        for i in trange(prefill, length_frames_padded, disable=not show_progress):
            if c_encs is not None:
                c_enc = c_encs[i*len(c_encs)//length_frames_padded]
            z_m = y_m_frames[:, i]
            z_phi = y_phi_frames[:, i]
            if sample_encs_dirty:
                ctx_encs = sample_encs[:, max(0, i//self.frames_per_token - (self.length_tokens-1)):i//self.frames_per_token]
                c_ctxs = seq_predictor(ctx_encs, include_scratch_outputs=1)[:, -1, :self.frames_per_token*self.d_token].view(batch_size, self.frames_per_token, self.d_token)
            c_ctx = c_ctxs[:, i % self.frames_per_token]
            if steps > 1:
                c = self.get_c_v(t, c_enc[:, None].expand(-1, steps, -1), c_ctx[:, None].expand(-1, steps, -1), ema=ema)
            else:
                c = self.get_c_y(c_enc, c_ctx, ema=ema)
            for j in range(steps):
                v_m_pred = decoder_m(
                    torch.cat((*prev_m_frames, z_m), dim=-1),
                    aux=c if steps == 1 else c[:, j]
                )
                #z_m = self.clip_frames(z_m + (1.0/steps)*v_pred)
                z_m = z_m + (1.0/steps) * v_m_pred
            y_m_frames[:, i] = z_m
            if eps_ctx > 0.0:
                z_m = (1.0 - eps_ctx) * z_m + eps_ctx * self.sample_prior_like(z_m)
            x_m_frames[:, i] = z_m
            for j in range(steps):
                v_phi_pred = decoder_phi(
                    torch.cat((*prev_m_frames, z_m, *prev_phi_frames, z_phi), dim=-1),
                    aux=c if steps == 1 else c[:, j]
                )
                z_phi = self.clip_frames(z_phi + (1.0/steps)*v_phi_pred)
            # Synchronize predicted magnitudes
            z_phi = torch.view_as_real(rescale_complex(
                torch.view_as_complex(z_phi.view(*z_phi.shape[:-1], z_phi.shape[-1]//2, 2)),
                torch.abs(y_m_frames[:, i])
            )).flatten(-2)
            
            y_phi_frames[:, i] = z_phi
            if eps_ctx > 0.0:
                z_phi = (1.0 - eps_ctx) * z_phi + eps_ctx * self.sample_prior_like(z_phi)
            x_phi_frames[:, i] = z_phi
            prev_m_frames = prev_m_frames[1:]
            prev_m_frames.append(z_m)
            prev_phi_frames = prev_phi_frames[1:]
            prev_phi_frames.append(z_phi)
            if i//self.frames_per_token < sample_encs.size(1) and (i+1) % self.frames_per_token == 0:
                sample_encs[:, i//self.frames_per_token] = self.frames_to_tokens(torch.stack(prev_m_frames[-self.frames_per_token:], dim=1), ema=ema).squeeze(1)
                sample_encs_dirty = True

        return self.frames_to_signal(y_phi_frames[:, :length_tokens*self.frames_per_token])

    @torch.no_grad()
    def sample_single_frame(self, i, x_m_frames, x_phi_frames, y_m_frames, y_phi_frames, prev_m_frames, prev_phi_frames, sample_encs, c_ctx, c_lat, steps=1, eps_ctx=None, ema_decoders=('m','phi')):
        batch_size = x_m_frames.size(0)
        device = x_m_frames.device
        if steps > 1:
            t = torch.linspace(0.0, 1.0 - 1.0/steps, steps, device=device)[None].expand(batch_size, -1)
            r = torch.linspace(1.0/steps, 1.0, steps, device=device)[None].expand(batch_size, -1)
        else:
            t = None
        z_m = y_m_frames[:, i]
        z_phi = y_phi_frames[:, i]
        sample_encs_dirty = i % self.frames_per_token == 0
        if sample_encs_dirty:
            ctx_encs = sample_encs[:, max(0, i//self.frames_per_token - (self.length_tokens-1)):i//self.frames_per_token]
            c_ctx[:] = self.seq_predictor_ema(ctx_encs, include_scratch_outputs=1)[:, -1, :self.frames_per_token*self.d_token].view(batch_size, self.frames_per_token, self.d_token)
        c_ctx = c_ctx[:, i % self.frames_per_token]
        if steps > 1:
            decoder_m = self.decoder_u_m_ema if 'm' in ema_decoders else self.decoder_u_m
            decoder_phi = self.decoder_u_phi_ema if 'phi' in ema_decoders else self.decoder_u_phi
            c = self.get_c_u(t, r, c_lat[:, None].expand(-1, steps, -1), c_ctx[:, None].expand(-1, steps, -1), ema=True)
        else:
            decoder_m = self.decoder_y_m_ema if 'm' in ema_decoders else self.decoder_y_m
            decoder_phi = self.decoder_y_phi_ema if 'phi' in ema_decoders else self.decoder_y_phi
            c = self.get_c_y(c_lat, c_ctx, ema=True)
        for j in range(steps):
            v_m_pred = decoder_m(
                torch.cat((*prev_m_frames, z_m.to(dtype=c_ctx.dtype)), dim=-1),
                aux=c if steps == 1 else c[:, j]
            ).to(dtype=x_m_frames.dtype)
            #z_m = self.clip_frames(z_m + (1.0/steps)*v_pred)
            z_m = z_m + (1.0/steps) * v_m_pred
        y_m_frames[:, i] = z_m
        if eps_ctx > 0.0:
            z_m = (1.0 - eps_ctx) * z_m + eps_ctx * self.sample_prior_like(z_m)
        x_m_frames[:, i] = z_m
        for j in range(steps):
            decoder_phi_input = torch.cat((*prev_m_frames, z_m.to(dtype=c_ctx.dtype), *prev_phi_frames, z_phi.to(dtype=c_ctx.dtype)), dim=-1)
            v_phi_pred = decoder_phi(
                decoder_phi_input, aux=c if steps == 1 else c[:, j]
            ).to(dtype=x_phi_frames.dtype)
            z_phi = z_phi + (1.0/steps)*v_phi_pred
        # Synchronize predicted magnitudes
        z_phi = torch.view_as_real(rescale_complex(
            torch.view_as_complex(z_phi.view(*z_phi.shape[:-1], z_phi.shape[-1]//2, 2)),
            torch.abs(y_m_frames[:, i])
        )).flatten(-2)
        
        y_phi_frames[:, i] = z_phi
        if eps_ctx > 0.0:
            z_phi = (1.0 - eps_ctx) * z_phi + eps_ctx * self.sample_prior_like(z_phi)
        x_phi_frames[:, i] = z_phi
        prev_m_frames.pop(0)
        prev_m_frames.append(z_m.to(dtype=c_ctx.dtype))
        prev_phi_frames.pop(0)
        prev_phi_frames.append(z_phi.to(dtype=c_ctx.dtype))
        if i//self.frames_per_token < sample_encs.size(1) and (i+1) % self.frames_per_token == 0:
            sample_encs[:, i//self.frames_per_token] = self.frames_to_tokens(torch.stack(prev_m_frames[-self.frames_per_token:], dim=1), ema=True).squeeze(1)

    def update_ema(self):
        update_ema(self.decoder_v_m_ema, self.decoder_v_m, self.ema_weight)
        update_ema(self.decoder_u_m_ema, self.decoder_u_m, self.ema_weight)
        update_ema(self.decoder_y_m_ema, self.decoder_y_m, self.ema_weight)
        update_ema(self.decoder_v_phi_ema, self.decoder_v_phi, self.ema_weight)
        update_ema(self.decoder_u_phi_ema, self.decoder_u_phi, self.ema_weight)
        update_ema(self.decoder_y_phi_ema, self.decoder_y_phi, self.ema_weight)
        update_ema(self.c_v_net_ema, self.c_v_net, self.ema_weight)
        update_ema(self.c_u_net_ema, self.c_u_net, self.ema_weight)
        update_ema(self.c_y_net_ema, self.c_y_net, self.ema_weight)
        update_ema(self.seq_predictor_ema, self.seq_predictor, self.ema_weight)
        update_ema(self.seq_encoder_ema, self.seq_encoder, self.ema_weight)
        update_ema(self.token_encoder_ema, self.token_encoder, self.ema_weight)

    def grad_clip_groups(self):
        return {
            'main': (p for m in self.children() for p in m.parameters() if m is not self.w_net),
            'w_net': self.w_net.parameters()
        }
