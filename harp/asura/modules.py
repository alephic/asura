
import torch
try:
    from flash_attn import flash_attn_func
except ImportError as e:
    flash_attn_func = None
from harp.modeling.util import sinusoid_encoding, apply_rope, get_act_fn, unitary_log
import torch.utils.checkpoint

class AttnLayer(torch.nn.Module):
    def __init__(self, 
            *,
            d_model: int,
            attn_d_head: int,
            attn_query_grouping: int,
            num_scratch_states: int,
            pos_enc_length: int,
            causal: bool,
            zero_out: bool=True,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None
        ):
        super().__init__()
        # assert flash_attn_func is not None
        self.attn_d_head = attn_d_head
        self.num_q_heads = d_model // self.attn_d_head
        self.num_kv_heads = max(1, self.num_q_heads // attn_query_grouping)
        self.causal = causal
        self.pos_enc_length = pos_enc_length

        self.attn_qkv_proj = torch.nn.Linear(
            d_model,
            self.attn_d_head * (self.num_q_heads + 2*self.num_kv_heads),
            bias=False, device=device, dtype=dtype
        )
        self.attn_out_proj = torch.nn.Linear(
            self.attn_d_head*self.num_q_heads,
            d_model,
            bias=False, device=device, dtype=dtype
        )

        if zero_out:
            torch.nn.init.zeros_(self.attn_out_proj.weight)

        self.num_scratch_states = num_scratch_states

    def forward(self, x: torch.Tensor):
        bsz, l, d_model = x.shape

        pos_encs = sinusoid_encoding(
            torch.linspace(0, l-1, steps=l, device=x.device, dtype=x.dtype),
            2*self.attn_d_head, self.pos_enc_length
        )[None, :, None]
        
        qkv = self.attn_qkv_proj(x)
        q = qkv[..., :self.attn_d_head*self.num_q_heads] \
            .view(bsz, l, self.num_q_heads, self.attn_d_head)
        kv = qkv[..., self.attn_d_head*self.num_q_heads:] \
            .view(bsz, l, 2*self.num_kv_heads, self.attn_d_head)
        k, v = torch.chunk(kv, 2, dim=2)
        q = torch.nn.functional.layer_norm(q, q.shape[-1:])
        q = apply_rope(q, pos_encs).to(dtype=torch.bfloat16)
        k = torch.nn.functional.layer_norm(k, k.shape[-1:])
        k = apply_rope(k, pos_encs).to(dtype=torch.bfloat16)
        v = v.to(dtype=torch.bfloat16)

        if flash_attn_func is not None:
            attn_out = flash_attn_func(q, k, v, causal=self.causal)
            out = self.attn_out_proj(
                attn_out.view(bsz, l, -1).to(dtype=x.dtype)
            )
        else:
            attn_out = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=self.causal, dropout_p=0.0, enable_gqa=True
            ).transpose(1, 2)
            out = self.attn_out_proj(
                attn_out.reshape(bsz, l, -1).to(dtype=x.dtype)
            )
        return out
    
class FFLayer(torch.nn.Module):
    def __init__(self, 
            *,
            d_model: int,
            ff_mult: int,
            act_fn: str,
            zero_out: bool=True,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None
        ):
        super().__init__()

        d_hidden = ff_mult * d_model
        d_hidden_in = d_hidden * (2 if act_fn.endswith('glu') else 1)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_hidden_in, device=device, dtype=dtype),
            get_act_fn(act_fn, d_hidden_in, device=device, dtype=dtype),
            torch.nn.Linear(d_hidden, d_model, device=device, dtype=dtype, bias=False)
        )
        if zero_out:
            torch.nn.init.zeros_(self.ff[-1].weight)

    def forward(self, x: torch.Tensor):
        return self.ff(x)
    
class ConvLayer(torch.nn.Module):
    def __init__(self,
            *,
            d_in: int,
            d_out: int,
            d_aux: int | None = None,
            k: int=2,
            downsample: bool=False,
            upsample: bool=False,
            ff_mult: int,
            act_fn: str,
            norm_in: bool=True,
            zero_out: bool=True,
            zero_shortcut: bool=False,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None
        ):
        super().__init__()
        d_hidden = d_out * ff_mult
        d_hidden_in = d_hidden * (2 if act_fn.endswith('glu') else 1)
        d_proj_in = d_hidden_in + d_out
        self.d_in = d_in
        self.d_out = d_out
        self.proj_in_d = torch.nn.Conv1d(d_in, d_proj_in, 1, device=device, dtype=dtype)
        self.proj_in_w = torch.nn.Conv1d(
            d_proj_in, d_proj_in,
            2*k-1,
            stride=k if downsample else 1,
            padding=k - 1,
            groups=d_proj_in,
            bias=False,
            device=device, dtype=dtype
        )
        self.act_fn = get_act_fn(act_fn, d_hidden_in, dim=1, device=device, dtype=dtype)
        self.proj_out = torch.nn.Conv1d(d_hidden, d_out, 1, bias=False, device=device, dtype=dtype)
        if d_aux is not None:
            self.aux_proj = torch.nn.Linear(d_aux, d_hidden_in, bias=False, device=device, dtype=dtype)
        self.norm_in = norm_in
        self.upsample = upsample
        self.downsample = downsample
        self.k = k
        if zero_out:
            torch.nn.init.zeros_(self.proj_out.weight)
        if zero_shortcut:
            torch.nn.init.zeros_(self.proj_in_w.weight[:self.d_out])
    
    def forward(self, x: torch.Tensor, aux: torch.Tensor | None) -> torch.Tensor:
        batch_size, length, d_in = x.shape
        # aux: (batch_size, d_aux)
        h = x
        if self.norm_in:
            h = torch.nn.functional.layer_norm(h, h.shape[-1:])
        h = self.proj_in_d(h.transpose(1, 2))
        if self.upsample:
            h = h[:, :, :, None].expand(-1, -1, -1, self.k) \
                .reshape(batch_size, h.size(1), self.k*length)
        h = self.proj_in_w(h)
        shortcut = h[:, :self.d_out]
        h = h[:, self.d_out:]
        if aux is not None:
            h = h + self.aux_proj(aux)[:, :, None]
        h = self.proj_out(self.act_fn(h)) + shortcut
        return h.transpose(1, 2)

class UNet(torch.nn.Module):
    def __init__(self,
            *,
            d_in: int,
            d_out: int | None = None,
            d_aux: int | None = None,
            d_conv_stages: list[int],
            d_bottleneck: int,
            conv_layers_per_stage: int,
            conv_layers_per_stage_out: int | None = None,
            conv_k_stages: list[int],
            bins_in: int | None = None,
            min_aux_rank_per_bin: int = 4,
            ff_mult: int,
            act_fn: str,
            down_only: bool = False,
            zero_out: bool = True,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None
        ):
        super().__init__()

        if d_aux is not None:
            assert bins_in is not None
            stage_bins = [bins_in]
            for conv_k in conv_k_stages[:-1]:
                stage_bins.append(stage_bins[-1]//conv_k)
            self.per_bin_aux_projs = torch.nn.ModuleList(
                torch.nn.Linear(d_aux, max(min_aux_rank_per_bin*bins, d_aux), bias=False, device=device, dtype=dtype)
                for bins in stage_bins
            )
            d_bin_aux_stages = [max(min_aux_rank_per_bin*bins, d_aux) // bins for bins in stage_bins]
        else:
            d_bin_aux_stages = [0 for _ in conv_k_stages]

        self.conv_layers_in = torch.nn.ModuleList(
            ConvLayer(
                d_in=(d_stage_in if layer_idx == 0 else d_stage) + stage_d_bin_aux,
                d_out=d_stage_out if layer_idx == conv_layers_per_stage-1 else d_stage,
                d_aux=d_aux,
                k=conv_k,
                downsample=layer_idx == conv_layers_per_stage-1,
                norm_in=not (layer_idx == 0 and stage_idx == 0),
                ff_mult=ff_mult,
                act_fn=act_fn,
                device=device, dtype=dtype
            ) for stage_idx, d_stage, d_stage_in, d_stage_out, conv_k, stage_d_bin_aux in zip(
                range(len(d_conv_stages)),
                d_conv_stages,
                [d_in, *d_conv_stages[1:]],
                [*d_conv_stages[1:], d_bottleneck],
                conv_k_stages,
                d_bin_aux_stages
            ) for layer_idx in range(conv_layers_per_stage)
        )
        self.down_only = down_only
        if not down_only:
            assert d_out is not None
            conv_layers_per_stage_out = conv_layers_per_stage if conv_layers_per_stage_out is None else conv_layers_per_stage_out
            self.conv_layers_out = torch.nn.ModuleList(
                ConvLayer(
                    d_in=d_stage_in if layer_idx == 0 else d_stage,
                    d_out=d_stage_out if layer_idx == conv_layers_per_stage_out-1 else d_stage,
                    d_aux=d_aux,
                    k=conv_k,
                    upsample=layer_idx == 0,
                    ff_mult=ff_mult,
                    act_fn=act_fn,
                    zero_shortcut=True,
                    zero_out=zero_out or not (layer_idx == conv_layers_per_stage_out-1 and stage_idx == len(d_conv_stages)-1),
                    device=device, dtype=dtype
                ) for stage_idx, d_stage, d_stage_in, d_stage_out, conv_k in zip(
                    range(len(d_conv_stages)),
                    reversed(d_conv_stages),
                    [d_bottleneck, *reversed(d_conv_stages[1:])],
                    [*reversed(d_conv_stages[1:]), d_out],
                    reversed(conv_k_stages)
                ) for layer_idx in range(conv_layers_per_stage_out)
            )

        self.conv_k_stages = conv_k_stages
        self.cumulative_k = 1
        for k in conv_k_stages:
            self.cumulative_k *= k
        self.d_bottleneck = d_bottleneck
        self.ff_mult = ff_mult
        self.act_fn = act_fn
        # self.z_proj = torch.nn.Linear(self.d_conv_bottleneck, self.d_conv_bottleneck, dtype=dtype, device=device)

    def forward(self,
            x: torch.Tensor,
            aux: torch.Tensor | None = None,
            do_checkpoint = False
        ):
        bsz, l, d_in = x.shape
        # aux shape: bsz, l, d_aux
        do_checkpoint = do_checkpoint and torch.is_grad_enabled()

        if aux is not None:
            bin_auxs = [aux_proj(aux) for aux_proj in self.per_bin_aux_projs]

        h = x
        conv_skips = []
        stage_idx = 0
        for conv_layer in self.conv_layers_in:
            layer_in = h if aux is None else torch.cat((h, bin_auxs[stage_idx].view(h.size(0), h.size(1), -1)), dim=-1)
            h = torch.utils.checkpoint.checkpoint(
                conv_layer, layer_in, aux, use_reentrant=False
            ) if do_checkpoint else conv_layer(layer_in, aux)
            conv_skips.append(h)
            if conv_layer.downsample:
                stage_idx += 1

        # --- BOTTLENECK ---
        if self.down_only:
            return h.flatten(1)

        if len(self.conv_layers_out) > len(conv_skips):
            num_resamples = sum(l.downsample for l in self.conv_layers_in)
            repeat_count = (len(self.conv_layers_out) - num_resamples) // (len(self.conv_layers_in) - num_resamples)
            conv_skips = [skip for skip, l in zip(conv_skips, self.conv_layers_in) for _ in range(1 if l.downsample else repeat_count)]

        for conv_layer, skip in zip(self.conv_layers_out, reversed(conv_skips)):
            h = h[:, :skip.size(1)]
            h = torch.utils.checkpoint.checkpoint(
                conv_layer, h + skip, aux, use_reentrant=False
            ) if do_checkpoint else conv_layer(h + skip, aux)

        h = h[:, :l]

        return h
    
class FFTNet(torch.nn.Module):
    def __init__(self,
            *,
            d_in: int,
            d_out: int,
            n_fft: int,
            ff_mult: int,
            act_fn: str,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None
        ):
        super().__init__()
        self.n_fft = n_fft
        self.register_buffer('window', torch.hann_window(n_fft+1, device=device, dtype=dtype)[1:], persistent=False)
        self.norm_factor = 2.0/self.window.sum().item()
        d_fft_out = n_fft//2 * d_in
        d_hidden = d_out * ff_mult
        d_hidden_in = d_hidden * (2 if act_fn.endswith('glu') else 1)
        self.d_out = d_out
        self.ff_proj = torch.nn.Linear(d_fft_out, d_out + d_hidden_in, device=device, dtype=dtype)
        self.out_proj = torch.nn.Sequential(
            get_act_fn(act_fn, d_hidden_in, device=device, dtype=dtype),
            torch.nn.Linear(d_hidden, d_out, device=device, dtype=dtype)
        )
    
    def forward(self, x: torch.Tensor):
        assert x.shape[-2] == self.n_fft
        fft_out = unitary_log(self.norm_factor*torch.abs(
            torch.fft.rfft(
                x.transpose(-1, -2).reshape(-1, self.n_fft) * self.window[None]
            )[:, :self.n_fft//2]
        ))
        projd = self.ff_proj(fft_out.view(-1, self.n_fft//2 * x.shape[-1]))
        out = projd[:, :self.d_out] + self.out_proj(projd[:, self.d_out:])
        return out.view(*x.shape[:-2], self.d_out)
    
class Transformer(torch.nn.Module):
    def __init__(self,
            *,
            d_in: int,
            d_out: int | None = None,
            d_model: int,
            d_aux: int | None = None,
            num_layers: int,
            layer_pattern: list[str] | None = None,
            attn_d_head: int,
            attn_query_grouping: int,
            ff_mult: int,
            act_fn: str,
            num_scratch_states: int,
            pos_enc_length: int,
            causal: bool = False,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None
        ):
        super().__init__()
        self.d_model = d_model

        d_out = d_in if d_out is None else d_out
        self.in_proj = torch.nn.Linear(d_in + (0 if d_aux is None else d_aux), d_model, device=device, dtype=dtype)
        self.out_proj = torch.nn.Linear(d_model, d_out, device=device, dtype=dtype)

        layer_pattern = ['a', 'ff']
        layer_factories = {
            'a': lambda: AttnLayer(
                d_model=d_model,
                attn_d_head=attn_d_head,
                attn_query_grouping=attn_query_grouping,
                num_scratch_states=num_scratch_states,
                pos_enc_length=pos_enc_length,
                causal=causal,
                device=device,
                dtype=dtype
            ),
            'ff': lambda: FFLayer(
                d_model=d_model,
                ff_mult=ff_mult,
                act_fn=act_fn,
                device=device,
                dtype=dtype
            )
        }
        self.attn_d_head = attn_d_head
        self.layers = torch.nn.ModuleList(
            layer_factories[layer_pattern[i % len(layer_pattern)]]()
            for i in range(num_layers)
        )
        self.num_scratch_states = num_scratch_states
        self.scratch_inputs = torch.nn.Parameter(torch.zeros((num_scratch_states, d_in), device=device, dtype=dtype))

    def forward(self,
            x: torch.Tensor,
            aux: torch.Tensor | None = None,
            include_scratch_outputs: int = 0,
            do_checkpoint = False
        ):
        bsz = x.shape[0]
        do_checkpoint = do_checkpoint and torch.is_grad_enabled()

        scratch_inputs = self.scratch_inputs[None].expand(bsz, -1, -1)
        inp = torch.cat((scratch_inputs, x), dim=1)
        if aux is not None:
            inp = torch.cat((inp, aux[:, None, :].expand(-1, inp.size(1), -1)), dim=-1)
        h = self.in_proj(inp)

        side = h
        for layer in self.layers:
            o = torch.utils.checkpoint.checkpoint(
                layer, h, use_reentrant=False
            ) if do_checkpoint else layer(h)
            h = torch.nn.functional.layer_norm(h + o, h.shape[-1:])
            side = side + o

        h = self.out_proj(
            h[:, self.num_scratch_states-include_scratch_outputs:] \
            + torch.nn.functional.layer_norm(
                side[:, self.num_scratch_states-include_scratch_outputs:], side.shape[-1:]
            )
        )
        return h
    
class MLP(torch.nn.Module):
    def __init__(self, d_x, *, d_h, layers, d_out=None, act_fn='silu', zero_out=True, device=None):
        super().__init__()
        self.in_proj = torch.nn.Linear(d_x, d_h, device=device)
        self.layers = torch.nn.ModuleList([
            FFLayer(d_model=d_h, ff_mult=4, act_fn=act_fn, zero_out=zero_out, device=device)
            for _ in range(layers)
        ])
        d_out = d_x if d_out is None else d_out
        self.out_proj = torch.nn.Linear(d_h, d_out, bias=False, device=device)
        if zero_out:
            torch.nn.init.zeros_(self.out_proj.weight)
    def forward(self, x):
        h = self.in_proj(x)
        for l in self.layers:
            h = h + l(h)
        return self.out_proj(h)