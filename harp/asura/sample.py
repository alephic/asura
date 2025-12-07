from .asura import ASURA
import torch
from harp.modeling.util import log_mel_spectrogram
from harp.analysis.util import save_image
from harp.data.vector_sequence import VectorSequencesDataset
import os
import glob
import argparse
from scipy.io.wavfile import write
import numpy as np

def main(checkpoint_path, ctx_dataset_path, emb_path=None, ids=[42], prefill=0, steps=1, eps_ctx=None, ema=True):
    device = torch.device('cuda', 0)
    model = ASURA.from_checkpoint(checkpoint_path, device=device)
    checkpoint_name = os.path.basename(os.path.abspath(checkpoint_path))
    if ctx_dataset_path is not None:
        d = VectorSequencesDataset(ctx_dataset_path, model.example_length_samples)
        c_samples = d[ids[0]].to(device=device)[None]
        ref_spec = log_mel_spectrogram(c_samples[:, :, 0], 44100, 2048, 256)[0].cpu()
    else:
        c_samples = None
        ref_spec = None
    if emb_path is not None:
        embs = np.load(emb_path)
        embs = [torch.from_numpy(embs[i:i+1]).to(device=device) for i in ids]
    else:
        embs = None

    out = model.sample(1, model.length_tokens, c_samples=c_samples, c_encs=embs, prefill=prefill, steps=steps, eps_ctx=eps_ctx, ema=ema, show_progress=True)
    spec = log_mel_spectrogram(out[:, :, 0], 44100, 2048, 256)
    os.makedirs('outputs/sample', exist_ok=True)
    n = len(glob.glob(f'outputs/sample/{checkpoint_name}-*-gen.png'))
    save_image(spec[0].cpu(), f'outputs/sample/{checkpoint_name}-{n}-gen.png')
    if ref_spec is not None:
        save_image(ref_spec, f'outputs/sample/{checkpoint_name}-{n}-ref.png')
    write(f'outputs/test_sample/{checkpoint_name}-{n}.wav', 44100, out[0].cpu().numpy())


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument('checkpoint', type=str)
    argp.add_argument('-d', '--dataset', type=str, default=None)
    argp.add_argument('-m', '--embeddings', type=str, default=None)
    argp.add_argument('-i', '--id', type=str, default='42')
    argp.add_argument('-p', '--prefill', type=int, default=0)
    argp.add_argument('-s', '--steps', type=int, default=1)
    argp.add_argument('-e', '--eps', type=float, default=None)
    argp.add_argument('-g', '--guidance', type=float, default=1.0)
    args = argp.parse_args()
    main(
        checkpoint_path=args.checkpoint,
        ctx_dataset_path=args.dataset,
        emb_path=args.embeddings,
        ids=list(map(int, args.id.split(','))),
        prefill=args.prefill,
        steps=args.steps,
        eps_ctx=args.eps,
        g=args.guidance
    )