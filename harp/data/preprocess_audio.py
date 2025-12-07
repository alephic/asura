import numpy as np
import soundfile
import os
import glob
import argparse
from tqdm import tqdm

if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument('base_dir', type=str)
    argp.add_argument('-f', '--input_formats', type=str, default='mp3')
    argp.add_argument('-o', '--output_dir', type=str, required=True)
    argp.add_argument('--filter_ids', type=str, default=None)
    argp.add_argument('--output_format', type=str, default='npy')
    argp.add_argument('--skip_existing', action='store_true')
    argp.add_argument('--sample_rate', type=int, default=44100)
    argp.add_argument('--block_length', type=int, default=None)
    args = argp.parse_args()
    
    selected_ids = None
    if args.filter_ids is not None:
        selected_ids = set()
        with open(args.filter_ids, mode='r') as id_filter_file:
            for line in id_filter_file:
                selected_ids.add(line.rstrip())
    
    sr = args.sample_rate

    os.makedirs(args.output_dir, exist_ok=True)
    for ext in args.input_formats.split(','):
        for path in tqdm(glob.glob(os.path.join(args.base_dir, f'*.{ext}')), desc='Source file'):
            entry_name = os.path.splitext(os.path.basename(path))[0]
            if selected_ids is not None and entry_name not in selected_ids:
                continue
            if args.block_length is None:
                out_filename = os.path.join(args.output_dir, f'{entry_name}.{args.output_format}')
                if args.skip_existing and os.path.exists(out_filename):
                    continue
                arr = soundfile.read(path, dtype='int16')[0]
                if args.output_format == 'npy':
                    np.save(out_filename, arr)
                elif args.output_format == 'wav':
                    soundfile.write(out_filename, arr, sr, subtype='PCM_16')
            else:
                out_dir = os.path.join(args.output_dir, entry_name)
                os.makedirs(out_dir, exist_ok=True)
                for block_idx, block in enumerate(soundfile.blocks(path, dtype='int16', blocksize=args.block_length*sr, fill_value=0)):
                    out_filename = os.path.join(out_dir, f'{block_idx}.{args.output_format}')
                    if args.skip_existing and os.path.exists(out_filename):
                        continue
                    if args.output_format == 'npy':
                        np.save(out_filename, block)
                    elif args.output_format =='wav':
                        soundfile.write(out_filename, block, sr, subtype='PCM_16')

