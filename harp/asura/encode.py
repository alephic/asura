import argparse
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import os
from harp.data.vector_sequence import VectorSequencesDataset
import json
from .asura import ASURA
from harp.data.sampler import DistributedSampler
from tqdm import tqdm
import numpy as np

SPECT_CMAP = 'BuPu_r'
SPECT_DIFF_CMAP = 'managua'

def is_main_worker():
    return not dist.is_initialized() or dist.get_rank() == 0

def barrier():
    if dist.is_initialized():
        dist.barrier()

def encode_worker_fn(
        rank: int,
        world_size: int,
        config_path: str
    ):
    torch.manual_seed(1337+rank)
    device = torch.device('cuda', rank)
    torch.cuda.set_device(device)

    if world_size > 1:
        print(f"Hello from worker {rank}")
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group('nccl', rank=rank, world_size=world_size)
        dist.barrier()

    torch.cuda.manual_seed(1337+rank)

    with open(config_path, encoding='utf-8') as config_file:
        config = json.load(config_file)

    ckpt_path = os.path.dirname(config_path)

    model = ASURA.from_checkpoint(ckpt_path, device=device)
    
    local_batch_size = config.get('batch_size_per_device')
    step_batch_size = config.get('batch_size')
    if local_batch_size is None:
        local_batch_size = step_batch_size//world_size
    elif step_batch_size is None:
        step_batch_size = local_batch_size*world_size
    
    global_batch_size = local_batch_size*world_size
    assert step_batch_size % global_batch_size == 0

    example_length = model.seq_enc_length_samples

    train_dataset = VectorSequencesDataset(
        config['train_data'], example_length,
        do_offset=True,
        do_channel_flip=True
    )
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        sampler=train_sampler,
        drop_last=True
    )

    model.eval()

    all_encs = []
    save_audio = []
    save_audio_count = 100
    batch_idx = 0

    total_count = 100000

    train_dataloader_prog = tqdm(train_dataloader, disable=rank > 0, total=total_count//global_batch_size)
    for batch in train_dataloader_prog:
        batch = batch.to(device=device)

        if batch_idx*global_batch_size < save_audio_count:
            if world_size > 1:
                batch_gathered = [torch.empty_like(batch) for _ in range(world_size)]
                dist.all_gather(batch_gathered, batch.contiguous())
                batch_gathered = torch.cat(batch_gathered, dim=0)
            else:
                batch_gathered = batch
            if is_main_worker():
                save_audio.append(batch_gathered.cpu().numpy())

        with torch.no_grad():
            batch_encs = model.encode(batch)

        if world_size > 1:
            encs_gathered = [torch.empty_like(batch_encs) for _ in range(world_size)]
            dist.all_gather(encs_gathered, batch_encs.contiguous())
            encs_gathered = torch.cat(encs_gathered, dim=0)
        else:
            encs_gathered = batch_encs

        if is_main_worker():
            all_encs.append(encs_gathered.cpu().numpy())
        
        batch_idx += 1
        if batch_idx * global_batch_size >= total_count:
            break

    if is_main_worker():
        all_encs = np.concatenate(all_encs, axis=0)
        np.save('embeddings.npy', all_encs)
        save_audio = np.concatenate(save_audio, axis=0)
        np.save('embed_audio.npy', save_audio)

    if world_size > 1:
        dist.destroy_process_group()

def main():
    argp = argparse.ArgumentParser()
    argp.add_argument('config_or_ckpt_dir', type=str)
    argp.add_argument('-n', '--num-workers', type=int, default=1)
    argp.add_argument('--debug', action='store_true')

    args = argp.parse_args()
    resume = os.path.isdir(args.config_or_ckpt_dir)
    config_path = os.path.join(args.config_or_ckpt_dir, 'config.json') if resume else args.config_or_ckpt_dir

    if args.num_workers == 1:
        encode_worker_fn(0, 1, config_path)
    else:
        mp.spawn(
            encode_worker_fn, args=(args.num_workers, config_path),
            nprocs=args.num_workers,
            join=True
        )

if __name__ == "__main__":
    main()
