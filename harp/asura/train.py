import argparse
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.distributed.optim.zero_redundancy_optimizer import ZeroRedundancyOptimizer
import torch.multiprocessing as mp
import os
from datetime import datetime
from harp.data.vector_sequence import VectorSequencesDataset
from harp.data.synthetic import NoiseBurstDataset, MultiWobbleDataset
import json
from .asura import ASURA
from harp.data.sampler import DistributedSampler
from tqdm import tqdm
from itertools import count
from torch.utils.tensorboard import SummaryWriter
from harp.analysis.util import apply_colormap
from harp.modeling.util import noop_ctx, log_mel_spectrogram, log_mel_spectrogram_diff

SPECT_CMAP = 'BuPu_r'
SPECT_DIFF_CMAP = 'managua'

def is_main_worker():
    return not dist.is_initialized() or dist.get_rank() == 0

def barrier():
    if dist.is_initialized():
        dist.barrier()

def save_checkpoint(
        path: str,
        config: dict,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        sched: torch.optim.lr_scheduler.LRScheduler
    ):
    if is_main_worker():
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'config.json'), mode='w', encoding='utf-8') as config_file:
            json.dump(config, config_file, indent=4)
        torch.save(model.state_dict(), os.path.join(path, 'model.pt'))
        torch.save(optim.state_dict(), os.path.join(path, 'optimizer.pt'))
        torch.save(sched.state_dict(), os.path.join(path, 'scheduler.pt'))
    barrier()

def load_checkpoint(
        path: str,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        sched: torch.optim.lr_scheduler.LRScheduler
    ):
    map_location = {'cuda:0': f'cuda:{dist.get_rank()}'}
    model.load_state_dict(torch.load(os.path.join(path, 'model.pt'), map_location=map_location))
    optim.load_state_dict(torch.load(os.path.join(path, 'optimizer.pt')))
    sched.load_state_dict(torch.load(os.path.join(path, 'scheduler.pt')))

@torch.no_grad()
def evaluate(
        config: dict,
        model: ASURA,
        eval_dataloader: DataLoader,
        tb_writer: SummaryWriter,
        device: torch.device
    ):
    histogram_keys = []
    c_real = None
    if is_main_worker():
        model.eval()
        total_eval_losses = {}
        total_eval_examples = 0
        log_example_idxs = list(reversed(range(0, config['eval_examples'], config['eval_examples']//config['eval_log_examples'])[:config['eval_log_examples']]))
        log_examples_gold = []
        log_examples_pred = []
        log_examples_pred_null = []
        c_real = []
        all_histogram_samples = {}
        for batch_raw in eval_dataloader:
            batch_raw = batch_raw[:config['eval_examples']-total_eval_examples].to(device=device)
            with torch.random.fork_rng(devices=(device,)):
                torch.cuda.manual_seed(42 + total_eval_examples)
                batch_outputs = model(batch_raw)
            while len(log_example_idxs) > 0 and log_example_idxs[-1] - total_eval_examples < batch_raw.size(0):
                log_sample_idx = log_example_idxs.pop() - total_eval_examples
                log_examples_gold.append(batch_outputs["y"][log_sample_idx, :, 0].cpu())
                log_examples_pred.append(batch_outputs["y_pred"][log_sample_idx, :, 0].cpu())
                if model.seq_encoder is not None:
                    c_real.append(batch_outputs['c_enc'][log_sample_idx])

            losses = {k: v for k, v in batch_outputs.items() if k.startswith('loss')}
            for loss_type, loss_value in losses.items():
                total_eval_losses[loss_type] = total_eval_losses.get(loss_type, 0.0) + loss_value.item() * batch_raw.size(0)
            for hist_key in histogram_keys:
                if hist_key in batch_outputs:
                    if hist_key not in all_histogram_samples:
                        all_histogram_samples[hist_key] = []
                    all_histogram_samples[hist_key].append(batch_outputs[hist_key].cpu())
            total_eval_examples += batch_raw.size(0)
            if total_eval_examples >= config['eval_examples']:
                break
        
        for loss_type, total_loss_value in total_eval_losses.items():
            tb_writer.add_scalar(f'eval/{loss_type}', total_loss_value/total_eval_examples, global_step=config['global_step'])
        for hist_key, hist_samples in all_histogram_samples.items():
            tb_writer.add_histogram(f'eval/{hist_key}', torch.cat(hist_samples, dim=0), global_step=config['global_step'])

        first_log = config['global_step'] == config['eval_every']
        log_examples_pred = torch.stack(log_examples_pred)
        log_examples_gold = torch.stack(log_examples_gold)
        if first_log:
            spectrograms_gold = torch.from_numpy(apply_colormap(log_mel_spectrogram(log_examples_gold, 44100, 2048, 256).numpy(), cmap=SPECT_CMAP))
            tb_writer.add_image('eval/gold', spectrograms_gold.flatten(0, 1), global_step=config['global_step'], dataformats='HWC')
            for i in range(config.get('sample_count_gold', 0)):
                tb_writer.add_audio(f'samples/gold_{i}', log_examples_gold[i][None], global_step=config['global_step'])
        spectrograms_pred = torch.from_numpy(apply_colormap(log_mel_spectrogram(log_examples_pred, 44100, 2048, 256).numpy(), cmap=SPECT_CMAP))
        tb_writer.add_image('eval/pred', spectrograms_pred.flatten(0, 1), global_step=config['global_step'], dataformats='HWC')
        spect_diffs = torch.from_numpy(apply_colormap(log_mel_spectrogram_diff(log_examples_pred, log_examples_gold, 44100, 2048, 256).numpy(), cmap=SPECT_DIFF_CMAP))
        tb_writer.add_image('eval/diff', spect_diffs.flatten(0, 1), global_step=config['global_step'], dataformats='HWC')
        
        if model.seq_encoder is not None:
            c_real = torch.stack(c_real[:config['sample_count_fixed']], dim=0)

        model.train()
    barrier()
    if config['global_step'] % config['sample_every'] == 0:
        sample(config, model, tb_writer, device, c_fixed=c_real, c_random=None)

@torch.no_grad()
def sample(
        config: dict,
        model: ASURA,
        tb_writer: SummaryWriter,
        device: torch.device,
        c_fixed: torch.Tensor | None = None,
        c_random: torch.Tensor | None = None
    ):
    if is_main_worker():
        model.eval()
        sample_length = config.get('sample_length', model.length_tokens)
        sample_count_fixed = config.get('sample_count_fixed', 0)
        if sample_count_fixed > 0:
            with torch.random.fork_rng(devices=(device,)):
                torch.cuda.manual_seed(42)
                samples = model.sample(
                    sample_count_fixed, sample_length, c_enc=c_fixed, ema=False, device=device
                )[:, :, 0].cpu()
                spectrograms = torch.from_numpy(apply_colormap(log_mel_spectrogram(samples, 44100, 2048, 256).numpy(), cmap=SPECT_CMAP))
                tb_writer.add_image('samples/fixed', spectrograms.flatten(0, 1), global_step=config['global_step'], dataformats='HWC')
                for i, sample in enumerate(samples.unbind(0)):
                    tb_writer.add_audio(f'samples/fixed_{i}', sample[None].clamp(min=-1.0, max=1.0), global_step=config['global_step'])
                samples_ema = model.sample(
                    sample_count_fixed, sample_length, c_enc=c_fixed, ema=True, device=device
                )[:, :, 0].cpu()
                spectrograms = torch.from_numpy(apply_colormap(log_mel_spectrogram(samples_ema, 44100, 2048, 256).numpy(), cmap=SPECT_CMAP))
                tb_writer.add_image('samples/fixed_ema', spectrograms.flatten(0, 1), global_step=config['global_step'], dataformats='HWC')
                for i, sample in enumerate(samples_ema.unbind(0)):
                    tb_writer.add_audio(f'samples/fixed_ema_{i}', sample[None].clamp(min=-1.0, max=1.0), global_step=config['global_step'])
        sample_count_random = config.get('sample_count_random', 0)
        if sample_count_random > 0:
            samples = model.sample(
                sample_count_random, sample_length, c_enc=c_random, ema=False, device=device
            )[:, :, 0].cpu()
            spectrograms = torch.from_numpy(apply_colormap(log_mel_spectrogram(samples, 44100, 2048, 256).numpy(), cmap=SPECT_CMAP))
            tb_writer.add_image('samples/random', spectrograms.flatten(0, 1), global_step=config['global_step'], dataformats='HWC')
            for i, sample in enumerate(samples.unbind(0)):
                tb_writer.add_audio(f'samples/random_{i}', sample[None].clamp(min=-1.0, max=1.0), global_step=config['global_step'])
            samples_ema = model.sample(
                sample_count_random, sample_length, c_enc=c_random, ema=True, device=device
            )[:, :, 0].cpu()
            spectrograms = torch.from_numpy(apply_colormap(log_mel_spectrogram(samples_ema, 44100, 2048, 256).numpy(), cmap=SPECT_CMAP))
            tb_writer.add_image('samples/random_ema', spectrograms.flatten(0, 1), global_step=config['global_step'], dataformats='HWC')
            for i, sample in enumerate(samples_ema.unbind(0)):
                tb_writer.add_audio(f'samples/random_ema_{i}', sample[None].clamp(min=-1.0, max=1.0), global_step=config['global_step'])
        model.train()
    barrier()

def train_worker_fn(
        rank: int,
        world_size: int,
        config_path: str,
        resume: bool,
        debug: bool=False
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
    config_name = os.path.splitext(os.path.basename(config_path))[0]

    if resume:
        run_name = config['run_name']
        global_step = config['global_step']
        ckpt_path = os.path.dirname(config_path)
    else:
        config['run_name'] = run_name = f"{config_name}_{datetime.now().strftime('%y%m%d-%H%M')}"
        global_step = 0
        ckpt_path = os.path.join('checkpoints', run_name)
    
    tb_writer = SummaryWriter(log_dir=os.path.join('logs', run_name)) if is_main_worker() else None

    model = model_module = ASURA(**config['model'], device=device)
    if rank == 0:
        print(f'Frame hop length: {model.frame_hop/44100:.2f}s')
        print(f'Context window: {model.length_samples/44100:.1f}s')

    optim_config = dict(config['optim'])
    optim_class = {
        'adam': torch.optim.Adam,
        'adamw': torch.optim.AdamW,
        'sgd': torch.optim.SGD
    }[optim_config.pop('type').lower()]
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[rank], static_graph=True
        )
    optim = optim_class(
        model.parameters(),
        **optim_config
    )
    
    local_batch_size = config.get('batch_size_per_device')
    step_batch_size = config.get('batch_size')
    if local_batch_size is None:
        local_batch_size = step_batch_size//world_size
    elif step_batch_size is None:
        step_batch_size = local_batch_size*world_size
    
    global_batch_size = local_batch_size*world_size
    assert step_batch_size % global_batch_size == 0
    batches_per_step = step_batch_size // global_batch_size

    example_length = model_module.example_length_samples

    if config['train_data'] == 'burst':
        train_dataset = NoiseBurstDataset(step_batch_size, example_length, 9450, 18900)
    elif config['train_data'] == 'wobble':
        train_dataset = MultiWobbleDataset(step_batch_size, example_length, 3, 50, 5000, 0.25, 4)
    else:
        train_dataset = VectorSequencesDataset(
            config['train_data'], example_length,
            do_offset=True,
            do_channel_flip=True
        )
    if config.get('sanity_check'):
        train_dataset = VectorSequencesDataset(
            [train_dataset[100].numpy()]*step_batch_size,
            example_length
        )
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        sampler=train_sampler,
        drop_last=True
    )

    eval_dataloader = None
    if rank == 0:
        eval_dataset = train_dataset if (config.get('sanity_check') or config['eval_data'] == config['train_data']) \
            else VectorSequencesDataset(config['eval_data'], example_length)
        if isinstance(eval_dataset, (MultiWobbleDataset, NoiseBurstDataset)):
            eval_dataset = VectorSequencesDataset([eval_dataset[i].numpy() for i in range(len(eval_dataset))], example_length)
        eval_sampler = DistributedSampler(eval_dataset, num_replicas=1, rank=0)
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=local_batch_size,
            sampler=eval_sampler
        )

    epoch_steps = len(train_dataloader)//batches_per_step
    start_epoch = global_step // epoch_steps

    lr_decay_type = config.get('lr_decay')
    if lr_decay_type is None:
        lr_decay_fn = lambda step: 1.0
    if lr_decay_type == 'linear':
        lr_decay_fn = lambda step: 1.0 - step / (epoch_steps*config['epochs'] - config['lr_warmup'])
    else:
        halflife = config.get('lr_decay_halflife', epoch_steps)
        lr_decay_fn = {
            'exponential': lambda step: 2.0**(-step/halflife),
            'reciprocal': lambda step: 1.0 / (1.0 + step/halflife),
            'quadratic': lambda step: 1.0 / (1.0 + (step/halflife)**2.0)
        }[lr_decay_type]
    lr_factor_lambda = lambda step: step / config['lr_warmup'] if step < config['lr_warmup'] else lr_decay_fn(step - config['lr_warmup'])
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_factor_lambda
    )

    skipped_batches = 0
    if resume:
        load_checkpoint(ckpt_path, model, optim, schedule)
        skipped_batches = (global_step % epoch_steps) * batches_per_step
        train_sampler.set_skip(skipped_batches * global_batch_size)

    if 'epochs' in config:
        epochs = range(start_epoch, config['epochs'])
    else:
        epochs = count(start_epoch)

    accumulated_batches = 0
    accumulated_losses = {}

    for epoch in epochs:
        train_sampler.set_epoch(epoch)
        train_dataloader_prog = tqdm(train_dataloader, desc=f"Epoch {epoch}", disable=rank > 0, initial=skipped_batches)
        for batch in train_dataloader_prog:
            batch = batch.to(device=device)

            with torch.autograd.detect_anomaly() if debug else noop_ctx():
                batch_outputs = model(batch)
                losses = {k: v for k, v in batch_outputs.items() if k.startswith('loss')}
                (losses['loss_all'] / batches_per_step).backward()

            for loss_type in sorted(losses.keys()):
                if world_size > 1:
                    loss_gathered = losses[loss_type].detach().clone()
                    dist.all_reduce(loss_gathered)
                    batch_loss_cpu = loss_gathered.item() * local_batch_size
                else:
                    batch_loss_cpu = losses[loss_type].item() * local_batch_size
                accumulated_losses[loss_type] = accumulated_losses.get(loss_type, 0.0) + batch_loss_cpu
            
            del losses
            del batch_outputs

            accumulated_batches += 1
            if accumulated_batches == batches_per_step:
                if 'clip_grad' in config:
                    for name, group in model_module.grad_clip_groups().items():
                        norm = torch.nn.utils.clip_grad_norm_(group, config['clip_grad'])
                        if is_main_worker():
                            tb_writer.add_scalar(f'train/grad_norm_{name}', norm, global_step=global_step)
                optim.step()
                optim.zero_grad(set_to_none=True)
                schedule.step()

                model_module.update_ema()

                if is_main_worker():
                    for loss_type, accumulated_loss_value in accumulated_losses.items():
                        step_loss = accumulated_loss_value/step_batch_size
                        tb_writer.add_scalar(f'train/{loss_type}', step_loss, global_step=global_step)
                    tb_writer.add_scalar('train/lr', schedule.get_last_lr()[0], global_step=global_step)
                    train_dataloader_prog.set_postfix(loss=accumulated_losses['loss_all']/step_batch_size, refresh=False)

                accumulated_batches = 0
                accumulated_losses.clear()
                global_step += 1
                config['global_step'] = global_step
                
                if global_step % config['save_every'] == 0:
                    save_checkpoint(ckpt_path, config, model, optim, schedule)
                if global_step % config['eval_every'] == 0:
                    evaluate(config, model_module, eval_dataloader, tb_writer, device)
        
        train_sampler.set_skip(0)
        skipped_batches = 0

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
        train_worker_fn(0, 1, config_path, resume, debug=args.debug)
    else:
        mp.spawn(
            train_worker_fn, args=(args.num_workers, config_path, resume, args.debug),
            nprocs=args.num_workers,
            join=True
        )

if __name__ == "__main__":
    main()