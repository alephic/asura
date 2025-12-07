# Asura's Harp

This repository contains the implementation of the paper "Asura's Harp: Direct Latent Control of Neural Sound".

[🔗Paper: openreview.net/pdf?id=cckgZBnUVC](https://openreview.net/pdf?id=cckgZBnUVC)

[🔈Samples: bostromk.net/ASURA/](https://bostromk.net/ASURA/)

As I am not in a position to redistribute others' music in bulk, or artifacts derived from such work, I am electing not to release the dataset or model checkpoints from the main training run described in the paper. However, all code to reproduce an equivalent model is available here -- all you need to do is pick out some data, not much audio is needed!

To train your own model, you'll need a CUDA-capable GPU, preferably of a recent enough architecture to support Flash Attention. I used a machine with 4 RTX A6000 48G cards, but smaller batch sizes + memory footprints are viable.

To get this repo ready to train, just clone it, create a virtual environment, and `pip install -r requirements.txt`. Note that due to a known issue with the distributed data sampler, `torch==2.1.1` is required for training, which is quite old. Resolving that issue to unblock more recent `torch.compile` capabilities is on my todo list.

All hyperparameters (model architecture, data, and optimization) are exposed in the [config file](./configs/asura/asura.json).

To prep data, use the `harp.data.preprocess_audio` script. This decompresses audio to PCM 16-bit for faster dataloading during training.

To launch training, from the root of the repo run:
```
python -m harp.asura.train configs/asura/asura.json -n <your GPU count>
```

To cache a set of latent embeddings for control mapping, run:
```
python -m harp.asura.encode <path to your trained checkpoint>
```

To train a control map on those embeddings (saved as `embeddings.npy` in the repo root by default), run:
```
python -m harp.asura.train_control_map
```

You can then play with the model interactively by launching this command on your desktop:
```
python -m harp.asura.play <main checkpoint> <control map checkpoint>
```

Currently, streaming output is not *quite* fast enough to send to speakers in realtime, but I am planning to add that ability to the player UI once I have squeezed a little more efficiency out of the model architecture.

### Notes on differences with the paper formalism

The implementation of amortized flow in this repository has developed a little beyond what's described in the paper: time delta sampling for the `u` decoder is now bidirectional, and since I had extra memory headroom I elected to train a `v` decoder alongside the `u` decoder to lower the gradient noise passed into the `u` objective from stochastic CFM pairings. There is an additional loss term that enforces round-trip velocity consistency using the bidirectional `u` mapping, and there is a third `y` decoder that explicitly learns only the `t==0` to `t==1` single-step jump condition (this part may be overkill). Despite the extra decoder networks, the past- and latent- conditioning encoders are still shared, so it is still more efficient to train all components simultaneously than to first train `v` and then distill `u` and `y` in successive rounds.
