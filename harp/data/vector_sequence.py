from torch.utils.data.dataset import Dataset
import numpy as np
import os
import glob
import torch

class VectorSequenceDataset(Dataset):
    def __init__(self, path_or_arr: str | np.ndarray, example_length: int, do_channel_flip = False, do_offset = False, overlap = 0):
        if isinstance(path_or_arr, str):
            self._arr = np.load(path_or_arr, mmap_mode='r')
        else:
            self._arr = path_or_arr
        self._example_length = example_length
        self._do_offset = do_offset
        self._do_channel_flip = do_channel_flip
        self._stride = example_length - overlap
        self._len = (self._arr.shape[0] - self._example_length)//self._stride + 1
        if do_offset:
            self._len -= 1

    def __len__(self):
        return self._len

    def __getitem__(self, i):
        frame_start = i*self._stride
        if self._do_offset:
            frame_start += torch.randint(0, self._stride, tuple()).item()
        out = self._arr[frame_start:frame_start+self._example_length]
        if self._do_channel_flip and torch.rand(tuple()).item() < 0.5:
            out = out[:, ::-1]
        if out.dtype == np.int16:
            out = out.astype(np.float32) / 32768.0
        else:
            out = out.copy()
        return torch.from_numpy(out)
    
class VectorSequencesDataset(Dataset):
    def __init__(self, path_or_arrs: str | list[np.ndarray], example_length: int, do_channel_flip = False, do_offset = False, overlap = 0):
        if isinstance(path_or_arrs, str):
            if not os.path.exists(path_or_arrs):
                raise RuntimeError(f"Couldn't find dataset at {path_or_arrs}")
            source_iterable = sorted(glob.glob(os.path.join(path_or_arrs, '*')))
        else:
            source_iterable = path_or_arrs
        self.sequences = [
            VectorSequenceDataset(
                e,
                example_length=example_length,
                do_channel_flip=do_channel_flip,
                do_offset=do_offset,
                overlap=overlap
            ) for e in source_iterable
        ]
        self.sequence_end_idxs = np.cumsum(list(map(len, self.sequences)), axis=0)

    def __len__(self):
        return self.sequence_end_idxs[-1]

    def __getitem__(self, i):
        sequence_idx = np.searchsorted(self.sequence_end_idxs, i, side='right')
        if sequence_idx > 0:
            i -= self.sequence_end_idxs[sequence_idx-1]
        return self.sequences[sequence_idx][i]
