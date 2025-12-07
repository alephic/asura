from math import sqrt, floor
import math
from typing import TypeVar, Optional, Iterator

import torch
from torch.utils.data import Sampler, Dataset
import torch.distributed as dist

T_co = TypeVar('T_co', covariant=True)

# Memory-efficient DistributedSampler by Aleksandr Dubinsky (github.com/almson)
# NOTE: Broken sometime after torch 2.1.1

class Range:
    size: int

    # for slicing
    start: int
    stop: int
    step: int

    # for iterating
    index: int

    def __init__(self, size: int, start: int = None, stop: int = None, step: int = None):
        self.size = size
        self.start = start if start is not None else 0
        self.stop = stop if stop is not None else size
        self.step = step if step is not None else 1
        if not (0 <= self.start <= self.stop):
            raise ValueError('not (0 <= start <= stop)')
        if self.stop > self.size:
            pass  # Allow wrap-around
        if self.step <= 0:
            raise ValueError('step <= 0')
        if (size == 0) != (len(self) == 0):
            raise ValueError('if size==0 then stop-start must also be 0')

    def __len__(self):
        if self.start == self.stop:
            return 0
        return (self.stop - self.start - 1) // self.step + 1

    def __getitem__(self, index):
        if isinstance(index, slice):
            sl = slice(index.start if index.start is not None else 0,
                       index.stop if index.stop is not None else len(self),
                       index.step if index.step is not None else 1)
            if not (0 <= sl.start and sl.start <= sl.stop and 0 < sl.step):
                raise IndexError
            return self._slice(sl)

        if not 0 <= index < len(self):
            raise IndexError

        abs_index = (self.start + index * self.step) % self.size

        return self._get(abs_index)

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index == len(self):
            raise StopIteration
        next = self[self.index]
        self.index = self.index + 1
        return next

    def _get(self, index):
        return index

    def _slice(self, sl):
        return Range(self.size,
                     self.start + sl.start * self.step,
                     self.start + sl.stop * self.step,
                     self.step * sl.step)


# We can use a faster primehood test, but there is no readily available implementation.
# But even this brute-force test is reasonably fast, O(sqrt(n))
def _is_prime(n):
    if n == 2:
        return True
    if n == 1 or n % 2 == 0:
        return False

    for d in range(3, floor(sqrt(n)) + 1, 2):  # can use isqrt in Python 3.8
        if n % d == 0:
            return False

    return True


class Permutation(Range):
    """
    Generates a random permutation of integers from 0 up to size.
    Inspired by https://preshing.com/20121224/how-to-generate-a-sequence-of-unique-random-integers/
    """

    prime: int
    seed: int

    def __init__(self, size: int, seed: int, start: int = None, stop: int = None, step: int = None, _prime: int = None):
        super().__init__(size, start, stop, step)
        self.prime = self._get_prime(size) if _prime is None else _prime
        self.seed = seed % self.prime

    def _get(self, index):
        x = self._map(index)

        while x >= self.size:
            # If we map to a number greater than size, then the cycle of successive mappings must eventually result
            # in a number less than size. Proof: The cycle of successive mappings traces a path
            # that either always stays in the set n>=size or it enters and leaves it,
            # else the 1:1 mapping would be violated (two numbers would map to the same number).
            # Moreover, `set(range(size)) - set(map(n) for n in range(size) if map(n) < size)`
            # equals the `set(map(n) for n in range(size, prime) if map(n) < size)`
            # because the total mapping is exhaustive.
            # Which means we'll arrive at a number that wasn't mapped to by any other valid index.
            # This will take at most `prime-size` steps, and `prime-size` is on the order of log(size), so fast.
            # But usually we just need to remap once.
            x = self._map(x)

        return x

    def _slice(self, sl):
        return Permutation(self.size,
                           self.seed,
                           self.start + sl.start * self.step,
                           self.start + sl.stop * self.step,
                           self.step * sl.step,
                           self.prime)

    @staticmethod
    def _get_prime(size):
        """
        Returns the prime number >= size which has the form (4n-1)
        """
        n = size + (3 - size % 4)
        while not _is_prime(n):
            # We expect to find a prime after O(log(size)) iterations
            # Using a brute-force primehood test, total complexity is O(log(size)*sqrt(size)), which is pretty good.
            n = n + 4
        return n

    def _map(self, index):
        a = self._permute_qpr(index)
        b = (a + self.seed) % self.prime
        c = self._permute_qpr(b)
        return c

    def _permute_qpr(self, x):
        residue = pow(x, 2, self.prime)

        if x * 2 < self.prime:
            return residue
        else:
            return self.prime - residue
        
class DistributedSampler(Sampler[T_co]):
    r"""Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class:`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.
    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.
    Example::
        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # `type:ignore` is required because Dataset cannot provide a default __len__
        # see NOTE in pytorch/torch/utils/data/sampler.py
        self.dataset_size = len(self.dataset)  # type: ignore
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and self.dataset_size % self.num_replicas != 0:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil((self.dataset_size - self.num_replicas) / self.num_replicas)
        else:
            self.num_samples = math.ceil(self.dataset_size / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.skip = 0

    def __iter__(self) -> Iterator[T_co]:
        if not self.shuffle:
            indices = Range(self.dataset_size, start=self.skip, stop=self.total_size)
        else:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            seed = int(torch.randint(high=self.dataset_size, size=(1,), dtype=torch.int64, generator=g).item())
            indices = Permutation(self.dataset_size, seed, start=self.skip, stop=self.total_size)

        # subsample
        indices = indices[self.rank:self.total_size-self.skip:self.num_replicas]
        assert len(indices) == self.num_samples - self.skip//self.num_replicas

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples - self.skip//self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.
        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch

    def set_skip(self, skip: int) -> None:
        """ Sets how many examples to skip. """
        self.skip = skip