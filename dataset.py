from megengine.data.dataset import Dataset as Base
from megengine.data.sampler import SequentialSampler
from megengine.data import DataLoader
import numpy as np
from utils import pixel_unshuffle
from copy import deepcopy


__all__ = ['make_dataloader']

class Dataset(Base):
    def __init__(self, data_path):
        super().__init__()
        with open(data_path, 'rb') as content:
            x = np.frombuffer(content.read(), dtype='uint16').reshape((-1, 256, 256))
        x = np.expand_dims(x, 1).astype(np.float32)
        x = pixel_unshuffle(x) / 65535.0
        self.x = x

    def __getitem__(self, index):
        return deepcopy(self.x[index])

    def __len__(self):
        return len(self.x)


def make_dataloader(data_path, batch_size, workers=0):
    dataset = Dataset(data_path)
    loader = DataLoader(
        dataset,
        SequentialSampler(
            dataset,
            batch_size=batch_size,
            drop_last=False,
            world_size=1,
            rank=0,
        ),
        num_workers=workers
    )
    return loader
