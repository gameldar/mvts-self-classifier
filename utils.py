from torch.utils.data import Dataset
import torch
import random

import numpy as np
from tsaug import AddNoise, TimeWarp, Quantize, Drift, Reverse, Dropout, Pool

class MTSDataset(Dataset):
    def __init__(self, dataframe, transform=None, has_labels=True):
        super(MTSDataset, self).__init__()
        self.has_labels = has_labels

        if has_labels:
          self.df = dataframe.drop(columns=['label'])
          self.labels = dataframe['label'].values
        else:
          self.df = dataframe
        self.transform = transform

    def __getitem__(self, index):
        values = self.df.loc[index].values
        if self.transform:
            values = self.transform(values)
        if self.has_labels:
          lbls = torch.tensor([self.labels[index]])
          return (values, lbls, index)
        return (values, index)

    def __len__(self):
        return len(self.df)


class MTSAugmentation(object):
    def __init__(self):
        self.augs = [("Noise", AddNoise()),
                ("TimeWarp", TimeWarp()),
                ("Reverse", Reverse()),
                ("Drift", Drift(max_drift=0.7, n_drift_points=5)),
                ("Dropout", Dropout(p = 0.1, size=(1,5), fill=float("0.0"), per_channel=True)),
                ("Pool", Pool(size=2)),
                ("Quantize", Quantize(n_levels=20))
                ]

    @staticmethod
    def reshape_array_data(data):
        """Blocked window ([x..., y..., z...]) to the (1, time, axis) view tsaug wants."""
        axis_len, remainder = divmod(data.shape[0], 3)
        if remainder != 0:
            raise ValueError(f"window length {data.shape[0]} is not divisible by 3 axes")
        axes = data.reshape(3, axis_len)
        return np.ascontiguousarray(axes.T[np.newaxis, ...])

    @staticmethod
    def flatten_array_data(data, shape):
        """Inverse of reshape_array_data, back to the blocked layout train.parquet uses.

        Reshaping the (1, time, axis) view directly would reinterpret the memory and
        interleave the axes, leaving the view in a different layout to the input.
        """
        axes = data.reshape(data.shape[-2], 3).T
        return np.ascontiguousarray(axes).reshape(shape)

    def __call__(self, data):
        changed = list()
        orig_shape = data.shape
        data = self.reshape_array_data(data)

        for i in range(0, 2):
            name, aug = random.choice(self.augs)
            aug = self.flatten_array_data(aug.augment(data), orig_shape)
            if np.isnan(aug).any():
                print(f"Applied {name} and resulted in nan")
            changed.append(aug)

        return changed
