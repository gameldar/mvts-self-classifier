import torch.nn as nn

AXES = 3


def as_axis_channels(flat, channels=AXES):
    """Flat blocked windows ([x..., y..., z...]) to (batch, channel, time) for the backbone.

    The blocked layout maps straight onto the channel dimension, so this is a plain
    reshape. It is only correct because MTSAugmentation preserves that layout. Two fused
    sensors are six blocks rather than three, so the count is a parameter.
    """
    axis_len, remainder = divmod(flat.shape[-1], channels)
    if remainder != 0:
        raise ValueError(f"window length {flat.shape[-1]} is not divisible by {channels}")
    return flat.reshape(flat.shape[0], channels, axis_len)


class CNN(nn.Module):
    def __init__(
        self,
        num_cols,
        features=115,
        filters=128,
        dropout=0.5,
        maxpool=2,
        kernel_size=6,
        channels=AXES,
    ):
        super(CNN, self).__init__()
        _, remainder = divmod(num_cols, channels)
        if remainder != 0:
            raise ValueError(f"window length {num_cols} is not divisible by {channels} channels")
        # convolve over time within each axis, with the axes as input channels so a
        # filter sees all of them at once. Kept on the module so callers that reshape a
        # flat window have a single source for the count.
        self.channels = channels
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=channels, out_channels=filters, kernel_size=kernel_size),
            nn.ReLU(),
            nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=kernel_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(maxpool),
        )
        self.project = nn.Linear(in_features=filters, out_features=features)

    def forward(self, x):
        # (batch, axis, time) -> (batch, time, filters), so the LSTM that follows steps
        # over time and the projection mixes filters within each step. Projecting the
        # time axis instead would leave the LSTM recurring over filter index.
        return self.project(self.cnn(x).transpose(1, 2))

    def freeze(self, warmup=False):
        self.requires_grad_(False)


class LSTM(nn.Module):
    def __init__(self, num_classes, features=115, bidirectional=False):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=features, hidden_size=features, batch_first=True, bidirectional=bidirectional
        )
        self.dense = nn.Linear(in_features=features, out_features=num_classes)

    def forward(self, x):
        x, (hn, cn) = self.lstm(x)
        results = self.dense(hn[-1])
        return results

    def freeze(self, warmup=False):
        self.lstm.requires_grad_(False)


class CNNLSTM(nn.Module):
    def __init__(
        self,
        num_cols,
        num_classes,
        features=115,
        filters=128,
        dropout=0.5,
        maxpool=2,
        kernel_size=6,
        bidirectional=False,
        channels=AXES,
    ):
        super(CNNLSTM, self).__init__()
        self.cnn = CNN(
            num_cols,
            features=features,
            filters=filters,
            dropout=dropout,
            maxpool=maxpool,
            kernel_size=kernel_size,
            channels=channels,
        )
        self.lstm = LSTM(num_classes, features, bidirectional=bidirectional)

    def forward(self, x):
        x = self.cnn(x)
        return self.lstm(x)

    def freeze(self, warmup=False):
        self.cnn.freeze(warmup=warmup)
        if warmup:
            self.lstm.freeze(warmup=warmup)
