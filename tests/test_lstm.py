import pytest
import torch

from lstm import CNN, LSTM, as_axis_channels
from model import Model

DATA_SIZE = 450
AXIS_LEN = DATA_SIZE // 3
FEATURES = 115
FILTERS = 128


def test_as_axis_channels_splits_a_blocked_window_into_axes():
    flat = torch.tensor([[1.0, 1.0, 2.0, 2.0, 3.0, 3.0]])

    channels = as_axis_channels(flat)

    assert channels.shape == (1, 3, 2)
    assert channels[0, 0].tolist() == [1.0, 1.0]
    assert channels[0, 1].tolist() == [2.0, 2.0]
    assert channels[0, 2].tolist() == [3.0, 3.0]


def test_as_axis_channels_rejects_windows_that_are_not_tri_axial():
    with pytest.raises(ValueError):
        as_axis_channels(torch.zeros(1, 8))


def test_backbone_consumes_the_three_axes_as_channels():
    backbone = CNN(DATA_SIZE, FEATURES)

    features = backbone(torch.randn(4, 3, AXIS_LEN))

    assert features.shape == (4, FILTERS, FEATURES)


def test_backbone_rejects_a_flattened_single_channel_window():
    """The axes must arrive as channels, not concatenated along time."""
    backbone = CNN(DATA_SIZE, FEATURES)

    with pytest.raises(RuntimeError):
        backbone(torch.randn(4, 1, DATA_SIZE))


def test_backbone_rejects_a_window_length_that_is_not_tri_axial():
    with pytest.raises(ValueError):
        CNN(DATA_SIZE - 1, FEATURES)


def test_backbone_output_is_sensitive_to_cross_axis_alignment():
    """Two windows with identical per-axis content but different alignment in time.

    A backbone that only ever sees one axis per filter cannot tell these apart beyond
    the shift within each axis; one that takes the axes as channels can.
    """
    torch.manual_seed(0)
    backbone = CNN(DATA_SIZE, FEATURES).eval()

    x = torch.randn(AXIS_LEN)
    y = torch.randn(AXIS_LEN)
    z = torch.randn(AXIS_LEN)
    aligned = torch.cat([x, y, z]).unsqueeze(0)
    shifted = torch.cat([x, y.roll(7), z]).unsqueeze(0)

    with torch.no_grad():
        first = backbone(as_axis_channels(aligned))
        second = backbone(as_axis_channels(shifted))

    assert not torch.allclose(first, second)


def test_model_takes_flat_views_and_returns_logits_per_head():
    model = Model(
        base_model=CNN(DATA_SIZE, FEATURES),
        classifier=LSTM(FEATURES),
        dim=FEATURES,
        cls_size=[4],
    )
    views = [torch.randn(2, DATA_SIZE), torch.randn(2, DATA_SIZE)]

    embeddings = model(views, return_embds=True)
    probs = model(embeddings, return_embds=False)

    assert len(embeddings) == len(views)
    assert probs[0][0].shape == (2, 4)
