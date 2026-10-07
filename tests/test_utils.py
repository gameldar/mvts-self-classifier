import numpy as np
import pytest
from tsaug import Reverse

from utils import MTSAugmentation


def blocked_window(axis_len=4):
    """A window in the layout train.parquet uses: all x, then all y, then all z."""
    return np.concatenate(
        [
            np.full(axis_len, 1.0),
            np.full(axis_len, 2.0),
            np.full(axis_len, 3.0),
        ]
    ).astype("float32")


def test_reshape_separates_the_three_axes_over_time():
    window = blocked_window(axis_len=3)

    reshaped = MTSAugmentation.reshape_array_data(window)

    assert reshaped.shape == (1, 3, 3)
    # every timestep holds one sample from each axis
    assert reshaped[0, 0].tolist() == [1.0, 2.0, 3.0]


def test_flatten_inverts_reshape():
    window = np.arange(12, dtype="float32")

    round_tripped = MTSAugmentation.flatten_array_data(
        MTSAugmentation.reshape_array_data(window), window.shape
    )

    assert round_tripped.tolist() == window.tolist()


def test_reshape_rejects_windows_that_are_not_tri_axial():
    with pytest.raises(ValueError):
        MTSAugmentation.reshape_array_data(np.arange(8, dtype="float32"))


def test_augmented_views_keep_the_blocked_axis_layout():
    """Augmented views must come back in the same layout they went in as.

    Each axis is held constant, so reversing time cannot change any value. Anything
    other than the original blocked window means the axes were mixed by the reshape.
    """
    augmentation = MTSAugmentation()
    augmentation.augs = [("Reverse", Reverse())]
    window = blocked_window(axis_len=4)

    views = augmentation(window)

    for view in views:
        assert view.shape == window.shape
        assert view.tolist() == window.tolist()


def test_augmented_views_still_vary_the_signal():
    """Guard against the layout fix accidentally turning augmentation into a no-op."""
    augmentation = MTSAugmentation()
    augmentation.augs = [("Reverse", Reverse())]
    window = np.arange(12, dtype="float32")

    views = augmentation(window)

    assert all(view.tolist() != window.tolist() for view in views)
