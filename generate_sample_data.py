"""Generate synthetic tri-axial accelerometer data for testing the classifier.

Each row is one window of accelerometer samples laid out as all x samples, then all
y, then all z (matching MTSAugmentation.reshape_array_data). Windows are drawn from a
few behaviour prototypes so the data has real structure for the self-supervised model
to find.

The windows file carries no label column, because train.py trains without labels. The
ground-truth behaviour per window is written to a row-aligned sidecar instead, so
cluster_eval.py can score what the model found against what generated it.
"""

import argparse
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

# Each prototype describes a behaviour: the axis gravity rests on, plus the
# frequency, amplitude and noise of the motion. Loosely models grazing animals.
BEHAVIOURS = {
    "resting": {"gravity_axis": 2, "freq": 0.3, "amp": 0.05, "noise": 0.02},
    "grazing": {"gravity_axis": 2, "freq": 1.5, "amp": 0.25, "noise": 0.05},
    "walking": {"gravity_axis": 1, "freq": 4.0, "amp": 0.6, "noise": 0.08},
    "running": {"gravity_axis": 0, "freq": 8.0, "amp": 1.1, "noise": 0.15},
}

LABELS_SUFFIX = "_labels"


class SyntheticDataset(NamedTuple):
    """Windows and their ground truth, kept apart so training stays unsupervised."""

    windows: pd.DataFrame
    labels: pd.DataFrame
    counts: dict[str, int]


def make_window(rng, spec, window):
    t = np.linspace(0.0, 1.0, window)
    axes = []
    for axis in range(3):
        phase = rng.uniform(0, 2 * np.pi)
        signal = spec["amp"] * np.sin(2 * np.pi * spec["freq"] * t + phase)
        signal += rng.normal(0.0, spec["noise"], size=window)
        if axis == spec["gravity_axis"]:
            signal += 1.0  # gravity, ~1g on the resting axis
        axes.append(signal)
    return np.concatenate(axes).astype("float32")


def generate(rows, window, seed):
    rng = np.random.default_rng(seed)
    names = list(BEHAVIOURS)
    drawn = rng.integers(0, len(names), size=rows)
    behaviours = [names[index] for index in drawn]

    data = np.stack([make_window(rng, BEHAVIOURS[name], window) for name in behaviours])
    columns = [f"{axis}_{i}" for axis in ("x", "y", "z") for i in range(window)]
    timestamps = pd.date_range("2016-06-26 09:40:25", periods=rows, freq="min")

    windows = pd.DataFrame(data, columns=columns)
    windows.insert(0, "Timestamp", timestamps)
    labels = pd.DataFrame({"Timestamp": timestamps, "behaviour": behaviours})

    counts = labels["behaviour"].value_counts().to_dict()
    return SyntheticDataset(windows=windows, labels=labels, counts=counts)


def labels_path_for(output):
    """Sidecar path for a windows file: data/train.parquet -> data/train_labels.parquet."""
    output = Path(output)
    return output.with_name(f"{output.stem}{LABELS_SUFFIX}{output.suffix}")


def write(dataset, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.windows.to_parquet(output)

    labels_path = labels_path_for(output)
    dataset.labels.to_parquet(labels_path)
    return labels_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=500, help="number of windows to generate")
    parser.add_argument(
        "--window",
        type=int,
        default=150,
        help="samples per axis; total columns (--data-size) is 3x this",
    )
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--output", default="data/train.parquet")
    args = parser.parse_args()

    dataset = generate(args.rows, args.window, args.seed)
    labels_path = write(dataset, args.output)

    data_columns = dataset.windows.shape[1] - 1
    print(f"Wrote {dataset.windows.shape[0]} rows x {data_columns} data columns to {args.output}")
    print(f"Wrote ground-truth behaviours to {labels_path}")
    print(f"data-size for train.py: {3 * args.window}")
    print(f"behaviour mix: {dataset.counts}")


if __name__ == "__main__":
    main()
