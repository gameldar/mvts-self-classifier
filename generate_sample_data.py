"""Generate synthetic tri-axial accelerometer data for testing the classifier.

Each row is one window of accelerometer samples laid out as all x samples, then all
y, then all z (matching MTSAugmentation.reshape_array_data). Windows are drawn from a
few behaviour prototypes so the data has real structure for the self-supervised model
to find, but no label column is written since train.py trains without labels.
"""
import argparse

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
    labels = rng.integers(0, len(names), size=rows)

    data = np.stack([make_window(rng, BEHAVIOURS[names[label]], window) for label in labels])
    columns = [f"{axis}_{i}" for axis in ("x", "y", "z") for i in range(window)]
    df = pd.DataFrame(data, columns=columns)
    df.insert(0, "Timestamp", pd.date_range("2016-06-26 09:40:25", periods=rows, freq="min"))

    counts = {names[label]: int((labels == label).sum()) for label in range(len(names))}
    return df, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=500, help="number of windows to generate")
    parser.add_argument("--window", type=int, default=150,
                        help="samples per axis; total columns (--data-size) is 3x this")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--output", default="data/train.parquet")
    args = parser.parse_args()

    df, counts = generate(args.rows, args.window, args.seed)

    import os
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_parquet(args.output)

    print(f"Wrote {df.shape[0]} rows x {df.shape[1] - 1} data columns to {args.output}")
    print(f"data-size for train.py: {3 * args.window}")
    print(f"behaviour mix: {counts}")


if __name__ == "__main__":
    main()
