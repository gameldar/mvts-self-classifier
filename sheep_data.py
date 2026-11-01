"""Convert observed sheep behaviour recordings into the parquet pair the model trains on.

Each row of the source CSVs is one 10 second segment carrying two accelerometers - a
halter at 30 Hz and an ear tag at 25 Hz - plus a `class` naming every action observed in
that segment. The model takes one tri-axial window, so a dataset is built per sensor.

The windows file carries no label column, because train.py trains without labels. The
observed behaviour per window goes to a row-aligned sidecar instead, so cluster_eval.py
can score what the model found against what the observers recorded.
"""

import argparse
import glob
import os
from fractions import Fraction
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from generate_sample_data import write

AXES = ("x", "y", "z")


class Sensor(NamedTuple):
    """One accelerometer's share of a segment: its column prefix, length and rate."""

    name: str
    prefix: str
    samples: int
    rate_hz: float


SENSORS = {
    "halter": Sensor("halter", "halt", 300, 30.0),
    "ear": Sensor("ear", "ear", 250, 25.0),
}


class Fusion(NamedTuple):
    """Several sensors laid out as channels of one window, resampled to a common length."""

    name: str
    sensors: tuple[Sensor, ...]
    samples: int


# The halter runs at 30 Hz and the ear at 25 Hz over the same 10 s, so one has to move.
# The ear is resampled up rather than the halter down, keeping the halter's full
# bandwidth in case fast jaw motion is what separates chewing from rumination.
FUSIONS = {
    "combined": Fusion("combined", (SENSORS["halter"], SENSORS["ear"]), 300),
}

# A segment is labelled with every action seen in it, so 26 compound classes appear.
# Collapsing takes the most specific action present: what the sheep was doing outranks
# the posture it was doing it in.
PRECEDENCE = ("ruminating", "grazing", "walking", "standing", "sitting")


class SheepDataset(NamedTuple):
    """Windows and their observed behaviour, kept apart so training stays unsupervised."""

    windows: pd.DataFrame
    labels: pd.DataFrame
    counts: dict[str, int]


def collapse_label(compound):
    """One behaviour for a compound class, by PRECEDENCE."""
    actions = compound.split("_")
    unknown = [action for action in actions if action not in PRECEDENCE]
    if unknown:
        raise ValueError(f"class {compound!r} names unknown actions: {', '.join(unknown)}")
    return next(action for action in PRECEDENCE if action in actions)


def sensor_columns(sensor):
    """The sensor's columns in blocked axis order, as the source CSVs name them."""
    return [f"{sensor.prefix}_{axis}_{i}" for axis in AXES for i in range(sensor.samples)]


def fused_columns(fusion):
    """The output column names of a fusion, blocked per channel in sensor order."""
    return [
        f"{sensor.prefix}_{axis}_{i}"
        for sensor in fusion.sensors
        for axis in AXES
        for i in range(fusion.samples)
    ]


def channels(layout):
    """How many input channels a sensor or fusion produces."""
    sensors = layout.sensors if isinstance(layout, Fusion) else (layout,)
    return len(AXES) * len(sensors)


def data_size(layout):
    """The --data-size train.py needs for this sensor or fusion."""
    return channels(layout) * layout.samples


def resample_axis(values, target):
    """One axis of every segment resampled to target samples, with unity DC gain."""
    native = values.shape[1]
    if native == target:
        return values
    factor = Fraction(target, native)
    return resample_poly(values, factor.numerator, factor.denominator, axis=1)


def fuse(frame, fusion):
    """Every sensor's axes resampled to the fusion's common length, in channel order."""
    blocks = []
    for sensor in fusion.sensors:
        for axis in AXES:
            columns = [f"{sensor.prefix}_{axis}_{i}" for i in range(sensor.samples)]
            missing = [column for column in columns if column not in frame.columns]
            if missing:
                raise ValueError(
                    f"segments are missing {len(missing)} {sensor.name} columns, from {missing[0]}"
                )
            blocks.append(resample_axis(frame[columns].to_numpy(dtype="float64"), fusion.samples))
    return np.hstack(blocks).astype("float32")


def convert(frame, layout):
    """One sensor's or fusion's windows out of a segment frame, with labels row-aligned."""
    timestamps = pd.date_range("2016-06-26 09:40:25", periods=len(frame), freq="min")

    if isinstance(layout, Fusion):
        samples = pd.DataFrame(fuse(frame, layout), columns=fused_columns(layout))
    else:
        columns = sensor_columns(layout)
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(
                f"segments are missing {len(missing)} {layout.name} columns, from {missing[0]}"
            )
        samples = frame[columns].astype("float32").reset_index(drop=True)
        samples.columns = [f"{axis}_{i}" for axis in AXES for i in range(layout.samples)]
    # concatenated rather than inserted: these frames run to 900 columns, and inserting
    # into one selected out of a wider frame leaves it badly fragmented
    windows = pd.concat([pd.Series(timestamps, name="Timestamp"), samples], axis=1)

    labels = pd.DataFrame(
        {
            "Timestamp": timestamps,
            "behaviour": [collapse_label(value) for value in frame["class"]],
        }
    )
    if "source" in frame.columns:
        labels["source"] = frame["source"].values
        labels["location"] = labels["source"].map(location_of)

    return SheepDataset(
        windows=windows, labels=labels, counts=labels["behaviour"].value_counts().to_dict()
    )


def location_of(source):
    """The recording location a sheep's file belongs to, without its number."""
    return source.rsplit("_", 1)[0]


def select_locations(segments, locations):
    """Only the segments recorded at the named locations."""
    available = sorted({location_of(source) for source in segments["source"]})
    unknown = [location for location in locations if location not in available]
    if unknown:
        raise ValueError(
            f"no segments from {', '.join(unknown)}; the data holds {', '.join(available)}"
        )
    keep = segments["source"].map(location_of).isin(locations)
    return segments[keep].reset_index(drop=True)


def read_segments(directory):
    """Every segment in the directory's CSVs, tagged with the sheep it came from."""
    paths = sorted(glob.glob(os.path.join(directory, "*.csv")))
    if not paths:
        raise ValueError(f"no CSV files in {directory}")

    frames, sources = [], []
    for path in paths:
        frame = pd.read_csv(path)
        frames.append(frame)
        sources += [os.path.splitext(os.path.basename(path))[0]] * len(frame)

    # joined as a column rather than inserted: these frames are 1651 columns wide and
    # inserting into them one at a time leaves the result badly fragmented
    segments = pd.concat(frames, ignore_index=True)
    return pd.concat([segments, pd.Series(sources, name="source")], axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="directory holding the per-sheep CSVs")
    parser.add_argument(
        "--sensor",
        choices=sorted(SENSORS) + sorted(FUSIONS),
        default="halter",
        help="one sensor, or 'combined' for both fused as six channels",
    )
    parser.add_argument(
        "--location",
        nargs="+",
        default=None,
        help="only these recording locations, e.g. 'Muresk Dry Pasture' 'Muresk Stubble'",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="windows file to write; defaults to data/sheep-<sensor>/train.parquet",
    )
    args = parser.parse_args()

    layout = FUSIONS.get(args.sensor) or SENSORS[args.sensor]
    output = args.output or os.path.join(f"data/sheep-{layout.name}", "train.parquet")

    segments = read_segments(args.source)
    if args.location:
        segments = select_locations(segments, args.location)
    dataset = convert(segments, layout)
    labels_path = write(dataset, output)

    print(f"Wrote {len(dataset.windows)} segments x {data_size(layout)} data columns to {output}")
    print(f"Wrote observed behaviours to {labels_path}")
    print(f"data-size for train.py: {data_size(layout)}")
    print(f"channels for train.py:  {channels(layout)}")
    print(f"cls-size for train.py:  {len(dataset.counts)}")
    print(f"behaviour mix: {dataset.counts}")


if __name__ == "__main__":
    main()
