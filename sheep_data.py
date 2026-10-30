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
from typing import NamedTuple

import pandas as pd

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


def data_size(sensor):
    """The --data-size train.py needs for this sensor."""
    return len(AXES) * sensor.samples


def convert(frame, sensor):
    """One sensor's windows out of a segment frame, with labels row-aligned."""
    timestamps = pd.date_range("2016-06-26 09:40:25", periods=len(frame), freq="min")

    columns = sensor_columns(sensor)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(
            f"segments are missing {len(missing)} {sensor.name} columns, from {missing[0]}"
        )
    samples = frame[columns].astype("float32").reset_index(drop=True)
    samples.columns = [f"{axis}_{i}" for axis in AXES for i in range(sensor.samples)]
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
        choices=sorted(SENSORS),
        default="halter",
        help="which accelerometer to build the dataset from",
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

    sensor = SENSORS[args.sensor]
    output = args.output or os.path.join(f"data/sheep-{sensor.name}", "train.parquet")

    segments = read_segments(args.source)
    if args.location:
        segments = select_locations(segments, args.location)
    dataset = convert(segments, sensor)
    labels_path = write(dataset, output)

    print(f"Wrote {len(dataset.windows)} segments x {data_size(sensor)} data columns to {output}")
    print(f"Wrote observed behaviours to {labels_path}")
    print(f"data-size for train.py: {data_size(sensor)}")
    print(f"cls-size for train.py:  {len(dataset.counts)}")
    print(f"behaviour mix: {dataset.counts}")


if __name__ == "__main__":
    main()
