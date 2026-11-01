import pandas as pd

from generate_sample_data import BEHAVIOURS, generate, write


def test_windows_carry_no_label_column():
    dataset = generate(rows=8, window=4, seed=1)

    assert "behaviour" not in dataset.windows.columns
    assert "label" not in dataset.windows.columns


def test_labels_are_row_aligned_with_windows():
    dataset = generate(rows=8, window=4, seed=1)

    assert len(dataset.labels) == len(dataset.windows)
    assert dataset.labels["Timestamp"].equals(dataset.windows["Timestamp"])


def test_labels_name_known_behaviours():
    dataset = generate(rows=32, window=4, seed=1)

    assert set(dataset.labels["behaviour"]) <= set(BEHAVIOURS)


def test_counts_match_the_emitted_labels():
    dataset = generate(rows=32, window=4, seed=1)

    emitted = dataset.labels["behaviour"].value_counts().to_dict()
    assert dataset.counts == emitted


def test_gravity_axis_distinguishes_resting_from_running():
    dataset = generate(rows=200, window=50, seed=7)
    windows = dataset.windows.drop(columns=["Timestamp"])
    behaviours = dataset.labels["behaviour"].values

    # resting rests gravity on z, running on x, so the per-axis means separate them.
    resting_z = windows[behaviours == "resting"].iloc[:, 100:150].values.mean()
    running_z = windows[behaviours == "running"].iloc[:, 100:150].values.mean()

    assert resting_z > 0.9
    assert running_z < 0.5


def test_writing_emits_a_sidecar_next_to_the_windows(tmp_path):
    output = tmp_path / "train.parquet"
    dataset = generate(rows=8, window=4, seed=1)

    labels_path = write(dataset, output)

    assert labels_path == tmp_path / "train_labels.parquet"
    assert "behaviour" not in pd.read_parquet(output).columns
    assert list(pd.read_parquet(labels_path)["behaviour"]) == list(dataset.labels["behaviour"])
