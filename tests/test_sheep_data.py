import warnings

import pandas as pd
import pytest
from pandas.errors import PerformanceWarning

from sheep_data import (
    SENSORS,
    collapse_label,
    convert,
    data_size,
    location_of,
    read_segments,
    select_locations,
    sensor_columns,
)


def segment_frame(rows=2):
    """A frame shaped like the sheep CSVs: both sensors blocked, plus a class column."""
    columns = {}
    for prefix, samples in (("halt", SENSORS["halter"].samples), ("ear", SENSORS["ear"].samples)):
        for axis in "xyz":
            base = {"x": 0.0, "y": 10.0, "z": 20.0}[axis] + (0.0 if prefix == "halt" else 100.0)
            for i in range(samples):
                columns[f"{prefix}_{axis}_{i}"] = [base + i for _ in range(rows)]
    frame = pd.DataFrame(columns)
    frame["class"] = ["standing_grazing" for _ in range(rows)]
    return frame


def test_collapse_label_keeps_a_single_action_unchanged():
    assert collapse_label("sitting") == "sitting"


def test_collapse_label_prefers_ruminating_over_every_other_action():
    assert collapse_label("sitting_standing_walking_grazing_ruminating") == "ruminating"


def test_collapse_label_follows_the_full_precedence_order():
    assert collapse_label("standing_walking_grazing") == "grazing"
    assert collapse_label("sitting_standing_walking") == "walking"
    assert collapse_label("sitting_standing") == "standing"


def test_collapse_label_rejects_an_unknown_action():
    with pytest.raises(ValueError, match="flying"):
        collapse_label("standing_flying")


def test_sensor_columns_are_blocked_by_axis():
    columns = sensor_columns(SENSORS["ear"])

    assert columns[0] == "ear_x_0"
    assert columns[249] == "ear_x_249"
    assert columns[250] == "ear_y_0"
    assert columns[-1] == "ear_z_249"
    assert len(columns) == 750


def test_data_size_is_three_axes_of_samples():
    assert data_size(SENSORS["halter"]) == 900
    assert data_size(SENSORS["ear"]) == 750


def test_windows_hold_only_a_timestamp_and_the_chosen_sensor():
    dataset = convert(segment_frame(), SENSORS["halter"])

    assert list(dataset.windows.columns[:1]) == ["Timestamp"]
    assert len(dataset.windows.columns) == 1 + 900
    assert "class" not in dataset.windows.columns
    assert not [c for c in dataset.windows.columns if c.startswith(("halt", "ear"))]


def test_windows_are_renamed_to_the_blocked_layout_the_loader_expects():
    dataset = convert(segment_frame(), SENSORS["ear"])

    assert list(dataset.windows.columns[1:4]) == ["x_0", "x_1", "x_2"]
    assert dataset.windows.columns[-1] == "z_249"


def test_windows_take_their_values_from_the_chosen_sensor_only():
    dataset = convert(segment_frame(), SENSORS["ear"])

    # the fixture offsets the ear sensor by 100, so the first x sample proves the source
    assert dataset.windows["x_0"].iloc[0] == 100.0
    assert dataset.windows["y_0"].iloc[0] == 110.0


def test_labels_are_row_aligned_with_windows():
    dataset = convert(segment_frame(rows=5), SENSORS["halter"])

    assert len(dataset.labels) == len(dataset.windows)
    assert dataset.labels["Timestamp"].equals(dataset.windows["Timestamp"])


def test_labels_carry_the_collapsed_behaviour():
    dataset = convert(segment_frame(), SENSORS["halter"])

    assert set(dataset.labels["behaviour"]) == {"grazing"}


def test_counts_match_the_emitted_labels():
    dataset = convert(segment_frame(rows=4), SENSORS["halter"])

    assert dataset.counts == dataset.labels["behaviour"].value_counts().to_dict()


def test_labels_keep_the_source_file_as_provenance_when_given():
    frame = segment_frame()
    frame["source"] = ["Muresk Stubble_3.csv", "Muresk Stubble_3.csv"]

    dataset = convert(frame, SENSORS["halter"])

    assert list(dataset.labels["source"]) == ["Muresk Stubble_3.csv", "Muresk Stubble_3.csv"]


def test_convert_rejects_a_frame_missing_the_sensor_columns():
    frame = segment_frame().drop(columns=["halt_z_299"])

    with pytest.raises(ValueError, match="halt_z_299"):
        convert(frame, SENSORS["halter"])


def test_windows_are_float32_so_the_parquet_stays_small():
    dataset = convert(segment_frame(), SENSORS["halter"])

    assert (dataset.windows.drop(columns=["Timestamp"]).dtypes == "float32").all()


def write_csv(directory, name, rows=2, klass="sitting"):
    frame = segment_frame(rows)
    frame["class"] = [klass for _ in range(rows)]
    frame.to_csv(directory / name, index=False)


def test_read_segments_concatenates_every_csv_in_the_directory(tmp_path):
    write_csv(tmp_path, "Muresk Stubble_1.csv", rows=2)
    write_csv(tmp_path, "Muresk Stubble_2.csv", rows=3)

    segments = read_segments(tmp_path)

    assert len(segments) == 5


def test_read_segments_tags_each_segment_with_the_sheep_it_came_from(tmp_path):
    write_csv(tmp_path, "Muresk Stubble_1.csv", rows=2)
    write_csv(tmp_path, "Murdoch Green Pasture_4.csv", rows=1)

    segments = read_segments(tmp_path)

    assert sorted(set(segments["source"])) == ["Murdoch Green Pasture_4", "Muresk Stubble_1"]


def test_read_segments_rejects_a_directory_with_no_csvs(tmp_path):
    with pytest.raises(ValueError, match="no CSV files"):
        read_segments(tmp_path)


def test_convert_does_not_fragment_the_frame_it_builds(tmp_path):
    """Segment frames are 1651 columns wide, so column-at-a-time building is costly."""
    write_csv(tmp_path, "Muresk Stubble_1.csv", rows=2)
    write_csv(tmp_path, "Muresk Stubble_2.csv", rows=2)
    segments = read_segments(tmp_path)

    with warnings.catch_warnings():
        warnings.simplefilter("error", PerformanceWarning)
        convert(segments, SENSORS["halter"])


def test_location_of_a_source_drops_the_sheep_number():
    assert location_of("Muresk Stubble_3") == "Muresk Stubble"
    assert location_of("Murdoch Green Pasture_10") == "Murdoch Green Pasture"


def test_select_locations_keeps_only_the_named_locations():
    frame = segment_frame(rows=3)
    frame["source"] = ["Muresk Stubble_1", "Murdoch Green Pasture_2", "Muresk Dry Pasture_3"]

    kept = select_locations(frame, ["Muresk Stubble", "Muresk Dry Pasture"])

    assert list(kept["source"]) == ["Muresk Stubble_1", "Muresk Dry Pasture_3"]


def test_select_locations_rejects_a_location_that_is_not_in_the_data():
    frame = segment_frame(rows=2)
    frame["source"] = ["Muresk Stubble_1", "Muresk Stubble_2"]

    with pytest.raises(ValueError, match="Mureck Stuble"):
        select_locations(frame, ["Mureck Stuble"])


def test_labels_carry_the_location_alongside_the_source():
    frame = segment_frame(rows=2)
    frame["source"] = ["Muresk Stubble_1", "Muresk Stubble_2"]

    dataset = convert(frame, SENSORS["halter"])

    assert list(dataset.labels["location"]) == ["Muresk Stubble", "Muresk Stubble"]
