import pandas as pd
import pytest

from sweep import (
    SeedResult,
    cluster_histogram,
    format_sweep_report,
    parse_args,
    prepare_run_dir,
    run_sweep,
    seed_list,
    summarise,
    train_command,
    write_sweep_csv,
)


def result(seed, mapping_accuracy=0.7, purity=0.8, clusters=4, final_loss=0.2):
    return SeedResult(
        seed=seed,
        final_loss=final_loss,
        clusters_used=clusters,
        purity=purity,
        mapping_accuracy=mapping_accuracy,
    )


def test_summarise_reports_mean_spread_and_range():
    stats = summarise([0.6, 0.8, 1.0])

    assert stats.mean == pytest.approx(0.8)
    assert stats.sd == pytest.approx(0.2)
    assert stats.minimum == pytest.approx(0.6)
    assert stats.maximum == pytest.approx(1.0)


def test_summarise_has_no_spread_for_a_single_seed():
    stats = summarise([0.6])

    assert stats.mean == pytest.approx(0.6)
    assert stats.sd is None


def test_summarise_rejects_an_empty_sweep():
    with pytest.raises(ValueError):
        summarise([])


def test_cluster_histogram_counts_how_many_runs_used_each_number_of_clusters():
    results = [result(1, clusters=4), result(2, clusters=4), result(3, clusters=1)]

    assert cluster_histogram(results) == {4: 2, 1: 1}


def test_seed_list_counts_up_from_the_starting_seed():
    assert seed_list(count=3, start=100) == [100, 101, 102]


def test_run_sweep_trains_once_per_seed_in_order():
    trained = []

    def train_and_score(seed):
        trained.append(seed)
        return result(seed)

    results = run_sweep([7, 1, 42], train_and_score)

    assert trained == [7, 1, 42]
    assert [r.seed for r in results] == [7, 1, 42]


def test_report_shows_every_seed_the_spread_and_the_cluster_counts():
    results = [
        result(100, mapping_accuracy=0.6, clusters=4),
        result(101, mapping_accuracy=0.8, clusters=1),
    ]

    report = format_sweep_report(results)

    assert "100" in report
    assert "101" in report
    assert "mean 0.700" in report
    assert "sd 0.141" in report
    assert "4 clusters in 1/2 runs" in report
    assert "1 cluster in 1/2 runs" in report


def test_report_marks_collapsed_runs_so_a_good_mean_cannot_hide_them():
    results = [result(100, clusters=4), result(101, clusters=1)]

    assert "collapsed: 1 of 2 runs" in format_sweep_report(results)


def test_sweep_csv_has_a_row_per_seed(tmp_path):
    path = tmp_path / "sweep.csv"

    write_sweep_csv([result(100), result(101)], path)

    written = pd.read_csv(path)
    assert list(written.seed) == [100, 101]
    assert list(written.columns) == [
        "seed",
        "final_loss",
        "clusters_used",
        "purity",
        "mapping_accuracy",
    ]


def test_train_command_passes_the_seed_and_run_directory_to_train_py():
    command = train_command("python", "data/", "experiments/base/seed-7", 7, [])

    assert "--seed" in command
    assert command[command.index("--seed") + 1] == "7"
    assert command[command.index("--save-path") + 1] == "experiments/base/seed-7"
    assert command[-1] == "data/"


def test_train_command_puts_overrides_last_so_they_win():
    command = train_command("python", "data/", "run", 7, ["--col-tau", "0.05"])

    assert command.index("--col-tau") > command.index("--data-size")


def test_prepare_run_dir_refuses_an_existing_run_rather_than_resuming_it(tmp_path):
    existing = tmp_path / "seed-7"
    existing.mkdir()
    (existing / "model_last.pth.tar").write_bytes(b"")

    with pytest.raises(FileExistsError):
        prepare_run_dir(tmp_path, seed=7, overwrite=False)


def test_prepare_run_dir_clears_an_existing_run_when_told_to_overwrite(tmp_path):
    existing = tmp_path / "seed-7"
    existing.mkdir()
    (existing / "model_last.pth.tar").write_bytes(b"")

    prepared = prepare_run_dir(tmp_path, seed=7, overwrite=True)

    assert prepared.exists()
    assert not (prepared / "model_last.pth.tar").exists()


def test_args_after_the_separator_are_handed_to_train_py():
    args, _ = parse_args(["data/", "--name", "x", "--", "--col-tau", "0.05"])

    assert args.train_args == ["--col-tau", "0.05"]


def test_explicit_seeds_replace_the_counted_ones():
    _, seeds = parse_args(["data/", "--name", "x", "--seeds", "5", "--seed-list", "1", "7"])

    assert seeds == [1, 7]
