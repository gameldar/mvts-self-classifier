"""Train the same configuration over several seeds and report the spread of the metrics.

One run says very little here. With everything but the seed held fixed, mapping accuracy
ranges from total collapse to 0.88 on the same data, and a repeated seed does not even
reproduce itself on CPU. So a single number cannot tell you whether a change helped; a
distribution over seeds can.

The statistics and the report are pure functions over `SeedResult`s. Training a seed and
scoring its checkpoint is injected into `run_sweep`, so the aggregation is testable
without torch or subprocesses.
"""

import argparse
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from cluster_eval import (
    clusters_used,
    confusion_matrix,
    load_training_config,
    mapping_accuracy,
    purity,
)

# Matches the Makefile's default ARGS, so a bare sweep reproduces `make run` per seed.
DEFAULT_TRAIN_ARGS = [
    "--dim", "115",
    "--hidden-dim", "115",
    "--cls-size", "4",
    "--epochs", "30",
    "--batch-size", "32",
    "--warmup-epochs", "0",
    "--lr", "0.001",
    "--data-size", "450",
]  # fmt: skip


class SeedResult(NamedTuple):
    """What one training run of the sweep produced."""

    seed: int
    final_loss: float
    clusters_used: int
    purity: float
    mapping_accuracy: float


class Stats(NamedTuple):
    """Spread of one metric across the sweep. `sd` is None when there is only one run."""

    mean: float
    sd: float | None
    minimum: float
    maximum: float


def summarise(values):
    values = list(values)
    if not values:
        raise ValueError("a sweep needs at least one run to summarise")
    return Stats(
        mean=statistics.fmean(values),
        sd=statistics.stdev(values) if len(values) > 1 else None,
        minimum=min(values),
        maximum=max(values),
    )


def cluster_histogram(results):
    """How many runs ended up using each number of clusters."""
    counts = {}
    for seed_result in results:
        counts[seed_result.clusters_used] = counts.get(seed_result.clusters_used, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: -item[1]))


def results_frame(results):
    return pd.DataFrame([seed_result._asdict() for seed_result in results])


def write_sweep_csv(results, path):
    results_frame(results).to_csv(path, index=False)


def format_stats(name, stats):
    spread = f"{stats.sd:.3f}" if stats.sd is not None else "n/a"
    return (
        f"{name:<17} mean {stats.mean:.3f}  sd {spread}  "
        f"min {stats.minimum:.3f}  max {stats.maximum:.3f}"
    )


def format_sweep_report(results):
    runs = len(results)
    histogram = cluster_histogram(results)
    usage = ", ".join(
        f"{count} cluster{'s' if count != 1 else ''} in {runs_with}/{runs} runs"
        for count, runs_with in histogram.items()
    )
    collapsed = sum(1 for seed_result in results if seed_result.clusters_used == 1)
    return "\n".join(
        [
            results_frame(results).to_string(
                index=False, float_format=lambda value: f"{value:.3f}"
            ),
            "",
            format_stats("mapping accuracy", summarise([r.mapping_accuracy for r in results])),
            format_stats("purity", summarise([r.purity for r in results])),
            format_stats("final loss", summarise([r.final_loss for r in results])),
            "",
            f"clusters used: {usage}",
            f"collapsed: {collapsed} of {runs} runs",
            "",
            "sd is the spread over seeds, not a significance test.",
        ]
    )


def run_sweep(seeds, train_and_score):
    return [train_and_score(seed) for seed in seeds]


def seed_list(count, start):
    return list(range(start, start + count))


def train_command(python_executable, data, run_dir, seed, extra_args):
    """argv for one training run. Overrides go last so they beat the defaults."""
    summary_file = Path(run_dir).parent / "train-summary.csv"
    return [
        python_executable,
        "train.py",
        *DEFAULT_TRAIN_ARGS,
        "--seed",
        str(seed),
        "--save-path",
        str(run_dir),
        "--summary-file",
        str(summary_file),
        *extra_args,
        str(data),
    ]


def prepare_run_dir(parent, seed, overwrite):
    """Empty directory for one seed's run.

    train.py resumes from `model_last.pth.tar` whenever it finds one, so reusing a
    directory would silently train nothing and score the previous run again.
    """
    run_dir = Path(parent) / f"seed-{seed}"
    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{run_dir} already exists; train.py would resume from its checkpoint and "
                "train no further. Pass --overwrite, or sweep under a different --name."
            )
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    return run_dir


def final_loss(run_dir):
    losses = pd.read_csv(Path(run_dir) / "epoch_loss.csv")
    return float(losses["loss"].iloc[-1])


def score_run(run_dir, data, checkpoint):
    """Cluster metrics for one finished run, scored the way cluster_eval.py scores."""
    from cluster_eval import assign_clusters

    config = load_training_config(run_dir)
    windows = pd.read_parquet(os.path.join(data, "train.parquet")).drop(columns=["Timestamp"])
    behaviours = pd.read_parquet(os.path.join(data, "train_labels.parquet"))["behaviour"].values

    clusters, num_clusters, _ = assign_clusters(os.path.join(run_dir, checkpoint), config, windows)
    matrix = confusion_matrix(behaviours, clusters, num_clusters)
    return clusters_used(matrix), purity(matrix), mapping_accuracy(matrix)


def make_trainer(data, sweep_dir, checkpoint, extra_args, overwrite):
    def train_and_score(seed):
        run_dir = prepare_run_dir(sweep_dir, seed, overwrite)
        print(f"==> seed {seed}: training into {run_dir}", flush=True)
        command = train_command(sys.executable, data, run_dir, seed, extra_args)
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)

        used, cluster_purity, accuracy = score_run(run_dir, data, checkpoint)
        print(
            f"    clusters {used}  purity {cluster_purity:.3f}  mapping accuracy {accuracy:.3f}",
            flush=True,
        )
        return SeedResult(
            seed=seed,
            final_loss=final_loss(run_dir),
            clusters_used=used,
            purity=cluster_purity,
            mapping_accuracy=accuracy,
        )

    return train_and_score


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Arguments after -- are passed to train.py, overriding the sweep defaults, "
        "e.g. sweep.py data/ --name coltau-0.05 -- --col-tau 0.05",
    )
    parser.add_argument("data", help="directory holding train.parquet and train_labels.parquet")
    parser.add_argument("--name", required=True, help="sweep name; runs go in <experiments>/<name>")
    parser.add_argument("--seeds", type=int, default=5, help="number of seeds to train")
    parser.add_argument("--seed-start", type=int, default=100, help="first seed")
    parser.add_argument(
        "--seed-list", type=int, nargs="+", help="explicit seeds, instead of --seeds/--seed-start"
    )
    parser.add_argument("--experiments", default="experiments", help="root for sweep directories")
    parser.add_argument(
        "--checkpoint",
        default="model_last.pth.tar",
        help="checkpoint to score; model_best.pth.tar is the lowest-loss epoch, which is "
        "not the best-clustering one",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="delete existing run directories first"
    )
    parser.add_argument("train_args", nargs="*", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    seeds = args.seed_list if args.seed_list else seed_list(args.seeds, args.seed_start)
    if not seeds:
        parser.error("a sweep needs at least one seed")
    return args, seeds


def main():
    args, seeds = parse_args()

    sweep_dir = Path(args.experiments) / args.name
    sweep_dir.mkdir(parents=True, exist_ok=True)
    print(f"Sweeping {len(seeds)} seeds {seeds} into {sweep_dir}\n")

    results = run_sweep(
        seeds, make_trainer(args.data, sweep_dir, args.checkpoint, args.train_args, args.overwrite)
    )

    print(f"\n{format_sweep_report(results)}")
    csv_path = sweep_dir / "sweep.csv"
    write_sweep_csv(results, csv_path)
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
