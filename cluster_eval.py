"""Score the self-supervised model's clusters against the behaviours that generated them.

train.py never sees a label, so its cluster ids are arbitrary: cluster 2 has no reason to
be "walking". These metrics are therefore label-permutation invariant.

- purity rewards clusters that each hold a single behaviour, but a model that put every
  window in its own cluster would score 1.0, so read it alongside the matrix.
- mapping accuracy pairs each behaviour with one distinct cluster, choosing the pairing
  that maximises agreement, then reports the fraction of windows explained. Collapse onto
  one cluster is penalised because the pairing must be one-to-one.

The functions here take arrays and return numbers; loading the checkpoint and running
inference lives in main().
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from scipy.optimize import linear_sum_assignment


def confusion_matrix(behaviours, clusters, num_clusters):
    """Counts of behaviour (rows) against assigned cluster (columns).

    Unused cluster columns are kept so collapse is visible as empty columns.
    """
    counted = pd.crosstab(
        pd.Series(behaviours, name="behaviour"),
        pd.Series(clusters, name="cluster"),
    )
    return counted.reindex(columns=range(num_clusters), fill_value=0)


def purity(matrix):
    total = matrix.values.sum()
    if total == 0:
        return 0.0
    return float(matrix.values.max(axis=0).sum() / total)


def clusters_used(matrix):
    """How many clusters hold at least one window.

    Reported next to the metrics because partial collapse is otherwise invisible in
    them: two behaviours sharing a cluster and two clean clusters can score alike.
    """
    return int((matrix.values.sum(axis=0) > 0).sum())


def mapping_accuracy(matrix):
    total = matrix.values.sum()
    if total == 0:
        return 0.0
    rows, columns = linear_sum_assignment(matrix.values, maximize=True)
    return float(matrix.values[rows, columns].sum() / total)


def best_mapping(matrix):
    """Behaviour -> cluster under the pairing mapping_accuracy scores."""
    rows, columns = linear_sum_assignment(matrix.values, maximize=True)
    return {matrix.index[row]: int(matrix.columns[column]) for row, column in zip(rows, columns)}


def format_report(matrix):
    mapping = best_mapping(matrix)
    lines = [
        "cluster assignment counts (rows = true behaviour, columns = cluster id)",
        "",
        matrix.to_string(),
        "",
        f"cluster purity:   {purity(matrix):.3f}",
        f"mapping accuracy: {mapping_accuracy(matrix):.3f}",
        f"clusters used:    {clusters_used(matrix)} of {len(matrix.columns)}",
        "",
        "best behaviour -> cluster pairing:",
    ]
    for behaviour, cluster in sorted(mapping.items()):
        matched = matrix.loc[behaviour, cluster]
        in_behaviour = matrix.loc[behaviour].sum()
        share = matched / in_behaviour if in_behaviour else 0.0
        lines.append(
            f"  {behaviour:<10} -> cluster {cluster}  ({matched}/{in_behaviour}, {share:.0%})"
        )
    return "\n".join(lines)


def load_training_config(save_path):
    """Read the hyperparameters train.py recorded, so the model is rebuilt to match."""
    with open(Path(save_path) / "configuration.json") as handle:
        return json.load(handle)


def assign_clusters(checkpoint_path, config, windows, batch_size=64):
    """Run the trained model over each window and return its arg-max cluster id."""
    import torch

    from lstm import LSTM, CNN
    from model import Model

    cls_size = config["cls_size"]
    model = Model(
        base_model=CNN(config["data_size"], config["hidden_dim"]),
        classifier=LSTM(config["hidden_dim"]),
        dim=config["dim"],
        cls_size=cls_size,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # MTSAugmentation round-trips to the blocked layout train.parquet uses, so the raw
    # windows are already in the layout the model was trained on.
    values = torch.from_numpy(windows.values.astype("float32"))
    assigned = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = values[start : start + batch_size]
            logits = model(batch.reshape(batch.shape[0], 1, batch.shape[-1]))
            # Model returns one logit tensor per classifier head; the first is cls_0.
            assigned.append(logits[0].argmax(dim=1))

    return torch.cat(assigned).numpy(), cls_size[0], checkpoint.get("epoch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", help="directory holding train.parquet and train_labels.parquet")
    parser.add_argument(
        "--save-path",
        default="experiments/run",
        help="run directory holding configuration.json and the checkpoints",
    )
    parser.add_argument(
        "--checkpoint",
        default="model_best.pth.tar",
        help="checkpoint file inside --save-path",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    config = load_training_config(args.save_path)

    windows = pd.read_parquet(os.path.join(args.data, "train.parquet")).drop(columns=["Timestamp"])
    labels_file = os.path.join(args.data, "train_labels.parquet")
    if not os.path.exists(labels_file):
        raise SystemExit(
            f"No ground truth at {labels_file}. Regenerate the data with "
            "generate_sample_data.py, which writes the labels sidecar."
        )
    behaviours = pd.read_parquet(labels_file)["behaviour"].values

    if len(behaviours) != len(windows):
        raise SystemExit(
            f"{labels_file} has {len(behaviours)} rows but train.parquet has {len(windows)}; "
            "they must come from the same generator run."
        )

    clusters, num_clusters, epoch = assign_clusters(
        os.path.join(args.save_path, args.checkpoint), config, windows, args.batch_size
    )

    print(f"Scoring {args.checkpoint} (epoch {epoch}) over {len(windows)} windows\n")
    print(format_report(confusion_matrix(behaviours, clusters, num_clusters)))


if __name__ == "__main__":
    main()
