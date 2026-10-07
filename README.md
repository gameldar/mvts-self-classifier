# Self-Supervised Multivariate Time Series Classification Network

This is a work-in-progress conversion of the Self-Supervised Classification Network as detailed in the paper: [Self-Supervised Classification Network](https://arxiv.org/abs/2103.10994)

The source code is a modified version of the PyTorch implementation, taken from: https://github.com/elad-amrani/self-classifier
This is largely been simplified, to remove the distributed running support nature, and converted to work with accelerometer data.
The internals are mostly the same but the augmentation was changed and the backbone and networks need to change.

The conversion was originally written as a literate walkthrough; to follow along, read [accelerometer.md](accelerometer.md).

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/). Install uv, then sync
the environment, selecting the torch build for your hardware:

```shell
uv sync --extra cpu     # CPU-only
uv sync --extra cu124   # NVIDIA GPU (CUDA 12.4)
```

The `cpu` and `cu124` extras are mutually exclusive; exactly one must be chosen. Run the
classifier through uv so it uses the synced environment (see the example below).

The legacy `environment.yml` is kept for reference only; the uv lockfile is the source of
truth for dependencies.

### apex

The optional `--lars` optimizer relies on NVIDIA [apex](https://github.com/NVIDIA/apex),
which must be compiled against your CUDA toolchain and is not installed by uv. All other
functionality, including mixed precision, uses native PyTorch.

## Docker

The image runs anywhere by default. The CUDA torch wheels bundle their own CUDA userspace,
so a GPU run only needs `--gpus all` at runtime; no CUDA base image is required.

```shell
# CPU
docker build -t mvts:cpu .
docker run --rm -v "$PWD/data:/app/data" -v "$PWD/experiments:/app/experiments" \
    mvts:cpu --dim 115 --hidden-dim 115 --cls-size 3 --epochs 30 --batch-size 100 \
    --data-size 9 --save-path experiments/run1 data/

# NVIDIA GPU
docker build --build-arg TORCH_EXTRA=cu124 -t mvts:gpu .
docker run --rm --gpus all -v "$PWD/data:/app/data" -v "$PWD/experiments:/app/experiments" \
    mvts:gpu --dim 115 --hidden-dim 115 --cls-size 3 --epochs 30 --batch-size 100 \
    --data-size 9 --save-path experiments/run1 data/
```

Or use Docker Compose, which wires up the volume mounts and GPU reservation:

```shell
docker compose run --rm cpu --dim 115 --hidden-dim 115 --cls-size 3 --data-size 9 data/
docker compose run --rm gpu --dim 115 --hidden-dim 115 --cls-size 3 --data-size 9 data/
```

### Makefile

The `Makefile` wraps the build and run commands. The container engine is configurable via
`ENGINE` (default `docker`), so podman works too. With no arguments, `make run` uses a
short training run matched to `make gen` output, so `make gen && make run` works out of the
box. Override with `ARGS`:

```shell
make gen && make run                    # generate sample data, then train with defaults
make run ARGS='--dim 115 --hidden-dim 115 --cls-size 3 --data-size 450 data/'
make run-gpu ARGS='--dim 115 --hidden-dim 115 --cls-size 3 --data-size 450 data/'

# podman instead of docker
make run ENGINE=podman
```

The default run writes checkpoints and logs to `experiments/run/` and appends a summary
line to `experiments/results-summary.csv`, both on the mounted `experiments/` volume so
they survive the container. `train.py`'s own `--summary-file` default (`../results-summary.csv`)
lands at the container root and is lost, so always point it inside a mounted directory.

Generate sample data into `data/` through the container with `make gen` (pass generator
options via `GEN_ARGS`, e.g. `make gen GEN_ARGS='--rows 1000'`).

Older podman needs CDI for GPUs: `make run-gpu ENGINE=podman GPU_FLAG='--device nvidia.com/gpu=all'`.
Other targets: `build`, `build-gpu`, `gen`, `eval`, `shell`, `clean`.

The classifier runs on GPU when one is available and otherwise falls back to CPU with a
warning. `--use-amp` (mixed precision) requires a GPU.

## Evaluation

The data is loaded from a Pandas dataframe, stored in parquet format in a file named `train.parquet`.
The dataframe consists of a column `Timestamp` followed but the individual points for the MTVS data. e.g.:

| Timestamp           |    x_0 |    x_1 |   x_2 |    y_0 |    y_1 |    y_2 |   z_0 |   z_1 |   z_2 |
|---------------------|--------|--------|-------|--------|--------|--------|-------|-------|-------|
| 2016-06-26 09:40:25 |  0.168 |  0.172 | 0.176 | -0.859 | -0.852 | -0.867 | 0.477 | 0.488 | 0.496 |
| 2016-06-26 09:41:25 | -0.230 | -0.062 | 0.215 | -0.949 | -0.941 | -0.836 | 0.434 | 0.441 | 0.414 |

To generate a synthetic `data/train.parquet` for testing (tri-axial accelerometer windows
drawn from a few behaviour prototypes so the data has structure), run:

```shell
uv run --extra cpu python generate_sample_data.py
```

This writes 500 rows of 450 data columns (150 samples per axis), so pass `--data-size 450`
to `train.py`. Use `--rows`, `--window` and `--output` to change the shape.

It also writes `data/train_labels.parquet`, a row-aligned sidecar naming the behaviour each
window was drawn from. The windows file stays label-free so `train.py` still trains
unsupervised; the sidecar exists only so the clusters can be scored afterwards.

The evaluation of the model (this uses a CNN-LSTM based upon [Deep learning based classification of sheep behaviour from accelerometer data with imbalance](https://linkinghub.elsevier.com/retrieve/pii/S2214317322000415)) can then be performed with the following:

```shell
uv run --extra cpu python train.py --dim 115 --hidden-dim 115 --cls-size 3 --epochs 30 --batch-size 100 --save-path experiments/mtvs-test --warmup-epochs 0 --lr 0.001 --data-size 9 --summary-file epoch_results.csv data/
```

where:

- `cls-size` is the number of classes to identify in the data
- `data-size` is the number of columns in the data (without the `Timestamp` column)
- `data/` is the path where the `train.parquet` is located.

More details for the parameters can be found by running without arguments.

### Scoring the clusters

Training only reports a loss, which falls even when the model collapses every window into
one cluster. `cluster_eval.py` loads a checkpoint, assigns each window to a cluster and
scores those clusters against the generated behaviours:

```shell
uv run --extra cpu python cluster_eval.py data/ --save-path experiments/run \
    --checkpoint model_last.pth.tar
```

Hyperparameters are read from the run's `configuration.json`, so the model is always
rebuilt to match the checkpoint. Through the container: `make eval`.

Cluster ids are arbitrary - nothing makes cluster 2 "walking" - so both metrics are
invariant to how the clusters are numbered:

- **cluster purity**: the fraction of windows sharing their cluster's dominant behaviour.
  Read it with the matrix, since one-window-per-cluster would also score 1.0.
- **mapping accuracy**: each behaviour is paired with one distinct cluster, choosing the
  pairing that maximises agreement, and the explained fraction is reported. Because the
  pairing is one-to-one, collapsing everything into a single cluster is penalised.

A fully collapsed model scores the share of the largest behaviour (0.29 here); four clean
clusters would score 1.0.

## Tests

```shell
uv run --extra cpu --group dev python -m pytest
```

These cover the generator's label sidecar and the cluster metrics. The model code itself is
unchanged from upstream and remains untested.

## License

See the [LICENSE](LICENSE) file for more details.
