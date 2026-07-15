# Container engine, override to use podman: `make build ENGINE=podman`.
ENGINE ?= docker
IMAGE ?= mvts

# Host directories mounted into the container so checkpoints and logs persist.
DATA ?= data
EXPERIMENTS ?= experiments

# GPU passthrough flag. Docker and recent podman accept `--gpus all`; older podman
# uses CDI instead: `make run-gpu ENGINE=podman GPU_FLAG='--device nvidia.com/gpu=all'`.
GPU_FLAG ?= --gpus all

# train.py arguments. Defaults to a short run matched to `make gen` output so
# `make run` works out of the box; override with ARGS='...'.
ARGS ?= --dim 115 --hidden-dim 115 --cls-size 4 --epochs 30 --batch-size 32 \
        --warmup-epochs 0 --lr 0.001 --data-size 450 --save-path experiments/run \
        --summary-file experiments/results-summary.csv data/

# generate_sample_data.py arguments, e.g. `make gen GEN_ARGS='--rows 1000'`.
GEN_ARGS ?=

VOLUMES = -v $(CURDIR)/$(DATA):/app/data -v $(CURDIR)/$(EXPERIMENTS):/app/experiments

.PHONY: build build-gpu gen run run-gpu shell clean

build:
	$(ENGINE) build -t $(IMAGE):cpu .

gen: build
	$(ENGINE) run --rm $(VOLUMES) --entrypoint python $(IMAGE):cpu generate_sample_data.py $(GEN_ARGS)

build-gpu:
	$(ENGINE) build --build-arg TORCH_EXTRA=cu124 -t $(IMAGE):gpu .

run: build
	$(ENGINE) run --rm $(VOLUMES) $(IMAGE):cpu $(ARGS)

run-gpu: build-gpu
	$(ENGINE) run --rm $(GPU_FLAG) $(VOLUMES) $(IMAGE):gpu $(ARGS)

shell: build
	$(ENGINE) run --rm -it $(VOLUMES) --entrypoint /bin/bash $(IMAGE):cpu

clean:
	-$(ENGINE) rmi $(IMAGE):cpu $(IMAGE):gpu
