#!/usr/bin/env bash
set -e

NUM_SAMPLES_TRAIN=10000
NUM_SAMPLES_TEST=1000
ELECTRODES=16
PATTERN=adjacent
SEED=42

# training dataset
uv run python scripts/data/generate_dataset.py --samples $NUM_SAMPLES_TRAIN --electrodes $ELECTRODES --pattern $PATTERN --seed $SEED

# testing id dataset
uv run python scripts/data/generate_test_dataset.py --samples $NUM_SAMPLES_TEST --electrodes $ELECTRODES --pattern $PATTERN --dbar --mode id --seed $SEED

# testing ood dataset
uv run python scripts/data/generate_test_dataset.py --samples $NUM_SAMPLES_TEST --electrodes $ELECTRODES --pattern $PATTERN --dbar --mode ood --seed $SEED