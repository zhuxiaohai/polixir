#!/bin/bash

# Use the baseline folder as context root
BASELINE_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Remove the possible 'cp -i' alias
unalias cp 2>/dev/null

# Install baseline folder to PYTHONPATH with base dependencies
pushd $BASELINE_ROOT
pip install -e .
popd

# Enter data directory
pushd $BASELINE_ROOT/data
# Download public data in development phase
wget -q https://codalab.lisn.upsaclay.fr/my/datasets/download/eea9f5b7-3933-47cf-ba6f-394218eeb913 -O public_data_dev.zip
unzip -o public_data_dev.zip

# Preprocess offline data
python $BASELINE_ROOT/data_preprocess.py offline_592_1000.csv
popd

pushd $BASELINE_ROOT
# Download Revive SDK
git clone https://agit.ai/Polixir/revive.git
popd

# Install Revive SDK and its dependencies
pushd $BASELINE_ROOT/revive
git fetch
git checkout 0.6.0 # Use version 0.6.0
pip install -e .
export PYARMOR_LICENSE=$BASELINE_ROOT/data/license.lic
popd

pushd $BASELINE_ROOT/data
# Create config.json if not exist
if ! [ -r config.json ]; then
    cp -f $BASELINE_ROOT/revive/data/config.json config.json
fi

# Start learning virtual environment (use ctrl+z and bg to bring it to background)
python $BASELINE_ROOT/revive/train.py -rcf config.json --data_file venv.npz --config_file venv.yaml --run_id venv_baseline --venv_metric wdist --venv_mode tune --policy_mode None

# Acquire learned virtual environment
cp -f $BASELINE_ROOT/revive/logs/venv_baseline/env.pkl venv.pkl
popd

pushd $BASELINE_ROOT/data
# Start learning policy validation based on virtual environment (use ctrl+z and bg to bring it to background)
mkdir -p logs
mkdir -p model_checkpoints
python $BASELINE_ROOT/train_policy.py
popd

# Acquire learned policy validation
pushd $BASELINE_ROOT/data/model_checkpoints
cp -f $(ls -Art . | tail -n 1) $BASELINE_ROOT/data/rl_model.zip
popd

# Update sample submission and create bundle
pushd $BASELINE_ROOT/../sample_submission
cp -f $BASELINE_ROOT/data/evaluation_start_states.npy ./data/evaluation_start_states.npy
cp -f $BASELINE_ROOT/data/rl_model.zip ./data/rl_model.zip
cp -f $BASELINE_ROOT/user_states.py ./user_states.py
zip -o -r --exclude='*.git*' --exclude='*__pycache__*' --exclude='*.DS_Store*' --exclude='*public_data*' ../sample_submission .;
popd
