#!/bin/bash
set -euo pipefail

source /fsx/sroutray/miniconda3/etc/profile.d/conda.sh
conda activate lwm

echo "Using python     : $(which python)"
echo "LD_LIBRARY_PATH  : $LD_LIBRARY_PATH"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# Get the directory where this script is located (scripts/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Get the parent directory (vipra/)
VIPRA_DIR="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH=${PYTHONPATH:-}:$VIPRA_DIR

CUDA_VISIBLE_DEVICES=${1:-0}
PORT=${2:-8005}

export CUDA_VISIBLE_DEVICES
export PORT

python inference/dynamics_action_cont_server.py
