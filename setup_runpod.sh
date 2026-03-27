#!/bin/bash
# Parameter Golf — RunPod Environment Setup
# Run once after creating the pod. Installs to /workspace so it survives spot interruptions.
#
# Usage:
#   bash setup_runpod.sh          # first time (clone + data + env)
#   bash setup_runpod.sh --quick  # after spot restart (just activate env)

set -e

VENV_DIR="/workspace/venv"
REPO_DIR="/workspace/parameter-golf"

# ---------- Quick mode: just reactivate ----------
if [[ "$1" == "--quick" ]]; then
    source "$VENV_DIR/bin/activate"
    echo "Activated venv. GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo 'not found')"
    echo "Run:  cd $REPO_DIR"
    # Print this so the user can source it
    echo ""
    echo "To activate in your current shell, run:"
    echo "  source $VENV_DIR/bin/activate && cd $REPO_DIR"
    exit 0
fi

# ---------- Full setup ----------
echo "=== Setting up Parameter Golf environment ==="

# 1. Clone repo if not present
if [ ! -d "$REPO_DIR" ]; then
    echo "Cloning repository..."
    git clone https://github.com/openai/parameter-golf.git "$REPO_DIR"
else
    echo "Repo already exists at $REPO_DIR, pulling latest..."
    cd "$REPO_DIR" && git pull && cd /workspace
fi

# 2. Create persistent venv (uses system PyTorch + CUDA)
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR" --system-site-packages
else
    echo "Venv already exists at $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# 3. Install dependencies (PyTorch comes from system via --system-site-packages)
echo "Installing Python dependencies..."
pip install --quiet --upgrade pip
pip install --quiet \
    sentencepiece \
    huggingface-hub \
    tqdm \
    datasets \
    tiktoken \
    numpy \
    setuptools \
    "typing-extensions==4.15.0"

# 4. Verify GPU access
echo ""
echo "=== Verification ==="
python3 -c "
import torch
print(f'PyTorch:  {torch.__version__}')
print(f'CUDA:     {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:      {torch.cuda.get_device_name(0)}')
    print(f'VRAM:     {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
import sentencepiece; print('sentencepiece: OK')
import tiktoken; print('tiktoken: OK')
"

# 5. Download dataset (1 shard for smoke testing)
cd "$REPO_DIR"
if [ ! -d "data/datasets/fineweb10B_sp1024" ]; then
    echo ""
    echo "=== Downloading dataset (1 shard) ==="
    python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
else
    echo "Dataset already downloaded."
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "To start training:"
echo "  source /workspace/venv/bin/activate"
echo "  cd /workspace/parameter-golf"
echo "  RUN_ID=test DATA_PATH=./data/datasets/fineweb10B_sp1024/ TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model VOCAB_SIZE=1024 torchrun --standalone --nproc_per_node=1 train_gpt.py"
echo ""
echo "After a spot restart, just run:"
echo "  source /workspace/venv/bin/activate && cd /workspace/parameter-golf"
