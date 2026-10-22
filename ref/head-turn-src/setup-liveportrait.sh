#!/bin/zsh
set -e
LP="$1"
cd "$LP"
echo "== clone =="
[ -d LivePortrait ] || git clone --depth 1 https://github.com/KwaiVGI/LivePortrait.git
cd LivePortrait
echo "== venv (python 3.10) =="
[ -d .venv ] || uv venv --python 3.10 .venv
source .venv/bin/activate
echo "== deps =="
ls requirements*.txt
if [ -f requirements_macOS.txt ]; then uv pip install -r requirements_macOS.txt; else uv pip install -r requirements.txt; fi
uv pip install "huggingface_hub[cli]"
echo "== weights =="
hf download KwaiVGI/LivePortrait --local-dir pretrained_weights --exclude "*.git*" "README.md" "docs/*" 2>&1 | tail -3 || huggingface-cli download KwaiVGI/LivePortrait --local-dir pretrained_weights --exclude "*.git*" "README.md" "docs/*" 2>&1 | tail -3
echo "== check =="
du -sh pretrained_weights
python -c "import torch, sys; print('torch', torch.__version__, 'mps', torch.backends.mps.is_available(), sys.version)"
echo "SETUP DONE"
