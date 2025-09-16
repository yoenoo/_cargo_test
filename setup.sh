set -e

pip install uv
uv venv --python=3.10
source .venv/bin/activate

uv pip install torch datasets transformers trl peft accelerate
uv pip install vllm
uv pip install wandb
uv pip install matplotlib

apt-get update && apt-get install -y python3.10-dev build-essential

# install rust cargo
curl https://sh.rustup.rs -sSf | sh
. "$HOME/.cargo/env"





git config --global user.name "Yeonwoo Jang"
git config --global user.email "yjang385@gmail.com"
git config pull.rebase false