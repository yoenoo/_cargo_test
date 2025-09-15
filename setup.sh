set -e

pip install uv
uv venv --python=3.10
source .venv/bin/activate

uv pip install torch datasets transformers trl peft accelerate
uv pip install matplotlib


# instasll rust toolchain
apt-get update -y
apt-get install -y curl build-essential pkg-config libssl-dev cmake
curl https://sh.rustup.rs -sSf | sh -s -- -y
source "$HOME/.cargo/env"
rustup component add clippy rustfmt

# persist cargo in PATH for future shells
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> "$HOME/.bashrc"
echo 'source "$HOME/.cargo/env"' >> "$HOME/.bashrc"
