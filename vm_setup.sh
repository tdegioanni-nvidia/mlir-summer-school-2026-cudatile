#!/bin/bash

set -euo pipefail

REPOSITORY_URL="${CUDA_TILE_REPOSITORY_URL:-https://github.com/tdegioanni-nvidia/mlir-summer-school-2026-cudatile.git}"
PROJECT_DIR="${1:-${CUDA_TILE_PROJECT_DIR:-/cuda-tile-project}}"

if (( $# > 1 )); then
    echo "Usage: $0 [project-directory]" >&2
    exit 2
fi

if ! command -v apt-get >/dev/null 2>&1; then
    echo "This setup script requires an Ubuntu or Debian system with apt-get." >&2
    exit 1
fi

if (( EUID != 0 )) && ! command -v sudo >/dev/null 2>&1; then
    echo "Run this script as root or install sudo so it can install system packages." >&2
    exit 1
fi

run_as_root() {
    if (( EUID == 0 )); then
        "$@"
    else
        sudo "$@"
    fi
}

# Ensure NVIDIA's Ubuntu 22.04 repository and matching kernel headers exist.
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
ca-certificates \
wget \
"linux-headers-$(uname -r)"

wget -q \
https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
-O /tmp/cuda-keyring_1.1-1_all.deb

sudo dpkg -i /tmp/cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Brev holds its NVIDIA packages, preventing driver upgrades.
mapfile -t held_packages < <(
apt-mark showhold |
    grep -E '^(cuda|libcuda|libnvidia|nvidia)' || true
)

if ((${#held_packages[@]})); then
sudo apt-mark unhold "${held_packages[@]}"
fi

# CUDA 13.3 requires driver 610+. Keep the driver on the 610 branch.
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
nvidia-driver-pinning-610

# Explicitly remove Brev's old versioned 565 driver packages. This is needed
# because packages such as libnvidia-fbc1-565 conflict with the new names.
remove_args=()

while IFS= read -r package; do
[[ -n "$package" ]] && remove_args+=("${package}-")
done < <(
dpkg --get-selections |
    awk '
    $2 == "install" &&
    $1 ~ /^(cuda-drivers|libnvidia|nvidia|xserver-xorg-video-nvidia).*565/ {
        print $1
    }
    '
)

sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
"${remove_args[@]}" \
nvidia-open \
cuda-toolkit-13-3

# A reboot is required to replace the loaded 565 kernel module with 610.
echo "CUDA 13.3 installed. Reboot required."

echo "Installing Git, Git LFS, and the system Python tools..."
run_as_root apt-get update
run_as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    git-lfs \
    python3 \
    python3-pip \
    python3-venv \
    python-is-python3

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132
pip3 install pytest cuda-python==13.3.1

# Configure the LFS clean/smudge filters for every user before cloning.
run_as_root git lfs install --system --skip-repo

if [[ -e "$PROJECT_DIR" ]]; then
    if [[ ! -d "$PROJECT_DIR" ]]; then
        echo "The destination exists and is not a directory: $PROJECT_DIR" >&2
        exit 1
    fi
    if [[ -n "$(find "$PROJECT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        echo "The destination directory is not empty: $PROJECT_DIR" >&2
        exit 1
    fi
else
    run_as_root install -d -m 0755 -o "$(id -u)" -g "$(id -g)" "$PROJECT_DIR"
fi

echo "Cloning $REPOSITORY_URL into $PROJECT_DIR..."
GIT_LFS_SKIP_SMUDGE=1 git clone --branch main --single-branch "$REPOSITORY_URL" "$PROJECT_DIR"

echo "Downloading Git LFS objects..."
git -C "$PROJECT_DIR" lfs pull

if [[ ! -s "$PROJECT_DIR/cuda-tile-translate" ]]; then
    echo "Git LFS did not produce cuda-tile-translate." >&2
    exit 1
fi

chmod -R a+rX "$PROJECT_DIR"
chmod a+rx "$PROJECT_DIR/cuda-tile-translate"

sudo chown -R root:root "$PROJECT_DIR"

echo "CUDA Tile project setup complete: $PROJECT_DIR"

sudo reboot
