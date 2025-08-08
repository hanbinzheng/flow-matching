#!/bin/bash
set -e

LOG_FILE="packages_install_log.txt"
touch $LOG_FILE

function install_package() {
    local pkg=$1
    local extra_args=$2
    echo "Installing $pkg..."
    if pip install $pkg $extra_args >> $LOG_FILE 2>&1; then
        echo "$pkg done"
    else
        echo "error: $pkg installation failed, see $LOG_FILE for details"
        deactivate
        exit 1
    fi
}

# create virtual environment
python3 -m venv .venv >> $LOG_FILE 2>&1
echo "create python virtual environment .venv"
source .venv/bin/activate

# upgrade pip
install_package "pip --upgrade"

# install required packages except pytorch
grep -v '^torch' requirements.txt > req_no_torch.txt
echo "Installing packages from requirements.txt except pytorch..."
while IFS= read -r pkg || [[ -n "$pkg" ]]; do
    install_package "$pkg"
done < req_no_torch.txt
rm req_no_torch.txt

# install pytorch according to cuda version
CUDA_RUNTIME_VER=$(nvidia-smi | grep "CUDA Version" | sed 's/.*CUDA Version: \([0-9\.]*\).*/\1/')
CUDA_VER_DIGIT=${CUDA_RUNTIME_VER//./}  # eg. 12.4 -> 124

if [[ -n "$CUDA_RUNTIME_VER" ]]; then
    echo "Detected CUDA runtime version: $CUDA_RUNTIME_VER"
    install_package "torch torchvision" "--index-url https://download.pytorch.org/whl/cu${CUDA_VER_DIGIT}"
else
    echo "CUDA not detected or nvidia-smi not available."
    install_package "torch torchvision" ""
fi

echo "All packages done."
rm -f $LOG_FILE

deactivate
