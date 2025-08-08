#!/bin/bash
set -e

LOG_FILE="packages_install_log.txt"
touch $LOG_FILE

function install_package() {
    local pkg=$1
    local extra_args=$2
    echo "pip install $pkg $extra_args ......"
    if pip install $pkg $extra_args >> $LOG_FILE 2>&1; then
        echo "pip install $pkg $extra_args done"
    else
        echo "error: pip install $pkg $extra_args failed, see $LOG_FILE for details"
        deactivate
        exit 1
    fi
}

function install_package() {
    local pkg=$1
    local extra_args=$2
    local timeout_seconds=18000 # 设置超时时间为 1800 秒（30 分钟）
    local log_prefix="pip install $pkg $extra_args"
    echo "$log_prefix ......"

    (pip install $pkg $extra_args >> $LOG_FILE 2>&1) &
    local pip_pid=$! # 获取 pip 进程的 PID

    local elapsed_seconds=0
    while kill -0 "$pip_pid" 2>/dev/null; do
        if (( elapsed_seconds >= timeout_seconds )); then
            echo -e "\nError: $log_prefix timed out after $timeout_seconds seconds. Terminating process."
            kill "$pip_pid" 2>/dev/null || true # 强制终止 pip 进程
            deactivate
            exit 1
        fi

        echo -n "#" # 每隔一段时间打印一个 #
        sleep 1
        ((elapsed_seconds+=10))
    done

    wait "$pip_pid"
    local exit_status=$?
    
    if [[ $exit_status -eq 0 ]]; then
        echo -e "\n$log_prefix done"
    else
        echo -e "\nError: $log_prefix failed, see $LOG_FILE for details"
        deactivate
        exit 1
    fi
}

# create virtual environment
python3 -m venv .venv >> $LOG_FILE 2>&1
echo -e "\ncreate python virtual environment .venv"
source .venv/bin/activate

# upgrade pip
install_package "pip" "--upgrade"

# install pytorch according to cuda version
CUDA_RUNTIME_VER=$(nvidia-smi | grep "CUDA Version" | sed 's/.*CUDA Version: \([0-9\.]*\).*/\1/')
CUDA_VER_DIGIT=${CUDA_RUNTIME_VER//./}  # eg. 12.4 -> 124

if [[ -n "$CUDA_RUNTIME_VER" ]]; then
    echo -e "\nDetected CUDA runtime version: $CUDA_RUNTIME_VER"
    install_package "torch torchvision" "--index-url https://download.pytorch.org/whl/cu${CUDA_VER_DIGIT}"
else
    echo -e "\nCUDA not detected or nvidia-smi not available."
    install_package "torch torchvision" ""
fi

# install required packages except pytorch
grep -v '^torch' requirements.txt > req_no_torch.txt
echo -e "\nInstalling packages from requirements.txt except pytorch...\n"
while IFS= read -r pkg || [[ -n "$pkg" ]]; do
    install_package "$pkg"
done < req_no_torch.txt
rm req_no_torch.txt

echo -e "\nAll packages done."
rm -f $LOG_FILE

deactivate
