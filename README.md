# Flow Matching

> Inspired by [Ezra Erives](https://github.com/eje24/iap-diffusion-labs) and [Jiarui Hai](https://github.com/haidog-yaqub/MeanFlow)

This repository provides the codebase and environment setup for the Flow Matching experiment, designed to deliver a clean, minimal, and extensible framework.

## Quick Start

### 1. Clone the repository

```bash
# clone the repository
git clone https://github.com/hanbinzheng/flow-matching.git
cd flow-matching
```

---

### 2. Set up the environment

If you are using `WSL`, install the appropriate python3-venv package before proceeding. To detect and install the correct package version, run:

```bash
# install python3-venv package for wsl users
ver=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
sudo apt install -y python${ver}-venv
```

Then, execute the environment setup script:

```bash
# set up the environment
chmod +x setup_env.sh
./setup_env.sh
```

This process may take several minutes depending on your internet connection. Please be patient.

---

### 3. Activate the virtual environment and run training

```bash
# run the training code
source .venv/bin/activate
python train.py
```

Training duration depends on your hardware configuration, but typically completes within an hour.

Training outputs will be saved under the `flow-matching/results/` directory.

---

### Notes

- The `setup_env.sh` script automates the creation of a Python virtual environment, upgrades pip, installs required packages (excluding PyTorch), detects your system’s CUDA version, and installs the appropriate PyTorch build accordingly.

- This setup supports WSL, macOS, Ubuntu, and the vast majority of Unix-like operating systems.

- `CPU-only training is fully supported`, with automatic adaptation for environments lacking compatible GPUs.

- The .venv directory is excluded from version control to maintain a clean repository.

- (`Only for GPU Training`) Multi-GPU configurations are not supported by the automated setup and require manual intervention.

- (`Only for GPU Training`) Ensure NVIDIA drivers and CUDA are properly installed, and that the nvidia-smi command is available for CUDA-based GPU detection.

- For troubleshooting installation issues, consult the `packages_install_log.txt` file generated during setup (only for those who failed the `2: Set up the environment`).


---

This setup guarantees a reproducible and efficient environment for your Flow Matching experiments.

---
