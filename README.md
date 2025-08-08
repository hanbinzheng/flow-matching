# Flow Matching

> Inspired by [Ezra Erives](https://github.com/eje24/iap-diffusion-labs) and [Jiarui Hai](https://github.com/haidog-yaqub/MeanFlow)

This repository provides the codebase and environment setup for the Flow Matching experiment, designed to deliver a clean, minimal, and extensible framework.

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/hanbinzheng/flow-matching.git
cd flow-matching
```

---

### 2. Set up the environment

```bash
chmod +x setup_env.sh
./setup_env.sh
```

---

### 3. Activate the virtual environment and run training

```bash
source .venv/bin/activate
python train.py
```

---

### Notes

- The `setup_env.sh` script automatically creates a python virtual environment, upgrades pip, installs required packages (except PyTorch), detects your system’s CUDA version, and installs the matching PyTorch build accordingly.

- The .venv directory is excluded from version control to keep the repository clean.

- Multi-GPU setups are not supported by the automated script and require manual configuration.

- Ensure that NVIDIA drivers and CUDA are properly installed and that the nvidia-smi command is available.

- For troubleshooting installation errors, refer to the packages_install_log.txt generated during setup.

---

This setup guarantees a reproducible and efficient environment for your Flow Matching experiments.

---