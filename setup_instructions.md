# AI Workspace Setup Instructions

## System Requirements Met
- ✅ Intel i9-13900HX (32 threads)
- ✅ NVIDIA RTX 4090 Laptop GPU
- ✅ 15GB RAM, 954GB free storage
- ✅ Node.js v20.19.3
- ✅ MCP servers installed (filesystem, memory)

## Manual Installation Required

Run these commands to complete the setup:

### 1. Install Python Development Tools
```bash
sudo apt update
sudo apt install python3.12-venv python3-pip python3-dev
```

### 2. Install CUDA Toolkit (for GPU acceleration)
```bash
# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# Install CUDA toolkit
sudo apt install cuda-toolkit-12-3
```

### 3. Install Docker (optional, for containerized AI models)
```bash
sudo apt install docker.io
sudo usermod -aG docker $USER
# Log out and back in for group changes to take effect
```

## After Installation

Once the above packages are installed, I can help you:
1. Set up a Python virtual environment with ML packages
2. Install PyTorch with CUDA support
3. Download and run local AI models (LLaMA, Mistral, etc.)
4. Set up training pipelines
5. Create experiment tracking systems

## Quick Test Commands

After installation, verify everything works:
```bash
# Test CUDA
nvidia-smi
nvcc --version

# Test Python environment
cd ~/ai-workspace
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available())"
```