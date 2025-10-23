#!/bin/bash

# ===================================================================
# Kaggle VQA Challenge - Installation Script (T4 GPU Compatible)
# ===================================================================

set -e  # Exit on error

echo "============================================================"
echo "  Kaggle VQA Challenge - Installing Dependencies"
echo "============================================================"
echo ""
echo "CRITICAL NOTES:"
echo "- T4 GPU: BFloat16 NOT supported → Using Float16"
echo "- FlashAttention 2: NOT supported on T4 → Removed"
echo "- Transformers: Git install required for Qwen2.5-VL"
echo ""
echo "============================================================"

# Step 1: Install PyTorch
echo ""
echo "[1/4] Installing PyTorch 2.3.0..."
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118

# Step 2: Install Transformers from Git
echo ""
echo "[2/4] Installing Transformers from Git (for Qwen2.5-VL support)..."
pip install git+https://github.com/huggingface/transformers.git

# Step 3: Install Qwen VL Utils
echo ""
echo "[3/4] Installing qwen-vl-utils (REQUIRED)..."
pip install qwen-vl-utils[decord]==0.0.8

# Step 4: Install remaining packages
echo ""
echo "[4/4] Installing remaining packages..."
pip install peft==0.12.0 bitsandbytes==0.43.3 accelerate==0.33.0
pip install datasets==2.20.0 pillow==10.4.0 opencv-python==4.10.0
pip install scikit-learn==1.5.1 pandas==2.2.2 numpy==1.26.4 tqdm==4.66.4
pip install wandb==0.17.5 optuna==3.6.1
pip install pyyaml==6.0.1 python-dotenv==1.0.1
pip install jupyter==1.0.0 ipywidgets==8.1.3
pip install matplotlib==3.9.1 seaborn==0.13.2

# Verification
echo ""
echo "============================================================"
echo "  Installation Complete!"
echo "============================================================"
echo ""
echo "Verifying critical packages..."

python3 -c "
import torch
import transformers
print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ CUDA Available: {torch.cuda.is_available()}')
print(f'✓ CUDA Version: {torch.version.cuda}')
print(f'✓ Transformers: {transformers.__version__}')

try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    print('✓ Qwen2.5-VL classes: Available')
except ImportError as e:
    print(f'✗ Qwen2.5-VL classes: NOT Available ({e})')
    print('  Please reinstall transformers from Git')

try:
    import qwen_vl_utils
    print('✓ qwen-vl-utils: Installed')
except ImportError:
    print('✗ qwen-vl-utils: NOT Installed')
    print('  Please run: pip install qwen-vl-utils[decord]==0.0.8')
"

echo ""
echo "============================================================"
echo "  Ready to start!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "1. Login to WandB: wandb login"
echo "2. Run EDA: python scripts/eda.py"
echo "3. Train model: python scripts/train_lora.py --fold 0"
echo ""
