#!/usr/bin/env bash
# Upgrade PyTorch for RTX 5090 (CUDA sm_120) support

set -e

echo "🚀 Upgrading PyTorch for RTX 5090 compatibility..."

# Activate the virtual environment
source .venv/bin/activate

# Check current PyTorch version
echo "Current PyTorch version:"
python -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null || echo "PyTorch not properly installed"

# Remove existing PyTorch packages
echo "Removing existing PyTorch packages..."
uv remove torch torchvision torchaudio || true

# Install PyTorch with CUDA 12.4 support (compatible with RTX 5090)
echo "Installing PyTorch with CUDA 12.4 support..."
uv add torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124

# Reinstall sentence-transformers to ensure compatibility
echo "Reinstalling sentence-transformers..."
uv remove sentence-transformers || true
uv add sentence-transformers>=2.2.2

echo "✅ PyTorch upgrade complete!"
echo ""
echo "Testing CUDA compatibility:"
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
    try:
        # Test tensor operations
        x = torch.zeros(1).cuda()
        print('✅ CUDA operations working correctly')
    except Exception as e:
        print(f'❌ CUDA error: {e}')
        print('💡 This is expected if your GPU is too new for this PyTorch version')
else:
    print('ℹ️ CUDA not available, will use CPU')
"

echo ""
echo "🎯 If you still see CUDA compatibility warnings, you can:"
echo "   1. Force CPU usage by setting: export EMBEDDING_DEVICE=cpu"
echo "   2. Or wait for PyTorch to release support for sm_120 architecture"
echo "   3. The application will automatically fallback to CPU if CUDA fails"
