#!/bin/bash
# Setup script for sensor-consciousness environment

echo "Setting up sensor-consciousness environment..."

# Activate virtual environment
source sensor_venv/bin/activate

# Install core requirements
pip install opencv-python numpy matplotlib pillow scipy scikit-learn

# Check installation
python -c "import cv2; print(f'✅ OpenCV {cv2.__version__} installed')"
python -c "import numpy; print(f'✅ NumPy {numpy.__version__} installed')"

echo "✅ Environment ready!"
echo "To activate: source sensor_venv/bin/activate"