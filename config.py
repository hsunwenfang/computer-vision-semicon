"""
Configuration settings for the computer vision project.
"""

import os

# Project settings
PROJECT_NAME = "Computer Vision Semiconductor Analysis"
VERSION = "1.0.0"

# Image generation settings
IMAGE_SIZE = (128, 128)
NUM_IMAGES = 512
NUM_RECTANGLES_PER_IMAGE = 10
RANDOM_SEED = 42

# Rectangle generation parameters
RECT_MIN_SIZE = 10
RECT_MAX_SIZE = 40
BORDER_MIN_WIDTH = 1
BORDER_MAX_WIDTH = 5
INTENSITY_MIN = 100
INTENSITY_MAX = 255

# Classification criteria
LARGE_RECT_THRESHOLD = 800  # Area threshold for "large" rectangles
THICK_BORDER_THRESHOLD = 4  # Border width threshold for "thick" borders

# Gaussian blur settings
DEFAULT_KERNEL_SIZE = 7
DEFAULT_SIGMA = 1.5

# Feature extraction settings
FEATURE_KERNELS = {
    'edge_x': [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
    'edge_y': [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
    'blur': [[1/9] * 3] * 3,
    'sharpen': [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]
}

# Classification settings
TEST_SIZE = 0.2
N_ESTIMATORS = 100

# Output directories
RESULTS_BASE_DIR = "results"
DATA_DIR = "data"  # Relative to timestamped directory
IMG_DIR = "img"    # Relative to timestamped directory  
REPORT_DIR = "report"  # Relative to timestamped directory

# Output structure: results/{timestamp}-{image_count}/
#   ├── img/           # Individual PNG images ({id}.png)
#   ├── report/        # CSV results (result.csv)
#   ├── data/          # NumPy arrays and metadata
#   └── *.png          # Summary visualizations

# Ensure base directory exists
import os
os.makedirs(RESULTS_BASE_DIR, exist_ok=True)

# Visualization settings
VISUALIZATION_SAMPLES = 8
DPI = 300
FIGURE_SIZE = (16, 8)

# MLX/GPU settings
GPU_ENABLED = True
BATCH_SIZE = 32

# Data type settings
IMAGE_DTYPE = 'int8'
FEATURE_DTYPE = 'float32'

# File formats
IMAGE_FORMAT = 'npy'
RESULTS_FORMAT = 'csv'
MODEL_FORMAT = 'pkl'
