#!/bin/bash

# Computer Vision Pipeline Runner
# Usage: ./run.sh [main|demo|test|quick|structure]

set -e

PYTHON_CMD="/Users/hsunwenfang/Documents/computer-vision-semicon/.venv/bin/python"

case "${1:-main}" in
    "main")
        echo "Running main computer vision pipeline..."
        $PYTHON_CMD main.py
        ;;
    "demo")
        echo "Running demonstrations..."
        $PYTHON_CMD demo.py
        ;;
    "test")
        echo "Running setup tests..."
        $PYTHON_CMD test_setup.py
        ;;
    "structure")
        echo "Testing directory structure with small dataset..."
        $PYTHON_CMD test_directory_structure.py
        ;;
    "quick")
        echo "Running quick test..."
        $PYTHON_CMD -c "
from main import SemiconImageProcessor
processor = SemiconImageProcessor(seed=42)
processor.num_images = 20
images, labels = processor.generate_synthetic_images()
features = processor.extract_features(images)
results = processor.train_classifier(features, labels)
print(f'Quick test completed! Accuracy: {results[\"accuracy\"]:.4f}')
"
        ;;
    *)
        echo "Usage: $0 [main|demo|test|quick|structure]"
        echo ""
        echo "  main      - Run the complete computer vision pipeline (512 images)"
        echo "  demo      - Run demonstration scripts"
        echo "  test      - Test environment setup"
        echo "  quick     - Quick functionality test (20 images)"
        echo "  structure - Test new directory structure (20 images)"
        exit 1
        ;;
esac
