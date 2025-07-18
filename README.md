# Computer Vision Semiconductor Analysis

A comprehensive Python project for computer vision analysis using MLX GPU acceleration on Apple Silicon MacBooks. This project demonstrates synthetic image generation, advanced filtering, classification, and performance evaluation for semiconductor wafer analysis.

## Features

🚀 **GPU Acceleration**: Leverages MLX for optimized performance on Apple Silicon  
🖼️ **Synthetic Dataset**: Generates 512 grayscale images (128x128) with geometric patterns  
🔍 **Advanced Filtering**: Gaussian blur, edge detection, and morphological operations  
🤖 **ML Classification**: Kernel-based feature extraction and Random Forest classification  
📊 **Performance Metrics**: Comprehensive evaluation with accuracy, precision, and recall  
📈 **Visualization**: Interactive plots and result analysis  
💾 **Data Management**: Automated saving and loading of datasets and results  

## Project Structure

```
computer-vision-semicon/
├── main.py              # Main pipeline implementation
├── demo.py              # Demonstration scripts
├── test_setup.py        # Environment testing
├── test_directory_structure.py  # Directory structure testing
├── image_utils.py       # MLX-accelerated image processing utilities
├── config.py            # Configuration settings
├── requirements.txt     # Package dependencies
├── run.sh              # Convenient execution script
├── results/            # Timestamped output directories
│   └── {datetime}-{count}/
│       ├── img/        # Individual PNG images ({id}.png)
│       ├── report/     # CSV results (result.csv)
│       ├── data/       # NumPy arrays and metadata
│       └── *.png       # Summary visualizations
└── models/             # Trained models (if saved)
```

## Requirements

- Python 3.12+
- Apple Silicon Mac (for MLX acceleration)
- 8GB+ RAM recommended

## Installation

1. Clone or navigate to the project directory:
```bash
cd /Users/hsunwenfang/Documents/computer-vision-semicon
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Test the installation:
```bash
python test_setup.py
```

## Quick Start

### Run the Complete Pipeline
```bash
./run.sh main
```

This will:
- Generate 512 synthetic images with random rectangles
- Apply Gaussian blur filtering
- Extract features using various kernels
- Train a Random Forest classifier
- Evaluate performance and save results
- Create visualizations

**Output Structure:**
```
results/{timestamp}-512/
├── img/           # Individual PNG images (0.png, 1.png, ...)
├── report/        # result.csv with performance metrics
├── data/          # NumPy arrays and metadata
└── *.png          # Summary visualizations
```

### Test Directory Structure
```bash
./run.sh structure   # Test with 20 images
```

### Run Demonstrations
```bash
python demo.py
```

Includes demos for:
- Basic pipeline functionality
- Advanced image filters
- MLX GPU operations
- Performance comparisons

## Technical Details

### Image Generation
- **Format**: 128x128 grayscale images (int8)
- **Content**: 10 random rectangles per image with varying:
  - Dimensions (10-40 pixels)
  - Positions (random placement)
  - Border widths (1-5 pixels)
  - Intensities (100-255)

### Classification Categories
- **Class 0**: No special features
- **Class 1**: Large rectangles (area > 800px²) OR thick borders (≥4px)
- **Class 2**: Both large rectangles AND thick borders

### Feature Extraction
- Edge detection (Sobel X/Y)
- Gaussian blur response
- Sharpening filter response
- Statistical measures (mean, std, percentiles)
- Pixel intensity distributions

### Performance Metrics
- **Accuracy**: Overall classification correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)

## Usage Examples

### Custom Image Generation
```python
from main import SemiconImageProcessor

processor = SemiconImageProcessor(seed=42)
images, labels = processor.generate_synthetic_images()
print(f"Generated {len(images)} images")
```

### Apply Custom Filters
```python
# Apply Gaussian blur with custom parameters
blurred = processor.apply_gaussian_blur(
    kernel_size=9, 
    sigma=2.0
)
```

### Train Custom Classifier
```python
features = processor.extract_features(images)
results = processor.train_classifier(features, labels, test_size=0.3)
print(f"Accuracy: {results['accuracy']:.4f}")
```

## Output Files

### Organized by Timestamp and Image Count
All outputs are saved in timestamped directories: `results/{timestamp}-{image_count}/`

### Generated Data
- `data/images.npy`: Original synthetic images
- `data/labels.npy`: Classification labels  
- `data/blurred_images.npy`: Gaussian-filtered images
- `data/metadata.json`: Dataset metadata

### Individual Images
- `img/{id}.png`: Individual PNG images with IDs (0.png, 1.png, ...)

### Results
- `report/result.csv`: Performance metrics (accuracy, precision, recall)
- `sample_images_summary.png`: Grid visualization
- `blur_comparison.png`: Before/after filtering comparison

### Example Directory Structure
```
results/20250718_163604-512/
├── img/
│   ├── 0.png
│   ├── 1.png
│   ├── ...
│   └── 511.png
├── report/
│   └── result.csv
├── data/
│   ├── images.npy
│   ├── labels.npy
│   ├── blurred_images.npy
│   └── metadata.json
├── sample_images_summary.png
└── blur_comparison.png
```

## MLX GPU Acceleration

This project leverages Apple's MLX framework for GPU acceleration on Apple Silicon:

- **Array Operations**: Efficient tensor computations
- **Memory Management**: Optimized GPU memory usage
- **Parallel Processing**: Batch operations for image processing

### Performance Benefits
- 2-5x faster image processing compared to CPU-only
- Efficient memory utilization for large datasets
- Seamless integration with existing NumPy workflows

## Configuration

Modify `config.py` to customize:
- Image dimensions and count
- Rectangle generation parameters
- Classification thresholds
- Filter parameters
- Output directories

## Troubleshooting

### Common Issues

1. **MLX Import Error**:
   ```bash
   pip install mlx
   ```

2. **OpenCV Issues**:
   ```bash
   pip install opencv-python-headless
   ```

3. **Memory Issues**:
   - Reduce `NUM_IMAGES` in `config.py`
   - Process images in smaller batches

### Performance Optimization

- Ensure you're running on Apple Silicon
- Close other GPU-intensive applications
- Monitor memory usage with Activity Monitor

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Acknowledgments

- Apple MLX team for GPU acceleration framework
- OpenCV community for computer vision tools
- scikit-learn for machine learning utilities

---

For questions or issues, please check the troubleshooting section or create an issue in the repository.
