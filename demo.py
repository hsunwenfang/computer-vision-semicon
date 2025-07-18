"""
Demo script showcasing the computer vision pipeline.
"""

from main import SemiconImageProcessor
from image_utils import MLXImageUtils, AdvancedFilters
import numpy as np
import matplotlib.pyplot as plt
import time


def demo_basic_pipeline():
    """Demonstrate the basic pipeline functionality."""
    print("=" * 60)
    print("DEMO: Basic Computer Vision Pipeline")
    print("=" * 60)
    
    # Initialize processor
    processor = SemiconImageProcessor(seed=42)
    
    # Generate images
    start_time = time.time()
    images, labels = processor.generate_synthetic_images()
    gen_time = time.time() - start_time
    print(f"Image generation took: {gen_time:.2f} seconds")
    
    # Apply blur
    start_time = time.time()
    blurred = processor.apply_gaussian_blur(kernel_size=5, sigma=1.0)
    blur_time = time.time() - start_time
    print(f"Gaussian blur took: {blur_time:.2f} seconds")
    
    # Extract features and train classifier
    start_time = time.time()
    features = processor.extract_features(images)
    feature_time = time.time() - start_time
    print(f"Feature extraction took: {feature_time:.2f} seconds")
    
    start_time = time.time()
    results = processor.train_classifier(features, labels)
    train_time = time.time() - start_time
    print(f"Training took: {train_time:.2f} seconds")
    
    # Evaluate and visualize
    processor.evaluate_performance(results)
    processor.visualize_results(num_samples=6)
    processor.save_data()
    
    total_time = gen_time + blur_time + feature_time + train_time
    print(f"\nTotal pipeline time: {total_time:.2f} seconds")


def demo_advanced_filters():
    """Demonstrate advanced filtering techniques."""
    print("\n" + "=" * 60)
    print("DEMO: Advanced Image Filters")
    print("=" * 60)
    
    # Generate a few test images
    processor = SemiconImageProcessor(seed=123)
    images, _ = processor.generate_synthetic_images()
    
    # Convert first image to numpy for demo
    test_image = np.array(images[0], dtype=np.uint8)
    
    # Apply different filters
    filters = AdvancedFilters()
    
    # Sobel edge detection
    grad_x, grad_y = filters.sobel_edge_detection(test_image)
    edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Laplacian sharpening
    sharpened = filters.laplacian_sharpening(test_image, strength=0.8)
    
    # Adaptive thresholding
    threshold = filters.adaptive_threshold(test_image, block_size=11)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(test_image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(edge_magnitude, cmap='gray')
    axes[0, 1].set_title('Edge Detection (Sobel)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(sharpened, cmap='gray')
    axes[0, 2].set_title('Laplacian Sharpening')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(threshold, cmap='gray')
    axes[1, 0].set_title('Adaptive Threshold')
    axes[1, 0].axis('off')
    
    # Show gradient components
    axes[1, 1].imshow(grad_x, cmap='gray')
    axes[1, 1].set_title('Gradient X')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(grad_y, cmap='gray')
    axes[1, 2].set_title('Gradient Y')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/advanced_filters_demo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Advanced filters demo completed!")
    print("Visualization saved to: results/advanced_filters_demo.png")


def demo_mlx_operations():
    """Demonstrate MLX GPU operations."""
    print("\n" + "=" * 60)
    print("DEMO: MLX GPU Operations")
    print("=" * 60)
    
    try:
        import mlx.core as mx
        
        # Create some test data
        test_images = mx.random.uniform(0, 255, (10, 128, 128))
        print(f"Created test tensor with shape: {test_images.shape}")
        
        # Test MLX utilities
        utils = MLXImageUtils()
        
        # Create Gaussian kernel
        kernel = utils.create_gaussian_kernel(7, 1.5)
        print(f"Created Gaussian kernel with shape: {kernel.shape}")
        
        # Enhance contrast
        enhanced = utils.enhance_contrast_mlx(test_images, alpha=1.2, beta=10)
        print(f"Enhanced contrast for {enhanced.shape[0]} images")
        
        # Batch normalize
        normalized = utils.batch_normalize_mlx(test_images)
        print(f"Normalized batch of images")
        
        print("MLX operations demo completed successfully!")
        
    except ImportError as e:
        print(f"MLX not available: {e}")
        print("Install MLX with: pip install mlx")


def demo_performance_comparison():
    """Compare performance of different operations."""
    print("\n" + "=" * 60)
    print("DEMO: Performance Comparison")
    print("=" * 60)
    
    processor = SemiconImageProcessor(seed=42)
    
    # Test different image sizes
    sizes = [64, 128, 256]
    num_images = [100, 200, 500]
    
    results = []
    
    for size in sizes:
        for num in num_images:
            processor.image_size = (size, size)
            processor.num_images = num
            
            # Time image generation
            start_time = time.time()
            images, labels = processor.generate_synthetic_images()
            gen_time = time.time() - start_time
            
            # Time feature extraction
            start_time = time.time()
            features = processor.extract_features(images)
            feature_time = time.time() - start_time
            
            results.append({
                'image_size': f"{size}x{size}",
                'num_images': num,
                'generation_time': gen_time,
                'feature_time': feature_time,
                'total_pixels': size * size * num
            })
            
            print(f"Size: {size}x{size}, Images: {num}, "
                  f"Gen: {gen_time:.2f}s, Features: {feature_time:.2f}s")
    
    # Create performance visualization
    import pandas as pd
    df = pd.DataFrame(results)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot generation time vs total pixels
    ax1.scatter(df['total_pixels'], df['generation_time'])
    ax1.set_xlabel('Total Pixels')
    ax1.set_ylabel('Generation Time (s)')
    ax1.set_title('Image Generation Performance')
    
    # Plot feature extraction time vs total pixels
    ax2.scatter(df['total_pixels'], df['feature_time'])
    ax2.set_xlabel('Total Pixels')
    ax2.set_ylabel('Feature Extraction Time (s)')
    ax2.set_title('Feature Extraction Performance')
    
    plt.tight_layout()
    plt.savefig('results/performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save performance data
    df.to_csv('results/performance_data.csv', index=False)
    print("Performance comparison completed!")
    print("Results saved to: results/performance_data.csv")


def main():
    """Run all demos."""
    print("Computer Vision Pipeline Demos")
    print("=" * 60)
    
    # Run demos
    demo_basic_pipeline()
    demo_advanced_filters()
    demo_mlx_operations()
    demo_performance_comparison()
    
    print("\n" + "=" * 60)
    print("All demos completed!")
    print("Check the 'results' directory for output files.")
    print("=" * 60)


if __name__ == "__main__":
    main()
