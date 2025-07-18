#!/usr/bin/env python3
"""
Quick demo to test the new directory structure with a small dataset.
"""

from main import SemiconImageProcessor
import os

def demo_directory_structure():
    """Test the new directory structure with a small dataset."""
    print("Testing New Directory Structure")
    print("=" * 40)
    
    # Create processor with small dataset for quick testing
    processor = SemiconImageProcessor(seed=42)
    processor.num_images = 20  # Small dataset for quick demo
    
    print(f"Generating {processor.num_images} test images...")
    images, labels = processor.generate_synthetic_images()
    
    print("Applying blur...")
    blurred = processor.apply_gaussian_blur(kernel_size=5, sigma=1.0)
    
    print("Extracting features...")
    features = processor.extract_features(images)
    
    print("Training classifier...")
    results = processor.train_classifier(features, labels)
    
    print("Evaluating and saving results...")
    csv_path = processor.evaluate_performance(results)
    
    print("Creating visualizations...")
    processor.visualize_results(num_samples=6)
    
    print("Saving all data...")
    output_dir = processor.save_data()
    
    # Show the directory structure
    print("\n" + "=" * 40)
    print("DIRECTORY STRUCTURE CREATED:")
    print("=" * 40)
    
    def show_tree(path, prefix="", max_files=5):
        """Show directory tree structure."""
        if not os.path.exists(path):
            return
        
        items = sorted(os.listdir(path))
        for i, item in enumerate(items):
            item_path = os.path.join(path, item)
            is_last = i == len(items) - 1
            current_prefix = "└── " if is_last else "├── "
            
            if os.path.isdir(item_path):
                print(f"{prefix}{current_prefix}{item}/")
                next_prefix = prefix + ("    " if is_last else "│   ")
                show_tree(item_path, next_prefix, max_files)
            else:
                print(f"{prefix}{current_prefix}{item}")
                # Limit file listing to avoid clutter
                if i >= max_files - 1 and len(items) > max_files:
                    remaining = len(items) - i - 1
                    if remaining > 0:
                        print(f"{prefix}... and {remaining} more files")
                    break
    
    show_tree(output_dir)
    
    # Show some key files
    print(f"\n📁 Results saved in: {output_dir}")
    print(f"📊 CSV report: {csv_path}")
    img_dir = os.path.join(output_dir, "img")
    if os.path.exists(img_dir):
        img_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
        print(f"🖼️  Individual images: {len(img_files)} PNG files in {img_dir}")
    
    print("\n✅ Directory structure test completed!")

if __name__ == "__main__":
    demo_directory_structure()
