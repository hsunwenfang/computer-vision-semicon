"""
Computer Vision Semiconductor Analysis Project

This project implements a complete computer vision pipeline for semiconductor wafer analysis
using MLX for GPU acceleration on Apple Silicon MacBooks.

Features:
- GPU-accelerated image processing using MLX
- Synthetic dataset generation with geometric shapes
- Gaussian blur filtering
- Kernel-based classification
- Performance evaluation and visualization
"""

import mlx.core as mx
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import json
from datetime import datetime
from typing import Tuple, List, Dict, Optional
import random


class SemiconImageProcessor:
    """Main class for semiconductor image processing and analysis."""
    
    def __init__(self, seed: int = 42):
        """
        Initialize the image processor.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        mx.random.seed(seed)
        
        self.image_size = (128, 128)
        self.num_images = 512
        self.images = None
        self.labels = None
        self.blurred_images = None
        
    def generate_synthetic_images(self) -> Tuple[mx.array, mx.array]:
        """
        Generate 512 grayscale images with random rectangles.
        
        Returns:
            Tuple of (images, labels) as MLX arrays
        """
        print("Generating synthetic images...")
        images = []
        labels = []
        
        for i in tqdm(range(self.num_images), desc="Creating images"):
            # Create blank image
            img = np.zeros(self.image_size, dtype=np.uint8)
            
            # Track rectangle properties for classification
            has_large_rect = False
            has_thick_border = False
            
            # Add 10 random rectangles
            for _ in range(10):
                # Random rectangle parameters
                width = np.random.randint(10, 40)
                height = np.random.randint(10, 40)
                x = np.random.randint(0, self.image_size[1] - width)
                y = np.random.randint(0, self.image_size[0] - height)
                border_width = np.random.randint(1, 6)
                intensity = np.random.randint(100, 255)
                
                # Check for classification criteria
                if width * height > 800:  # Large rectangle
                    has_large_rect = True
                if border_width >= 4:  # Thick border
                    has_thick_border = True
                
                # Draw rectangle border
                cv2.rectangle(img, (x, y), (x + width, y + height), 
                             intensity, border_width)
            
            images.append(img)
            
            # Create label based on rectangle properties
            # 0: Neither large nor thick, 1: Large or thick, 2: Both large and thick
            if has_large_rect and has_thick_border:
                label = 2
            elif has_large_rect or has_thick_border:
                label = 1
            else:
                label = 0
                
            labels.append(label)
        
        # Convert to MLX arrays
        self.images = mx.array(np.array(images, dtype=np.int8))
        self.labels = mx.array(np.array(labels, dtype=np.int32))
        
        print(f"Generated {len(images)} images with shape {self.image_size}")
        print(f"Label distribution: {np.bincount(labels)}")
        
        return self.images, self.labels
    
    def apply_gaussian_blur(self, images: Optional[mx.array] = None, 
                           kernel_size: int = 5, sigma: float = 1.0) -> mx.array:
        """
        Apply Gaussian blur to images using GPU acceleration.
        
        Args:
            images: Input images (if None, uses self.images)
            kernel_size: Size of the Gaussian kernel
            sigma: Standard deviation for Gaussian kernel
            
        Returns:
            Blurred images as MLX array
        """
        if images is None:
            images = self.images
            
        if images is None:
            raise ValueError("No images available. Generate images first.")
        
        print(f"Applying Gaussian blur (kernel_size={kernel_size}, sigma={sigma})...")
        
        # Convert MLX array to numpy for OpenCV processing
        images_np = np.array(images, dtype=np.uint8)
        blurred_images = []
        
        for img in tqdm(images_np, desc="Blurring images"):
            blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
            blurred_images.append(blurred)
        
        # Convert back to MLX array
        self.blurred_images = mx.array(np.array(blurred_images, dtype=np.int8))
        
        return self.blurred_images
    
    def extract_features(self, images: mx.array) -> mx.array:
        """
        Extract features from images using various kernels.
        
        Args:
            images: Input images
            
        Returns:
            Feature vectors
        """
        # Convert to numpy for feature extraction
        images_np = np.array(images, dtype=np.float32)
        features = []
        
        # Define various kernels for feature extraction
        edge_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        edge_kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        blur_kernel = np.ones((3, 3), dtype=np.float32) / 9
        sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        
        for img in tqdm(images_np, desc="Extracting features"):
            # Apply different kernels
            edge_x = cv2.filter2D(img, -1, edge_kernel_x)
            edge_y = cv2.filter2D(img, -1, edge_kernel_y)
            blurred = cv2.filter2D(img, -1, blur_kernel)
            sharpened = cv2.filter2D(img, -1, sharpen_kernel)
            
            # Extract statistical features
            feature_vector = [
                np.mean(img), np.std(img), np.max(img), np.min(img),
                np.mean(edge_x), np.std(edge_x),
                np.mean(edge_y), np.std(edge_y),
                np.mean(blurred), np.std(blurred),
                np.mean(sharpened), np.std(sharpened),
                np.sum(edge_x > 50),  # Count of strong edges
                np.sum(img > 200),    # Count of bright pixels
                np.percentile(img, 25), np.percentile(img, 75)  # Quartiles
            ]
            
            features.append(feature_vector)
        
        return mx.array(np.array(features, dtype=np.float32))
    
    def train_classifier(self, features: mx.array, labels: mx.array, 
                        test_size: float = 0.2) -> Dict:
        """
        Train a classifier using extracted features.
        
        Args:
            features: Feature vectors
            labels: Corresponding labels
            test_size: Fraction of data to use for testing
            
        Returns:
            Dictionary containing classifier and evaluation results
        """
        from sklearn.ensemble import RandomForestClassifier
        
        # Convert MLX arrays to numpy
        features_np = np.array(features)
        labels_np = np.array(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_np, labels_np, test_size=test_size, 
            random_state=self.seed, stratify=labels_np
        )
        
        # Train classifier
        print("Training classifier...")
        classifier = RandomForestClassifier(
            n_estimators=100, random_state=self.seed, n_jobs=-1
        )
        classifier.fit(X_train, y_train)
        
        # Make predictions
        y_pred = classifier.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        
        print(f"Classifier trained successfully!")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        
        return {
            'classifier': classifier,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'classification_report': classification_report(y_test, y_pred)
        }
    
    def evaluate_performance(self, results: Dict, base_output_dir: str = "results") -> str:
        """
        Evaluate classifier performance and save results to CSV.
        
        Args:
            results: Results from train_classifier
            base_output_dir: Base directory to save results
            
        Returns:
            Path to the saved CSV file
        """
        # Create timestamped directory structure
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(base_output_dir, f"{timestamp}-{self.num_images}")
        report_dir = os.path.join(output_dir, "report")
        os.makedirs(report_dir, exist_ok=True)
        
        # Create evaluation dataframe
        eval_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall'],
            'Score': [results['accuracy'], results['precision'], results['recall']],
            'Timestamp': [datetime.now().isoformat()] * 3
        }
        
        df = pd.DataFrame(eval_data)
        
        # Save to CSV with specified naming
        csv_path = os.path.join(report_dir, "result.csv")
        df.to_csv(csv_path, index=False)
        
        print(f"Evaluation results saved to: {csv_path}")
        print("\nDetailed Classification Report:")
        print(results['classification_report'])
        
        # Store the output directory for other methods
        self.current_output_dir = output_dir
        
        return csv_path
    
    def visualize_results(self, num_samples: int = 8, base_output_dir: str = "results"):
        """
        Visualize sample images and save them individually as PNG files.
        
        Args:
            num_samples: Number of sample images to visualize
            base_output_dir: Base directory to save visualization
        """
        # Use existing output directory if available, otherwise create new one
        if hasattr(self, 'current_output_dir') and self.current_output_dir:
            output_dir = self.current_output_dir
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = os.path.join(base_output_dir, f"{timestamp}-{self.num_images}")
        
        img_dir = os.path.join(output_dir, "img")
        os.makedirs(img_dir, exist_ok=True)
        
        if self.images is None:
            raise ValueError("No images available. Generate images first.")
        
        # Convert MLX arrays to numpy
        images_np = np.array(self.images, dtype=np.uint8)
        labels_np = np.array(self.labels) if self.labels is not None else None
        
        # Select random samples
        indices = np.random.choice(len(images_np), num_samples, replace=False)
        
        # Save individual images as PNG files
        label_names = ['No Special Features', 'Large or Thick', 'Large and Thick']
        
        for i, idx in enumerate(indices):
            img = images_np[idx]
            
            # Create individual plot for each image
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.imshow(img, cmap='gray')
            
            if labels_np is not None:
                title = f"Image {idx} - Label: {label_names[labels_np[idx]]}"
            else:
                title = f"Image {idx}"
                
            ax.set_title(title, fontsize=12)
            ax.axis('off')
            
            # Save individual image with ID as filename
            img_path = os.path.join(img_dir, f"{idx}.png")
            plt.savefig(img_path, dpi=300, bbox_inches='tight')
            plt.close()  # Close to free memory
            
            print(f"Saved image {idx} to: {img_path}")
        
        # Also create a summary grid visualization
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i, idx in enumerate(indices):
            ax = axes[i]
            img = images_np[idx]
            
            # Display image
            ax.imshow(img, cmap='gray')
            
            if labels_np is not None:
                title = f"Image {idx}\nLabel: {label_names[labels_np[idx]]}"
            else:
                title = f"Image {idx}"
                
            ax.set_title(title, fontsize=10)
            ax.axis('off')
        
        plt.tight_layout()
        
        # Save summary visualization
        summary_path = os.path.join(output_dir, f"sample_images_summary.png")
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Summary visualization saved to: {summary_path}")
        print(f"Individual images saved in: {img_dir}")
        
        # Also create a comparison between original and blurred images if available
        if self.blurred_images is not None:
            self._visualize_blur_comparison(num_samples, output_dir)
    
    def _visualize_blur_comparison(self, num_samples: int, output_dir: str):
        """Create comparison visualization between original and blurred images."""
        
        images_np = np.array(self.images, dtype=np.uint8)
        blurred_np = np.array(self.blurred_images, dtype=np.uint8)
        
        fig, axes = plt.subplots(2, num_samples, figsize=(2*num_samples, 6))
        
        indices = np.random.choice(len(images_np), num_samples, replace=False)
        
        for i, idx in enumerate(indices):
            # Original image
            axes[0, i].imshow(images_np[idx], cmap='gray')
            axes[0, i].set_title(f"Original {idx}", fontsize=10)
            axes[0, i].axis('off')
            
            # Blurred image
            axes[1, i].imshow(blurred_np[idx], cmap='gray')
            axes[1, i].set_title(f"Blurred {idx}", fontsize=10)
            axes[1, i].axis('off')
        
        plt.tight_layout()
        
        # Save comparison
        comp_path = os.path.join(output_dir, "blur_comparison.png")
        plt.savefig(comp_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Blur comparison saved to: {comp_path}")
    
    def save_data(self, base_output_dir: str = "results"):
        """
        Save generated images and metadata in the organized directory structure.
        
        Args:
            base_output_dir: Base directory to save data
        """
        # Use existing output directory if available, otherwise create new one
        if hasattr(self, 'current_output_dir') and self.current_output_dir:
            output_dir = self.current_output_dir
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = os.path.join(base_output_dir, f"{timestamp}-{self.num_images}")
        
        data_dir = os.path.join(output_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        
        if self.images is None:
            raise ValueError("No images available. Generate images first.")
        
        # Save images as numpy arrays
        images_path = os.path.join(data_dir, "images.npy")
        labels_path = os.path.join(data_dir, "labels.npy")
        
        np.save(images_path, np.array(self.images))
        if self.labels is not None:
            np.save(labels_path, np.array(self.labels))
        
        if self.blurred_images is not None:
            blurred_path = os.path.join(data_dir, "blurred_images.npy")
            np.save(blurred_path, np.array(self.blurred_images))
        
        # Save metadata
        metadata = {
            'num_images': self.num_images,
            'image_size': self.image_size,
            'seed': self.seed,
            'timestamp': datetime.now().isoformat(),
            'label_distribution': np.bincount(np.array(self.labels)).tolist() if self.labels is not None else None,
            'output_structure': {
                'base_dir': output_dir,
                'data_dir': data_dir,
                'img_dir': os.path.join(output_dir, "img"),
                'report_dir': os.path.join(output_dir, "report")
            }
        }
        
        metadata_path = os.path.join(data_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Data saved to directory: {output_dir}")
        print(f"- Images: {images_path}")
        print(f"- Labels: {labels_path}")
        if self.blurred_images is not None:
            print(f"- Blurred images: {blurred_path}")
        print(f"- Metadata: {metadata_path}")
        
        return output_dir
    
    def save_all_images_individually(self, base_output_dir: str = "results"):
        """
        Save all generated images as individual PNG files.
        
        Args:
            base_output_dir: Base directory to save images
        """
        # Use existing output directory if available, otherwise create new one
        if hasattr(self, 'current_output_dir') and self.current_output_dir:
            output_dir = self.current_output_dir
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = os.path.join(base_output_dir, f"{timestamp}-{self.num_images}")
        
        img_dir = os.path.join(output_dir, "img")
        os.makedirs(img_dir, exist_ok=True)
        
        if self.images is None:
            raise ValueError("No images available. Generate images first.")
        
        # Convert MLX arrays to numpy
        images_np = np.array(self.images, dtype=np.uint8)
        labels_np = np.array(self.labels) if self.labels is not None else None
        
        label_names = ['No Special Features', 'Large or Thick', 'Large and Thick']
        
        print(f"Saving all {len(images_np)} images individually...")
        
        for idx in tqdm(range(len(images_np)), desc="Saving images"):
            img = images_np[idx]
            
            # Create individual plot for each image
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.imshow(img, cmap='gray')
            
            if labels_np is not None:
                title = f"Image {idx} - Label: {label_names[labels_np[idx]]}"
            else:
                title = f"Image {idx}"
                
            ax.set_title(title, fontsize=12)
            ax.axis('off')
            
            # Save individual image with ID as filename
            img_path = os.path.join(img_dir, f"{idx}.png")
            plt.savefig(img_path, dpi=300, bbox_inches='tight')
            plt.close()  # Close to free memory
        
        print(f"All images saved in: {img_dir}")
        return img_dir


def main():
    """Main function to run the complete pipeline."""
    
    print("Computer Vision Semiconductor Analysis Pipeline")
    print("=" * 50)
    
    # Initialize processor
    processor = SemiconImageProcessor(seed=42)
    
    # Step 1: Generate synthetic images
    images, labels = processor.generate_synthetic_images()
    
    # Step 2: Apply Gaussian blur
    blurred_images = processor.apply_gaussian_blur(kernel_size=7, sigma=1.5)
    
    # Step 3: Extract features
    print("\nExtracting features...")
    features = processor.extract_features(images)
    
    # Step 4: Train classifier
    print("\nTraining classifier...")
    results = processor.train_classifier(features, labels)
    
    # Step 5: Evaluate performance and create timestamped directory structure
    print("\nEvaluating performance...")
    csv_path = processor.evaluate_performance(results)
    
    # Step 6: Visualize results (sample images saved individually)
    print("\nCreating visualizations...")
    processor.visualize_results(num_samples=8)
    
    # Step 7: Save all data in organized structure
    print("\nSaving data...")
    output_dir = processor.save_data()
    
    # Step 8: Optionally save all images individually (uncomment if needed)
    # print("\nSaving all images individually...")
    # processor.save_all_images_individually()
    
    print("\nPipeline completed successfully!")
    print(f"All results saved in: {output_dir}")
    print("Directory structure:")
    print(f"  ├── {output_dir}/")
    print(f"  ├── ├── img/           # Individual PNG images ({processor.num_images}.png)")
    print(f"  ├── ├── report/        # CSV results (result.csv)")
    print(f"  ├── ├── data/          # NumPy arrays and metadata")
    print(f"  └── └── *.png          # Summary visualizations")


if __name__ == "__main__":
    main()
