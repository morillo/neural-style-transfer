"""
Basic usage example for neural style transfer
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.neural_style_transfer import run_style_transfer_inference

def main():
    # Example content images (replace with your actual paths)
    content_images = [
        "assets/content1.jpg",
        "assets/content2.jpg",
    ]
    
    # Style image path (Van Gogh's Starry Night or any style image)
    style_image = "assets/starry_night.jpg"
    
    # Check if files exist
    for img_path in content_images + [style_image]:
        if not os.path.exists(img_path):
            print(f"Warning: {img_path} does not exist. Please add sample images to the assets/ directory.")
            return
    
    print("Running neural style transfer...")
    
    # Run style transfer
    results = run_style_transfer_inference(
        content_image_paths=content_images,
        style_image_path=style_image,
        output_dir="./outputs"
    )
    
    print(f"Processed {len(results)} images successfully!")
    print("Check the ./outputs directory for results.")

if __name__ == "__main__":
    main()