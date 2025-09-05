# Neural Style Transfer with Ray

A distributed neural style transfer implementation using PyTorch and Ray for batch processing of images.

## Features

- Neural style transfer using VGG19 features
- Distributed batch processing with Ray
- Support for various image formats
- GPU acceleration support
- Easy-to-use API

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

```python
from src.neural_style_transfer import run_style_transfer_inference

content_images = ["content1.jpg", "content2.jpg"]
style_image = "style.jpg"

results = run_style_transfer_inference(
    content_image_paths=content_images,
    style_image_path=style_image,
    output_dir="./outputs"
)
```

### Command Line Usage

```bash
python src/neural_style_transfer.py
```

## Requirements

- Python 3.8+
- PyTorch
- Ray
- Pillow (PIL)
- torchvision

## Project Structure

```
neural-style-transfer/
├── src/
│   └── neural_style_transfer.py    # Main implementation
├── examples/                       # Example usage
├── tests/                         # Unit tests
├── assets/                        # Sample images
├── docs/                          # Documentation
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## License

MIT License