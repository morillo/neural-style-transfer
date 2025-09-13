import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import ray
from ray import data
import numpy as np
import io
import base64

# Initialize Ray
ray.init()

def get_best_device():
    """Get the best available device with proper fallback order: CUDA -> MPS -> CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

class VGGFeatures(nn.Module):
    """Extract features from VGG19 for style transfer"""
    def __init__(self):
        super(VGGFeatures, self).__init__()
        # Load pre-trained VGG19
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.layers = nn.ModuleList(vgg[:29])  # Up to conv4_4
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return features

class GramMatrix(nn.Module):
    """Compute Gram matrix for style representation"""
    def forward(self, x):
        batch, channels, height, width = x.size()
        features = x.view(batch * channels, height * width)
        gram = torch.mm(features, features.t())
        return gram.div(batch * channels * height * width)

class StyleTransferModel(nn.Module):
    """Neural Style Transfer model"""
    def __init__(self):
        super(StyleTransferModel, self).__init__()
        self.vgg = VGGFeatures()
        self.gram = GramMatrix()
        
        # Style layers (conv layers where we compute style loss)
        self.style_layers = [0, 5, 10, 19, 28]  # conv1_1, conv2_1, conv3_1, conv4_1, conv5_1
        # Content layer
        self.content_layer = 21  # conv4_2
        
    def get_style_features(self, style_image):
        """Extract style features from style image"""
        features = self.vgg(style_image)
        style_features = []
        for i in self.style_layers:
            style_features.append(self.gram(features[i]))
        return style_features
    
    def get_content_features(self, content_image):
        """Extract content features from content image"""
        features = self.vgg(content_image)
        return features[self.content_layer]

def preprocess_image(image_path, size=512):
    """Preprocess image for neural style transfer"""
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    if isinstance(image_path, str):
        image = Image.open(image_path).convert('RGB')
    else:
        # Assume it's already a PIL Image or bytes
        if isinstance(image_path, bytes):
            image = Image.open(io.BytesIO(image_path)).convert('RGB')
        else:
            image = image_path.convert('RGB')
    
    return transform(image).unsqueeze(0)

def deprocess_image(tensor):
    """Convert tensor back to PIL Image"""
    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to PIL
    tensor = tensor.squeeze(0)
    transform = transforms.ToPILImage()
    return transform(tensor)

def style_transfer_step(target, content_features, style_features, model, 
                       content_weight=1, style_weight=1000, iterations=300):
    """Perform style transfer optimization"""
    target = target.clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS([target])
    
    def closure():
        optimizer.zero_grad()
        
        # Get features from target
        target_features = model.vgg(target)
        
        # Content loss
        content_loss = F.mse_loss(target_features[model.content_layer], content_features)
        
        # Style loss
        style_loss = 0
        for i, style_layer_idx in enumerate(model.style_layers):
            target_gram = model.gram(target_features[style_layer_idx])
            style_loss += F.mse_loss(target_gram, style_features[i])
        
        # Total loss
        total_loss = content_weight * content_loss + style_weight * style_loss
        total_loss.backward()
        
        return total_loss
    
    # Optimization loop
    for i in range(iterations):
        optimizer.step(closure)
    
    return target

class StyleTransferPredictor:
    """Ray predictor class for style transfer"""
    def __init__(self, style_image_path):
        self.device = get_best_device()
        self.model = StyleTransferModel().to(self.device)
        self.model.eval()
        
        # Load and preprocess style image (Starry Night)
        self.style_image = preprocess_image(style_image_path).to(self.device)
        self.style_features = self.model.get_style_features(self.style_image)
        
    def __call__(self, batch):
        """Process a batch of content images"""
        results = []
        
        for item in batch["image"]:
            try:
                # Preprocess content image
                if isinstance(item, str):
                    content_image = preprocess_image(item).to(self.device)
                else:
                    # Handle base64 encoded images or raw bytes
                    if isinstance(item, str) and item.startswith('data:image'):
                        # Base64 encoded image
                        image_data = base64.b64decode(item.split(',')[1])
                        content_image = preprocess_image(image_data).to(self.device)
                    else:
                        content_image = preprocess_image(item).to(self.device)
                
                # Extract content features
                with torch.no_grad():
                    content_features = self.model.get_content_features(content_image)
                
                # Initialize target as content image
                target = content_image.clone()
                
                # Perform style transfer
                with torch.enable_grad():
                    stylized = style_transfer_step(
                        target, content_features, self.style_features, self.model,
                        content_weight=1, style_weight=1000, iterations=100
                    )
                
                # Convert back to PIL and then to base64
                stylized_pil = deprocess_image(stylized.cpu())
                
                # Convert to base64 for storage/transmission
                buffer = io.BytesIO()
                stylized_pil.save(buffer, format='PNG')
                img_str = base64.b64encode(buffer.getvalue()).decode()
                
                results.append({
                    "stylized_image": f"data:image/png;base64,{img_str}",
                    "status": "success"
                })
                
            except Exception as e:
                results.append({
                    "stylized_image": None,
                    "status": f"error: {str(e)}"
                })
        
        return {"stylized_image": [r["stylized_image"] for r in results],
                "status": [r["status"] for r in results]}

def create_sample_dataset(image_paths):
    """Create a Ray dataset from image paths"""
    data_dict = {"image": image_paths}
    return ray.data.from_items([{"image": path} for path in image_paths])

def run_style_transfer_inference(content_image_paths, style_image_path, output_dir="./outputs"):
    """Main function to run style transfer inference using Ray Data"""
    
    print("Creating dataset...")
    dataset = create_sample_dataset(content_image_paths)
    
    print("Setting up style transfer predictor...")
    # Create predictor with the style image
    predictor = StyleTransferPredictor(style_image_path)
    
    print("Running inference...")
    # Apply style transfer using Ray Data
    # Note: Ray doesn't have built-in MPS support, so we use CPU resources for MPS devices
    device = get_best_device()
    use_gpu_resources = torch.cuda.is_available()
    
    results = dataset.map_batches(
        predictor,
        batch_size=2,  # Process 2 images at a time
        num_gpus=1 if use_gpu_resources else 0,
        num_cpus=2
    )
    
    print("Collecting results...")
    # Collect and save results
    output_data = results.take_all()
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for i, result in enumerate(output_data):
        if result["status"] == "success" and result["stylized_image"]:
            # Decode base64 and save
            img_data = base64.b64decode(result["stylized_image"].split(',')[1])
            output_path = os.path.join(output_dir, f"stylized_image_{i}.png")
            with open(output_path, 'wb') as f:
                f.write(img_data)
            print(f"Saved stylized image: {output_path}")
        else:
            print(f"Failed to process image {i}: {result['status']}")
    
    return output_data

# Example usage
if __name__ == "__main__":
    # Example paths - replace with your actual image paths
    content_images = [
        "house1.jpg",
        "building1.jpg", 
        "house2.jpg",
        "building2.jpg"
    ]
    
    # Path to Van Gogh's Starry Night image
    starry_night_path = "starry_night.jpg"
    
    # Run the style transfer inference
    results = run_style_transfer_inference(
        content_image_paths=content_images,
        style_image_path=starry_night_path,
        output_dir="./stylized_outputs"
    )
    
    print(f"Processed {len(results)} images")
    print("Style transfer inference complete!")
    
    # Shutdown Ray
    ray.shutdown()