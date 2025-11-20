"""
Inference script for breast MRI lesion detection using trained ResNet50 model
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import argparse
import numpy as np
from pathlib import Path
import json


class ResNet50Classifier(nn.Module):
    """ResNet50 model adapted for breast MRI lesion classification"""
    
    def __init__(self, num_classes=2, pretrained=True, input_channels=1):
        super(ResNet50Classifier, self).__init__()
        
        # Load pretrained ResNet50
        self.resnet = models.resnet50(pretrained=pretrained)
        
        # Modify first layer if input is grayscale
        if input_channels == 1:
            self.resnet.conv1 = nn.Conv2d(
                input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            if pretrained:
                pretrained_weights = models.resnet50(pretrained=True).conv1.weight
                self.resnet.conv1.weight = nn.Parameter(
                    pretrained_weights.mean(dim=1, keepdim=True)
                )
        
        # Replace final fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.resnet(x)


def load_model(checkpoint_path, num_classes=2, input_channels=1, device='cuda'):
    """Load trained model from checkpoint"""
    model = ResNet50Classifier(
        num_classes=num_classes,
        pretrained=False,
        input_channels=input_channels
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def preprocess_image(image_path, input_channels=1, image_size=224):
    """Preprocess image for inference"""
    # Load image
    img_path = Path(image_path)
    
    # Handle NIfTI files
    if img_path.suffix in ['.nii', '.gz']:
        try:
            import nibabel as nib
            nii_img = nib.load(str(img_path))
            img_array = nii_img.get_fdata()
            
            # Handle 3D volumes - take middle slice
            if len(img_array.shape) == 3:
                slice_idx = img_array.shape[2] // 2
                img_array = img_array[:, :, slice_idx]
            
            # Normalize to 0-255
            img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8)
            img_array = (img_array * 255).astype(np.uint8)
            
            img = Image.fromarray(img_array, mode='L')
        except ImportError:
            raise ImportError("nibabel is required for NIfTI files. Install with: pip install nibabel")
    else:
        img = Image.open(img_path)
        if input_channels == 1 and img.mode != 'L':
            img = img.convert('L')
        elif input_channels == 3 and img.mode != 'RGB':
            img = img.convert('RGB')
    
    # Apply transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229]) if input_channels == 1
        else transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    
    return img_tensor


def predict(model, image_path, class_names=None, input_channels=1, image_size=224, device='cuda'):
    """Predict class for a single image"""
    # Preprocess image
    img_tensor = preprocess_image(image_path, input_channels, image_size)
    img_tensor = img_tensor.to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    # Get class name
    if class_names:
        predicted_name = class_names[predicted_class]
    else:
        predicted_name = f"Class {predicted_class}"
    
    return {
        'predicted_class': predicted_class,
        'predicted_name': predicted_name,
        'confidence': confidence,
        'probabilities': probabilities[0].cpu().numpy().tolist()
    }


def main():
    parser = argparse.ArgumentParser(description='Inference for Breast MRI Lesion Detection')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model checkpoint (.pt file)')
    parser.add_argument('--image-path', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--input-channels', type=int, default=1, choices=[1, 3],
                        help='Number of input channels (1 for grayscale, 3 for RGB)')
    parser.add_argument('--image-size', type=int, default=224,
                        help='Input image size (default: 224)')
    parser.add_argument('--num-classes', type=int, default=2,
                        help='Number of classes (default: 2)')
    parser.add_argument('--class-names', type=str, nargs='+', default=None,
                        help='Class names (e.g., benign malignant)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use (default: cuda)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file to save results (JSON format)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    print(f'Loading model from {args.model_path}...')
    model = load_model(
        args.model_path,
        num_classes=args.num_classes,
        input_channels=args.input_channels,
        device=device
    )
    print('Model loaded successfully!')
    
    # Predict
    print(f'Processing image: {args.image_path}...')
    result = predict(
        model,
        args.image_path,
        class_names=args.class_names,
        input_channels=args.input_channels,
        image_size=args.image_size,
        device=device
    )
    
    # Print results
    print('\n' + '='*50)
    print('Prediction Results:')
    print('='*50)
    print(f'Predicted Class: {result["predicted_name"]} (Class {result["predicted_class"]})')
    print(f'Confidence: {result["confidence"]:.4f} ({result["confidence"]*100:.2f}%)')
    print('\nClass Probabilities:')
    for i, prob in enumerate(result['probabilities']):
        class_name = args.class_names[i] if args.class_names else f'Class {i}'
        print(f'  {class_name}: {prob:.4f} ({prob*100:.2f}%)')
    print('='*50)
    
    # Save results if output path is provided
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=4)
        print(f'\nResults saved to {args.output}')


if __name__ == '__main__':
    main()

