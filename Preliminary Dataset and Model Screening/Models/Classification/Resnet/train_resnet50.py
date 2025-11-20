"""
Training script for breast MRI lesion detection using ResNet50
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import os
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from datetime import datetime

from dataset import BreastMRIDataset


class ResNet50Classifier(nn.Module):
    """ResNet50 model adapted for breast MRI lesion classification"""
    
    def __init__(self, num_classes=2, pretrained=True, input_channels=1):
        """
        Args:
            num_classes: Number of output classes (default: 2 for benign/malignant)
            pretrained: Whether to use ImageNet pretrained weights
            input_channels: Number of input channels (1 for grayscale MRI, 3 for RGB)
        """
        super(ResNet50Classifier, self).__init__()
        
        # Load pretrained ResNet50
        self.resnet = models.resnet50(pretrained=pretrained)
        
        # Modify first layer if input is grayscale
        if input_channels == 1:
            # Replace first conv layer to accept single channel input
            self.resnet.conv1 = nn.Conv2d(
                input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            # Initialize with pretrained weights (average across RGB channels)
            if pretrained:
                pretrained_weights = models.resnet50(pretrained=True).conv1.weight
                self.resnet.conv1.weight = nn.Parameter(
                    pretrained_weights.mean(dim=1, keepdim=True)
                )
        
        # Replace final fully connected layer for our classification task
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.resnet(x)


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return {
        'loss': epoch_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='[Val]'):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return {
        'loss': epoch_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def plot_training_history(history, save_path):
    """Plot training and validation metrics"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy
    axes[0, 1].plot(epochs, history['train_accuracy'], 'b-', label='Train Acc')
    axes[0, 1].plot(epochs, history['val_accuracy'], 'r-', label='Val Acc')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Precision
    axes[0, 2].plot(epochs, history['train_precision'], 'b-', label='Train Precision')
    axes[0, 2].plot(epochs, history['val_precision'], 'r-', label='Val Precision')
    axes[0, 2].set_title('Precision')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Precision')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Recall
    axes[1, 0].plot(epochs, history['train_recall'], 'b-', label='Train Recall')
    axes[1, 0].plot(epochs, history['val_recall'], 'r-', label='Val Recall')
    axes[1, 0].set_title('Recall')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Recall')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # F1 Score
    axes[1, 1].plot(epochs, history['train_f1'], 'b-', label='Train F1')
    axes[1, 1].plot(epochs, history['val_f1'], 'r-', label='Val F1')
    axes[1, 1].set_title('F1 Score')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Learning Rate
    if 'lr' in history:
        axes[1, 2].plot(epochs, history['lr'], 'g-', label='Learning Rate')
        axes[1, 2].set_title('Learning Rate')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('LR')
        axes[1, 2].set_yscale('log')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
    else:
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training history plot saved to {save_path}")


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config = vars(args)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229]) if args.input_channels == 1 
        else transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229]) if args.input_channels == 1
        else transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = BreastMRIDataset(
        data_dir=args.data_dir,
        split='train',
        transform=train_transform,
        input_channels=args.input_channels
    )
    val_dataset = BreastMRIDataset(
        data_dir=args.data_dir,
        split='val',
        transform=val_transform,
        input_channels=args.input_channels
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f'Train samples: {len(train_dataset)}')
    print(f'Val samples: {len(val_dataset)}')
    print(f'Number of classes: {train_dataset.num_classes}')
    
    # Create model
    model = ResNet50Classifier(
        num_classes=train_dataset.num_classes,
        pretrained=args.pretrained,
        input_channels=args.input_channels
    ).to(device)
    
    # Loss function
    if args.class_weights:
        # Calculate class weights from training set
        class_counts = train_dataset.get_class_counts()
        total = sum(class_counts.values())
        class_weights = [total / (len(class_counts) * class_counts[i]) for i in range(len(class_counts))]
        class_weights = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f'Using class weights: {class_weights.cpu().numpy()}')
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    if args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    elif args.scheduler == 'step':
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    else:
        scheduler = None
    
    # Training history
    history = {
        'train_loss': [], 'train_accuracy': [], 'train_precision': [], 'train_recall': [], 'train_f1': [],
        'val_loss': [], 'val_accuracy': [], 'val_precision': [], 'val_recall': [], 'val_f1': [],
        'lr': []
    }
    
    best_val_loss = float('inf')
    best_val_f1 = 0.0
    patience_counter = 0
    
    # Training loop
    print('\nStarting training...')
    for epoch in range(1, args.epochs + 1):
        print(f'\nEpoch {epoch}/{args.epochs}')
        print('-' * 50)
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        if scheduler:
            if args.scheduler == 'plateau':
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()
        
        # Record history
        for key in ['loss', 'accuracy', 'precision', 'recall', 'f1']:
            history[f'train_{key}'].append(train_metrics[key])
            history[f'val_{key}'].append(val_metrics[key])
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Print metrics
        print(f'Train - Loss: {train_metrics["loss"]:.4f}, Acc: {train_metrics["accuracy"]:.4f}, '
              f'F1: {train_metrics["f1"]:.4f}')
        print(f'Val   - Loss: {val_metrics["loss"]:.4f}, Acc: {val_metrics["accuracy"]:.4f}, '
              f'F1: {val_metrics["f1"]:.4f}')
        print(f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_f1': val_metrics['f1'],
                'history': history
            }, output_dir / 'best_model.pt')
            print('✓ Saved best model (lowest validation loss)')
            patience_counter = 0
        elif val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_f1': val_metrics['f1'],
                'history': history
            }, output_dir / 'best_f1_model.pt')
            print('✓ Saved best model (highest validation F1)')
        
        # Early stopping
        patience_counter += 1
        if args.early_stopping_patience > 0 and patience_counter >= args.early_stopping_patience:
            print(f'\nEarly stopping triggered after {epoch} epochs')
            break
        
        # Save checkpoint
        if epoch % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history
            }, output_dir / f'checkpoint_epoch_{epoch}.pt')
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history
    }, output_dir / 'final_model.pt')
    
    # Plot training history
    plot_training_history(history, output_dir / 'training_history.png')
    
    # Save history as JSON
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=4)
    
    print(f'\nTraining completed! Results saved to {output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ResNet50 for Breast MRI Lesion Detection')
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--input-channels', type=int, default=1, choices=[1, 3],
                        help='Number of input channels (1 for grayscale, 3 for RGB)')
    parser.add_argument('--image-size', type=int, default=224,
                        help='Input image size (default: 224)')
    
    # Model arguments
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained ImageNet weights')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false',
                        help='Do not use pretrained weights')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay (default: 1e-4)')
    parser.add_argument('--scheduler', type=str, default='plateau', choices=['plateau', 'step', 'none'],
                        help='Learning rate scheduler (default: plateau)')
    parser.add_argument('--step-size', type=int, default=10,
                        help='Step size for StepLR scheduler (default: 10)')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='Gamma for StepLR scheduler (default: 0.1)')
    
    # Other arguments
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                        help='Output directory for models and logs (default: ./outputs)')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='Save checkpoint every N epochs (default: 10)')
    parser.add_argument('--early-stopping-patience', type=int, default=15,
                        help='Early stopping patience (default: 15, 0 to disable)')
    parser.add_argument('--class-weights', action='store_true',
                        help='Use class weights to handle imbalanced dataset')
    
    args = parser.parse_args()
    main(args)

