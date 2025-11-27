"""
Helper script to train ResNet50 with transfer learning on Odelia dataset
This script loads settings from config_odelia.json and runs training
"""

import json
import subprocess
import sys
from pathlib import Path

# Load config
config_path = Path(__file__).parent / "config_odelia.json"
with open(config_path, 'r') as f:
    config = json.load(f)

# Resolve data directory path
data_dir = config["data_dir"]
data_dir_path = Path(data_dir).resolve()

# Verify path exists
if not data_dir_path.exists():
    raise FileNotFoundError(
        f"Data directory not found: {data_dir}\n"
        f"Resolved path: {data_dir_path}\n"
        f"Please check the path in config_odelia.json"
    )

data_dir = str(data_dir_path)

# Build command
cmd = [
    sys.executable,
    str(Path(__file__).parent / "train_resnet50.py"),
    "--data-dir", data_dir,
    "--input-channels", str(config["input_channels"]),
    "--image-size", str(config["image_size"]),
    "--epochs", str(config["epochs"]),
    "--batch-size", str(config["batch_size"]),
    "--learning-rate", str(config["learning_rate"]),
    "--weight-decay", str(config["weight_decay"]),
    "--scheduler", config["scheduler"],
    "--num-workers", str(config["num_workers"]),
    "--output-dir", config["output_dir"],
    "--save-interval", str(config["save_interval"]),
    "--early-stopping-patience", str(config["early_stopping_patience"]),
]

# Add pretrained flag (transfer learning)
if config.get("pretrained", True):
    cmd.append("--pretrained")
    print("=" * 60)
    print("Training with TRANSFER LEARNING (ImageNet pretrained weights)")
    print("=" * 60)
else:
    cmd.append("--no-pretrained")
    print("=" * 60)
    print("Training from SCRATCH (no pretrained weights)")
    print("=" * 60)

# Add class weights if enabled
if config.get("class_weights", False):
    cmd.append("--class-weights")

# Print configuration
print("\nConfiguration:")
print(f"  Data directory: {data_dir}")
print(f"  Input channels: {config['input_channels']}")
print(f"  Image size: {config['image_size']}")
print(f"  Epochs: {config['epochs']}")
print(f"  Batch size: {config['batch_size']}")
print(f"  Learning rate: {config['learning_rate']}")
print(f"  Pretrained: {config.get('pretrained', True)}")
print(f"  Class weights: {config.get('class_weights', False)}")
print(f"  Output directory: {config['output_dir']}")
print("\n" + "=" * 60 + "\n")

# Run training
subprocess.run(cmd)

