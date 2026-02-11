"""
Generic Model-to-Config Converter

Converts any trained PyTorch model checkpoint or model file to a YAML config file.
Automatically extracts architecture information, hyperparameters, and metadata.

Usage:
    python model_to_config.py <input_file> <output_file>
    
    Examples:
        python model_to_config.py models/vae.pt experiments/exp009/config.yaml
        python model_to_config.py checkpoints/epoch_50.pt config.yaml
"""

import yaml
import torch
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import argparse

# ==================== CONFIGURATION ====================
# Set your parameters here instead of command-line arguments
INPUT_FILE = "path/to/model.pt"  # Path to model or checkpoint file
OUTPUT_FILE = "config.yaml"       # Path to save config YAML file

# Optional parameters
EPOCH = None                      # Epoch number (if applicable)
BATCH_SIZE = None                 # Batch size
LEARNING_RATE = None              # Learning rate
NUM_EPOCHS = None                 # Total number of epochs
DESCRIPTION = None                # Model description
# ========================================================


def extract_model_architecture(model: torch.nn.Module) -> Dict[str, Any]:
    """
    Automatically extract architecture information from a PyTorch model
    
    Args:
        model: PyTorch model instance
    
    Returns:
        Dictionary with model architecture details
    """
    architecture = {
        'type': model.__class__.__name__,
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'layers': []
    }
    
    # Extract layer information
    for name, module in model.named_modules():
        if name:  # Skip empty names (root module)
            layer_info = {
                'name': name,
                'type': module.__class__.__name__,
                'parameters': sum(p.numel() for p in module.parameters())
            }
            
            # Extract specific layer parameters
            if hasattr(module, 'out_features'):
                layer_info['out_features'] = module.out_features
            if hasattr(module, 'in_features'):
                layer_info['in_features'] = module.in_features
            if hasattr(module, 'kernel_size'):
                layer_info['kernel_size'] = module.kernel_size
            if hasattr(module, 'stride'):
                layer_info['stride'] = module.stride
            if hasattr(module, 'padding'):
                layer_info['padding'] = module.padding
            if hasattr(module, 'out_channels'):
                layer_info['out_channels'] = module.out_channels
            if hasattr(module, 'in_channels'):
                layer_info['in_channels'] = module.in_channels
            
            architecture['layers'].append(layer_info)
    
    # Try to extract model-specific attributes
    if hasattr(model, 'latent_dim'):
        architecture['latent_dim'] = model.latent_dim
    if hasattr(model, 'hidden_dim'):
        architecture['hidden_dim'] = model.hidden_dim
    
    return architecture


def load_checkpoint(file_path: str) -> tuple[Dict[str, Any] | torch.nn.Module, Dict[str, Any]]:
    """
    Load model or checkpoint from file
    
    Args:
        file_path: Path to model or checkpoint file (.pt, .pth, .tar, etc.)
    
    Returns:
        Tuple of (model or state_dict, metadata)
    """
    file_path_obj = Path(file_path)
    
    if not file_path_obj.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    print(f"Loading checkpoint from: {file_path}")
    checkpoint = torch.load(file_path, map_location='cpu')
    
    metadata = {
        'source_file': str(file_path),
        'file_size_bytes': file_path_obj.stat().st_size,
        'file_created': datetime.fromtimestamp(file_path_obj.stat().st_ctime).isoformat(),
    }
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        # Standard checkpoint format with state_dict
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
            metadata['checkpoint_keys'] = list(checkpoint.keys())
            metadata['epoch'] = checkpoint.get('epoch', None)
            metadata['includes_optimizer'] = 'optimizer_state_dict' in checkpoint
        else:
            model_state = checkpoint
    else:
        # Direct model object
        model_state = checkpoint
    
    return model_state, metadata


def model_to_config_generic(
    input_path: str,
    output_path: str,
    additional_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Generic function to convert model/checkpoint to config YAML
    
    Args:
        input_path: Path to model or checkpoint file
        output_path: Path to save config YAML
        additional_config: Optional dictionary with additional config info
    
    Returns:
        Complete configuration dictionary
    """
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    model_state, metadata = load_checkpoint(input_path)
    
    # Extract architecture
    if isinstance(model_state, torch.nn.Module):
        architecture = extract_model_architecture(model_state)
        model_object = model_state
    else:
        # It's a state_dict, we can extract some info
        architecture = {
            'type': 'Unknown',
            'total_parameters': sum(p.numel() if isinstance(p, torch.Tensor) else 0 
                                   for p in model_state.values() 
                                   if isinstance(p, torch.Tensor))
        }
        model_object = None
    
    # Create configuration
    config = {
        'model': architecture,
        'metadata': {
            'converted_at': datetime.now().isoformat(),
            'source': metadata,
            'source_file': str(input_path),
        }
    }
    
    # Add additional configuration if provided
    if additional_config:
        config['additional_config'] = additional_config
    
    # Save to YAML
    with open(output_path, 'w') as f:
        yaml.dump(config, f, 
                 default_flow_style=False,
                 sort_keys=False,
                 allow_unicode=True)
    
    print(f"✓ Configuration saved to: {output_path}")
    
    return config


def print_config_summary(config: Dict[str, Any]) -> None:
    """
    Print a formatted summary of the configuration
    
    Args:
        config: Configuration dictionary
    """
    print("\n" + "="*70)
    print("CONFIGURATION SUMMARY")
    print("="*70)
    
    # Model info
    print("\n[MODEL ARCHITECTURE]")
    model_cfg = config.get('model', {})
    print(f"  Type:                  {model_cfg.get('type', 'Unknown')}")
    print(f"  Total Parameters:      {model_cfg.get('total_parameters', 'N/A'):,}")
    print(f"  Trainable Parameters:  {model_cfg.get('trainable_parameters', 'N/A'):,}")
    
    if 'latent_dim' in model_cfg:
        print(f"  Latent Dimension:      {model_cfg.get('latent_dim')}")
    if 'hidden_dim' in model_cfg:
        print(f"  Hidden Dimension:      {model_cfg.get('hidden_dim')}")
    
    # Metadata
    metadata = config.get('metadata', {})
    if 'source' in metadata:
        print("\n[SOURCE FILE METADATA]")
        source = metadata['source']
        print(f"  Source File:           {source.get('source_file', 'N/A')}")
        print(f"  File Size (bytes):     {source.get('file_size_bytes', 'N/A'):,}")
        print(f"  File Created:          {source.get('file_created', 'N/A')}")
    
    print("\n[CONVERSION INFO]")
    print(f"  Converted At:          {metadata.get('converted_at', 'N/A')}")
    
    # Additional config
    if 'additional_config' in config:
        print("\n[ADDITIONAL CONFIGURATION]")
        for key, value in config['additional_config'].items():
            print(f"  {key}: {value}")
    
    print("\n" + "="*70)


def main():
    """Convert model/checkpoint to YAML configuration using top-level parameters"""
    
    # Build additional config from top-level parameters
    additional_config = {}
    if EPOCH is not None:
        additional_config['epoch'] = EPOCH
    if BATCH_SIZE is not None:
        additional_config['batch_size'] = BATCH_SIZE
    if LEARNING_RATE is not None:
        additional_config['learning_rate'] = LEARNING_RATE
    if NUM_EPOCHS is not None:
        additional_config['num_epochs'] = NUM_EPOCHS
    if DESCRIPTION is not None:
        additional_config['description'] = DESCRIPTION
    
    # Convert model to config
    try:
        config = model_to_config_generic(
            input_path=INPUT_FILE,
            output_path=OUTPUT_FILE,
            additional_config=additional_config if additional_config else None
        )
        
        # Print summary
        print_config_summary(config)
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
