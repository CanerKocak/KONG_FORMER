#!/usr/bin/env python3
"""
Setup script to create both a standard GPT-2 model and a compressed version
for use with the Neural Compression Model GUI.
"""

import os
import torch
import argparse
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from train_compression import CompressibleLanguageModel


def setup_base_model(output_dir="gpt2_base_model"):
    """Set up a standard GPT-2 model without compression."""
    print(f"Setting up base GPT-2 model in {output_dir}...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load model and tokenizer
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Save model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Base GPT-2 model saved to {output_dir}")
    return output_dir


def setup_compressed_model(output_dir="gpt2_compressed_model", compression_ratio=8.0):
    """Set up a GPT-2 model with compression layers."""
    print(f"Setting up compressed GPT-2 model in {output_dir}...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create a new model with GPT-2 as the base and add compression layers
    model = CompressibleLanguageModel(
        base_model_name="gpt2",
        compression_layer_indices=[5],  # Add compression after layer 5
        compression_factor=2,
        similarity_threshold=0.92,
        freeze_base_model=False,
        use_residuals=True,
        use_progressive_compression=False,
    )

    # Set the max_compression_ratio attribute
    model.max_compression_ratio = compression_ratio

    # Initialize max_compression_ratio on all compression layers
    if hasattr(model, "compression_layers"):
        for layer in model.compression_layers:
            layer.max_compression_ratio = model.max_compression_ratio

    # Save the model
    model.save_pretrained(output_dir)

    print(f"Compressed GPT-2 model saved to {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Set up models for the Neural Compression GUI"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="gpt2_base_model",
        help="Directory to save the base GPT-2 model",
    )
    parser.add_argument(
        "--compressed_dir",
        type=str,
        default="gpt2_compressed_model",
        help="Directory to save the compressed GPT-2 model",
    )
    parser.add_argument(
        "--compression_ratio",
        type=float,
        default=8.0,
        help="Maximum compression ratio for the compressed model",
    )

    args = parser.parse_args()

    # Setup both models
    base_dir = setup_base_model(args.base_dir)
    compressed_dir = setup_compressed_model(args.compressed_dir, args.compression_ratio)

    print("\nSetup complete!")
    print(f"Base model: {base_dir}")
    print(f"Compressed model: {compressed_dir}")
    print("\nYou can now run the GUI with either model:")
    print(f"python run_gui.py --model_dir {base_dir}")
    print(f"python run_gui.py --model_dir {compressed_dir}")


if __name__ == "__main__":
    main()
