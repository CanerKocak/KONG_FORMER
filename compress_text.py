#!/usr/bin/env python
"""
Simple script to demonstrate using the compression model for text generation.
"""

import torch
from transformers import AutoTokenizer
import argparse
import os
from train_compression import CompressibleLanguageModel
import json


def main():
    parser = argparse.ArgumentParser(
        description="Generate text using a compressed language model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model directory",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Neural networks",
        help="Text prompt for generation",
    )
    parser.add_argument(
        "--max_length", type=int, default=100, help="Maximum length of generated text"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature"
    )
    parser.add_argument(
        "--num_samples", type=int, default=1, help="Number of samples to generate"
    )
    parser.add_argument(
        "--disable_compression",
        action="store_true",
        help="Disable compression during generation",
    )
    parser.add_argument(
        "--show_stats", action="store_true", help="Show compression statistics"
    )

    args = parser.parse_args()

    # Check if model path exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model path '{args.model_path}' does not exist.")
        return

    # Load model configuration
    config_path = os.path.join(args.model_path, "config.json")
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at '{config_path}'.")
        return

    with open(config_path, "r") as f:
        config = json.load(f)

    # Load tokenizer
    print(f"Loading tokenizer for {config['base_model_name']}...")
    tokenizer = AutoTokenizer.from_pretrained(config["base_model_name"])

    # Load model
    print(f"Loading model from {args.model_path}...")
    model = CompressibleLanguageModel.load(args.model_path)

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    model.eval()

    # Set compression mode
    if args.disable_compression:
        print("Compression disabled for generation.")
        model.disable_compression()
    else:
        print("Compression enabled for generation.")
        model.enable_compression()

    # Generate text samples
    print(f"\nGenerating {args.num_samples} sample(s) with prompt: '{args.prompt}'")
    print(f"Max length: {args.max_length}, Temperature: {args.temperature}")
    print("-" * 50)

    for i in range(args.num_samples):
        if args.num_samples > 1:
            print(f"\nSample {i+1}:")

        # Generate text
        generated_text = model.generate_text(
            tokenizer,
            args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
        )

        print(generated_text)
        print("-" * 50)

    # Show compression statistics if requested
    if args.show_stats and not args.disable_compression:
        print("\nCompression Statistics:")
        stats = model.get_compression_stats()
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")


if __name__ == "__main__":
    main()
