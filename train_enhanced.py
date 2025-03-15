import torch
import argparse
import os
from train_compression import (
    CompressibleLanguageModel,
    TextDataset,
    train,
    generate_text,
)


def main():
    parser = argparse.ArgumentParser(
        description="Train with fixed adaptive compression"
    )
    parser.add_argument(
        "--model_name", type=str, default="gpt2", help="Base model to use"
    )
    parser.add_argument(
        "--data_path", type=str, default="sample_data.txt", help="Path to training data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="enhanced_model_fixed",
        help="Directory to save model",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs to train (increased to 5)",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="Learning rate"
    )
    parser.add_argument(
        "--max_length", type=int, default=128, help="Maximum sequence length"
    )
    parser.add_argument(
        "--eval_steps", type=int, default=50, help="Evaluate every N steps"
    )
    parser.add_argument(
        "--save_steps", type=int, default=100, help="Save model every N steps"
    )
    parser.add_argument(
        "--log_steps",
        type=int,
        default=20,
        help="Log stats every N steps (reduced for more frequent logging)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine compression layer indices - use 3 layers for better distribution
    compression_indices = [3, 7, 11]  # For GPT-2, distribute across the 12 layers

    # Create compressible model with conservative initial thresholds
    print("Creating compressible model with fixed adaptive thresholds...")
    model = CompressibleLanguageModel(
        base_model_name=args.model_name,
        compression_layer_indices=compression_indices,
        # More conservative initial thresholds
        similarity_threshold=[0.85, 0.75, 0.65],  # Conservative initial thresholds
        use_residuals=True,  # Enable residuals for better reconstruction
        percentile_threshold=95.0,  # Enable adaptive thresholding
        use_progressive_compression=True,  # Use progressive compression based on layer depth
    )

    # Load training data
    print(f"Loading dataset from: {args.data_path}")
    dataset = TextDataset(
        tokenizer=model.tokenizer, max_length=args.max_length, file_path=args.data_path
    )

    # Train model with more frequent logging and monitoring
    print("Starting training with adaptive compression...")
    train(
        model=model,
        train_dataset=dataset,
        batch_size=args.batch_size,
        epochs=args.epochs,  # Train for more epochs to allow adaptation
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        log_steps=args.log_steps,  # More frequent logging
    )

    # Generate sample text
    print("\nGenerating sample text with enhanced adaptive compression model:")
    prompts = [
        "Artificial intelligence will",
        "The future of technology is",
        "In the next decade, humans will",
    ]

    for prompt in prompts:
        generated_text = generate_text(model, prompt, max_length=100)
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated_text}")

    # Print compression statistics
    print("\nCompression Statistics:")
    stats = model.get_compression_stats()
    for key, value in stats.items():
        if isinstance(value, list) and value:
            avg_value = sum(value) / len(value)
            print(f"{key}: {avg_value:.4f}")
        else:
            print(f"{key}: {value}")

    print(f"\nTraining complete. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
