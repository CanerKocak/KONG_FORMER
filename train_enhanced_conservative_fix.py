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
        description="Train with EXTREMELY conservative compression"
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
        default="enhanced_model_conservative_fix",
        help="Directory to save model",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for training (reduced for stability)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs to train",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="Learning rate"
    )
    parser.add_argument(
        "--max_length", type=int, default=128, help="Maximum sequence length"
    )
    parser.add_argument(
        "--eval_steps", type=int, default=25, help="Evaluate every N steps"
    )
    parser.add_argument(
        "--save_steps", type=int, default=100, help="Save model every N steps"
    )
    parser.add_argument(
        "--log_steps",
        type=int,
        default=5,
        help="Log stats every N steps",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine compression layer indices - use only ONE layer in the middle
    compression_indices = [8]  # Single compression layer as recommended

    # Create compressible model with EXTREMELY conservative settings
    print("Creating compressible model with EXTREMELY conservative settings...")
    model = CompressibleLanguageModel(
        base_model_name=args.model_name,
        compression_layer_indices=compression_indices,
        # Much more conservative threshold as recommended
        similarity_threshold=0.85,  # Conservative but effective
        use_residuals=True,  # Enable residuals for better reconstruction
        percentile_threshold=0.0,  # Disable adaptive thresholding until stable
        use_progressive_compression=False,  # Disable progressive compression
    )

    # Load training data
    print(f"Loading dataset from: {args.data_path}")
    dataset = TextDataset(
        tokenizer=model.tokenizer, max_length=args.max_length, file_path=args.data_path
    )

    # Enhanced validation function with strict compression ratio limits
    def validation_callback(model, step):
        """Comprehensive quality check with strict compression ratio limits"""
        print("\n--- Generation Quality Check ---")

        # Track compression ratio
        stats = model.get_compression_stats()
        compression_ratio = stats.get("compression_ratio", 1.0)
        print(f"Current compression ratio: {compression_ratio:.4f}x")

        # CRITICAL: Enforce strict upper bound on compression ratio
        if compression_ratio > 20.0:
            print(
                "⚠️ CRITICAL: Compression ratio too high (>20x)! Resetting thresholds lower..."
            )
            for layer in model.compression_layers:
                # Reset threshold to a much lower value
                layer.base_threshold = max(0.75, layer.base_threshold - 0.05)
                print(f"Reset threshold to {layer.base_threshold:.4f}")

        # Test with various prompts
        test_prompts = [
            "Artificial intelligence will",
            "The future of technology is",
            "In the next decade, we will see",
        ]

        generation_quality_score = 0
        for prompt in test_prompts:
            generated = generate_text(model, prompt, max_length=50)
            print(f"Prompt: {prompt}")
            print(f"Generated: {generated}")

            # Simple quality heuristic: length of generation beyond prompt
            prompt_words = len(prompt.split())
            generated_words = len(generated.split())
            words_added = max(0, generated_words - prompt_words)

            # Score based on words added (simple heuristic)
            quality_score = min(1.0, words_added / 10)  # Max score at 10+ words
            generation_quality_score += quality_score

            print(f"Words added: {words_added}, Quality score: {quality_score:.2f}")
            print("-" * 40)

        # Average quality score across all prompts
        avg_quality = generation_quality_score / len(test_prompts)
        print(f"Average generation quality: {avg_quality:.2f}/1.0")

        # Adjust compression based on quality and target ratio
        target_ratio = 1.2  # Target only 20% improvement

        # If compression is too high OR quality is too low, reduce compression
        if compression_ratio > target_ratio * 1.1 or avg_quality < 0.5:
            print("⚠️ WARNING: Compression too aggressive or quality too low!")
            for layer in model.compression_layers:
                # Make threshold higher (less compression)
                layer.base_threshold = min(0.95, layer.base_threshold + 0.02)
                print(f"Increased threshold to {layer.base_threshold:.4f}")

        # If compression is too low AND quality is good, allow slightly more compression
        elif compression_ratio < target_ratio * 0.9 and avg_quality > 0.7:
            print(
                "ℹ️ Compression below target with good quality, allowing slight increase"
            )
            for layer in model.compression_layers:
                # Very slightly reduce threshold (more compression)
                layer.base_threshold = max(0.75, layer.base_threshold - 0.01)
                print(f"Decreased threshold to {layer.base_threshold:.4f}")

        print("--- End Quality Check ---\n")

    # Train model with more frequent validation and monitoring
    print("Starting training with EXTREMELY conservative compression...")
    train(
        model=model,
        train_dataset=dataset,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        log_steps=args.log_steps,
        validation_callback=validation_callback,  # Enhanced validation
    )

    # Generate sample text
    print("\nGenerating sample text with conservative compression model:")
    prompts = [
        "Artificial intelligence will",
        "The future of technology is",
        "In the next decade, humans will",
        "The most important scientific discovery is",
        "Climate change will affect",
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
