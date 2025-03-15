import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from train_compression import (
    CompressibleLanguageModel,
    TextDataset,
    train,
    generate_text,
)
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Train a compressible language model")
    parser.add_argument(
        "--model_name", type=str, default="gpt2", help="Base model to use"
    )
    parser.add_argument(
        "--data_path", type=str, default="sample_data.txt", help="Path to training data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="compressed_model",
        help="Directory to save model",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of epochs to train"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="Learning rate"
    )
    parser.add_argument(
        "--freeze_base", action="store_true", help="Freeze base model parameters"
    )
    parser.add_argument(
        "--compression_layers", type=int, default=1, help="Number of compression layers"
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.85,
        help="Similarity threshold for compression",
    )
    parser.add_argument(
        "--use_residuals",
        action="store_true",
        help="Use residual vectors in compression",
    )
    parser.add_argument(
        "--adaptive_threshold", action="store_true", help="Use adaptive thresholding"
    )
    parser.add_argument(
        "--contrastive_weight",
        type=float,
        default=0.1,
        help="Weight for contrastive loss",
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
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer and model
    print(f"Loading base model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name)

    # Determine compression layer indices based on number of layers
    num_layers = base_model.config.num_hidden_layers
    compression_indices = []
    if args.compression_layers > 0:
        step = max(1, num_layers // (args.compression_layers + 1))
        compression_indices = [i * step for i in range(1, args.compression_layers + 1)]

    # Create compressible model
    print("Creating compressible model...")
    model = CompressibleLanguageModel(
        base_model_name=args.model_name,
        compression_layer_indices=compression_indices,
        freeze_base_model=args.freeze_base,
        similarity_threshold=args.similarity_threshold,
        use_residuals=args.use_residuals,
        percentile_threshold=95.0 if args.adaptive_threshold else 0.0,
    )

    # Load training data
    print(f"Loading dataset from: {args.data_path}")
    dataset = TextDataset(
        tokenizer=model.tokenizer, max_length=args.max_length, file_path=args.data_path
    )

    # Train model
    print("Starting training...")
    train(
        model=model,
        train_dataset=dataset,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
    )

    # Generate sample text
    print("\nGenerating sample text with compressed model:")
    prompt = "Neural networks"
    generated_text = generate_text(model=model, prompt=prompt, max_length=100)
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")

    # Print compression statistics
    print("\nCompression Statistics:")
    stats = model.get_compression_stats()
    for key, value in stats.items():
        if isinstance(value, list) and value:
            avg_value = sum(value) / len(value)
            print(f"{key}: {avg_value:.4f}")

    print(f"\nTraining complete. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
