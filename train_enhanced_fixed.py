import torch
import argparse
import os
from train_compression import (
    CompressibleLanguageModel,
    TextDataset,
    train,
    generate_text,
)
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Train with properly fixed adaptive compression"
    )
    parser.add_argument(
        "--model_name", type=str, default="gpt2-xl", help="Base model to use"
    )
    parser.add_argument(
        "--data_path", type=str, default="sample_data.txt", help="Path to training data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="enhanced_model_conservative1",
        help="Directory to save model",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,  # Reduced to prevent overtraining
        help="Number of epochs to train",
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
        default=10,  # Even more frequent logging
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

    # Determine compression layer indices - use fewer layers (2 instead of 3)
    compression_indices = [4, 8]  # Reduced number of compression layers

    # Create compressible model with MUCH more conservative initial thresholds
    print("Creating compressible model with EXTREMELY conservative thresholds...")
    model = CompressibleLanguageModel(
        base_model_name=args.model_name,
        compression_layer_indices=compression_indices,
        # Ultra-conservative initial thresholds
        similarity_threshold=0.995,  # Extremely high threshold
        use_residuals=True,  # Enable residuals for better reconstruction
        percentile_threshold=99.5,  # Very high percentile threshold
        use_progressive_compression=True,  # Enable progressive compression for safer training
    )

    # Load training data
    print(f"Loading dataset from: {args.data_path}")
    dataset = TextDataset(
        tokenizer=model.tokenizer, max_length=args.max_length, file_path=args.data_path
    )

    # Custom validation function to check generation quality during training
    def validation_callback(model, step):
        """Check generation quality at regular intervals"""
        # Only print detailed generation every 5 validation steps
        if step % 5 == 0:
            print("\n--- Generation Quality Check ---")
            test_prompts = [
                "Artificial intelligence will",
                "The future of technology is",
            ]
            for prompt in test_prompts:
                generated = generate_text(model, prompt, max_length=50)
                print(f"Prompt: {prompt}")
                print(f"Generated: {generated}")
                print("-" * 40)

        # Get current compression stats
        stats = model.get_compression_stats()
        if stats:
            print(
                f"Step {step} - Compression ratio: {stats.get('compression_ratio', 'N/A')}"
            )

        # Only print end marker for detailed checks
        if step % 5 == 0:
            print("--- End Quality Check ---\n")

        # More conservative threshold adjustment
        if stats and stats.get("compression_ratio", 0) > 5:  # Reduced from 10 to 5
            print(f"⚠️ WARNING: Compression ratio too high! Adjusting threshold...")
            for layer in model.compression_layers:
                layer.base_threshold = min(
                    0.999, layer.base_threshold + 0.001
                )  # Smaller adjustments
                print(f"Increased threshold to {layer.base_threshold:.3f}")

    # Train model with more frequent logging and monitoring
    print("Starting training with CONSERVATIVE compression...")
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
        validation_callback=validation_callback,  # Add validation callback
    )

    # Compare base model vs compressed model
    print("\n" + "=" * 50)
    print("COMPARING BASE MODEL VS COMPRESSED MODEL")
    print("=" * 50)

    # Load the base model for comparison
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name)
    base_tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Function to generate text with the base model
    def generate_with_base(prompt, max_length=100):
        inputs = base_tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output_sequences = base_model.generate(
                input_ids=inputs["input_ids"],
                max_length=max_length,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )
        return base_tokenizer.decode(output_sequences[0], skip_special_tokens=True)

    # Custom prompt for comparison
    custom_prompt = "My name is Julien and I like to"

    print(f"\nPrompt: {custom_prompt}")
    print("-" * 40)
    print("BASE MODEL OUTPUT:")
    base_output = generate_with_base(custom_prompt)
    print(base_output)
    print("-" * 40)
    print("COMPRESSED MODEL OUTPUT:")
    compressed_output = generate_text(model, custom_prompt, max_length=100)
    print(compressed_output)
    print("-" * 40)

    # Generate sample text with other prompts
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
# enhanced_model_conservative/epoch-1
