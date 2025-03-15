import torch
import argparse
import time
from train_compression import CompressibleLanguageModel, generate_text


def generate_with_compression(
    model_path, prompts=None, max_length=200, temperature=0.7, top_p=0.9
):
    """Generate text with the enhanced compression model."""
    print(f"Loading model from {model_path}")
    model = CompressibleLanguageModel.from_pretrained(model_path)

    # Move model to appropriate device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model.to(device)

    # Default prompts if none provided
    if prompts is None:
        prompts = [
            "Recursive compression of neural networks works by",
            "The most efficient way to represent language is to",
            "Attention-weighted clustering in language models helps",
            "Progressive compression across transformer layers enables",
            "The future of efficient language models involves",
        ]

    print("\nGenerating text with enhanced compression:")
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")

        # Generate with compression enabled
        model.enable_compression()
        start_time = time.time()
        compressed_text = generate_text(
            model, prompt, max_length=max_length, temperature=temperature, top_p=top_p
        )
        compressed_time = time.time() - start_time
        print(f"With compression ({compressed_time:.2f}s): {compressed_text}")

        # Generate with compression disabled
        for layer in model.compression_layers:
            layer.compression_enabled = False

        start_time = time.time()
        uncompressed_text = generate_text(
            model, prompt, max_length=max_length, temperature=temperature, top_p=top_p
        )
        uncompressed_time = time.time() - start_time
        print(f"Without compression ({uncompressed_time:.2f}s): {uncompressed_text}")

        # Re-enable compression for next prompt
        for layer in model.compression_layers:
            layer.compression_enabled = True

        # Calculate speedup
        speedup = uncompressed_time / compressed_time if compressed_time > 0 else 0
        print(f"Speedup: {speedup:.2f}x")

    print("\nGeneration completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate text with enhanced compression model"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model directory"
    )
    parser.add_argument(
        "--max_length", type=int, default=200, help="Maximum length for generated text"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Temperature for sampling"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.9, help="Top-p sampling parameter"
    )
    parser.add_argument("--prompt", type=str, help="Custom prompt for text generation")

    args = parser.parse_args()

    # Use custom prompt if provided
    prompts = [args.prompt] if args.prompt else None

    generate_with_compression(
        args.model_path, prompts, args.max_length, args.temperature, args.top_p
    )
