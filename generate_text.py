import torch
import argparse
from train_compression import CompressibleLanguageModel, generate_text


def main():
    parser = argparse.ArgumentParser(
        description="Generate text with a trained compression model"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="compressed_model/final_model",
        help="Directory containing the trained model",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Artificial intelligence is",
        help="Prompt for text generation",
    )
    parser.add_argument(
        "--max_length", type=int, default=150, help="Maximum length for generated text"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Temperature for text generation"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.9, help="Top-p sampling parameter"
    )

    args = parser.parse_args()

    print(f"Loading model from {args.model_dir}...")
    model = CompressibleLanguageModel.from_pretrained(
        args.model_dir, load_compression_layers=True
    )

    # Move model to GPU if available, but avoid MPS (Apple GPU) due to compatibility issues
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)

    print(f"Generating text with prompt: '{args.prompt}'")

    # Use the generate_text function from train_compression.py
    generated_text = generate_text(
        model=model,
        prompt=args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print("\nGenerated text:")
    print("-" * 50)
    print(generated_text)
    print("-" * 50)


if __name__ == "__main__":
    main()
