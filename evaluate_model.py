import torch
import argparse
import numpy as np
from train_compression import CompressibleLanguageModel, TextDataset, evaluate
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained compression model")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory containing the trained model",
    )
    parser.add_argument(
        "--test_file", type=str, required=True, help="Path to test data file"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--max_length", type=int, default=128, help="Maximum sequence length"
    )

    args = parser.parse_args()

    print(f"Loading model from {args.model_dir}...")
    model = CompressibleLanguageModel.load(args.model_dir)

    # Move model to GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model.to(device)

    print(f"Loading test data from {args.test_file}...")
    with open(args.test_file, "r", encoding="utf-8") as f:
        test_texts = [line.strip() for line in f]

    test_dataset = TextDataset(test_texts, model.tokenizer, max_length=args.max_length)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    print("Evaluating model...")
    model.eval()  # Set model to evaluation mode

    # Calculate perplexity
    test_loss = evaluate(model, test_loader)
    perplexity = np.exp(test_loss)

    print("\nEvaluation Results:")
    print("-" * 50)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")
    print("-" * 50)

    # Get compression statistics
    compression_stats = model.get_compression_stats()
    if compression_stats:
        print("\nCompression Statistics:")
        print("-" * 50)
        for key, value in compression_stats.items():
            print(f"{key}: {value:.4f}")
        print("-" * 50)


if __name__ == "__main__":
    main()
