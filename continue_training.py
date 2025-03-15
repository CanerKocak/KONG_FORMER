import torch
import argparse
from train_compression import CompressibleLanguageModel, TextDataset, train


def main():
    parser = argparse.ArgumentParser(
        description="Continue training a compression model from a checkpoint"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory containing the trained model checkpoint",
    )
    parser.add_argument(
        "--train_file", type=str, required=True, help="Path to training data file"
    )
    parser.add_argument(
        "--val_file", type=str, default=None, help="Path to validation data file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./continued_training",
        help="Output directory for model checkpoints",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate (usually lower than initial training)",
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=0, help="Number of warmup steps"
    )
    parser.add_argument(
        "--max_length", type=int, default=128, help="Maximum sequence length"
    )
    parser.add_argument(
        "--save_steps", type=int, default=100, help="Save checkpoint every X steps"
    )
    parser.add_argument(
        "--eval_steps", type=int, default=50, help="Evaluate every X steps"
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

    print(f"Loading training data from {args.train_file}...")
    with open(args.train_file, "r", encoding="utf-8") as f:
        train_texts = [line.strip() for line in f]

    train_dataset = TextDataset(
        train_texts, model.tokenizer, max_length=args.max_length
    )

    # Load validation data if provided
    val_dataset = None
    if args.val_file:
        print(f"Loading validation data from {args.val_file}...")
        with open(args.val_file, "r", encoding="utf-8") as f:
            val_texts = [line.strip() for line in f]
        val_dataset = TextDataset(
            val_texts, model.tokenizer, max_length=args.max_length
        )

    print(f"Continuing training for {args.epochs} epochs...")
    model = train(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        output_dir=args.output_dir,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
    )

    print(f"Training complete. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
