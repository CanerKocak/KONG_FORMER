import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer
import os
import json
from train_compression import CompressibleLanguageModel
from main import RecursiveCompressionLayer


class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, seq_length=64):
        self.tokenizer = tokenizer
        self.seq_length = seq_length

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        self.tokenized_text = tokenizer.encode(text)

    def __len__(self):
        return max(0, len(self.tokenized_text) - self.seq_length)

    def __getitem__(self, idx):
        chunk = self.tokenized_text[idx : idx + self.seq_length]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def validate_model(model, tokenizer, prompt="The quick brown fox", max_length=50):
    """Generate text to validate model quality during training"""
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    # Simple greedy decoding
    generated_ids = []
    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(input_ids)

            # Handle dictionary output format
            if isinstance(outputs, dict):
                next_token_logits = outputs["logits"][:, -1, :]
            else:
                # Handle tuple output format (fallback)
                next_token_logits = outputs[1][:, -1, :]

            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            generated_ids.append(next_token_id.item())
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)

    generated_text = tokenizer.decode(generated_ids)
    print(f"\nValidation generation:\nPrompt: {prompt}\nGenerated: {generated_text}\n")
    model.train()
    return generated_text


def train_model(
    model, train_dataloader, optimizer, epochs, device, output_dir, tokenizer=None
):
    model.train()
    criterion = nn.CrossEntropyLoss()

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Store compression stats
    compression_stats = {
        "epoch": [],
        "loss": [],
        "validation_loss": [],
        "compression_ratio": [],
    }

    # Add get_compression_stats method to model if it doesn't exist
    if not hasattr(model, "get_compression_stats"):

        def get_compression_stats(self):
            # Simple implementation that returns the max_compression_ratio if set
            if hasattr(self, "max_compression_ratio"):
                current_ratio = getattr(self, "compression_ratio", 1.0)
                return {
                    "compression_ratio": current_ratio,
                    "max_compression_ratio": self.max_compression_ratio,
                    "target_ratio": min(current_ratio, self.max_compression_ratio),
                }
            return {"compression_ratio": 1.0}

        # Add the method to the model
        import types

        model.get_compression_stats = types.MethodType(get_compression_stats, model)

    # Add update_compression_schedule method if it doesn't exist
    if not hasattr(model, "update_compression_schedule"):

        def update_compression_schedule(self, epoch, total_epochs):
            # Simple implementation that enforces max_compression_ratio
            if hasattr(self, "max_compression_ratio"):
                # Store current compression ratio (capped by max_compression_ratio)
                current_ratio = getattr(self, "compression_ratio", 1.0)
                self.compression_ratio = min(current_ratio, self.max_compression_ratio)

                # Pass max_compression_ratio to compression layers if they exist
                if hasattr(self, "compression_layers"):
                    for layer in self.compression_layers:
                        # Set max_compression_ratio on the layer
                        layer.max_compression_ratio = self.max_compression_ratio

        # Add the method to the model
        import types

        model.update_compression_schedule = types.MethodType(
            update_compression_schedule, model
        )

    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (x, y) in enumerate(train_dataloader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            # Forward pass with compression
            outputs = model(x)

            # Handle dictionary output format
            if isinstance(outputs, dict):
                logits = outputs["logits"]
                # If loss is already calculated by the model, use it
                if outputs["loss"] is not None:
                    loss = outputs["loss"]
                else:
                    # Calculate loss
                    loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            else:
                # Handle tuple output format (fallback)
                loss, logits = outputs[:2]
                if loss is None:
                    # Calculate loss
                    loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 5 == 0:
                print(
                    f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_dataloader)}, Loss: {loss.item():.4f}"
                )

                # Get compression stats
                stats = model.get_compression_stats()
                if stats and "compression_ratio" in stats:
                    print(
                        f"Current compression ratio: {stats['compression_ratio']:.2f}x"
                    )
                else:
                    print("Compression stats not available yet")

                # Update compression schedule if method exists
                if hasattr(model, "update_compression_schedule"):
                    model.update_compression_schedule(epoch, epochs)

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{epochs} completed, Avg Loss: {avg_loss:.4f}")

        # Validate model by generating text
        if tokenizer:
            validate_model(model, tokenizer)

        # Save model checkpoint
        checkpoint_path = os.path.join(output_dir, f"model_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), checkpoint_path)

        # Update compression stats
        compression_stats["epoch"].append(epoch + 1)
        compression_stats["loss"].append(avg_loss)
        compression_stats["validation_loss"].append(
            None
        )  # We're not calculating validation loss

        # Get compression ratio if available
        stats = model.get_compression_stats()
        if stats and "compression_ratio" in stats:
            compression_stats["compression_ratio"].append(stats["compression_ratio"])
        else:
            compression_stats["compression_ratio"].append(
                1.0
            )  # Default to no compression

        # Save compression stats
        with open(os.path.join(output_dir, "compression_stats.json"), "w") as f:
            json.dump(compression_stats, f, indent=4)

    # Save final model
    final_model_path = os.path.join(output_dir, "final_model.pt")
    torch.save(model.state_dict(), final_model_path)

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train a compressible language model with ultra-safe settings"
    )
    parser.add_argument(
        "--model_name", type=str, default="gpt2", help="Base model name"
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to training data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="Learning rate"
    )
    parser.add_argument(
        "--seq_length", type=int, default=64, help="Sequence length for training"
    )
    parser.add_argument(
        "--accumulation_steps", type=int, default=4, help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--validate_every", type=int, default=1000, help="Validate every N steps"
    )
    parser.add_argument(
        "--max_compression_ratio",
        type=float,
        default=8.0,
        help="Maximum allowed compression ratio",
    )

    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Create dataset and dataloader
    dataset = TextDataset(args.data_path, tokenizer, args.seq_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    print(f"Dataset size: {len(dataset)} samples")

    # Create model with ULTRA SAFE settings
    print("Creating model with ULTRA SAFE compression settings...")
    model = CompressibleLanguageModel(
        base_model_name=args.model_name,
        compression_layer_indices=[5],  # Only add compression after layer 5
        compression_factor=2,  # Changed from 1.5 to 2 (integer value for compatibility)
        similarity_threshold=0.92,  # Conservative threshold for stability
        freeze_base_model=False,  # Train the whole model
        use_residuals=True,  # Use residuals for better reconstruction
        use_progressive_compression=False,  # Disable progressive compression
    )

    # Store the max_compression_ratio as an attribute instead
    model.max_compression_ratio = args.max_compression_ratio

    # Initialize compression_ratio attribute
    model.compression_ratio = 1.0  # Start with no compression

    # Initialize max_compression_ratio on all compression layers
    if hasattr(model, "compression_layers"):
        print(
            f"Setting max compression ratio to {args.max_compression_ratio} on all compression layers"
        )
        for layer in model.compression_layers:
            layer.max_compression_ratio = args.max_compression_ratio

    # Add device property to model for compatibility
    model.device = device
    model.to(device)

    # Print model parameters
    print(
        f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Train model
    print("Starting training...")
    train_model(
        model, dataloader, optimizer, args.epochs, device, args.output_dir, tokenizer
    )

    print("Training completed!")


if __name__ == "__main__":
    main()
