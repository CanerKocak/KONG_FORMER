import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import os
import json
from train_compression import CompressibleLanguageModel, TextDataset
from visualize import (
    visualize_token_embeddings,
    plot_compression_stats,
    visualize_attention_patterns,
    compare_original_vs_reconstructed,
    plot_residual_analysis,
    create_compression_dashboard,
)
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F


def calculate_perplexity(model, dataset, batch_size=4):
    """Calculate perplexity on a dataset."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for i in tqdm(
            range(0, len(dataset), batch_size), desc="Calculating perplexity"
        ):
            batch_indices = list(range(i, min(i + batch_size, len(dataset))))
            batch = [dataset[j] for j in batch_indices]

            # Prepare batch
            input_ids = torch.stack([item["input_ids"] for item in batch])
            attention_mask = torch.stack([item["attention_mask"] for item in batch])
            labels = input_ids.clone()

            # Move to device
            device = next(model.parameters()).device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )

            # Calculate loss
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
            total_loss += loss.item() * torch.sum(attention_mask).item()
            total_tokens += torch.sum(attention_mask).item()

    # Calculate perplexity
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return perplexity


def evaluate_compression(
    model_path: str,
    test_data_path: str,
    output_dir: str,
    batch_size: int = 4,
    max_samples: int = 10,
    generate_samples: bool = True,
    create_dashboard: bool = True,
):
    """
    Evaluate a trained compression model.

    Args:
        model_path: Path to the trained model directory
        test_data_path: Path to test data file
        output_dir: Directory to save evaluation results
        batch_size: Batch size for evaluation
        max_samples: Maximum number of samples to generate
        generate_samples: Whether to generate text samples
        create_dashboard: Whether to create a visualization dashboard
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load model configuration
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["base_model_name"])

    # Ensure padding token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print(f"Loading model from {model_path}...")
    model = CompressibleLanguageModel.load(model_path)

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Load test dataset
    print(f"Loading test data from {test_data_path}...")
    test_dataset = TextDataset(
        file_path=test_data_path,
        tokenizer=tokenizer,
        max_length=config.get("max_length", 128),
    )

    # Calculate perplexity
    print("Calculating perplexity...")
    perplexity = calculate_perplexity(model, test_dataset, batch_size)
    print(f"Perplexity: {perplexity:.4f}")

    # Get compression statistics
    print("Collecting compression statistics...")
    compression_stats = model.get_compression_stats()
    compression_stats["perplexity"] = perplexity

    # Save compression statistics
    stats_path = os.path.join(output_dir, "compression_stats.json")
    with open(stats_path, "w") as f:
        json.dump(compression_stats, f, indent=2)

    # Generate text samples
    sample_texts = {}
    if generate_samples:
        print("Generating text samples...")
        prompts = [
            "Neural networks",
            "Language models",
            "The transformer architecture",
            "Model compression techniques",
            "Training language models",
        ]

        for i, prompt in enumerate(prompts[:max_samples]):
            print(f"Generating sample {i+1}/{min(len(prompts), max_samples)}: {prompt}")

            # Instead of using model.generate_text, use the base model's generate method
            input_ids = tokenizer.encode(prompt, return_tensors="pt")

            # Generate text
            with torch.no_grad():
                output_sequences = model.base_model.generate(
                    input_ids=input_ids,
                    max_length=config.get("max_length", 128),
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            generated_text = tokenizer.decode(
                output_sequences[0], skip_special_tokens=True
            )
            sample_texts[f"sample_{i+1}"] = {
                "prompt": prompt,
                "generated_text": generated_text,
            }

            # Create text comparison visualization
            compare_path = os.path.join(output_dir, f"text_comparison_{i+1}.html")
            compare_original_vs_reconstructed(
                original_text=prompt,
                reconstructed_text=generated_text,
                output_path=compare_path,
            )

        # Save sample texts
        samples_path = os.path.join(output_dir, "text_samples.json")
        with open(samples_path, "w") as f:
            json.dump(sample_texts, f, indent=2)

    # Create token embedding visualization
    print("Creating token embedding visualization...")

    # Get token embeddings from a sample batch
    sample_batch = next(iter(DataLoader(test_dataset, batch_size=1)))
    input_ids = sample_batch["input_ids"]
    attention_mask = sample_batch["attention_mask"]

    # Forward pass with output_hidden_states=True
    with torch.no_grad():
        outputs = model.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

    # Extract token embeddings from the last hidden state
    token_embeddings = outputs.hidden_states[-1].squeeze(0)

    # Get token strings
    tokens = [tokenizer.decode([token_id]) for token_id in input_ids.squeeze(0)]

    # Get token importance and compressed indices
    token_importance = None
    compressed_indices = None
    if hasattr(model, "compression_layers") and len(model.compression_layers) > 0:
        first_layer = model.compression_layers[0]
        if hasattr(first_layer, "get_token_importance") and hasattr(
            first_layer, "get_compressed_indices"
        ):
            # Normalize embeddings for similarity calculation
            normalized_embeddings = F.normalize(token_embeddings, p=2, dim=1)

            # Get token importance and compressed indices
            token_importance = first_layer.get_token_importance(normalized_embeddings)

            # Get compression parameters from model or use defaults
            adaptive_threshold = getattr(model, "adaptive_threshold", False)
            percentile_threshold = getattr(model, "percentile_threshold", 95.0)
            min_similarity_threshold = getattr(model, "min_similarity_threshold", 0.8)

            # Call get_compressed_indices with only the normalized_embeddings parameter
            compressed_indices = first_layer.get_compressed_indices(
                normalized_embeddings
            )

    # Create visualization
    embedding_viz_path = os.path.join(output_dir, "token_embeddings.png")
    visualize_token_embeddings(
        embeddings=token_embeddings.cpu(),
        tokens=tokens,
        token_importance=(
            token_importance.cpu() if token_importance is not None else None
        ),
        compressed_indices=(
            compressed_indices if compressed_indices is not None else None
        ),
        output_path=embedding_viz_path,
        method="tsne",
        title="Token Embeddings with Compression",
    )

    # Visualize residuals if available
    if hasattr(model, "residuals") and model.residuals is not None:
        residuals_path = os.path.join(output_dir, "residual_analysis.png")
        plot_residual_analysis(
            residuals=model.residuals,
            gate_values=model.residual_gates,
            output_path=residuals_path,
        )

    # Visualize attention patterns if available
    if hasattr(outputs, "attentions") and outputs.attentions is not None:
        attention_path = os.path.join(output_dir, "attention_patterns.png")
        visualize_attention_patterns(
            attention_weights=outputs.attentions[-1][0],  # Last layer, first batch
            tokens=tokens,
            output_path=attention_path,
        )

    # Create compression dashboard
    if create_dashboard:
        print("Creating compression dashboard...")
        # Load compression history if available
        history_path = os.path.join(model_path, "compression_history.json")
        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                compression_history = json.load(f)
        else:
            # Create dummy history if not available
            compression_history = [compression_stats]

        # Get sample text for dashboard
        if sample_texts:
            first_sample = list(sample_texts.values())[0]
            dashboard_text = {
                "original": first_sample["prompt"],
                "reconstructed": first_sample["generated_text"],
            }
        else:
            dashboard_text = {
                "original": "No sample text available.",
                "reconstructed": "No sample text available.",
            }

        # Create dashboard
        dashboard_dir = os.path.join(output_dir, "dashboard")
        create_compression_dashboard(
            model_name=config["base_model_name"],
            stats=compression_stats,
            compression_history=compression_history,
            sample_text=dashboard_text,
            output_dir=dashboard_dir,
        )

    print(f"Evaluation complete. Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained compression model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model directory",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="sample_data.txt",
        help="Path to test data file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=5,
        help="Maximum number of samples to generate",
    )
    parser.add_argument(
        "--no_samples", action="store_true", help="Skip generating text samples"
    )
    parser.add_argument(
        "--no_dashboard",
        action="store_true",
        help="Skip creating visualization dashboard",
    )

    args = parser.parse_args()

    evaluate_compression(
        model_path=args.model_path,
        test_data_path=args.test_data,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        generate_samples=not args.no_samples,
        create_dashboard=not args.no_dashboard,
    )


if __name__ == "__main__":
    main()
