import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


def analyze_compression_stats(stats_file):
    """Analyze compression statistics from a JSON file."""
    print(f"Loading compression statistics from {stats_file}")
    with open(stats_file, "r") as f:
        stats = json.load(f)

    # Extract data
    steps = stats["steps"]
    thresholds = stats["thresholds"]
    compression_ratios = stats["compression_ratios"]
    loss_values = stats["loss_values"]
    val_loss_values = stats.get("val_loss_values", [])

    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Number of training steps: {len(steps)}")
    print(f"Final loss: {loss_values[-1]:.4f}")
    print(f"Average compression ratio: {np.mean(compression_ratios):.2f}x")

    # Print threshold information
    print("\nThreshold Information:")
    for i in range(len(thresholds[0])):
        layer_thresholds = [t[i] for t in thresholds]
        print(
            f"Layer {i}: Initial={layer_thresholds[0]:.4f}, Final={layer_thresholds[-1]:.4f}, Change={layer_thresholds[-1]-layer_thresholds[0]:.4f}"
        )

    # Create output directory for plots
    output_dir = os.path.dirname(stats_file) + "/analysis"
    os.makedirs(output_dir, exist_ok=True)

    # Plot thresholds over time
    plt.figure(figsize=(10, 6))
    for i in range(len(thresholds[0])):
        layer_thresholds = [t[i] for t in thresholds]
        plt.plot(steps, layer_thresholds, label=f"Layer {i}")

    plt.xlabel("Training Steps")
    plt.ylabel("Similarity Threshold")
    plt.title("Adaptive Thresholds During Training")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "thresholds.png"))

    # Plot compression ratio over time
    plt.figure(figsize=(10, 6))
    plt.plot(steps, compression_ratios)
    plt.xlabel("Training Steps")
    plt.ylabel("Compression Ratio")
    plt.title("Compression Ratio During Training")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "compression_ratio.png"))

    # Plot loss over time
    plt.figure(figsize=(10, 6))
    plt.plot(steps, loss_values, label="Training Loss")

    # Add validation loss if available
    if val_loss_values:
        # Interpolate validation steps to match training steps
        val_steps = np.linspace(min(steps), max(steps), len(val_loss_values))
        plt.plot(val_steps, val_loss_values, label="Validation Loss")

    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Loss During Training")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "loss.png"))

    # Plot loss vs compression ratio
    plt.figure(figsize=(10, 6))
    plt.scatter(compression_ratios, loss_values)
    plt.xlabel("Compression Ratio")
    plt.ylabel("Loss")
    plt.title("Loss vs Compression Ratio")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "loss_vs_compression.png"))

    print(f"\nAnalysis plots saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze compression statistics")
    parser.add_argument(
        "--stats_file",
        type=str,
        default="./enhanced_model/compression_stats.json",
        help="Path to the compression statistics JSON file",
    )

    args = parser.parse_args()
    analyze_compression_stats(args.stats_file)
