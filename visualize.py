import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
from typing import Dict, List, Tuple, Optional, Union
import seaborn as sns


def setup_plot_style():
    """Set up a clean, modern plot style."""
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("viridis")
    plt.rcParams["figure.figsize"] = (12, 8)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial"]
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 12


def visualize_token_embeddings(
    embeddings: torch.Tensor,
    tokens: List[str],
    centroids: Optional[torch.Tensor] = None,
    token_importance: Optional[torch.Tensor] = None,
    compressed_indices: Optional[List[int]] = None,
    output_path: str = "token_embeddings.png",
    method: str = "tsne",
    title: str = "Token Embeddings Visualization",
) -> None:
    """
    Visualize token embeddings in 2D space using dimensionality reduction.

    Args:
        embeddings: Token embeddings tensor [num_tokens, embedding_dim]
        tokens: List of token strings corresponding to embeddings
        centroids: Optional tensor of centroid embeddings [num_centroids, embedding_dim]
        token_importance: Optional tensor of importance scores for each token
        compressed_indices: Optional list of indices showing which tokens were compressed
        output_path: Path to save the visualization
        method: Dimensionality reduction method ('tsne' or 'pca')
        title: Plot title
    """
    setup_plot_style()

    # Convert embeddings to numpy for dimensionality reduction
    embeddings_np = embeddings.detach().cpu().numpy()

    # Apply dimensionality reduction
    if method.lower() == "tsne":
        reducer = TSNE(
            n_components=2, random_state=42, perplexity=min(30, len(embeddings_np) - 1)
        )
    else:  # PCA
        reducer = PCA(n_components=2, random_state=42)

    # Reduce embeddings to 2D
    embeddings_2d = reducer.fit_transform(embeddings_np)

    # Create figure
    plt.figure(figsize=(14, 10))

    # Determine point colors based on token importance if available
    if token_importance is not None:
        importance = token_importance.detach().cpu().numpy()
        colors = importance
        cmap = "viridis"
        norm = plt.Normalize(importance.min(), importance.max())
    else:
        colors = "blue"
        cmap = None
        norm = None

    # Plot token embeddings
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=colors,
        cmap=cmap,
        norm=norm,
        alpha=0.7,
        s=100,
    )

    # Add token labels
    for i, token in enumerate(tokens):
        plt.annotate(
            token,
            (embeddings_2d[i, 0], embeddings_2d[i, 1]),
            fontsize=8,
            alpha=0.7,
            xytext=(5, 5),
            textcoords="offset points",
        )

    # Highlight compressed tokens if indices are provided
    if compressed_indices is not None:
        plt.scatter(
            embeddings_2d[compressed_indices, 0],
            embeddings_2d[compressed_indices, 1],
            s=150,
            facecolors="none",
            edgecolors="red",
            linewidths=2,
            label="Compressed Tokens",
        )

    # Plot centroids if provided
    if centroids is not None:
        centroids_np = centroids.detach().cpu().numpy()

        # Project centroids to the same 2D space
        if method.lower() == "tsne":
            # For t-SNE, we need to use a different approach since it's not a linear projection
            # We'll use PCA first to get to the same dimensionality, then apply the t-SNE transform
            all_embeddings = np.vstack([embeddings_np, centroids_np])
            all_embeddings_2d = reducer.fit_transform(all_embeddings)
            centroids_2d = all_embeddings_2d[len(embeddings_np) :]
        else:
            # For PCA, we can directly transform the centroids
            centroids_2d = reducer.transform(centroids_np)

        plt.scatter(
            centroids_2d[:, 0],
            centroids_2d[:, 1],
            c="red",
            marker="*",
            s=200,
            label="Centroids",
        )

    # Add colorbar if using token importance
    if token_importance is not None:
        cbar = plt.colorbar(scatter)
        cbar.set_label("Token Importance")

    # Add legend, title and labels
    plt.legend(loc="upper right")
    plt.title(title)
    plt.xlabel(f"{method.upper()} Dimension 1")
    plt.ylabel(f"{method.upper()} Dimension 2")

    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Token embeddings visualization saved to {output_path}")


def plot_compression_stats(stats_history, output_path):
    """
    Plot compression statistics over time.

    Args:
        stats_history: List of dictionaries containing compression stats at each step or a JSON string
        output_path: Path to save the visualization
    """
    setup_plot_style()

    # Handle the case where stats_history is a string (JSON)
    if isinstance(stats_history, str):
        try:
            import json

            stats_history = json.loads(stats_history)
        except json.JSONDecodeError:
            # If it's not valid JSON, create a dummy history
            stats_history = [
                {"compression_ratio": 0, "token_redundancy": 0, "perplexity": 0}
            ]

    # Ensure stats_history is a list
    if not isinstance(stats_history, list):
        stats_history = [stats_history]

    # Extract metrics from history
    steps = list(range(len(stats_history)))
    compression_ratios = [stats.get("compression_ratio", 0) for stats in stats_history]
    token_redundancy = [stats.get("token_redundancy", 0) for stats in stats_history]
    perplexity = [stats.get("perplexity", 0) for stats in stats_history]

    # Create figure with multiple subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

    # Plot compression ratio
    axes[0].plot(steps, compression_ratios, "b-", linewidth=2)
    axes[0].set_ylabel("Compression Ratio")
    axes[0].set_title("Compression Ratio vs. Training Steps")
    axes[0].grid(True)

    # Plot token redundancy
    axes[1].plot(steps, token_redundancy, "g-", linewidth=2)
    axes[1].set_ylabel("Token Redundancy")
    axes[1].set_title("Token Redundancy vs. Training Steps")
    axes[1].grid(True)

    # Plot perplexity
    axes[2].plot(steps, perplexity, "r-", linewidth=2)
    axes[2].set_ylabel("Perplexity")
    axes[2].set_title("Perplexity vs. Training Steps")
    axes[2].set_xlabel("Training Steps")
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Compression statistics plot saved to {output_path}")


def visualize_attention_patterns(
    attention_weights: torch.Tensor,
    tokens: List[str],
    output_path: str = "attention_patterns.png",
    layer_idx: Optional[int] = None,
) -> None:
    """
    Visualize attention patterns between tokens.

    Args:
        attention_weights: Attention weights tensor [num_heads, seq_len, seq_len]
        tokens: List of token strings
        output_path: Path to save the visualization
        layer_idx: Optional layer index for multi-layer visualization
    """
    setup_plot_style()

    # Get attention weights as numpy array
    attn = attention_weights.detach().cpu().numpy()

    # Average across heads if multiple heads
    if len(attn.shape) > 2:
        attn = attn.mean(axis=0)

    # Create figure
    plt.figure(figsize=(12, 10))

    # Plot attention heatmap
    sns.heatmap(
        attn, annot=False, cmap="viridis", xticklabels=tokens, yticklabels=tokens
    )

    # Add title and labels
    layer_str = f" (Layer {layer_idx})" if layer_idx is not None else ""
    plt.title(f"Attention Patterns{layer_str}")
    plt.xlabel("Target Tokens")
    plt.ylabel("Source Tokens")

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Attention pattern visualization saved to {output_path}")


def compare_original_vs_reconstructed(
    original_text: str,
    reconstructed_text: str,
    output_path: str = "text_comparison.html",
) -> None:
    """
    Create an HTML visualization comparing original and reconstructed text.

    Args:
        original_text: Original text string
        reconstructed_text: Reconstructed text string
        output_path: Path to save the HTML visualization
    """
    # Split texts into words
    original_words = original_text.split()
    reconstructed_words = reconstructed_text.split()

    # Create HTML content
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Text Comparison</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .container { display: flex; }
            .column { flex: 1; padding: 20px; }
            .header { font-weight: bold; font-size: 18px; margin-bottom: 10px; }
            .match { color: green; }
            .mismatch { color: red; }
            .highlight { background-color: yellow; }
        </style>
    </head>
    <body>
        <h1>Original vs. Reconstructed Text Comparison</h1>
        <div class="container">
            <div class="column">
                <div class="header">Original Text</div>
                <div id="original">
    """

    # Add original text with highlighting
    for word in original_words:
        if word in reconstructed_words:
            html_content += f'<span class="match">{word}</span> '
        else:
            html_content += f'<span class="mismatch">{word}</span> '

    html_content += """
                </div>
            </div>
            <div class="column">
                <div class="header">Reconstructed Text</div>
                <div id="reconstructed">
    """

    # Add reconstructed text with highlighting
    for word in reconstructed_words:
        if word in original_words:
            html_content += f'<span class="match">{word}</span> '
        else:
            html_content += f'<span class="mismatch">{word}</span> '

    html_content += """
                </div>
            </div>
        </div>
        <div style="margin-top: 30px;">
            <h2>Statistics</h2>
            <p>Original word count: <strong id="original-count"></strong></p>
            <p>Reconstructed word count: <strong id="reconstructed-count"></strong></p>
            <p>Matching words: <strong id="matching-count"></strong></p>
            <p>Match percentage: <strong id="match-percentage"></strong>%</p>
        </div>
        <script>
            // Calculate statistics
            const originalWords = document.getElementById('original').innerText.split(' ').filter(w => w.trim());
            const reconstructedWords = document.getElementById('reconstructed').innerText.split(' ').filter(w => w.trim());
            const matchingWords = originalWords.filter(w => reconstructedWords.includes(w));
            
            // Update statistics
            document.getElementById('original-count').innerText = originalWords.length;
            document.getElementById('reconstructed-count').innerText = reconstructedWords.length;
            document.getElementById('matching-count').innerText = matchingWords.length;
            document.getElementById('match-percentage').innerText = 
                (matchingWords.length / Math.max(originalWords.length, 1) * 100).toFixed(2);
        </script>
    </body>
    </html>
    """

    # Write HTML to file
    with open(output_path, "w") as f:
        f.write(html_content)

    print(f"Text comparison visualization saved to {output_path}")


def plot_residual_analysis(
    residuals: torch.Tensor,
    gate_values: Optional[torch.Tensor] = None,
    output_path: str = "residual_analysis.png",
) -> None:
    """
    Analyze and visualize residual vectors.

    Args:
        residuals: Tensor of residual vectors [num_tokens, embedding_dim]
        gate_values: Optional tensor of gate values for residuals
        output_path: Path to save the visualization
    """
    setup_plot_style()

    # Convert to numpy
    residuals_np = residuals.detach().cpu().numpy()

    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Residual magnitude distribution
    residual_norms = np.linalg.norm(residuals_np, axis=1)
    axes[0, 0].hist(residual_norms, bins=30, alpha=0.7, color="blue")
    axes[0, 0].set_title("Residual Magnitude Distribution")
    axes[0, 0].set_xlabel("Residual Norm")
    axes[0, 0].set_ylabel("Frequency")

    # Plot 2: Residual component heatmap
    im = axes[0, 1].imshow(
        residuals_np[: min(50, residuals_np.shape[0])], aspect="auto", cmap="coolwarm"
    )
    axes[0, 1].set_title("Residual Components Heatmap (First 50 Tokens)")
    axes[0, 1].set_xlabel("Embedding Dimension")
    axes[0, 1].set_ylabel("Token Index")
    plt.colorbar(im, ax=axes[0, 1])

    # Plot 3: PCA of residuals
    if residuals_np.shape[0] > 1:  # Need at least 2 samples for PCA
        pca = PCA(n_components=2)
        residuals_2d = pca.fit_transform(residuals_np)
        axes[1, 0].scatter(residuals_2d[:, 0], residuals_2d[:, 1], alpha=0.7)
        axes[1, 0].set_title("PCA of Residual Vectors")
        axes[1, 0].set_xlabel("PC1")
        axes[1, 0].set_ylabel("PC2")

        # Add variance explained
        var_explained = pca.explained_variance_ratio_
        axes[1, 0].text(
            0.05,
            0.95,
            f"Variance explained: {var_explained[0]:.2f}, {var_explained[1]:.2f}",
            transform=axes[1, 0].transAxes,
            verticalalignment="top",
        )
    else:
        axes[1, 0].text(
            0.5,
            0.5,
            "Not enough samples for PCA",
            ha="center",
            va="center",
            transform=axes[1, 0].transAxes,
        )

    # Plot 4: Gate values distribution (if provided)
    if gate_values is not None:
        gate_np = gate_values.detach().cpu().numpy()
        if len(gate_np.shape) > 1:
            # If gate values are per-dimension, plot mean gate value per token
            gate_np = gate_np.mean(axis=1)

        axes[1, 1].hist(gate_np, bins=30, alpha=0.7, color="green")
        axes[1, 1].set_title("Gate Values Distribution")
        axes[1, 1].set_xlabel("Gate Value")
        axes[1, 1].set_ylabel("Frequency")
    else:
        axes[1, 1].text(
            0.5,
            0.5,
            "No gate values provided",
            ha="center",
            va="center",
            transform=axes[1, 1].transAxes,
        )

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Residual analysis visualization saved to {output_path}")


def create_compression_dashboard(
    model_name: str,
    stats: Dict[str, float],
    compression_history: List[Dict[str, float]],
    sample_text: Dict[str, str],
    output_dir: str = "compression_dashboard",
) -> None:
    """
    Create a comprehensive dashboard of compression visualizations.

    Args:
        model_name: Name of the model being analyzed
        stats: Dictionary of current compression statistics
        compression_history: List of compression statistics over time
        sample_text: Dictionary with 'original' and 'reconstructed' text samples
        output_dir: Directory to save the dashboard files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Compression Model Dashboard - {model_name}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; }}
            .header {{ background-color: #4CAF50; color: white; padding: 20px; text-align: center; }}
            .container {{ display: flex; flex-wrap: wrap; padding: 20px; }}
            .card {{ box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2); margin: 10px; padding: 20px; flex: 1; min-width: 300px; }}
            .stats-container {{ display: flex; flex-wrap: wrap; }}
            .stat-card {{ background-color: #f1f1f1; border-radius: 5px; margin: 5px; padding: 15px; flex: 1; min-width: 150px; text-align: center; }}
            .stat-value {{ font-size: 24px; font-weight: bold; margin: 10px 0; }}
            .stat-label {{ color: #666; }}
            img {{ max-width: 100%; height: auto; }}
            .text-sample {{ background-color: #f9f9f9; border-left: 5px solid #4CAF50; padding: 10px; margin: 10px 0; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Compression Model Dashboard</h1>
            <h2>{model_name}</h2>
        </div>
        
        <div class="container">
            <div class="card">
                <h2>Compression Statistics</h2>
                <div class="stats-container">
    """

    # Add stats cards
    for key, value in stats.items():
        formatted_value = f"{value:.4f}" if isinstance(value, float) else value
        html_content += f"""
                    <div class="stat-card">
                        <div class="stat-label">{key.replace('_', ' ').title()}</div>
                        <div class="stat-value">{formatted_value}</div>
                    </div>
        """

    html_content += """
                </div>
            </div>
        </div>
        
        <div class="container">
            <div class="card">
                <h2>Compression History</h2>
                <img src="compression_stats.png" alt="Compression Statistics">
            </div>
        </div>
        
        <div class="container">
            <div class="card">
                <h2>Token Embeddings</h2>
                <img src="token_embeddings.png" alt="Token Embeddings">
            </div>
            
            <div class="card">
                <h2>Residual Analysis</h2>
                <img src="residual_analysis.png" alt="Residual Analysis">
            </div>
        </div>
        
        <div class="container">
            <div class="card">
                <h2>Text Samples</h2>
                <h3>Original Text</h3>
                <div class="text-sample">
    """

    # Add original text
    html_content += sample_text.get("original", "No sample text provided").replace(
        "\n", "<br>"
    )

    html_content += """
                </div>
                <h3>Reconstructed Text</h3>
                <div class="text-sample">
    """

    # Add reconstructed text
    html_content += sample_text.get(
        "reconstructed", "No reconstructed text provided"
    ).replace("\n", "<br>")

    html_content += """
                </div>
                <p><a href="text_comparison.html" target="_blank">View detailed text comparison</a></p>
            </div>
        </div>
    </body>
    </html>
    """

    # Write HTML to file
    with open(os.path.join(output_dir, "index.html"), "w") as f:
        f.write(html_content)

    # Create supporting visualizations
    plot_compression_stats(
        compression_history, os.path.join(output_dir, "compression_stats.png")
    )

    # Create text comparison
    compare_original_vs_reconstructed(
        sample_text.get("original", ""),
        sample_text.get("reconstructed", ""),
        os.path.join(output_dir, "text_comparison.html"),
    )

    print(f"Compression dashboard created in {output_dir}")


if __name__ == "__main__":
    # Example usage
    print("This module provides visualization functions for compression analysis.")
    print("Import and use these functions in your training or evaluation scripts.")
