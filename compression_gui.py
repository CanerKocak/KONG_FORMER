import gradio as gr
import torch
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image
import seaborn as sns
from train_compression import CompressibleLanguageModel, generate_text
from visualize import (
    setup_plot_style,
    plot_compression_stats,
    visualize_token_embeddings,
)
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import psutil  # Added for memory profiling

# Set up the plot style
setup_plot_style()

# Global variables
loaded_model = None
model_path = None
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


def load_model(model_dir: str) -> Tuple[str, str]:
    """Load a trained compression model from the specified directory."""
    global loaded_model, model_path

    try:
        if not os.path.exists(model_dir):
            return f"Error: Directory {model_dir} does not exist", ""

        # Load the model
        loaded_model = CompressibleLanguageModel.from_pretrained(model_dir)
        loaded_model.to(device)
        model_path = model_dir

        # Get model info
        compression_stats = loaded_model.get_compression_stats()
        compression_ratio = compression_stats.get("compression_ratio", "N/A")

        # Get model parameters
        num_layers = (
            len(loaded_model.compression_layers)
            if hasattr(loaded_model, "compression_layers")
            else 0
        )
        base_model = (
            loaded_model.base_model_name
            if hasattr(loaded_model, "base_model_name")
            else "Unknown"
        )

        model_info = f"""
        Model loaded successfully from {model_dir}
        Base model: {base_model}
        Device: {device}
        Number of compression layers: {num_layers}
        Current compression ratio: {compression_ratio}
        """

        return "Model loaded successfully!", model_info

    except Exception as e:
        return f"Error loading model: {str(e)}", ""


def top_p_filtering(logits, top_p=0.9):
    """Filter logits using nucleus (top-p) sampling."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = False

    # Create a new tensor with the same shape as logits filled with False
    indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)

    # For each row in the batch, set the indices to remove
    for i in range(logits.size(0)):
        indices_to_remove[i, sorted_indices[i][sorted_indices_to_remove[i]]] = True

    logits[indices_to_remove] = -float("Inf")
    return logits


def get_memory_usage_mb():
    """Get current memory usage of the process in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def stream_generate_with_memory(
    prompt: str,
    max_length: int = 200,
    temperature: float = 0.7,
    top_p: float = 0.9,
    enable_compression: bool = True,
):
    """Generate text with streaming output and memory usage tracking."""
    global loaded_model

    if loaded_model is None:
        yield "Please load a model first.", 0, 0, ""
        return

    # Enable/disable compression
    if hasattr(loaded_model, "compression_layers"):
        for layer in loaded_model.compression_layers:
            layer.compression_enabled = enable_compression

    # Tokenize the prompt
    tokenizer = loaded_model.tokenizer
    input_text = prompt

    # Track initial memory usage
    initial_memory = get_memory_usage_mb()
    start_time = time.time()

    # Generate text incrementally
    generated_text = prompt

    for i in range(max_length):
        # Tokenize the current text
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

        # Forward pass through the model
        with torch.no_grad():
            outputs = loaded_model(input_ids)

            # Extract logits
            if isinstance(outputs, dict):
                logits = outputs.get("logits", outputs.get("last_hidden_state", None))
            else:
                logits = outputs.logits if hasattr(outputs, "logits") else outputs

            if logits is None:
                break

            # Get the last token's logits
            next_token_logits = logits[:, -1, :]

            # Apply filtering and sampling
            filtered_logits = top_p_filtering(next_token_logits, top_p=top_p)
            probs = torch.softmax(filtered_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            # Stop if we generate an EOS token
            if next_token == tokenizer.eos_token_id:
                break

            # Decode the next token and add it to the generated text
            next_token_text = tokenizer.decode([next_token], skip_special_tokens=True)
            generated_text += next_token_text

            # Update input for next iteration - use the full text to maintain context
            # This is less efficient but works without past_key_values
            input_text = generated_text

            # Calculate memory usage and elapsed time
            current_memory = get_memory_usage_mb()
            memory_overhead = current_memory - initial_memory
            elapsed_time = time.time() - start_time

            # Get compression stats
            stats_text = ""
            if enable_compression and hasattr(loaded_model, "get_compression_stats"):
                stats = loaded_model.get_compression_stats()
                stats_text = (
                    f"Compression ratio: {stats.get('compression_ratio', 'N/A')}\n"
                )
                if "max_compression_ratio" in stats:
                    stats_text += f"Max compression ratio: {stats.get('max_compression_ratio', 'N/A')}\n"
                if "target_ratio" in stats:
                    stats_text += f"Target ratio: {stats.get('target_ratio', 'N/A')}\n"

            # Yield the current state every few tokens or at the end
            if (
                i % 3 == 0
                or i == max_length - 1
                or next_token == tokenizer.eos_token_id
            ):
                yield generated_text, memory_overhead, elapsed_time, stats_text


def compare_generation(
    prompt: str, max_length: int = 200, temperature: float = 0.7, top_p: float = 0.9
) -> Tuple[str, str, float, float, float, str]:
    """Compare text generation with and without compression."""
    global loaded_model

    if loaded_model is None:
        return "Please load a model first.", "", 0.0, 0.0, 0.0, ""

    # Generate with compression
    if hasattr(loaded_model, "compression_layers"):
        for layer in loaded_model.compression_layers:
            layer.compression_enabled = True

    start_time = time.time()
    compressed_text = generate_text(
        loaded_model,
        prompt,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
    )
    compressed_time = time.time() - start_time

    # Generate without compression
    if hasattr(loaded_model, "compression_layers"):
        for layer in loaded_model.compression_layers:
            layer.compression_enabled = False

    start_time = time.time()
    uncompressed_text = generate_text(
        loaded_model,
        prompt,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
    )
    uncompressed_time = time.time() - start_time

    # Calculate speedup
    speedup = uncompressed_time / compressed_time if compressed_time > 0 else 0

    # Get compression stats
    stats_text = ""
    if hasattr(loaded_model, "get_compression_stats"):
        # Re-enable compression to get stats
        if hasattr(loaded_model, "compression_layers"):
            for layer in loaded_model.compression_layers:
                layer.compression_enabled = True

        stats = loaded_model.get_compression_stats()
        stats_text = f"Compression ratio: {stats.get('compression_ratio', 'N/A')}\n"
        if "max_compression_ratio" in stats:
            stats_text += (
                f"Max compression ratio: {stats.get('max_compression_ratio', 'N/A')}\n"
            )
        if "target_ratio" in stats:
            stats_text += f"Target ratio: {stats.get('target_ratio', 'N/A')}\n"

    return (
        compressed_text,
        uncompressed_text,
        compressed_time,
        uncompressed_time,
        speedup,
        stats_text,
    )


def update_compression_params(
    similarity_threshold: float = 0.75, max_compression_ratio: float = 8.0
) -> str:
    """Update compression parameters of the loaded model."""
    global loaded_model

    if loaded_model is None:
        return "Please load a model first."

    try:
        # Update parameters for all compression layers
        if hasattr(loaded_model, "compression_layers"):
            for layer in loaded_model.compression_layers:
                layer.base_threshold = similarity_threshold
                layer.max_compression_ratio = max_compression_ratio

        # Also set at the model level if applicable
        if hasattr(loaded_model, "max_compression_ratio"):
            loaded_model.max_compression_ratio = max_compression_ratio

        return f"Parameters updated: similarity_threshold={similarity_threshold}, max_compression_ratio={max_compression_ratio}"

    except Exception as e:
        return f"Error updating parameters: {str(e)}"


def visualize_compression_stats() -> Optional[Image.Image]:
    """Visualize compression statistics from the model's output directory."""
    global model_path

    if model_path is None or not os.path.exists(model_path):
        return None

    # Look for compression_stats.json in the model directory
    stats_path = os.path.join(model_path, "compression_stats.json")
    if not os.path.exists(stats_path):
        return None

    try:
        # Create a plot in memory
        plt.figure(figsize=(10, 6))
        plot_compression_stats(stats_path, output_path=None)

        # Save the plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)

        # Convert to PIL Image
        img = Image.open(buf)
        plt.close()

        return img

    except Exception as e:
        print(f"Error visualizing stats: {str(e)}")
        return None


def plot_layer_thresholds() -> Optional[Image.Image]:
    """Plot the similarity thresholds for each compression layer."""
    global loaded_model

    if loaded_model is None or not hasattr(loaded_model, "compression_layers"):
        return None

    try:
        # Get thresholds from each layer
        layers = []
        base_thresholds = []
        current_thresholds = []

        for i, layer in enumerate(loaded_model.compression_layers):
            layers.append(f"Layer {i+1}")
            base_thresholds.append(
                layer.base_threshold if hasattr(layer, "base_threshold") else 0.0
            )
            current_thresholds.append(
                layer.current_threshold
                if hasattr(layer, "current_threshold")
                else layer.base_threshold
            )

        # Create a DataFrame
        df = pd.DataFrame(
            {
                "Layer": layers,
                "Base Threshold": base_thresholds,
                "Current Threshold": current_thresholds,
            }
        )

        # Create a plot
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x="Layer",
            y="value",
            hue="variable",
            data=pd.melt(
                df,
                id_vars=["Layer"],
                value_vars=["Base Threshold", "Current Threshold"],
            ),
        )
        plt.title("Similarity Thresholds by Layer")
        plt.ylabel("Threshold Value")
        plt.ylim(0, 1)

        # Save the plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)

        # Convert to PIL Image
        img = Image.open(buf)
        plt.close()

        return img

    except Exception as e:
        print(f"Error plotting thresholds: {str(e)}")
        return None


def visualize_token_embeddings_gui(
    text: str, method: str = "tsne", layer_idx: int = 0
) -> Optional[Image.Image]:
    """Visualize token embeddings before and after compression."""
    global loaded_model

    if loaded_model is None or not hasattr(loaded_model, "compression_layers"):
        return None

    try:
        # Tokenize the input text
        tokenizer = loaded_model.tokenizer
        input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
        tokens = [tokenizer.decode([id]) for id in input_ids[0].tolist()]

        # Get the original embeddings
        with torch.no_grad():
            # Get embeddings from the model
            inputs_embeds = loaded_model.base_model.get_input_embeddings()(input_ids)

            # Forward pass through the base model to get hidden states
            outputs = loaded_model.base_model(
                inputs_embeds=inputs_embeds, output_hidden_states=True
            )
            hidden_states = outputs.hidden_states

            # Get the hidden states at the specified layer
            if layer_idx < len(hidden_states):
                original_embeddings = hidden_states[layer_idx][0].cpu()
            else:
                # Fallback to the last layer if index is out of range
                original_embeddings = hidden_states[-1][0].cpu()

        # Get the compressed embeddings
        # Enable compression
        for layer in loaded_model.compression_layers:
            layer.compression_enabled = True

        with torch.no_grad():
            # Forward pass with compression
            outputs = loaded_model(input_ids, output_hidden_states=True)

            # Get the compressed hidden states
            if isinstance(outputs, dict) and "hidden_states" in outputs:
                compressed_hidden_states = outputs["hidden_states"]
            else:
                # Fallback if hidden_states not in outputs
                compressed_hidden_states = None

            # Get the compressed embeddings at the specified layer
            if compressed_hidden_states and layer_idx < len(compressed_hidden_states):
                compressed_embeddings = compressed_hidden_states[layer_idx][0].cpu()
            else:
                # Fallback to the original embeddings if compressed not available
                compressed_embeddings = original_embeddings

        # Create a plot
        plt.figure(figsize=(15, 7))

        # Plot original embeddings
        plt.subplot(1, 2, 1)
        if method == "tsne":
            # Apply t-SNE
            tsne = TSNE(
                n_components=2, random_state=42, perplexity=min(30, len(tokens) - 1)
            )
            embeddings_2d = tsne.fit_transform(original_embeddings)
        else:
            # Apply PCA
            pca = PCA(n_components=2, random_state=42)
            embeddings_2d = pca.fit_transform(original_embeddings)

        plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=range(len(tokens)),
            cmap="viridis",
        )
        for i, token in enumerate(tokens):
            plt.annotate(token, (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=8)
        plt.title(f"Original Embeddings ({method.upper()})")

        # Plot compressed embeddings
        plt.subplot(1, 2, 2)
        if method == "tsne":
            # Apply t-SNE
            tsne = TSNE(
                n_components=2, random_state=42, perplexity=min(30, len(tokens) - 1)
            )
            embeddings_2d = tsne.fit_transform(compressed_embeddings)
        else:
            # Apply PCA
            pca = PCA(n_components=2, random_state=42)
            embeddings_2d = pca.fit_transform(compressed_embeddings)

        plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=range(len(tokens)),
            cmap="viridis",
        )
        for i, token in enumerate(tokens):
            plt.annotate(token, (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=8)
        plt.title(f"Compressed Embeddings ({method.upper()})")

        plt.tight_layout()

        # Save the plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150)
        buf.seek(0)

        # Convert to PIL Image
        img = Image.open(buf)
        plt.close()

        return img

    except Exception as e:
        print(f"Error visualizing token embeddings: {str(e)}")
        import traceback

        traceback.print_exc()
        return None


def visualize_compression_clusters(
    text: str, layer_idx: int = 0
) -> Optional[Image.Image]:
    """Visualize token clusters formed during compression."""
    global loaded_model

    if loaded_model is None or not hasattr(loaded_model, "compression_layers"):
        return None

    try:
        # Tokenize the input text
        tokenizer = loaded_model.tokenizer
        input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
        tokens = [tokenizer.decode([id]) for id in input_ids[0].tolist()]

        # Get the embeddings
        with torch.no_grad():
            # Get embeddings from the model
            inputs_embeds = loaded_model.base_model.get_input_embeddings()(input_ids)

            # Forward pass through the base model to get hidden states
            outputs = loaded_model.base_model(
                inputs_embeds=inputs_embeds, output_hidden_states=True
            )
            hidden_states = outputs.hidden_states

            # Get the hidden states at the specified layer
            if layer_idx < len(hidden_states):
                embeddings = hidden_states[layer_idx][0]
            else:
                # Fallback to the last layer if index is out of range
                embeddings = hidden_states[-1][0]

        # Get the compression layer
        if layer_idx < len(loaded_model.compression_layers):
            compression_layer = loaded_model.compression_layers[layer_idx]
        else:
            # Fallback to the first layer if index is out of range
            compression_layer = loaded_model.compression_layers[0]

        # Enable compression
        compression_layer.compression_enabled = True

        # Get similarity matrix
        similarity_matrix = compression_layer.compute_similarity_matrix(
            embeddings.unsqueeze(0)
        )[0].cpu()

        # Get the current threshold
        threshold = getattr(
            compression_layer, "current_threshold", compression_layer.base_threshold
        )
        if isinstance(threshold, torch.Tensor):
            threshold = threshold.item()

        # Create a plot
        plt.figure(figsize=(12, 10))

        # Plot similarity matrix
        plt.subplot(2, 1, 1)
        sns.heatmap(similarity_matrix.numpy(), annot=False, cmap="viridis")
        plt.title(f"Token Similarity Matrix (Threshold: {threshold:.4f})")
        plt.xlabel("Token Index")
        plt.ylabel("Token Index")

        # Plot token clusters
        plt.subplot(2, 1, 2)

        # Find clusters based on similarity threshold
        clusters = {}
        processed = set()

        for i in range(len(tokens)):
            if i in processed:
                continue

            # Find similar tokens
            similar_indices = torch.where(similarity_matrix[i] > threshold)[0].tolist()

            # Create a new cluster
            if similar_indices:
                clusters[i] = similar_indices
                processed.update(similar_indices)

        # Create a colormap for clusters
        colors = plt.cm.tab20(np.linspace(0, 1, len(clusters)))

        # Plot each cluster
        for idx, (center, members) in enumerate(clusters.items()):
            # Plot the center token
            plt.scatter(0, center, color=colors[idx], s=100, label=f"Cluster {center}")

            # Plot the member tokens
            for i, member in enumerate(members):
                if member != center:  # Skip the center token
                    plt.scatter(1, member, color=colors[idx], s=50)
                    plt.plot([0, 1], [center, member], color=colors[idx], alpha=0.5)

            # Add token labels
            plt.text(0, center, tokens[center], fontsize=8, ha="right")
            for i, member in enumerate(members):
                if member != center:
                    plt.text(1, member, tokens[member], fontsize=8, ha="left")

        plt.title("Token Clusters")
        plt.xlabel("Position")
        plt.ylabel("Token Index")
        plt.yticks(range(len(tokens)))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save the plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150)
        buf.seek(0)

        # Convert to PIL Image
        img = Image.open(buf)
        plt.close()

        return img

    except Exception as e:
        print(f"Error visualizing compression clusters: {str(e)}")
        import traceback

        traceback.print_exc()
        return None


def generate_with_compression(
    prompt: str,
    max_length: int = 200,
    temperature: float = 0.7,
    top_p: float = 0.9,
    enable_compression: bool = True,
) -> Tuple[str, float, str]:
    """Generate text with or without compression."""
    global loaded_model

    if loaded_model is None:
        return "Please load a model first.", 0.0, ""

    # Set compression state
    if hasattr(loaded_model, "compression_layers"):
        for layer in loaded_model.compression_layers:
            layer.compression_enabled = enable_compression

    # Generate text
    start_time = time.time()
    generated_text = generate_text(
        loaded_model,
        prompt,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
    )
    generation_time = time.time() - start_time

    # Get compression stats if enabled
    stats_text = ""
    if enable_compression and hasattr(loaded_model, "get_compression_stats"):
        stats = loaded_model.get_compression_stats()
        stats_text = f"Compression ratio: {stats.get('compression_ratio', 'N/A')}\n"
        if "max_compression_ratio" in stats:
            stats_text += (
                f"Max compression ratio: {stats.get('max_compression_ratio', 'N/A')}\n"
            )
        if "target_ratio" in stats:
            stats_text += f"Target ratio: {stats.get('target_ratio', 'N/A')}\n"

    return generated_text, generation_time, stats_text


def compare_memory_usage(
    prompt: str,
    max_length: int = 100,  # Shorter for comparison
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    """Compare memory usage between compressed and uncompressed generation."""
    global loaded_model

    if loaded_model is None:
        return "Please load a model first.", "", 0.0, 0.0, 0.0, "", None

    # Data for visualization
    token_counts = []
    compressed_memory = []
    uncompressed_memory = []

    # First generate with compression
    if hasattr(loaded_model, "compression_layers"):
        for layer in loaded_model.compression_layers:
            layer.compression_enabled = True

    # Track initial memory
    initial_memory = get_memory_usage_mb()
    start_time = time.time()

    # Tokenize the prompt
    tokenizer = loaded_model.tokenizer
    input_text = prompt
    output_ids_compressed = tokenizer.encode(prompt)

    # Generate with compression
    generated_text = prompt

    for i in range(max_length):
        # Tokenize the current text
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

        # Forward pass through the model
        with torch.no_grad():
            outputs = loaded_model(input_ids)

            # Extract logits
            if isinstance(outputs, dict):
                logits = outputs.get("logits", outputs.get("last_hidden_state", None))
            else:
                logits = outputs.logits if hasattr(outputs, "logits") else outputs

            if logits is None:
                break

            # Get the last token's logits
            next_token_logits = logits[:, -1, :]

            # Apply filtering and sampling
            filtered_logits = top_p_filtering(next_token_logits, top_p=top_p)
            probs = torch.softmax(filtered_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            # Stop if we generate an EOS token
            if next_token == tokenizer.eos_token_id:
                break

            # Decode the next token and add it to the generated text
            next_token_text = tokenizer.decode([next_token], skip_special_tokens=True)
            generated_text += next_token_text
            output_ids_compressed.append(next_token)

            # Update input for next iteration
            input_text = generated_text

            # Record memory usage
            current_memory = get_memory_usage_mb()
            token_counts.append(len(output_ids_compressed))
            compressed_memory.append(current_memory - initial_memory)

    # Get the compressed text
    compressed_text = generated_text
    compressed_time = time.time() - start_time

    # Reset memory state
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Now generate without compression
    if hasattr(loaded_model, "compression_layers"):
        for layer in loaded_model.compression_layers:
            layer.compression_enabled = False

    # Track initial memory again
    initial_memory = get_memory_usage_mb()
    start_time = time.time()

    # Generate without compression
    input_text = prompt
    generated_text = prompt
    output_ids_uncompressed = tokenizer.encode(prompt)

    for i in range(max_length):
        if i >= len(token_counts):
            break

        # Tokenize the current text
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

        # Forward pass through the model
        with torch.no_grad():
            outputs = loaded_model(input_ids)

            # Extract logits
            if isinstance(outputs, dict):
                logits = outputs.get("logits", outputs.get("last_hidden_state", None))
            else:
                logits = outputs.logits if hasattr(outputs, "logits") else outputs

            if logits is None:
                break

            # Get the last token's logits
            next_token_logits = logits[:, -1, :]

            # Apply filtering and sampling
            filtered_logits = top_p_filtering(next_token_logits, top_p=top_p)
            probs = torch.softmax(filtered_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            # Stop if we generate an EOS token
            if next_token == tokenizer.eos_token_id:
                break

            # Decode the next token and add it to the generated text
            next_token_text = tokenizer.decode([next_token], skip_special_tokens=True)
            generated_text += next_token_text
            output_ids_uncompressed.append(next_token)

            # Update input for next iteration
            input_text = generated_text

            # Record memory usage
            current_memory = get_memory_usage_mb()
            uncompressed_memory.append(current_memory - initial_memory)

    # Get the uncompressed text
    uncompressed_text = generated_text
    uncompressed_time = time.time() - start_time

    # Calculate speedup
    speedup = uncompressed_time / compressed_time if compressed_time > 0 else 0

    # Get compression stats
    if hasattr(loaded_model, "compression_layers"):
        # Re-enable compression to get stats
        for layer in loaded_model.compression_layers:
            layer.compression_enabled = True

    stats_text = ""
    if hasattr(loaded_model, "get_compression_stats"):
        stats = loaded_model.get_compression_stats()
        stats_text = f"Compression ratio: {stats.get('compression_ratio', 'N/A')}\n"
        if "max_compression_ratio" in stats:
            stats_text += (
                f"Max compression ratio: {stats.get('max_compression_ratio', 'N/A')}\n"
            )
        if "target_ratio" in stats:
            stats_text += f"Target ratio: {stats.get('target_ratio', 'N/A')}\n"

    # Create comparison plot
    fig = plt.figure(figsize=(12, 6))
    plt.plot(
        token_counts,
        compressed_memory,
        marker="o",
        linestyle="-",
        color="blue",
        label="With Compression",
    )
    plt.plot(
        token_counts[: len(uncompressed_memory)],
        uncompressed_memory,
        marker="x",
        linestyle="-",
        color="red",
        label="Without Compression",
    )
    plt.title("Memory Usage Comparison: Compressed vs Uncompressed")
    plt.xlabel("Number of Tokens")
    plt.ylabel("Memory Overhead (MB)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add memory savings annotation
    if len(compressed_memory) > 0 and len(uncompressed_memory) > 0:
        last_idx = min(len(compressed_memory), len(uncompressed_memory)) - 1
        if last_idx >= 0:
            memory_savings = uncompressed_memory[last_idx] - compressed_memory[last_idx]
            savings_percent = (
                (memory_savings / uncompressed_memory[last_idx]) * 100
                if uncompressed_memory[last_idx] > 0
                else 0
            )
            plt.annotate(
                f"Memory savings: {memory_savings:.2f} MB ({savings_percent:.1f}%)",
                xy=(
                    token_counts[last_idx],
                    (compressed_memory[last_idx] + uncompressed_memory[last_idx]) / 2,
                ),
                xytext=(
                    token_counts[last_idx] * 0.7,
                    (compressed_memory[last_idx] + uncompressed_memory[last_idx]) / 1.5,
                ),
                arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=8),
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3),
            )

    # Save the plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    buf.seek(0)

    # Convert to PIL Image
    comparison_plot = Image.open(buf)
    plt.close(fig)

    return (
        compressed_text,
        uncompressed_text,
        compressed_time,
        uncompressed_time,
        speedup,
        stats_text,
        comparison_plot,
    )


def stream_generate_with_memory_and_visualization(
    prompt: str,
    max_length: int = 200,
    temperature: float = 0.7,
    top_p: float = 0.9,
    enable_compression: bool = True,
):
    """Generate text with streaming output, memory usage tracking, and visualization."""
    global loaded_model

    if loaded_model is None:
        yield "Please load a model first.", 0, 0, "", None
        return

    # Enable/disable compression
    if hasattr(loaded_model, "compression_layers"):
        for layer in loaded_model.compression_layers:
            layer.compression_enabled = enable_compression

    # Tokenize the prompt
    tokenizer = loaded_model.tokenizer
    input_text = prompt

    # Track initial memory usage
    initial_memory = get_memory_usage_mb()
    start_time = time.time()

    # For visualization
    token_count = [len(tokenizer.encode(prompt))]
    memory_values = [0]  # Start with 0 overhead

    # Generate text incrementally
    generated_text = prompt

    for i in range(max_length):
        # Tokenize the current text
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

        # Forward pass through the model
        with torch.no_grad():
            outputs = loaded_model(input_ids)

            # Extract logits
            if isinstance(outputs, dict):
                logits = outputs.get("logits", outputs.get("last_hidden_state", None))
            else:
                logits = outputs.logits if hasattr(outputs, "logits") else outputs

            if logits is None:
                break

            # Get the last token's logits
            next_token_logits = logits[:, -1, :]

            # Apply filtering and sampling
            filtered_logits = top_p_filtering(next_token_logits, top_p=top_p)
            probs = torch.softmax(filtered_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            # Stop if we generate an EOS token
            if next_token == tokenizer.eos_token_id:
                break

            # Decode the next token and add it to the generated text
            next_token_text = tokenizer.decode([next_token], skip_special_tokens=True)
            generated_text += next_token_text

            # Update input for next iteration - use the full text to maintain context
            input_text = generated_text

            # Calculate memory usage and elapsed time
            current_memory = get_memory_usage_mb()
            memory_overhead = current_memory - initial_memory
            elapsed_time = time.time() - start_time

            # Update visualization data
            token_count.append(len(tokenizer.encode(generated_text)))
            memory_values.append(memory_overhead)

            # Create memory usage plot every few tokens or at the end
            if (
                i % 3 == 0
                or i == max_length - 1
                or next_token == tokenizer.eos_token_id
            ):
                fig = plt.figure(figsize=(10, 4))
                plt.plot(
                    token_count, memory_values, marker="o", linestyle="-", color="blue"
                )
                plt.title("Memory Usage During Generation")
                plt.xlabel("Number of Tokens")
                plt.ylabel("Memory Overhead (MB)")
                plt.grid(True, alpha=0.3)

                # Save the plot to a bytes buffer
                buf = io.BytesIO()
                plt.savefig(buf, format="png")
                buf.seek(0)

                # Convert to PIL Image
                memory_plot = Image.open(buf)
                plt.close(fig)

                # Get compression stats
                stats_text = ""
                if enable_compression and hasattr(
                    loaded_model, "get_compression_stats"
                ):
                    stats = loaded_model.get_compression_stats()
                    stats_text = (
                        f"Compression ratio: {stats.get('compression_ratio', 'N/A')}\n"
                    )
                    if "max_compression_ratio" in stats:
                        stats_text += f"Max compression ratio: {stats.get('max_compression_ratio', 'N/A')}\n"
                    if "target_ratio" in stats:
                        stats_text += (
                            f"Target ratio: {stats.get('target_ratio', 'N/A')}\n"
                        )

                # Yield the current state
                yield generated_text, memory_overhead, elapsed_time, stats_text, memory_plot


# Create the Gradio interface
with gr.Blocks(
    title="Neural Compression Model Interface", theme=gr.themes.Soft()
) as app:
    gr.Markdown("# Neural Compression Model Interface")
    gr.Markdown(
        "This interface allows you to interact with a trained neural compression model."
    )

    with gr.Tab("Load Model"):
        with gr.Row():
            model_dir_input = gr.Textbox(
                label="Model Directory", placeholder="Path to the model directory"
            )
            load_button = gr.Button("Load Model", variant="primary")

        with gr.Row():
            load_status = gr.Textbox(label="Status", interactive=False)
            model_info = gr.Textbox(
                label="Model Information", interactive=False, lines=10
            )

        load_button.click(
            load_model, inputs=[model_dir_input], outputs=[load_status, model_info]
        )

    with gr.Tab("Generate Text"):
        with gr.Row():
            with gr.Column():
                prompt_input = gr.Textbox(
                    label="Prompt", placeholder="Enter your prompt here", lines=3
                )
                with gr.Row():
                    max_length_slider = gr.Slider(
                        minimum=10, maximum=500, value=200, step=10, label="Max Length"
                    )
                    temperature_slider = gr.Slider(
                        minimum=0.1,
                        maximum=1.5,
                        value=0.7,
                        step=0.1,
                        label="Temperature",
                    )
                    top_p_slider = gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.9, step=0.1, label="Top-p"
                    )

                compression_checkbox = gr.Checkbox(
                    label="Enable Compression", value=True
                )
                generate_button = gr.Button("Generate", variant="primary")

            with gr.Column():
                generated_text = gr.Textbox(
                    label="Generated Text", interactive=False, lines=10
                )
                generation_time = gr.Number(
                    label="Generation Time (seconds)", interactive=False
                )
                compression_stats = gr.Textbox(
                    label="Compression Stats", interactive=False, lines=3
                )

        generate_button.click(
            generate_with_compression,
            inputs=[
                prompt_input,
                max_length_slider,
                temperature_slider,
                top_p_slider,
                compression_checkbox,
            ],
            outputs=[generated_text, generation_time, compression_stats],
        )

    with gr.Tab("Streaming Generation"):
        with gr.Row():
            with gr.Column(scale=1):
                stream_prompt_input = gr.Textbox(
                    label="Prompt", placeholder="Enter your prompt here", lines=3
                )
                with gr.Row():
                    stream_max_length_slider = gr.Slider(
                        minimum=10, maximum=500, value=200, step=10, label="Max Length"
                    )
                    stream_temperature_slider = gr.Slider(
                        minimum=0.1,
                        maximum=1.5,
                        value=0.7,
                        step=0.1,
                        label="Temperature",
                    )
                    stream_top_p_slider = gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.9, step=0.1, label="Top-p"
                    )

                stream_compression_checkbox = gr.Checkbox(
                    label="Enable Compression", value=True
                )
                stream_generate_button = gr.Button(
                    "Generate with Streaming", variant="primary"
                )

            with gr.Column(scale=1):
                stream_generated_text = gr.Textbox(
                    label="Generated Text", interactive=False, lines=15
                )
                with gr.Row():
                    memory_usage = gr.Number(
                        label="Memory Overhead (MB)", interactive=False
                    )
                    stream_generation_time = gr.Number(
                        label="Generation Time (seconds)", interactive=False
                    )
                stream_compression_stats = gr.Textbox(
                    label="Compression Stats", interactive=False, lines=3
                )

        with gr.Row():
            memory_plot = gr.Image(
                label="Memory Usage Visualization", interactive=False
            )

        gr.Markdown(
            """
        ### Streaming Generation with Memory Profiling
        
        This tab provides real-time token streaming with memory usage tracking. Watch as:
        
        - Text is generated token-by-token in real time
        - Memory usage is tracked to measure compression efficiency
        - Generation time is updated continuously
        - Memory usage visualization shows the impact of compression over time
        """
        )

        stream_generate_button.click(
            stream_generate_with_memory_and_visualization,
            inputs=[
                stream_prompt_input,
                stream_max_length_slider,
                stream_temperature_slider,
                stream_top_p_slider,
                stream_compression_checkbox,
            ],
            outputs=[
                stream_generated_text,
                memory_usage,
                stream_generation_time,
                stream_compression_stats,
                memory_plot,
            ],
        )

    with gr.Tab("Compare Generation"):
        with gr.Row():
            with gr.Column():
                compare_prompt = gr.Textbox(
                    label="Prompt", placeholder="Enter your prompt here", lines=3
                )
                with gr.Row():
                    compare_max_length = gr.Slider(
                        minimum=10, maximum=500, value=200, step=10, label="Max Length"
                    )
                    compare_temperature = gr.Slider(
                        minimum=0.1,
                        maximum=1.5,
                        value=0.7,
                        step=0.1,
                        label="Temperature",
                    )
                    compare_top_p = gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.9, step=0.1, label="Top-p"
                    )

                compare_button = gr.Button("Compare Generation", variant="primary")

        with gr.Row():
            with gr.Column():
                compressed_output = gr.Textbox(
                    label="With Compression", interactive=False, lines=10
                )
                compressed_time = gr.Number(label="Time (seconds)", interactive=False)

            with gr.Column():
                uncompressed_output = gr.Textbox(
                    label="Without Compression", interactive=False, lines=10
                )
                uncompressed_time = gr.Number(label="Time (seconds)", interactive=False)

        with gr.Row():
            speedup = gr.Number(label="Speedup (x)", interactive=False)
            compare_stats = gr.Textbox(
                label="Compression Stats", interactive=False, lines=3
            )

        compare_button.click(
            compare_generation,
            inputs=[
                compare_prompt,
                compare_max_length,
                compare_temperature,
                compare_top_p,
            ],
            outputs=[
                compressed_output,
                uncompressed_output,
                compressed_time,
                uncompressed_time,
                speedup,
                compare_stats,
            ],
        )

    with gr.Tab("Compression Parameters"):
        with gr.Row():
            similarity_threshold = gr.Slider(
                minimum=0.5,
                maximum=0.99,
                value=0.75,
                step=0.01,
                label="Similarity Threshold",
            )
            max_compression_ratio = gr.Slider(
                minimum=1.0,
                maximum=20.0,
                value=8.0,
                step=0.5,
                label="Max Compression Ratio",
            )

        update_button = gr.Button("Update Parameters", variant="primary")
        update_status = gr.Textbox(label="Status", interactive=False)

        update_button.click(
            update_compression_params,
            inputs=[similarity_threshold, max_compression_ratio],
            outputs=[update_status],
        )

    with gr.Tab("Visualizations"):
        with gr.Tabs():
            with gr.TabItem("Compression Stats"):
                stats_button = gr.Button("Visualize Compression Stats")
                stats_plot = gr.Image(label="Compression Statistics")

                stats_button.click(
                    visualize_compression_stats, inputs=[], outputs=[stats_plot]
                )

            with gr.TabItem("Layer Thresholds"):
                thresholds_button = gr.Button("Visualize Layer Thresholds")
                thresholds_plot = gr.Image(label="Layer Thresholds")

                thresholds_button.click(
                    plot_layer_thresholds, inputs=[], outputs=[thresholds_plot]
                )

            with gr.TabItem("Token Embeddings"):
                with gr.Row():
                    embedding_text = gr.Textbox(
                        label="Text to Visualize",
                        placeholder="Enter text to visualize token embeddings",
                        value="The quick brown fox jumps over the lazy dog.",
                    )

                with gr.Row():
                    embedding_method = gr.Radio(
                        choices=["tsne", "pca"],
                        value="tsne",
                        label="Dimensionality Reduction Method",
                    )
                    embedding_layer = gr.Slider(
                        minimum=0, maximum=11, value=0, step=1, label="Layer Index"
                    )

                embedding_button = gr.Button("Visualize Token Embeddings")
                embedding_plot = gr.Image(label="Token Embeddings Visualization")

                embedding_button.click(
                    visualize_token_embeddings_gui,
                    inputs=[embedding_text, embedding_method, embedding_layer],
                    outputs=[embedding_plot],
                )

            with gr.TabItem("Compression Clusters"):
                with gr.Row():
                    cluster_text = gr.Textbox(
                        label="Text to Visualize",
                        placeholder="Enter text to visualize compression clusters",
                        value="The quick brown fox jumps over the lazy dog.",
                    )
                    cluster_layer = gr.Slider(
                        minimum=0, maximum=11, value=0, step=1, label="Layer Index"
                    )

                cluster_button = gr.Button("Visualize Compression Clusters")
                cluster_plot = gr.Image(label="Compression Clusters Visualization")

                cluster_button.click(
                    visualize_compression_clusters,
                    inputs=[cluster_text, cluster_layer],
                    outputs=[cluster_plot],
                )

    with gr.Tab("Memory Comparison"):
        with gr.Row():
            with gr.Column():
                compare_memory_prompt = gr.Textbox(
                    label="Prompt", placeholder="Enter your prompt here", lines=3
                )
                with gr.Row():
                    compare_memory_max_length = gr.Slider(
                        minimum=10, maximum=200, value=50, step=10, label="Max Length"
                    )
                    compare_memory_temperature = gr.Slider(
                        minimum=0.1,
                        maximum=1.5,
                        value=0.7,
                        step=0.1,
                        label="Temperature",
                    )
                    compare_memory_top_p = gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.9, step=0.1, label="Top-p"
                    )

                compare_memory_button = gr.Button(
                    "Compare Memory Usage", variant="primary"
                )

        with gr.Row():
            with gr.Column():
                compare_memory_compressed = gr.Textbox(
                    label="With Compression", interactive=False, lines=5
                )
                compare_memory_compressed_time = gr.Number(
                    label="Time (seconds)", interactive=False
                )

            with gr.Column():
                compare_memory_uncompressed = gr.Textbox(
                    label="Without Compression", interactive=False, lines=5
                )
                compare_memory_uncompressed_time = gr.Number(
                    label="Time (seconds)", interactive=False
                )

        with gr.Row():
            compare_memory_speedup = gr.Number(label="Speedup (x)", interactive=False)
            compare_memory_stats = gr.Textbox(
                label="Compression Stats", interactive=False, lines=3
            )

        with gr.Row():
            compare_memory_plot = gr.Image(
                label="Memory Usage Comparison", interactive=False
            )

        gr.Markdown(
            """
        ### Memory Usage Comparison
        
        This tab compares memory usage between compressed and uncompressed generation:
        
        - See how memory grows with each token for both methods
        - Visualize memory savings from compression
        - Compare generation speed and quality
        
        Lower memory usage means more efficient inference and the ability to run larger models on the same hardware.
        """
        )

        compare_memory_button.click(
            compare_memory_usage,
            inputs=[
                compare_memory_prompt,
                compare_memory_max_length,
                compare_memory_temperature,
                compare_memory_top_p,
            ],
            outputs=[
                compare_memory_compressed,
                compare_memory_uncompressed,
                compare_memory_compressed_time,
                compare_memory_uncompressed_time,
                compare_memory_speedup,
                compare_memory_stats,
                compare_memory_plot,
            ],
        )

# Launch the app
if __name__ == "__main__":
    app.launch(share=True)
