import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import argparse
import os
import time
import math
import numpy as np
from tqdm import tqdm

# Import the compression model components
from main import CompressionState, RecursiveCompressionLayer


# Define the ProgressiveCompressionLayer here since it's not in main.py yet
class ProgressiveCompressionLayer(RecursiveCompressionLayer):
    """Enhanced compression layer with progressive thresholds based on layer depth."""

    def __init__(
        self,
        d_model: int,
        layer_position: float,  # Position in network (0-1)
        compression_factor: int = 4,
        early_threshold: float = 0.95,  # Conservative for early layers
        middle_threshold: float = 0.85,  # Moderate for middle layers
        late_threshold: float = 0.75,  # Aggressive for later layers
        position_weight: float = 0.2,
        percentile_threshold: float = 95.0,
        preserve_special_tokens: bool = True,
        residual_compression_factor: int = 4,
        use_residuals: bool = True,
        residual_gate_threshold: float = 0.1,
        attention_weight: float = 0.3,
        contrastive_margin: float = 0.2,
        residual_sparsity: float = 0.9,
        residual_bits: int = 8,
        residual_importance_threshold: float = 0.1,
        adaptive_compression: bool = True,
        min_compression_factor: int = 2,
        max_compression_factor: int = 8,
    ):
        # Determine base threshold based on layer position
        if layer_position < 0.3:
            base_threshold = early_threshold
        elif layer_position < 0.7:
            base_threshold = middle_threshold
        else:
            base_threshold = late_threshold

        super().__init__(
            d_model=d_model,
            compression_factor=compression_factor,
            similarity_threshold=base_threshold,  # Use position-based threshold
            position_weight=position_weight,
            percentile_threshold=percentile_threshold,
            preserve_special_tokens=preserve_special_tokens,
            residual_compression_factor=residual_compression_factor,
            use_residuals=use_residuals,
            residual_gate_threshold=residual_gate_threshold,
            attention_weight=attention_weight,
            contrastive_margin=contrastive_margin,
            residual_sparsity=residual_sparsity,
            residual_bits=residual_bits,
            residual_importance_threshold=residual_importance_threshold,
            adaptive_compression=adaptive_compression,
            min_compression_factor=min_compression_factor,
            max_compression_factor=max_compression_factor,
        )

        self.layer_position = layer_position


class CompressibleLanguageModel(nn.Module):
    """
    Language model with integrated compression layers that can be trained end-to-end.

    This model inserts compression/decompression layers between transformer blocks
    to learn how to effectively compress activations while preserving language modeling ability.

    Args:
        base_model_name (str): Name of the HuggingFace model to load as the backbone.
        compression_layer_indices (List[int]): Indices of transformer layers after which to insert compression.
        compression_factor (int): Factor by which each compression layer reduces dimensionality.
        freeze_base_model (bool): Whether to freeze the base model parameters.
        similarity_threshold (float): Threshold for redundancy detection in each compression layer.
        position_weight (float): Weight given to positional proximity in similarity calculation.
        percentile_threshold (float): Percentile to use for adaptive thresholding (0-100).
        preserve_special_tokens (bool): Whether to preserve special tokens from compression.
        residual_compression_factor (int): Factor by which residuals are compressed.
        use_residuals (bool): Whether to use residual vectors for reconstruction.
        residual_gate_threshold (float): Threshold for residual gating (0-1).
        attention_weight (float): Weight for attention-based decisions.
        contrastive_margin (float): Margin for contrastive loss.
        use_progressive_compression (bool): Whether to use progressive compression based on layer depth.
    """

    def __init__(
        self,
        base_model_name: str = "gpt2",
        compression_layer_indices: List[int] = [3, 7, 11],
        compression_factor: int = 4,
        freeze_base_model: bool = False,  # Default to training the whole model
        similarity_threshold: float = 0.75,
        position_weight: float = 0.2,
        percentile_threshold: float = 95.0,
        preserve_special_tokens: bool = True,
        residual_compression_factor: int = 4,
        use_residuals: bool = True,
        residual_gate_threshold: float = 0.1,
        attention_weight: float = 0.3,  # NEW: Weight for attention-based decisions
        contrastive_margin: float = 0.2,  # NEW: Margin for contrastive loss
        use_progressive_compression: bool = True,  # NEW: Use progressive compression
    ):
        super().__init__()
        # Load base model and tokenizer
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        # Ensure padding token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Get model configuration
        self.config = self.base_model.config
        self.d_model = self.config.hidden_size
        self.num_layers = self.config.num_hidden_layers

        # Validate compression layer indices
        self.compression_layer_indices = sorted(
            [idx for idx in compression_layer_indices if 0 <= idx < self.num_layers]
        )

        if not self.compression_layer_indices:
            raise ValueError(
                f"No valid compression layer indices provided. Must be between 0 and {self.num_layers-1}"
            )

        # Create compression layers
        if use_progressive_compression:
            # Create progressive compression layers with different thresholds based on depth
            self.compression_layers = nn.ModuleList()
            for i, layer_idx in enumerate(self.compression_layer_indices):
                # Calculate layer position (0-1 range)
                layer_position = layer_idx / (self.num_layers - 1)

                # Create progressive compression layer
                self.compression_layers.append(
                    ProgressiveCompressionLayer(
                        d_model=self.d_model,
                        layer_position=layer_position,
                        compression_factor=compression_factor,
                        position_weight=position_weight,
                        percentile_threshold=percentile_threshold,
                        preserve_special_tokens=preserve_special_tokens,
                        residual_compression_factor=residual_compression_factor,
                        use_residuals=use_residuals,
                        residual_gate_threshold=residual_gate_threshold,
                        attention_weight=attention_weight,
                        contrastive_margin=contrastive_margin,
                    )
                )
        else:
            # Create standard compression layers
            self.compression_layers = nn.ModuleList(
                [
                    RecursiveCompressionLayer(
                        d_model=self.d_model,
                        compression_factor=compression_factor,
                        similarity_threshold=similarity_threshold,
                        position_weight=position_weight,
                        percentile_threshold=percentile_threshold,
                        preserve_special_tokens=preserve_special_tokens,
                        residual_compression_factor=residual_compression_factor,
                        use_residuals=use_residuals,
                        residual_gate_threshold=residual_gate_threshold,
                        attention_weight=attention_weight,
                        contrastive_margin=contrastive_margin,
                    )
                    for _ in range(len(self.compression_layer_indices))
                ]
            )

        # Freeze base model if specified
        if freeze_base_model:
            for param in self.base_model.parameters():
                param.requires_grad = False

        # Add a contrastive loss component to ensure dissimilar tokens aren't merged
        self.contrastive_projection = nn.Linear(self.d_model, 128)

        # Track compression statistics
        self.compression_stats = {
            "compression_ratio": [],
            "token_redundancy": [],
            "residual_sparsity": [],
        }

    def _get_transformer_layers(self):
        """Get the transformer layers from the base model."""
        # This is model-specific - for GPT-2 it's base_model.transformer.h
        if hasattr(self.base_model, "transformer") and hasattr(
            self.base_model.transformer, "h"
        ):
            return self.base_model.transformer.h
        else:
            raise NotImplementedError(
                f"Unsupported model architecture: {type(self.base_model)}"
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        output_attentions: bool = False,  # NEW: Option to output attention
    ):
        """
        Forward pass with compression layers inserted between transformer blocks.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            labels (torch.Tensor, optional): Labels for language modeling loss.
            output_hidden_states (bool): Whether to return hidden states.
            return_dict (bool): Whether to return a dictionary of outputs.
            output_attentions (bool): Whether to output attention matrices.

        Returns:
            outputs: Model outputs including loss, logits, and hidden states.
        """
        # First, get attention scores if needed for compression
        if any(layer.attention_weight > 0 for layer in self.compression_layers):
            # Get attention scores from base model
            base_outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True,
            )
            all_hidden_states = base_outputs.hidden_states
            all_attentions = base_outputs.attentions
        else:
            # Regular forward pass without attention
            base_outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=output_attentions,
                return_dict=True,
            )
            all_hidden_states = base_outputs.hidden_states
            all_attentions = base_outputs.attentions if output_attentions else None

        # Initialize variables to track compression
        compression_states = []
        total_loss = 0

        # Apply compression to selected hidden states
        modified_hidden_states = list(all_hidden_states)
        batch_size, seq_length = input_ids.shape

        for i, layer_idx in enumerate(self.compression_layer_indices):
            if layer_idx < len(modified_hidden_states) - 1:
                compression_layer = self.compression_layers[i]
                layer_hidden_states = modified_hidden_states[layer_idx + 1]

                # Get corresponding attention if available
                layer_attention = None
                if all_attentions is not None and i < len(all_attentions):
                    layer_attention = all_attentions[i]

                # Apply compression with attention guidance
                compressed_states, layer_loss = compression_layer(
                    layer_hidden_states,
                    attention_scores=layer_attention,
                    labels=(
                        labels if i == len(self.compression_layer_indices) - 1 else None
                    ),
                )

                # Accumulate loss
                if layer_loss is not None:
                    total_loss += layer_loss

                # Store compression state
                compression_states.append(compressed_states)

                # Replace hidden states with compressed version
                modified_hidden_states[layer_idx + 1] = compressed_states

        # Use the final hidden state for the language model head
        lm_logits = self.base_model.lm_head(modified_hidden_states[-1])

        # Calculate language modeling loss if not already done in compression layers
        loss = None
        if labels is not None and total_loss == 0:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Calculate loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
        elif total_loss > 0:
            loss = total_loss

        # Calculate compression statistics
        compression_ratio = 1.0  # Default to no compression
        if compression_states:
            # Estimate compression ratio based on clusters
            total_tokens = batch_size * seq_length
            unique_tokens = sum(state.shape[0] for state in compression_states)
            if unique_tokens > 0:
                compression_ratio = total_tokens / unique_tokens

        if return_dict:
            return {
                "loss": loss,
                "logits": lm_logits,
                "hidden_states": (
                    modified_hidden_states if output_hidden_states else None
                ),
                "compression_states": compression_states,
                "compression_ratio": compression_ratio,
                "attentions": all_attentions if output_attentions else None,
            }
        else:
            return (
                loss,
                lm_logits,
                modified_hidden_states,
                compression_states,
                compression_ratio,
            )

    def get_compression_stats(self):
        """Get current compression statistics."""
        stats = {}
        for key, values in self.compression_stats.items():
            if values:
                stats[key] = values[-1]  # Get most recent value
        return stats

    def save_pretrained(self, output_dir):
        """Save the model to the specified directory."""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Save base model
        self.base_model.save_pretrained(os.path.join(output_dir, "base_model"))
        self.tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))

        # Save compression layers
        compression_layers_dir = os.path.join(output_dir, "compression_layers")
        os.makedirs(compression_layers_dir, exist_ok=True)

        # Save each compression layer
        for i, layer in enumerate(self.compression_layers):
            layer_dir = os.path.join(compression_layers_dir, f"layer_{i}")
            os.makedirs(layer_dir, exist_ok=True)

            # Save layer parameters
            torch.save(layer.state_dict(), os.path.join(layer_dir, "layer.pt"))

            # Save layer configuration
            layer_config = {
                "d_model": layer.d_model,
                "compression_factor": layer.compression_factor,
                "base_threshold": layer.base_threshold,
                "position_weight": layer.position_weight,
                "percentile_threshold": layer.percentile_threshold,
                "preserve_special_tokens": layer.preserve_special_tokens,
                "residual_compression_factor": layer.residual_compression_factor,
                "use_residuals": layer.use_residuals,
                "residual_gate_threshold": layer.residual_gate_threshold,
                "attention_weight": layer.attention_weight,
                "contrastive_margin": layer.contrastive_margin,
            }

            # Add progressive layer specific parameters if applicable
            if hasattr(layer, "layer_position"):
                layer_config["layer_position"] = layer.layer_position
                layer_config["is_progressive"] = True
            else:
                layer_config["is_progressive"] = False

            # Save configuration
            import json

            with open(os.path.join(layer_dir, "config.json"), "w") as f:
                json.dump(layer_config, f)

        # Save model configuration
        model_config = {
            "base_model_name": self.base_model.config._name_or_path,
            "compression_layer_indices": self.compression_layer_indices,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "use_progressive_compression": any(
                hasattr(layer, "layer_position") for layer in self.compression_layers
            ),
        }

        import json

        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(model_config, f)

        print(f"Model saved to {output_dir}")

    @classmethod
    def from_pretrained(cls, model_dir):
        """Load a model from the specified directory."""
        # Load model configuration
        import json

        with open(os.path.join(model_dir, "config.json"), "r") as f:
            model_config = json.load(f)

        # Load base model and tokenizer
        from transformers import AutoModelForCausalLM, AutoTokenizer

        base_model = AutoModelForCausalLM.from_pretrained(
            os.path.join(model_dir, "base_model")
        )
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))

        # Create model instance
        model = cls(
            base_model_name=model_config["base_model_name"],
            compression_layer_indices=model_config["compression_layer_indices"],
            use_progressive_compression=model_config["use_progressive_compression"],
        )

        # Replace base model and tokenizer
        model.base_model = base_model
        model.tokenizer = tokenizer

        # Load compression layers
        compression_layers_dir = os.path.join(model_dir, "compression_layers")

        # Clear existing compression layers
        model.compression_layers = nn.ModuleList()

        # Load each compression layer
        for i in range(len(model_config["compression_layer_indices"])):
            layer_dir = os.path.join(compression_layers_dir, f"layer_{i}")

            # Load layer configuration
            with open(os.path.join(layer_dir, "config.json"), "r") as f:
                layer_config = json.load(f)

            # Create appropriate layer type
            if layer_config["is_progressive"]:
                layer = ProgressiveCompressionLayer(
                    d_model=layer_config["d_model"],
                    layer_position=layer_config["layer_position"],
                    compression_factor=layer_config["compression_factor"],
                    similarity_threshold=layer_config["base_threshold"],
                    position_weight=layer_config["position_weight"],
                    percentile_threshold=layer_config["percentile_threshold"],
                    preserve_special_tokens=layer_config["preserve_special_tokens"],
                    residual_compression_factor=layer_config[
                        "residual_compression_factor"
                    ],
                    use_residuals=layer_config["use_residuals"],
                    residual_gate_threshold=layer_config["residual_gate_threshold"],
                    attention_weight=layer_config["attention_weight"],
                    contrastive_margin=layer_config["contrastive_margin"],
                )
            else:
                layer = RecursiveCompressionLayer(
                    d_model=layer_config["d_model"],
                    compression_factor=layer_config["compression_factor"],
                    similarity_threshold=layer_config["base_threshold"],
                    position_weight=layer_config["position_weight"],
                    percentile_threshold=layer_config["percentile_threshold"],
                    preserve_special_tokens=layer_config["preserve_special_tokens"],
                    residual_compression_factor=layer_config[
                        "residual_compression_factor"
                    ],
                    use_residuals=layer_config["use_residuals"],
                    residual_gate_threshold=layer_config["residual_gate_threshold"],
                    attention_weight=layer_config["attention_weight"],
                    contrastive_margin=layer_config["contrastive_margin"],
                )

            # Load layer state
            layer.load_state_dict(torch.load(os.path.join(layer_dir, "layer.pt")))

            # Add to model
            model.compression_layers.append(layer)

        print(f"Model loaded from {model_dir}")
        return model

    def enable_compression(self):
        """Enable compression during inference."""
        for layer in self.compression_layers:
            layer.enable_compression()

    def disable_compression(self):
        """Disable compression during inference."""
        for layer in self.compression_layers:
            layer.disable_compression()


class TextDataset(Dataset):
    """Simple dataset for text data."""

    def __init__(self, texts=None, tokenizer=None, max_length=512, file_path=None):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Handle either file path or direct text list
        if file_path is not None:
            with open(file_path, "r", encoding="utf-8") as f:
                texts = [line.strip() for line in f]

        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]

        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self):
        return len(self.encodings.input_ids)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = item["input_ids"].clone()
        return item


def evaluate(model, val_loader):
    """Evaluate the model on validation data with compression metrics."""
    model.eval()
    total_loss = 0
    total_compression = 0

    with torch.no_grad():
        for batch in val_loader:
            # Move batch to device
            batch = {k: v.to(model.base_model.device) for k, v in batch.items()}

            # Forward pass
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )

            # Accumulate metrics
            total_loss += outputs["loss"].item()
            total_compression += outputs["compression_ratio"]

    avg_loss = total_loss / len(val_loader)
    avg_compression = total_compression / len(val_loader)

    return avg_loss, avg_compression


def log_compression_stats(model, epoch, global_step, output_dir):
    """Log compression statistics to file."""
    # Create stats directory
    stats_dir = os.path.join(output_dir, "stats")
    os.makedirs(stats_dir, exist_ok=True)

    # Get thresholds from all compression layers
    thresholds = [layer.base_threshold for layer in model.compression_layers]

    # Log to file
    with open(os.path.join(stats_dir, "compression_stats.txt"), "a") as f:
        f.write(f"Epoch {epoch}, Step {global_step}:\n")
        f.write(f"  Thresholds: {thresholds}\n")
        f.write(f"  Average threshold: {sum(thresholds)/len(thresholds):.4f}\n")
        f.write("\n")

    # Create visualization of thresholds over time
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        # Load previous stats if they exist
        stats_file = os.path.join(output_dir, "compression_stats.json")
        if os.path.exists(stats_file):
            import json

            with open(stats_file, "r") as f:
                stats = json.load(f)

            # Plot thresholds over time
            plt.figure(figsize=(10, 6))
            steps = stats["steps"]

            # Plot each layer's threshold
            for i, layer_thresholds in enumerate(zip(*stats["thresholds"])):
                plt.plot(steps, layer_thresholds, label=f"Layer {i}")

            plt.xlabel("Training Steps")
            plt.ylabel("Similarity Threshold")
            plt.title("Adaptive Thresholds During Training")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(stats_dir, "thresholds.png"))

            # Plot compression ratio over time
            plt.figure(figsize=(10, 6))
            plt.plot(steps, stats["compression_ratios"])
            plt.xlabel("Training Steps")
            plt.ylabel("Compression Ratio")
            plt.title("Compression Ratio During Training")
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(stats_dir, "compression_ratio.png"))

            # Plot loss over time
            plt.figure(figsize=(10, 6))
            plt.plot(steps, stats["loss_values"], label="Training Loss")

            # Add validation loss if available
            if "val_loss_values" in stats and stats["val_loss_values"]:
                # Interpolate validation steps to match training steps
                val_steps = np.linspace(
                    min(steps), max(steps), len(stats["val_loss_values"])
                )
                plt.plot(val_steps, stats["val_loss_values"], label="Validation Loss")

            plt.xlabel("Training Steps")
            plt.ylabel("Loss")
            plt.title("Loss During Training")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(stats_dir, "loss.png"))

            plt.close("all")
    except Exception as e:
        print(f"Error creating visualizations: {e}")


def train(
    model,
    train_dataset,
    val_dataset=None,
    batch_size=4,
    epochs=3,
    learning_rate=5e-5,
    warmup_steps=0,
    output_dir="./model_output",
    save_steps=1000,
    eval_steps=500,
    log_steps=100,  # NEW: Steps between logging
):
    """Train the compression language model with enhanced monitoring."""
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None

    # Setup optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize compression statistics tracking
    compression_stats = {
        "thresholds": [],
        "compression_ratios": [],
        "loss_values": [],
        "val_loss_values": [],
        "steps": [],
    }

    # Training loop
    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(model.base_model.device) for k, v in batch.items()}

            # Forward pass with attention for compression guidance
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                output_attentions=True,  # Get attention for compression
            )
            loss = outputs["loss"]

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Update progress
            epoch_loss += loss.item()
            progress_bar.set_postfix(
                {"loss": loss.item(), "compression": outputs["compression_ratio"]}
            )

            # Track compression statistics
            if global_step % log_steps == 0:
                # Get current thresholds from all compression layers
                current_thresholds = [
                    layer.base_threshold for layer in model.compression_layers
                ]

                compression_stats["thresholds"].append(current_thresholds)
                compression_stats["compression_ratios"].append(
                    outputs["compression_ratio"]
                )
                compression_stats["loss_values"].append(loss.item())
                compression_stats["steps"].append(global_step)

                # Log to console
                avg_threshold = sum(current_thresholds) / len(current_thresholds)
                print(
                    f"\nStep {global_step}: Loss {loss.item():.4f}, "
                    f"Compression {outputs['compression_ratio']:.2f}x, "
                    f"Avg Threshold {avg_threshold:.2f}"
                )

            # Increment global step
            global_step += 1

            # Evaluation
            if val_loader and global_step % eval_steps == 0:
                val_loss, val_compression = evaluate(model, val_loader)
                print(
                    f"Validation loss: {val_loss:.4f}, Compression: {val_compression:.2f}x"
                )

                # Track validation metrics
                compression_stats["val_loss_values"].append(val_loss)

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    model.save_pretrained(os.path.join(output_dir, "best_model"))

                # Log compression statistics
                log_compression_stats(model, epoch, global_step, output_dir)

                # Back to training mode
                model.train()

            # Save checkpoint
            if global_step % save_steps == 0:
                model.save_pretrained(
                    os.path.join(output_dir, f"checkpoint-{global_step}")
                )

                # Save compression statistics
                with open(os.path.join(output_dir, "compression_stats.json"), "w") as f:
                    import json

                    json.dump(compression_stats, f)

        # End of epoch
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Average loss: {avg_epoch_loss:.4f}")

        # Save epoch checkpoint
        model.save_pretrained(os.path.join(output_dir, f"epoch-{epoch+1}"))

    # Save final model
    model.save_pretrained(os.path.join(output_dir, "final_model"))

    # Save final compression statistics
    with open(os.path.join(output_dir, "compression_stats.json"), "w") as f:
        import json

        json.dump(compression_stats, f)

    return model


def generate_text(model, prompt, max_length=100, temperature=0.7, top_p=0.9):
    """Generate text using the trained compression model."""
    model.eval()

    # Tokenize prompt
    inputs = model.tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(model.base_model.device)
    attention_mask = inputs.attention_mask.to(model.base_model.device)

    # Generate text
    with torch.no_grad():
        for _ in range(max_length):
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]

            # Get next token logits
            next_token_logits = logits[0, -1, :] / temperature

            # Apply top-p sampling
            sorted_logits, sorted_indices = torch.sort(
                next_token_logits, descending=True
            )
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[indices_to_remove] = -float("inf")

            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to input_ids
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (1, 1), device=model.base_model.device, dtype=torch.long
                    ),
                ],
                dim=1,
            )

            # Stop if EOS token is generated
            if next_token.item() == model.tokenizer.eos_token_id:
                break

    # Decode and return generated text
    generated_text = model.tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text


def main():
    """Main function to train and evaluate the compression language model."""
    parser = argparse.ArgumentParser(description="Train a compressible language model")

    # Model parameters
    parser.add_argument(
        "--base_model_name", type=str, default="gpt2", help="Base model name"
    )
    parser.add_argument(
        "--compression_layer_indices",
        type=str,
        default="3,7,11",
        help="Comma-separated list of layer indices for compression",
    )
    parser.add_argument(
        "--compression_factor", type=int, default=4, help="Compression factor"
    )
    parser.add_argument(
        "--freeze_base_model",
        action="store_true",
        help="Freeze base model parameters",
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.75,
        help="Similarity threshold for compression",
    )
    parser.add_argument(
        "--position_weight",
        type=float,
        default=0.2,
        help="Weight for positional similarity",
    )
    parser.add_argument(
        "--percentile_threshold",
        type=float,
        default=95.0,
        help="Percentile threshold for adaptive compression",
    )
    parser.add_argument(
        "--preserve_special_tokens",
        action="store_true",
        default=True,
        help="Preserve special tokens from compression",
    )
    parser.add_argument(
        "--residual_compression_factor",
        type=int,
        default=4,
        help="Compression factor for residuals",
    )
    parser.add_argument(
        "--use_residuals",
        action="store_true",
        default=True,
        help="Use residual vectors for reconstruction",
    )
    parser.add_argument(
        "--residual_gate_threshold",
        type=float,
        default=0.1,
        help="Threshold for residual gating",
    )
    parser.add_argument(
        "--attention_weight",
        type=float,
        default=0.3,
        help="Weight for attention-based decisions",
    )
    parser.add_argument(
        "--contrastive_margin",
        type=float,
        default=0.2,
        help="Margin for contrastive loss",
    )
    parser.add_argument(
        "--use_progressive_compression",
        action="store_true",
        default=True,
        help="Use progressive compression based on layer depth",
    )

    # Training parameters
    parser.add_argument(
        "--train_file", type=str, required=True, help="Training data file"
    )
    parser.add_argument(
        "--val_file", type=str, default=None, help="Validation data file"
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="Learning rate"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=0, help="Warmup steps for scheduler"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./model_output", help="Output directory"
    )
    parser.add_argument(
        "--save_steps", type=int, default=1000, help="Steps between saving checkpoints"
    )
    parser.add_argument(
        "--eval_steps", type=int, default=500, help="Steps between evaluations"
    )
    parser.add_argument(
        "--log_steps", type=int, default=100, help="Steps between logging"
    )
    parser.add_argument(
        "--max_length", type=int, default=512, help="Maximum sequence length"
    )

    args = parser.parse_args()

    # Parse compression layer indices
    compression_layer_indices = [
        int(idx) for idx in args.compression_layer_indices.split(",")
    ]

    # Create model
    model = CompressibleLanguageModel(
        base_model_name=args.base_model_name,
        compression_layer_indices=compression_layer_indices,
        compression_factor=args.compression_factor,
        freeze_base_model=args.freeze_base_model,
        similarity_threshold=args.similarity_threshold,
        position_weight=args.position_weight,
        percentile_threshold=args.percentile_threshold,
        preserve_special_tokens=args.preserve_special_tokens,
        residual_compression_factor=args.residual_compression_factor,
        use_residuals=args.use_residuals,
        residual_gate_threshold=args.residual_gate_threshold,
        attention_weight=args.attention_weight,
        contrastive_margin=args.contrastive_margin,
        use_progressive_compression=args.use_progressive_compression,
    )

    # Load training data
    print(f"Loading training data from {args.train_file}")
    with open(args.train_file, "r", encoding="utf-8") as f:
        train_texts = [line.strip() for line in f]

    # Create training dataset
    train_dataset = TextDataset(
        train_texts, model.tokenizer, max_length=args.max_length
    )

    # Load validation data if provided
    val_dataset = None
    if args.val_file:
        print(f"Loading validation data from {args.val_file}")
        with open(args.val_file, "r", encoding="utf-8") as f:
            val_texts = [line.strip() for line in f]

        val_dataset = TextDataset(
            val_texts, model.tokenizer, max_length=args.max_length
        )

    # Train model
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
        log_steps=args.log_steps,
    )

    # Generate sample text
    print("\nGenerating sample text:")
    prompts = [
        "Artificial intelligence will",
        "The future of technology is",
        "In the next decade, humans will",
    ]

    for prompt in prompts:
        generated_text = generate_text(model, prompt)
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated_text}")

    print(f"\nTraining complete. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
