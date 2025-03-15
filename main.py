import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import math
import argparse
import random


@dataclass
class CompressionState:
    """Holds the state of a compression operation."""

    centroids: torch.Tensor  # Compressed token representations
    compression_factors: torch.Tensor  # Compression factor used for each token
    original_tokens: torch.Tensor  # Original uncompressed tokens


class RecursiveCompressionLayer(nn.Module):
    """
    Layer that compresses its own activations based on semantic similarity.

    Args:
        d_model (int): Dimension of model embeddings.
        compression_factor (int): Factor by which embeddings are reduced.
        similarity_threshold (float): Threshold above which two tokens are considered redundant.
        position_weight (float): Weight given to positional proximity in similarity calculation.
        percentile_threshold (float): Percentile to use for adaptive thresholding (0-100).
        preserve_special_tokens (bool): Whether to preserve special tokens from compression.
        residual_compression_factor (int): Factor by which residuals are compressed.
        use_residuals (bool): Whether to use residual vectors for reconstruction.
        residual_gate_threshold (float): Threshold for residual gating (0-1).
        residual_sparsity (float): Percentage of residual components to keep.
        residual_bits (int): Number of bits to use for quantization.
        residual_importance_threshold (float): Threshold for adaptive residual storage.
        adaptive_compression (bool): Whether to use adaptive compression based on token importance.
        min_compression_factor (int): Minimum compression factor for important tokens.
        max_compression_factor (int): Maximum compression factor for unimportant tokens.
        attention_weight (float): Weight for attention-based decisions.
        contrastive_margin (float): Margin for contrastive loss.
    """

    def __init__(
        self,
        d_model: int,
        compression_factor: int = 4,
        similarity_threshold: float = 0.75,
        position_weight: float = 0.2,
        percentile_threshold: float = 95.0,
        preserve_special_tokens: bool = True,
        residual_compression_factor: int = 4,
        use_residuals: bool = True,
        residual_gate_threshold: float = 0.1,
        residual_sparsity: float = 0.9,  # Keep only top 10% components
        residual_bits: int = 8,  # 8-bit quantization for residuals
        residual_importance_threshold: float = 0.1,  # Only keep residuals above this reconstruction error
        adaptive_compression: bool = True,  # Enable adaptive compression
        min_compression_factor: int = 2,  # Minimum compression for important tokens
        max_compression_factor: int = 8,  # Maximum compression for unimportant tokens
        attention_weight: float = 0.3,  # NEW: Weight for attention-based decisions
        contrastive_margin: float = 0.2,  # NEW: Margin for contrastive loss
    ):
        super().__init__()
        self.d_model = d_model
        self.compression_factor = compression_factor
        self.base_threshold = similarity_threshold  # Renamed to base_threshold
        self.position_weight = position_weight
        self.percentile_threshold = percentile_threshold
        self.preserve_special_tokens = preserve_special_tokens
        self.residual_compression_factor = residual_compression_factor
        self.use_residuals = use_residuals
        self.residual_gate_threshold = residual_gate_threshold
        self.residual_sparsity = residual_sparsity
        self.residual_bits = residual_bits
        self.residual_importance_threshold = residual_importance_threshold
        self.adaptive_compression = adaptive_compression
        self.min_compression_factor = min_compression_factor
        self.max_compression_factor = max_compression_factor
        self.attention_weight = (
            attention_weight  # NEW: Weight for attention-based decisions
        )
        self.contrastive_margin = contrastive_margin  # NEW: Margin for contrastive loss

        # Compression state
        self.centroids = None
        self.compressed_residuals = None
        self.residual_gates = None

        # Flag to enable/disable compression during inference
        self.compression_enabled = True
        self.last_ce_loss = None  # NEW: Track last CE loss for emergency parachute

        # New learnable components for adaptive thresholding
        self.threshold_modulator = nn.Linear(d_model, 1)

        # Contrastive projection head
        self.contrastive_projector = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 128)
        )

        # Compression components - now we'll have multiple compressors for adaptive compression
        if adaptive_compression:
            # Create multiple compressors with different compression factors
            self.compressors = nn.ModuleDict(
                {
                    f"compressor_{factor}": nn.Linear(d_model, d_model // factor)
                    for factor in range(
                        min_compression_factor, max_compression_factor + 1
                    )
                }
            )
            # Default compressor for backward compatibility
            self.compressor = self.compressors[f"compressor_{compression_factor}"]
        else:
            # Single compressor with fixed compression factor
            self.compressor = nn.Linear(d_model, d_model // compression_factor)

        # Residual compression components
        if use_residuals:
            # Network to compress residuals
            self.residual_compressor = nn.Sequential(
                nn.Linear(d_model, d_model // residual_compression_factor),
                nn.GELU(),
                nn.Linear(d_model // residual_compression_factor, d_model),
            )

            # Network to learn which parts of residuals are important
            self.residual_gate = nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.GELU(),
                nn.Linear(d_model // 4, d_model),
                nn.Sigmoid(),
            )

            # Network to enhance reconstruction
            self.reconstruction_enhancer = nn.Sequential(
                nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Linear(d_model, d_model)
            )

        # Enhanced decompressor with attention mechanism
        self.compressed_dim = d_model // compression_factor
        self.decompressor_query = nn.Linear(self.compressed_dim, self.compressed_dim)
        self.decompressor_key = nn.Linear(self.compressed_dim, self.compressed_dim)
        self.decompressor_value = nn.Linear(self.compressed_dim, self.compressed_dim)

        self.decompressor_mlp = nn.Sequential(
            nn.Linear(self.compressed_dim, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model),
        )

        # Token importance estimator
        self.importance_estimator = nn.Linear(d_model, 1)

        # Adaptive decompressors for different compression factors
        if adaptive_compression:
            self.decompressors = nn.ModuleDict()
            for factor in range(min_compression_factor, max_compression_factor + 1):
                compressed_dim = d_model // factor
                self.decompressors[f"decompressor_{factor}"] = nn.Sequential(
                    nn.Linear(compressed_dim, d_model // 2),
                    nn.GELU(),
                    nn.Linear(d_model // 2, d_model),
                    nn.LayerNorm(d_model),
                )

        # Calculate intermediate dimensions for decompressor
        dims = []
        current_dim = self.compressed_dim
        while current_dim < d_model:
            dims.append(current_dim)
            current_dim = min(current_dim * 2, d_model)
        if dims[-1] != d_model:
            dims.append(d_model)

        # Initialize decompressor with proper dimension scaling
        layers = []
        for i in range(len(dims) - 1):
            layers.extend(
                [
                    nn.Linear(dims[i], dims[i + 1]),
                    nn.ReLU() if i < len(dims) - 2 else nn.Identity(),
                ]
            )
        self.decompressor = nn.Sequential(*layers)

    def compute_adaptive_threshold(self, hidden_states):
        """Dynamically compute threshold based on hidden states distribution."""
        # Calculate pairwise similarities for percentile-based threshold
        token_embeddings = hidden_states.view(-1, self.d_model)
        token_embeddings = F.normalize(token_embeddings, dim=1)
        pairwise_sims = torch.matmul(token_embeddings, token_embeddings.transpose(0, 1))

        # Get similarity threshold based on percentile
        sim_threshold = torch.quantile(
            pairwise_sims.view(-1), self.percentile_threshold / 100
        )

        # Learnable modulation based on context
        context_vector = hidden_states.mean(dim=1)  # Aggregate context
        threshold_mod = torch.sigmoid(self.threshold_modulator(context_vector)) * 0.1

        # Final threshold combines base + statistical + learned components
        # Base threshold is the floor, not the ceiling
        final_threshold = torch.clamp(
            self.base_threshold
            + (1 - self.base_threshold) * sim_threshold
            + threshold_mod,
            min=self.base_threshold,
            max=0.98,  # Safety cap to prevent threshold from getting too high
        )

        return final_threshold

    def compute_similarity_matrix(self, hidden_states, attention_scores=None):
        """Compute similarity matrix with optional attention weighting."""
        batch_size, seq_length, d_model = hidden_states.shape

        # Normalize embeddings for cosine similarity
        normalized = F.normalize(hidden_states, dim=2)

        # Compute embedding similarity matrix
        emb_similarity = torch.bmm(normalized, normalized.transpose(1, 2))

        # If attention scores are provided, use them to weight similarity
        if attention_scores is not None and self.attention_weight > 0:
            # Average across attention heads if needed
            if len(attention_scores.shape) == 4:  # [batch, heads, seq, seq]
                attn = attention_scores.mean(dim=1)  # [batch, seq, seq]
            else:
                attn = attention_scores

            # Compute attention pattern similarity
            # If two tokens attend to the same context similarly, they're functionally similar
            attn_similarity = torch.bmm(attn, attn.transpose(1, 2))

            # Combine embedding and attention similarities
            combined_sim = (
                emb_similarity * (1 - self.attention_weight)
                + attn_similarity * self.attention_weight
            )
        else:
            combined_sim = emb_similarity

        # Apply positional penalty
        pos_indices = torch.arange(seq_length, device=hidden_states.device)
        pos_diff = (pos_indices.unsqueeze(1) - pos_indices.unsqueeze(0)).abs()
        pos_penalty = torch.exp(-pos_diff / (seq_length * self.position_weight))

        # Final similarity with positional weighting
        final_sim = combined_sim * pos_penalty.unsqueeze(0)

        return final_sim

    def contrastive_loss(self, hidden_states, clusters):
        """Enforce separation between dissimilar token clusters."""
        # Project to contrastive space
        projections = self.contrastive_projector(hidden_states)
        projections = F.normalize(projections, dim=2)

        batch_size, seq_length, _ = projections.shape
        loss = 0

        for b in range(batch_size):
            # For each cluster, compute centroid
            unique_clusters = torch.unique(clusters[b])
            centroids = []

            for c in unique_clusters:
                mask = clusters[b] == c
                if mask.sum() > 0:
                    centroid = projections[b, mask].mean(0)
                    centroids.append((c, centroid))

            # Contrastive loss between centroids
            for i, (c1, cent1) in enumerate(centroids):
                for j, (c2, cent2) in enumerate(centroids):
                    if i != j:
                        # Push different clusters apart
                        sim = torch.dot(cent1, cent2)
                        loss += F.relu(sim - (1 - self.contrastive_margin))

        return loss / batch_size if batch_size > 0 else 0

    def compress(self, tokens, attention_scores=None):
        """Compress input tokens using adaptive compression."""
        batch_size, seq_len, d_model = tokens.shape
        device = tokens.device

        # Calculate token importances with enhanced metrics
        token_importances = self.calculate_token_importance(tokens)

        # Calculate adaptive threshold using dynamic percentile
        sorted_importances = torch.sort(token_importances)[0]
        # Use higher percentile for more selective compression
        percentile_idx = int(len(sorted_importances) * 0.90)  # Using 90th percentile
        adaptive_threshold = sorted_importances[percentile_idx]

        # Use the current_threshold (which may be adaptive) instead of base_threshold
        # for similarity comparisons in clustering
        similarity_threshold = getattr(self, "current_threshold", self.base_threshold)

        # Print thresholds for debugging - handle tensor values properly
        if isinstance(self.base_threshold, torch.Tensor):
            if self.base_threshold.numel() == 1:
                base_threshold_val = self.base_threshold.item()
            else:
                # If it's a multi-element tensor, use the first value
                base_threshold_val = self.base_threshold[0].item()
        else:
            base_threshold_val = self.base_threshold

        if isinstance(similarity_threshold, torch.Tensor):
            if similarity_threshold.numel() == 1:
                similarity_threshold_val = similarity_threshold.item()
            else:
                # If it's a multi-element tensor, use the first value
                similarity_threshold_val = similarity_threshold[0].item()
        else:
            similarity_threshold_val = similarity_threshold

        # Print thresholds for debugging - handle tensor values properly - MAKE LESS VERBOSE
        if (
            self.training and random.random() < 0.05
        ):  # Only print 5% of the time during training
            print(
                f"DEBUG: base_threshold={base_threshold_val:.4f}, current_threshold={similarity_threshold_val:.4f}"
            )

        # Find important tokens with stricter criteria
        important_indices = torch.where(token_importances > adaptive_threshold)[0]

        # Initialize storage for compressed tokens and their factors
        compressed_centroids = []
        token_compression_factors = []

        # CRITICAL FIX: If similarity threshold is too low, force it higher to prevent over-compression
        if similarity_threshold_val < 0.90 and self.training:
            print(
                f"⚠️ WARNING: Similarity threshold too low ({similarity_threshold_val:.4f}), forcing to minimum 0.90"
            )
            similarity_threshold_val = max(similarity_threshold_val, 0.90)
            # Convert back to tensor if needed
            if isinstance(similarity_threshold, torch.Tensor):
                similarity_threshold = torch.tensor(
                    similarity_threshold_val, device=device
                )
            else:
                similarity_threshold = similarity_threshold_val

        # NEW SAFETY CHECK: Estimate compression ratio before proceeding
        # If we're going to compress too aggressively, adjust threshold immediately
        estimated_compression = 0
        for i in range(seq_len):
            token = tokens[:, i, :]
            similarities = torch.cosine_similarity(token.unsqueeze(1), tokens, dim=2)
            similar_indices = torch.where(similarities > similarity_threshold_val)[1]
            if len(similar_indices) > 1:
                estimated_compression += 1

        # Get max_compression_ratio if it exists, otherwise use default of 20.0
        max_ratio = getattr(self, "max_compression_ratio", 5.0)  # Reduced from 20.0

        # Calculate how many tokens we need to keep to achieve max_ratio
        # We need to keep at least seq_len / max_ratio tokens
        # Use floor instead of ceil to ensure we don't exceed max_ratio
        tokens_to_keep = math.floor(seq_len / max_ratio)
        # Ensure we keep at least 4 tokens (more conservative)
        tokens_to_keep = max(tokens_to_keep, 4)
        tokens_to_compress = seq_len - tokens_to_keep

        # Print target compression info - MAKE LESS VERBOSE
        if estimated_compression > 0 and (
            not self.training or random.random() < 0.05
        ):  # Only print 5% of the time during training
            # Avoid division by zero by ensuring denominator is at least 1
            remaining_tokens = max(1, seq_len - estimated_compression)
            estimated_ratio = seq_len / remaining_tokens

            print(
                f"Target compression: {max_ratio:.2f}x (keeping {tokens_to_keep}/{seq_len} tokens, compressing {tokens_to_compress})"
            )

            if (
                estimated_ratio > max_ratio
            ):  # Use configurable upper bound on compression ratio
                print(
                    f"🚨 EMERGENCY: Estimated compression ratio too high ({estimated_ratio:.2f}x)!"
                )
                print(
                    f"Forcing similarity threshold higher to prevent model destruction..."
                )

                # Force much higher threshold to prevent catastrophic compression
                similarity_threshold_val = max(
                    similarity_threshold_val, 0.99
                )  # Increased from 0.95
                if isinstance(similarity_threshold, torch.Tensor):
                    similarity_threshold = torch.tensor(
                        similarity_threshold_val, device=device
                    )
                else:
                    similarity_threshold = similarity_threshold_val
                print(f"Emergency threshold adjustment: {similarity_threshold_val:.4f}")
                # Also update base threshold for future iterations
                self.base_threshold = max(
                    self.base_threshold, 0.95
                )  # Increased from 0.90

        # NEW: Create a list to track which tokens have been compressed
        compressed_token_count = 0
        already_compressed = set()

        # Process each token with enhanced similarity analysis
        for i in range(seq_len):
            # NEW: Skip if we've already compressed the maximum allowed tokens
            if (
                compressed_token_count >= tokens_to_compress
                and i not in already_compressed
            ):
                # Just use standard compression for this token without merging
                compressed_token = self.compressor(tokens[:, i, :])
                factor = self.compression_factor
                compressed_centroids.append(compressed_token)
                token_compression_factors.append(factor)
                continue

            token = tokens[:, i, :]  # Shape: [batch_size, d_model]

            # Enhanced importance-based compression
            if i in important_indices:
                # Important tokens get lighter compression
                compressed_token = self.compressor(
                    token
                )  # Shape: [batch_size, compressed_dim]
                factor = max(self.min_compression_factor, self.compression_factor - 1)
            else:
                # Find similar tokens with stricter similarity threshold
                similarities = torch.cosine_similarity(
                    token.unsqueeze(1), tokens, dim=2
                )
                similar_indices = torch.where(similarities > similarity_threshold_val)[
                    1
                ]

                if (
                    len(similar_indices) > 1
                    and compressed_token_count < tokens_to_compress
                ):
                    # Enhanced clustering with attention to token positions
                    cluster_tokens = tokens[:, similar_indices, :]
                    # Weight tokens by position similarity
                    pos_weights = torch.exp(
                        -0.1
                        * torch.abs(
                            torch.arange(len(similar_indices), device=device)
                            - similar_indices.float()
                        )
                    )
                    weighted_tokens = cluster_tokens * pos_weights.unsqueeze(
                        -1
                    ).unsqueeze(0)
                    centroid = torch.sum(weighted_tokens, dim=1) / torch.sum(
                        pos_weights
                    )

                    # Compress with adaptive factor based on cluster size
                    cluster_size = len(similar_indices)
                    # Get max_compression_ratio if it exists, otherwise use default max factor
                    max_factor = min(
                        getattr(
                            self, "max_compression_ratio", self.max_compression_factor
                        ),
                        self.max_compression_factor,
                    )

                    # More conservative adaptive factor calculation
                    # Use logarithm with base 4 instead of base 2 for slower growth
                    log_factor = math.log(max(cluster_size, 2)) / math.log(4)
                    adaptive_factor = min(
                        max_factor, self.compression_factor + int(log_factor)
                    )

                    # Cap the adaptive factor to avoid exceeding max_ratio
                    if hasattr(self, "max_compression_ratio"):
                        adaptive_factor = min(
                            adaptive_factor, self.max_compression_ratio
                        )

                    compressed_token = self.compressor(centroid)
                    factor = adaptive_factor

                    # Track which tokens have been compressed
                    compressed_token_count += (
                        len(similar_indices) - 1
                    )  # -1 because we're keeping one token
                    already_compressed.update(similar_indices.tolist())
                else:
                    # No similar tokens found - use standard compression
                    compressed_token = self.compressor(token)
                    factor = self.compression_factor

            # Project to uniform compressed dimension with enhanced precision
            if compressed_token.shape[-1] != self.compressed_dim:
                projection = nn.Linear(
                    compressed_token.shape[-1], self.compressed_dim, device=device
                )
                # Initialize projection weights to preserve variance
                nn.init.orthogonal_(projection.weight)
                compressed_token = projection(compressed_token)

            # Apply L2 normalization to maintain vector magnitudes
            compressed_token = F.normalize(compressed_token, p=2, dim=-1)

            compressed_centroids.append(compressed_token)
            token_compression_factors.append(factor)

        # Stack results
        compressed_centroids = torch.stack(
            compressed_centroids, dim=1
        )  # Shape: [batch_size, seq_len, compressed_dim]
        token_compression_factors = torch.tensor(
            token_compression_factors, device=device
        )

        # Create dummy clusters tensor for compatibility with our enhanced implementation
        clusters = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)

        # Print actual compression achieved - MAKE LESS VERBOSE
        if (
            not self.training or random.random() < 0.05
        ):  # Only print 5% of the time during training
            # Ensure we never compress all tokens (avoid division by zero)
            safe_compressed_token_count = min(compressed_token_count, seq_len - 1)
            actual_ratio = (
                seq_len / (seq_len - safe_compressed_token_count)
                if safe_compressed_token_count > 0
                else 1.0
            )
            print(
                f"Actual compression ratio: {actual_ratio:.2f}x (compressed {compressed_token_count}/{seq_len} tokens)"
            )

        return (
            CompressionState(
                centroids=compressed_centroids,
                compression_factors=token_compression_factors,
                original_tokens=tokens,
            ),
            clusters,
        )

    def forward(self, hidden_states, attention_scores=None, labels=None):
        """Forward pass with dynamic compression and emergency parachute."""
        # Store original states for reconstruction loss
        original_states = hidden_states.clone()

        # Calculate adaptive threshold for this forward pass if in training mode
        if self.adaptive_compression and self.training:
            adaptive_threshold = self.compute_adaptive_threshold(hidden_states)
            # Actually use the computed threshold instead of just the base_threshold
            self.current_threshold = adaptive_threshold
        else:
            self.current_threshold = self.base_threshold

        # Apply compression - pass the current threshold to compress method
        compressed_state, clusters = self.compress(hidden_states, attention_scores)

        # Decompress the state
        if isinstance(compressed_state, CompressionState):
            compressed_states = self.decompress(compressed_state)
        else:
            compressed_states = compressed_state

        # If we have labels, compute language modeling loss
        ce_loss = None
        if labels is not None and self.training:
            # This assumes we have a projection layer to vocabulary size
            if hasattr(self, "output_projection"):
                logits = self.output_projection(compressed_states)
                ce_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), labels.view(-1)
                )

                # Emergency parachute: if CE loss spikes, reduce compression
                if self.last_ce_loss is not None:
                    loss_increase = ce_loss / self.last_ce_loss

                    # If loss increases dramatically, adjust threshold for next iteration
                    # Lower the threshold from 1.5 to 1.2 for more sensitivity
                    if loss_increase > 1.2:  # Was 1.5
                        self.base_threshold = min(0.95, self.base_threshold + 0.05)
                        print(
                            f"⚠️ Emergency parachute deployed! Increasing threshold to {self.base_threshold:.2f}"
                        )
                    # Make this more aggressive too
                    elif loss_increase < 0.9:  # Was 0.8
                        self.base_threshold = max(
                            0.65, self.base_threshold - 0.02
                        )  # More reduction
                        print(
                            f"🚀 Compression successful! Decreasing threshold to {self.base_threshold:.2f}"
                        )

                self.last_ce_loss = ce_loss.detach()

        # Compute contrastive loss if training
        cont_loss = 0
        if self.training and clusters is not None:
            cont_loss = self.contrastive_loss(hidden_states, clusters)

        # Compute reconstruction loss
        rec_loss = F.mse_loss(compressed_states, original_states)

        # Total loss combines all components
        total_loss = 0
        if ce_loss is not None:
            total_loss = ce_loss + 0.1 * cont_loss + 0.05 * rec_loss

        return compressed_states, total_loss

    def enable_compression(self):
        """Enable compression during inference."""
        self.compression_enabled = True

    def disable_compression(self):
        """Disable compression during inference."""
        self.compression_enabled = False

    def calculate_token_importance(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Calculate importance scores for each token to determine which should be preserved.

        Args:
            activations (torch.Tensor): Token activations of shape (num_tokens, d_model)

        Returns:
            importance (torch.Tensor): Importance scores of shape (num_tokens,)
        """
        # Simple approach: use a learned projection to estimate importance
        importance = self.importance_estimator(activations).squeeze(-1)

        # Normalize to [0, 1] range
        importance = torch.sigmoid(importance)

        return importance

    def detect_redundancy(
        self,
        activations: torch.Tensor,
        positions: List[int] = None,
        token_ids: List[int] = None,
    ) -> Tuple[torch.Tensor, Dict[int, int]]:
        """
        Detect redundant tokens in the input activations.

        Args:
            activations (torch.Tensor): Token activations of shape (num_tokens, d_model)
            positions (List[int], optional): Token positions for positional weighting
            token_ids (List[int], optional): Token IDs for special token preservation

        Returns:
            Tuple[torch.Tensor, Dict[int, int]]: Centroids and token mapping
        """
        # Calculate pairwise cosine similarity
        activations_norm = F.normalize(activations, p=2, dim=1)
        similarity = torch.mm(activations_norm, activations_norm.t())

        # Set diagonal to zero to avoid self-similarity
        similarity.fill_diagonal_(0)

        # Apply positional weighting if positions are provided
        if positions is not None:
            # Create position distance matrix
            num_tokens = similarity.shape[0]
            pos_array = torch.tensor(positions, device=similarity.device)
            pos_dist = torch.abs(pos_array.unsqueeze(1) - pos_array.unsqueeze(0))

            # Convert to similarity (closer = more similar)
            max_dist = float(torch.max(pos_dist))
            if max_dist > 0:  # Avoid division by zero
                pos_sim = 1.0 - (pos_dist / max_dist)

                # Combine semantic and positional similarity
                similarity = (
                    1 - self.position_weight
                ) * similarity + self.position_weight * pos_sim

        # Use current_threshold (which may be adaptive) instead of just base_threshold
        current_threshold = getattr(self, "current_threshold", self.base_threshold)

        # Adaptive thresholding based on percentile
        flat_sim = similarity.view(-1)
        k = int(len(flat_sim) * (self.percentile_threshold / 100.0))
        if k > 0:  # Ensure we have enough elements
            threshold = torch.kthvalue(flat_sim, k).values.item()
            print(
                f"Adaptive threshold based on {self.percentile_threshold}th percentile: {threshold:.4f}"
            )
            # Use the higher of adaptive threshold or current threshold
            effective_threshold = max(threshold, current_threshold)
        else:
            effective_threshold = current_threshold

        print(f"Effective similarity threshold: {effective_threshold:.4f}")
        print(
            f"Max similarity between any two tokens: {torch.max(similarity).item():.4f}"
        )

        # Identify important tokens that should be preserved
        important_indices = set()
        if token_ids is not None and self.preserve_special_tokens:
            # Identify special tokens (this is a simple heuristic for GPT-2)
            for i, token_id in enumerate(token_ids):
                if token_id < 50:  # Most special tokens have low IDs
                    important_indices.add(i)

        # Add tokens with high importance scores
        importance_threshold = (
            self.calculate_token_importance(activations).median()
            + self.calculate_token_importance(activations).std()
        )
        for i, imp in enumerate(self.calculate_token_importance(activations)):
            if imp > importance_threshold:
                important_indices.add(i)

        print(f"Preserving {len(important_indices)} important tokens")

        redundancy_map = {}
        used_indices = set()

        # Greedy approach: For each token, find all that exceed the threshold and map them.
        for i in range(similarity.shape[0]):
            if i in used_indices or i in important_indices:
                continue
            similar_indices = torch.where(similarity[i] > effective_threshold)[
                0
            ].tolist()
            for idx in similar_indices:
                # Only map forward to avoid cycles, and don't map important tokens
                if idx not in used_indices and idx > i and idx not in important_indices:
                    redundancy_map[idx] = i
                    used_indices.add(idx)

        # Unique tokens are those that aren't assigned to anything
        unique_indices = [
            i for i in range(similarity.shape[0]) if i not in redundancy_map
        ]
        compressed_activations = activations[unique_indices]

        # Visualize token clusters (for debugging)
        self.visualize_token_clusters(similarity, redundancy_map, token_ids)

        return (
            compressed_activations,
            redundancy_map,
            self.calculate_token_importance(activations).tolist(),
        )

    def visualize_token_clusters(
        self,
        similarity: torch.Tensor,
        redundancy_map: Dict[int, int],
        token_ids: List[int] = None,
    ):
        """
        Create a simple visualization of token clusters for debugging.

        Args:
            similarity (torch.Tensor): Similarity matrix
            redundancy_map (Dict[int, int]): Mapping of redundant tokens
            token_ids (List[int], optional): Original token IDs
        """
        # Build clusters
        clusters = {}
        for idx, rep_idx in redundancy_map.items():
            # Follow the chain to find the root representative
            root = rep_idx
            while root in redundancy_map:
                root = redundancy_map[root]

            if root not in clusters:
                clusters[root] = []
            clusters[root].append(idx)

        # Print cluster information
        print(f"\nFound {len(clusters)} clusters:")
        for root, members in clusters.items():
            if token_ids is not None:
                try:
                    from transformers import AutoTokenizer

                    tokenizer = AutoTokenizer.from_pretrained("gpt2")
                    root_token = tokenizer.decode([token_ids[root]])
                    member_tokens = [tokenizer.decode([token_ids[m]]) for m in members]
                    print(f"  Cluster {root} ('{root_token}'): {member_tokens}")
                except:
                    print(f"  Cluster {root}: {members}")
            else:
                print(f"  Cluster {root}: {members}")
        print()  # Empty line for readability

    def get_token_importance_by_type(self, token_ids: List[int]) -> torch.Tensor:
        """
        Calculate importance weights for tokens based on their types.
        Prioritizes content words (nouns, verbs, adjectives) over function words.

        Args:
            token_ids: List of token IDs

        Returns:
            Tensor of importance weights for each token
        """
        # Initialize with default importance
        importance_weights = torch.ones(
            len(token_ids),
            device=self.centroids.device if self.centroids is not None else None,
        )

        # Try to load GPT-2 tokenizer for better token type detection
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained("gpt2")

            # Decode each token to get its text
            for i, token_id in enumerate(token_ids):
                token_text = tokenizer.decode([token_id]).strip()

                # Special tokens and punctuation get lower importance
                if token_id < 50 or token_text in [
                    ".",
                    ",",
                    "!",
                    "?",
                    ";",
                    ":",
                    '"',
                    "'",
                    "(",
                    ")",
                    "[",
                    "]",
                    "{",
                    "}",
                ]:
                    importance_weights[i] = 0.5
                # Content words (longer tokens) get higher importance
                elif len(token_text) > 2 and token_text[0] != " ":
                    importance_weights[i] = 3.0  # Increased from 2.0
                # Beginning of words (tokens with leading space) get medium-high importance
                elif token_text.startswith(" ") and len(token_text) > 2:
                    importance_weights[i] = 2.0  # Increased from 1.5

                # Extra boost for likely semantic keywords
                if any(
                    keyword in token_text.lower()
                    for keyword in [
                        "neural",
                        "model",
                        "language",
                        "compress",
                        "semantic",
                        "efficient",
                        "represent",
                        "memory",
                        "footprint",
                    ]
                ):
                    importance_weights[i] *= 1.5

            print("Using enhanced token-type aware importance weights")
        except:
            # Fallback if tokenizer can't be loaded
            print("Using simple token ID-based importance weights")
            for i, token_id in enumerate(token_ids):
                # Special tokens get lower importance
                if token_id < 50:
                    importance_weights[i] = 0.5
                # Higher token IDs (likely content words) get higher importance
                elif token_id > 1000:
                    importance_weights[i] = 2.0  # Increased from 1.5

        return importance_weights

    def compress_residuals(
        self,
        residuals: torch.Tensor,
        original_tokens: torch.Tensor,
        centroids: torch.Tensor,
        assignments: List[int],
        token_ids: List[int] = None,
    ) -> Dict:
        """
        Aggressively compress residuals through sparsification and quantization.
        Now with adaptive storage - only keep residuals for tokens where they matter.
        Uses token-type awareness to prioritize important tokens.

        Args:
            residuals: Tensor of shape (num_tokens, d_model) containing residual vectors
            original_tokens: Original token activations
            centroids: Compressed centroids
            assignments: Which centroid each token maps to
            token_ids: Original token IDs for token-type awareness

        Returns:
            Dictionary containing compressed residual information
        """
        # First, determine which tokens actually need residuals
        # For each token, calculate reconstruction error without residuals
        reconstructed_without_residuals = torch.zeros_like(original_tokens)
        for i, centroid_idx in enumerate(assignments):
            reconstructed_without_residuals[i] = centroids[centroid_idx]

        # Calculate reconstruction error for each token
        reconstruction_error = torch.norm(
            original_tokens - reconstructed_without_residuals, dim=1
        )

        # Get token importance weights based on token types
        token_importance = torch.ones_like(reconstruction_error)
        if token_ids is not None:
            token_importance = self.get_token_importance_by_type(token_ids)

        # Apply token importance to reconstruction error
        weighted_error = reconstruction_error * token_importance

        # Normalize errors to [0,1] range for easier thresholding
        if weighted_error.max() > 0:
            normalized_error = weighted_error / weighted_error.max()
        else:
            normalized_error = weighted_error

        # NEW: Analyze semantic importance of tokens in context
        # We'll use a sliding window approach to identify tokens that are important for context
        semantic_importance = torch.zeros_like(normalized_error)
        window_size = min(
            5, len(normalized_error)
        )  # Use a context window of 5 tokens or less

        # For each token, check if it's semantically important in its local context
        for i in range(len(normalized_error)):
            # Get window around current token
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(normalized_error), i + window_size // 2 + 1)
            window = normalized_error[start_idx:end_idx]

            # If this token has higher error than average in its window, it's important
            if normalized_error[i] > window.mean():
                semantic_importance[i] = normalized_error[i] / window.mean()
            else:
                semantic_importance[i] = 0.5  # Baseline importance

        # Combine semantic importance with normalized error
        combined_importance = (normalized_error + semantic_importance) / 2

        # Use a much lower threshold for determining which tokens need residuals
        # This ensures we keep more residuals, especially for semantically important tokens
        adaptive_threshold = max(0.01, self.residual_importance_threshold / 2)
        needs_residual = combined_importance > adaptive_threshold

        # NEW: Always keep residuals for special tokens and sentence boundaries
        if token_ids is not None:
            for i, token_id in enumerate(token_ids):
                # Special tokens and punctuation often mark important boundaries
                if token_id < 50 or token_id in [
                    13,
                    30,
                    50,
                    764,
                    837,
                    13959,
                    13782,
                ]:  # Common punctuation in GPT-2
                    needs_residual[i] = True

        # Count how many tokens need residuals
        num_tokens_with_residuals = torch.sum(needs_residual).item()
        tokens_with_residuals_pct = 100 * num_tokens_with_residuals / len(assignments)

        print(
            f"Tokens needing residuals: {num_tokens_with_residuals}/{len(assignments)} ({tokens_with_residuals_pct:.1f}%)"
        )

        # If no tokens need residuals, return empty
        if num_tokens_with_residuals == 0:
            return {"format": "empty", "shape": residuals.shape, "sparsity": 1.0}

        # Create mask for tokens that need residuals
        token_mask = needs_residual.unsqueeze(1).expand_as(residuals)

        # Zero out residuals for tokens that don't need them
        masked_residuals = residuals * token_mask

        # Now apply component-wise sparsity to the remaining residuals
        magnitudes = torch.abs(masked_residuals)

        # For each token that needs residuals, find threshold that keeps only top (1-sparsity)% components
        # We need to handle each token separately
        sparse_residuals = torch.zeros_like(masked_residuals)

        # Get token importance for adaptive sparsity
        token_importance_for_sparsity = torch.ones(
            len(assignments), device=residuals.device
        )
        if token_ids is not None:
            token_importance_for_sparsity = self.get_token_importance_by_type(token_ids)

        # NEW: Analyze which dimensions are most important across all tokens
        # This helps identify which embedding dimensions carry the most semantic information
        global_dim_importance = torch.sum(magnitudes, dim=0)
        global_dim_importance = global_dim_importance / global_dim_importance.sum()

        # Sort dimensions by importance
        _, sorted_dims = torch.sort(global_dim_importance, descending=True)

        # Keep track of which dimensions we preserve for each token
        preserved_dims = []

        for i in range(len(assignments)):
            if needs_residual[i]:
                # Get magnitudes for this token
                token_magnitudes = magnitudes[i]

                # Adaptive sparsity based on token importance
                # More important tokens keep more components
                # More aggressive scaling to keep more components for important tokens
                token_sparsity = max(
                    0.5,  # Never remove more than 50% of components for any token
                    self.residual_sparsity
                    - (0.4 * (token_importance_for_sparsity[i] - 1.0)),
                )

                # NEW: For very important tokens, keep even more components
                if token_importance_for_sparsity[i] > 2.0:
                    token_sparsity = max(0.3, token_sparsity - 0.2)

                # NEW: Hybrid approach to component selection
                # 1. Always keep the globally important dimensions
                num_global_dims_to_keep = int(
                    self.d_model * 0.1
                )  # Keep top 10% global dims
                global_dims_to_keep = sorted_dims[:num_global_dims_to_keep]

                # 2. For remaining dimensions, use token-specific thresholding
                remaining_dims = sorted_dims[num_global_dims_to_keep:]
                token_specific_magnitudes = token_magnitudes[remaining_dims]

                # Calculate how many more dimensions to keep
                remaining_to_keep = (
                    int(self.d_model * (1 - token_sparsity)) - num_global_dims_to_keep
                )
                remaining_to_keep = max(0, remaining_to_keep)

                if remaining_to_keep > 0 and len(token_specific_magnitudes) > 0:
                    # Find threshold for token-specific dimensions
                    threshold = torch.quantile(
                        token_specific_magnitudes,
                        1.0 - (remaining_to_keep / len(remaining_dims)),
                    )

                    # Create mask for token-specific dimensions to keep
                    token_specific_mask = token_magnitudes[remaining_dims] > threshold
                    token_specific_dims_to_keep = remaining_dims[token_specific_mask]
                else:
                    token_specific_dims_to_keep = torch.tensor(
                        [], device=residuals.device, dtype=torch.long
                    )

                # Combine global and token-specific dimensions
                all_dims_to_keep = torch.cat(
                    [global_dims_to_keep, token_specific_dims_to_keep]
                )

                # Create the final mask
                component_mask = torch.zeros_like(token_magnitudes, dtype=torch.bool)
                component_mask[all_dims_to_keep] = True

                # Store which dimensions we preserved
                preserved_dims.append(all_dims_to_keep.tolist())

                # Apply mask
                sparse_residuals[i] = residuals[i] * component_mask

                # Print debug info for a few tokens
                if i < 3 or i > len(assignments) - 3:
                    print(
                        f"Token {i}: importance={token_importance_for_sparsity[i]:.2f}, sparsity={token_sparsity:.2f}, components kept={component_mask.sum().item()}/{len(component_mask)}"
                    )

        # Count non-zeros to report sparsity
        total_elements = sparse_residuals.numel()
        nonzero_elements = torch.count_nonzero(sparse_residuals).item()
        achieved_sparsity = 1.0 - (nonzero_elements / total_elements)

        print(f"Residual sparsity: {achieved_sparsity:.2%}")

        # Quantize the non-zero values to reduced precision
        if nonzero_elements > 0:
            # NEW: Adaptive bit precision based on token importance
            # More important tokens get higher precision
            token_bits = []
            for i in range(len(assignments)):
                if needs_residual[i]:
                    # Base bits on token importance
                    importance = token_importance_for_sparsity[i]
                    # Scale from 8 bits (low importance) to 16 bits (high importance)
                    adaptive_bits = min(16, max(8, int(8 + 8 * (importance - 1) / 2)))
                    token_bits.append(adaptive_bits)
                else:
                    token_bits.append(0)  # No bits for tokens without residuals

            # Find min and max for scaling
            nonzero_mask = sparse_residuals != 0
            nonzero_values = sparse_residuals[nonzero_mask]
            min_val = nonzero_values.min().item()
            max_val = nonzero_values.max().item()

            # Avoid division by zero
            if max_val == min_val:
                scale = 1.0
            else:
                scale = (max_val - min_val) / (2**self.residual_bits - 1)

            # Quantize to specified bits
            quantized = torch.round((sparse_residuals - min_val) / scale)
            quantized = torch.clamp(quantized, 0, 2**self.residual_bits - 1)

            # Store in efficient sparse format (COO format)
            indices = torch.nonzero(nonzero_mask)
            values = quantized[nonzero_mask].to(
                torch.uint8 if self.residual_bits <= 8 else torch.int16
            )

            return {
                "format": "sparse_quantized",
                "indices": indices,
                "values": values,
                "min_val": min_val,
                "scale": scale,
                "shape": residuals.shape,
                "sparsity": achieved_sparsity,
                "bits": self.residual_bits,
                "token_bits": token_bits,
                "preserved_dims": preserved_dims,
                "tokens_with_residuals": needs_residual.sum().item(),
                "total_tokens": len(assignments),
            }
        else:
            # No residuals needed
            return {"format": "empty", "shape": residuals.shape, "sparsity": 1.0}

    def decompress_residuals(self, compressed_residuals: Dict) -> torch.Tensor:
        """
        Decompress residuals from sparse, quantized format.

        Args:
            compressed_residuals: Dictionary containing compressed residual information

        Returns:
            Tensor of decompressed residuals
        """
        if compressed_residuals["format"] == "empty":
            return torch.zeros(
                compressed_residuals["shape"], device=self.centroids.device
            )

        # Extract components
        indices = compressed_residuals["indices"]
        values = compressed_residuals["values"]
        min_val = compressed_residuals["min_val"]
        scale = compressed_residuals["scale"]
        shape = compressed_residuals["shape"]

        # Create empty tensor
        decompressed = torch.zeros(shape, device=indices.device)

        # Convert quantized values back to original scale
        dequantized_values = values.float() * scale + min_val

        # Place values back in tensor
        for i in range(indices.shape[0]):
            idx = tuple(indices[i].tolist())
            decompressed[idx] = dequantized_values[i]

        return decompressed

    def decompress(self, state: CompressionState) -> torch.Tensor:
        """Decompress tokens back to their original dimension."""
        batch_size, seq_len, compressed_dim = state.centroids.shape

        # Process each token in the sequence
        decompressed_tokens = []
        for i in range(seq_len):
            # Get the compressed token for this position
            compressed = state.centroids[:, i, :]  # Shape: [batch_size, compressed_dim]

            # Decompress the token
            decompressed = self.decompressor(compressed)  # Shape: [batch_size, d_model]
            decompressed_tokens.append(decompressed)

        # Stack along sequence dimension
        decompressed = torch.stack(
            decompressed_tokens, dim=1
        )  # Shape: [batch_size, seq_len, d_model]

        return decompressed

    def get_token_importance(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Get importance scores for tokens.

        Args:
            activations (torch.Tensor): Token activations of shape (num_tokens, d_model)

        Returns:
            torch.Tensor: Importance scores of shape (num_tokens,)
        """
        return self.calculate_token_importance(activations)

    def get_compressed_indices(self, activations: torch.Tensor) -> List[int]:
        """
        Get indices of tokens that would be compressed.

        Args:
            activations (torch.Tensor): Token activations of shape (num_tokens, d_model)

        Returns:
            List[int]: Indices of tokens that would be compressed
        """
        # Calculate importance
        importance = self.calculate_token_importance(activations)

        # Calculate similarity
        similarity = torch.mm(
            F.normalize(activations, p=2, dim=1),
            F.normalize(activations, p=2, dim=1).t(),
        )

        # Set diagonal to zero to avoid self-similarity
        similarity.fill_diagonal_(0)

        # Determine threshold
        if self.percentile_threshold > 0:
            # Use adaptive thresholding based on percentile
            flat_sim = similarity.flatten()
            threshold = torch.quantile(
                flat_sim, self.percentile_threshold / 100.0
            ).item()
            effective_threshold = max(threshold, self.base_threshold)
        else:
            effective_threshold = self.base_threshold

        # Identify important tokens that should be preserved
        importance_threshold = importance.median() + importance.std()
        important_indices = set(
            [i for i, imp in enumerate(importance) if imp > importance_threshold]
        )

        # Find redundant tokens
        redundancy_map = {}
        used_indices = set()

        # Greedy approach: For each token, find all that exceed the threshold and map them
        for i in range(similarity.shape[0]):
            if i in used_indices or i in important_indices:
                continue
            similar_indices = torch.where(similarity[i] > effective_threshold)[
                0
            ].tolist()
            for idx in similar_indices:
                # Only map forward to avoid cycles, and don't map important tokens
                if idx not in used_indices and idx > i and idx not in important_indices:
                    redundancy_map[idx] = i
                    used_indices.add(idx)

        # Return indices of tokens that would be compressed
        return list(redundancy_map.keys())

    def update_compression_schedule(self, epoch, total_epochs):
        """
        Update compression parameters based on training progress.
        Implements a very gradual compression schedule targeting only 10% improvement.

        Args:
            epoch (int): Current epoch (0-indexed)
            total_epochs (int): Total number of epochs for training
        """
        if not self.training:
            return

        # Calculate progress factor (0 at start, 1 at end)
        progress = epoch / max(1, total_epochs - 1)

        # Handle the case when base_threshold is a list or tuple
        if isinstance(self.base_threshold, (list, tuple)):
            # Just use the first value for simplicity
            base_value = float(self.base_threshold[0])
        else:
            base_value = float(self.base_threshold)

        # MINIMAL COMPRESSION SCHEDULE:
        # Start with extremely conservative thresholds and make tiny adjustments
        # Target only 10% improvement in compression ratio

        # First 40% of training: No compression (learning phase)
        if progress < 0.4:
            # No compression at all - just learn the task
            target_threshold = 0.999  # Almost no compression
        # Middle 40% of training: Very minimal compression
        elif progress < 0.8:
            # Very slight compression - linear decrease from 0.999 to base_value
            # This ensures extremely gradual introduction of compression
            target_threshold = (
                0.999 - (0.999 - base_value) * ((progress - 0.4) / 0.4) * 0.5
            )
        # Final 20% of training: Reach target compression (still conservative)
        else:
            # Reach the target threshold (which is already conservative)
            target_threshold = base_value

        # Smoothly interpolate current threshold
        self.current_threshold = target_threshold

        print(
            f"Epoch {epoch+1}/{total_epochs}: Updated compression threshold to {self.current_threshold:.4f}"
        )

    def set_similarity_threshold(self, threshold):
        """Update the similarity threshold used for compression"""
        self.similarity_threshold = threshold
        # If there are any sub-layers or components that need the threshold, update them too
        # For example, if you have attention mechanisms that use this threshold


class RecursiveCompressionModel(nn.Module):
    """
    Model that uses recursive compression internally.

    Args:
        base_model_name (str): Name of the HuggingFace model to load as the backbone.
        num_compression_layers (int): Number of compression layers to stack.
        compression_factor (int): Factor by which each compression layer reduces dimensionality.
        freeze_base_model (bool): Whether to freeze the base model parameters.
        similarity_threshold (float): Threshold for redundancy detection in each compression layer.
        position_weight (float): Weight given to positional proximity in similarity calculation.
        percentile_threshold (float): Percentile to use for adaptive thresholding (0-100).
        preserve_special_tokens (bool): Whether to preserve special tokens from compression.
        residual_compression_factor (int): Factor by which residuals are compressed.
        use_residuals (bool): Whether to use residual vectors for reconstruction.
        residual_gate_threshold (float): Threshold for residual gating (0-1).
        residual_sparsity (float): Percentage of residual components to prune (0-1).
        residual_bits (int): Number of bits to use for residual quantization.
        residual_importance_threshold (float): Only keep residuals for tokens above this threshold.
        adaptive_compression (bool): Whether to use adaptive compression based on token importance.
        min_compression_factor (int): Minimum compression factor for important tokens.
        max_compression_factor (int): Maximum compression factor for unimportant tokens.
    """

    def __init__(
        self,
        base_model_name: str = "gpt2",
        num_compression_layers: int = 1,
        compression_factor: int = 2,
        freeze_base_model: bool = True,
        similarity_threshold: float = 0.95,
        position_weight: float = 0.2,
        percentile_threshold: float = 95.0,
        preserve_special_tokens: bool = True,
        residual_compression_factor: int = 4,
        use_residuals: bool = True,
        residual_gate_threshold: float = 0.1,
        residual_sparsity: float = 0.9,
        residual_bits: int = 8,
        residual_importance_threshold: float = 0.1,
        adaptive_compression: bool = False,
        min_compression_factor: int = 2,
        max_compression_factor: int = 8,
    ):
        super().__init__()
        # Load base model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.base_model = AutoModel.from_pretrained(base_model_name)

        # Freeze or unfreeze base model as specified
        if freeze_base_model:
            for param in self.base_model.parameters():
                param.requires_grad = False

        self.d_model = self.base_model.config.hidden_size

        # Build stacked compression layers with all parameters properly forwarded
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
                    residual_sparsity=residual_sparsity,
                    residual_bits=residual_bits,
                    residual_importance_threshold=residual_importance_threshold,
                    adaptive_compression=adaptive_compression,
                    min_compression_factor=min_compression_factor,
                    max_compression_factor=max_compression_factor,
                )
                for _ in range(num_compression_layers)
            ]
        )

        # Enhanced output projection with a small transformer-like block
        self.output_projection = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.tokenizer.vocab_size),
        )

    def forward(self, input_ids):
        """Forward pass with compression."""
        # Get embeddings and initial hidden states
        inputs_embeds = self.base_model.get_input_embeddings()(input_ids)
        hidden_states = inputs_embeds

        # Store compression states for each layer
        compression_states = []

        # Process through each layer
        for layer in self.compression_layers:
            # Compress the current hidden states
            state, clusters = layer.compress(hidden_states)
            compression_states.append((state, clusters))

            # Decompress for the next layer
            hidden_states = layer.decompress(state)

        # Final projection to vocabulary
        logits = self.output_projection(hidden_states)

        return logits, compression_states

    def calculate_compression_ratio(
        self, compression_states: List[Tuple[CompressionState, torch.Tensor]]
    ) -> float:
        """Calculate the effective compression ratio across all layers."""
        total_original_size = 0
        total_compressed_size = 0

        for state, clusters in compression_states:
            # Original size: batch_size * seq_len * d_model
            original_size = (
                state.original_tokens.shape[0]
                * state.original_tokens.shape[1]
                * state.original_tokens.shape[2]
            )

            # Compressed size: batch_size * seq_len * compressed_dim
            compressed_size = (
                state.centroids.shape[0]
                * state.centroids.shape[1]
                * state.centroids.shape[2]
            )

            total_original_size += original_size
            total_compressed_size += compressed_size

        # Calculate overall compression ratio
        compression_ratio = (
            total_original_size / total_compressed_size
            if total_compressed_size > 0
            else 1.0
        )

        return compression_ratio


class CompressionAwareLanguageModelLoss(nn.Module):
    """Custom loss function that balances compression and quality."""

    def __init__(
        self,
        quality_weight=0.6,  # Increased from 0.5
        compression_weight=0.3,  # Reduced from 0.4
        residual_weight=0.1,
        min_quality_threshold=0.3,  # Reduced from 0.6 for smoother ramp-up
        target_compression_ratio=3.0,
        quality_warmup_steps=50,  # Reduced for faster quality emphasis
        ce_loss_scale=20.0,  # Increased from 15.0 for softer quality metric
    ):
        super().__init__()
        self.quality_weight = quality_weight
        self.compression_weight = compression_weight
        self.residual_weight = residual_weight
        self.min_quality_threshold = min_quality_threshold
        self.target_compression_ratio = target_compression_ratio
        self.quality_warmup_steps = quality_warmup_steps
        self.ce_loss_scale = ce_loss_scale
        self.ce_loss = nn.CrossEntropyLoss()
        self.step_counter = 0

        # Exponential moving averages for adaptive scaling
        self.ema_ce_loss = None
        self.ema_alpha = 0.9

    def forward(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        compression_states: List[Tuple[CompressionState, torch.Tensor]],
        original_activations: torch.Tensor,
        reconstructed_activations: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Calculate the combined loss with dynamic weighting and adaptive scaling."""
        # Increment step counter
        self.step_counter += 1

        # Calculate warmup factor for quality weight
        quality_warmup = min(self.step_counter / self.quality_warmup_steps, 1.0)

        # Cross-entropy loss for language modeling
        ce_loss = self.ce_loss(logits.view(-1, logits.size(-1)), target_ids.view(-1))

        # Update exponential moving average of CE loss
        if self.ema_ce_loss is None:
            self.ema_ce_loss = ce_loss.item()
        else:
            self.ema_ce_loss = self.ema_ce_loss * self.ema_alpha + ce_loss.item() * (
                1 - self.ema_alpha
            )

        # Adaptive scaling based on EMA
        ce_scale = max(self.ce_loss_scale, self.ema_ce_loss * 1.5)

        # Calculate quality metric with adaptive scaling
        base_quality = 1.0 - min(ce_loss.item() / ce_scale, 1.0)
        quality_metric = base_quality * quality_warmup

        # Dynamic quality threshold based on training progress
        current_threshold = self.min_quality_threshold * (1.0 - quality_warmup * 0.5)

        # Quality threshold penalty with smoother transition
        if quality_metric < current_threshold:
            penalty_factor = (current_threshold - quality_metric) / current_threshold
            quality_penalty = torch.tensor(
                2.0 * torch.sigmoid(torch.tensor(penalty_factor * 5.0)),
                device=logits.device,
            )
        else:
            quality_penalty = torch.tensor(0.0, device=logits.device)

        # Calculate compression ratio and loss
        total_original_size = sum(
            state.original_tokens.numel() for state, clusters in compression_states
        )
        total_compressed_size = sum(
            state.centroids.numel() for state, clusters in compression_states
        )
        compression_ratio = (
            total_original_size / total_compressed_size
            if total_compressed_size > 0
            else 1.0
        )

        # Adaptive compression loss
        if compression_ratio >= self.target_compression_ratio:
            compression_loss = torch.tensor(0.1, device=logits.device)
        else:
            # Smoother ramp-up of compression loss
            ratio_factor = compression_ratio / self.target_compression_ratio
            compression_loss = torch.tensor(
                1.0 - torch.sigmoid(torch.tensor(ratio_factor * 5.0)),
                device=logits.device,
            )

        # Enhanced residual loss with attention to important features
        residual_loss = F.mse_loss(
            F.normalize(reconstructed_activations, dim=-1),
            F.normalize(original_activations, dim=-1),
        )

        # Dynamic weighting based on current metrics and training progress
        effective_quality_weight = self.quality_weight * (1.0 + quality_penalty)
        effective_compression_weight = (
            self.compression_weight
            * (0.5 if compression_ratio >= self.target_compression_ratio else 1.0)
            * (1.0 - quality_warmup * 0.3)
        )  # Reduce compression importance during warmup

        # Scale residual weight based on quality
        effective_residual_weight = self.residual_weight * (
            2.0 if quality_metric < current_threshold else 1.0
        )

        # Combine losses with dynamic weights
        total_loss = (
            (1.0 - quality_metric) * effective_quality_weight
            + compression_loss * effective_compression_weight
            + residual_loss * effective_residual_weight
        )

        return total_loss, {
            "total_loss": total_loss.item(),
            "ce_loss": ce_loss.item(),
            "quality_metric": quality_metric,
            "compression_ratio": compression_ratio,
            "compression_loss": compression_loss.item(),
            "residual_loss": residual_loss.item(),
            "quality_penalty": quality_penalty.item(),
            "effective_quality_weight": effective_quality_weight,
            "effective_compression_weight": effective_compression_weight,
            "effective_residual_weight": effective_residual_weight,
            "ce_scale": ce_scale,
            "current_threshold": current_threshold,
        }


class CompressibleLanguageModelTrainer:
    """
    Trainer for end-to-end training of compressible language models.

    Args:
        model: The compressible language model to train
        optimizer: PyTorch optimizer
        loss_fn: Custom loss function for compression-aware training
        scheduler: Optional learning rate scheduler
        device: Device to train on
    """

    def __init__(
        self,
        model: RecursiveCompressionModel,
        optimizer: torch.optim.Optimizer,
        loss_fn: CompressionAwareLanguageModelLoss,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.device = device
        self.model.to(device)

        # Training metrics
        self.metrics = {
            "train_loss": [],
            "val_loss": [],
            "compression_ratio": [],
            "quality_metric": [],
        }

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform a single training step.

        Args:
            batch: Dictionary containing input_ids and labels

        Returns:
            Dictionary of loss metrics
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Move batch to device
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        # Get original activations before compression
        with torch.no_grad():
            original_outputs = self.model.base_model(input_ids)
            original_activations = original_outputs.last_hidden_state

        # Forward pass with compression
        logits, compression_states = self.model(input_ids)

        # Get the final layer's reconstructed activations by properly decompressing
        final_state, clusters = compression_states[-1]
        reconstructed_activations = self.model.compression_layers[-1].decompress(
            final_state
        )

        # Debug prints to verify shapes
        print(f"\nShape check before potential unsqueeze:")
        print(f"Reconstructed activations shape: {reconstructed_activations.shape}")
        print(f"Original activations shape: {original_activations.shape}")

        # Only unsqueeze if needed (if decompress didn't return batched tensor)
        if len(reconstructed_activations.shape) == 2:
            reconstructed_activations = reconstructed_activations.unsqueeze(0)
            print("Applied unsqueeze to add batch dimension")

        # Assert shapes match exactly
        assert (
            reconstructed_activations.shape == original_activations.shape
        ), f"Shape mismatch! reconstructed: {reconstructed_activations.shape}, original: {original_activations.shape}"

        print(f"Final shapes after adjustment:")
        print(f"Reconstructed activations shape: {reconstructed_activations.shape}")
        print(f"Original activations shape: {original_activations.shape}\n")

        # Calculate compression ratio
        compression_ratio = self.model.calculate_compression_ratio(compression_states)

        # Calculate loss
        loss, loss_components = self.loss_fn(
            logits,
            labels,
            compression_states,
            original_activations,
            reconstructed_activations,  # Now using properly decompressed activations
        )

        # Backward pass
        loss.backward()

        # Optimizer step
        self.optimizer.step()

        # Scheduler step if provided
        if self.scheduler is not None:
            self.scheduler.step()

        # Update metrics
        self.metrics["train_loss"].append(loss_components["total_loss"])
        self.metrics["compression_ratio"].append(compression_ratio)
        self.metrics["quality_metric"].append(loss_components["quality_metric"])

        return loss_components

    def validate(self, val_dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Validate the model on a validation dataset.

        Args:
            val_dataloader: DataLoader for validation data

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_compression_ratio = 0.0
        total_quality_metric = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_dataloader:
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Get original activations
                original_outputs = self.model.base_model(input_ids)
                original_activations = original_outputs.last_hidden_state

                # Forward pass with compression
                logits, compression_states = self.model(input_ids)

                # Get reconstructed activations
                reconstructed_activations = logits  # Simplification

                # Calculate compression ratio
                compression_ratio = self.model.calculate_compression_ratio(
                    compression_states
                )

                # Calculate loss
                loss, loss_components = self.loss_fn(
                    logits,
                    labels,
                    compression_states,
                    original_activations,
                    reconstructed_activations,
                )

                # Update metrics
                total_loss += loss_components["total_loss"]
                total_compression_ratio += compression_ratio
                total_quality_metric += loss_components["quality_metric"]
                num_batches += 1

        # Calculate averages
        avg_loss = total_loss / num_batches
        avg_compression_ratio = total_compression_ratio / num_batches
        avg_quality_metric = total_quality_metric / num_batches

        # Update metrics
        self.metrics["val_loss"].append(avg_loss)

        return {
            "val_loss": avg_loss,
            "val_compression_ratio": avg_compression_ratio,
            "val_quality_metric": avg_quality_metric,
        }

    def train(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: Optional[torch.utils.data.DataLoader] = None,
        num_epochs: int = 3,
        log_interval: int = 10,
        save_path: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the model for the specified number of epochs.

        Args:
            train_dataloader: DataLoader for training data
            val_dataloader: Optional DataLoader for validation data
            num_epochs: Number of epochs to train for
            log_interval: How often to log metrics
            save_path: Optional path to save model checkpoints

        Returns:
            Dictionary of training metrics
        """
        print(f"Starting training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")

            # Training loop
            for batch_idx, batch in enumerate(train_dataloader):
                loss_components = self.train_step(batch)

                # Log metrics
                if batch_idx % log_interval == 0:
                    print(
                        f"Batch {batch_idx}: "
                        f"Loss: {loss_components['total_loss']:.4f}, "
                        f"Compression: {loss_components['compression_ratio']:.2f}x, "
                        f"Quality: {loss_components['quality_metric']:.4f}"
                    )

            # Validation
            if val_dataloader is not None:
                val_metrics = self.validate(val_dataloader)
                print(
                    f"Validation: "
                    f"Loss: {val_metrics['val_loss']:.4f}, "
                    f"Compression: {val_metrics['val_compression_ratio']:.2f}x, "
                    f"Quality: {val_metrics['val_quality_metric']:.4f}"
                )

            # Save checkpoint
            if save_path is not None:
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "metrics": self.metrics,
                        "epoch": epoch,
                    },
                    f"{save_path}/checkpoint_epoch_{epoch+1}.pt",
                )

        print("Training complete!")
        return self.metrics


def train_compression_model():
    """Train a compressible language model with the custom loss function."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model with optimized parameters
    model = RecursiveCompressionModel(
        base_model_name="gpt2",
        num_compression_layers=1,  # Start with single layer for better control
        compression_factor=3,  # Reduced from 4 to improve quality
        freeze_base_model=False,
        similarity_threshold=0.92,  # Increased from 0.90 for stricter merging
        position_weight=0.15,  # Increased from 0.10 for better context preservation
        percentile_threshold=90.0,  # Increased from 85.0 for more selective compression
        preserve_special_tokens=True,
        residual_compression_factor=3,  # Reduced from 4 to preserve more detail
        use_residuals=True,
        residual_gate_threshold=0.03,  # Reduced from 0.05 for finer control
        residual_sparsity=0.80,  # Reduced from 0.85 to keep more components
        residual_bits=16,  # Keep at 16 for high precision
        residual_importance_threshold=0.005,  # Reduced from 0.01 for more residuals
        adaptive_compression=True,
        min_compression_factor=2,
        max_compression_factor=6,  # Reduced from 8 for better quality
    ).to(device)

    # Initialize custom loss function with optimized weights and thresholds
    loss_fn = CompressionAwareLanguageModelLoss(
        compression_weight=0.3,  # Reduced from 0.4 to prioritize quality
        residual_weight=0.2,  # Increased from 0.1 for better reconstruction
        quality_weight=0.5,  # Keep at 0.5 as base weight
        min_quality_threshold=0.6,  # Set minimum quality threshold
        target_compression_ratio=3.0,  # Target 3x compression
        quality_warmup_steps=100,  # Gradual quality importance
    )

    # Initialize optimizer with lower learning rate and higher weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-5,  # Reduced from 5e-5 for more stable training
        weight_decay=0.02,  # Increased from 0.01 for better regularization
        betas=(0.9, 0.999),  # Default AdamW betas
        eps=1e-8,  # Default AdamW epsilon
    )

    # Training loop (simplified for demonstration)
    def train_step(input_ids, labels):
        # Move tensors to device
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Get original activations before compression
        with torch.no_grad():
            original_outputs = model.base_model(input_ids)
            original_activations = original_outputs.last_hidden_state

        # Forward pass with compression
        logits, compression_states = model(input_ids)

        # Get the final layer's reconstructed activations by properly decompressing
        final_state, clusters = compression_states[-1]
        reconstructed_activations = model.compression_layers[-1].decompress(final_state)

        # Debug prints to verify shapes
        print(f"\nShape check before potential unsqueeze:")
        print(f"Reconstructed activations shape: {reconstructed_activations.shape}")
        print(f"Original activations shape: {original_activations.shape}")

        # Only unsqueeze if needed (if decompress didn't return batched tensor)
        if len(reconstructed_activations.shape) == 2:
            reconstructed_activations = reconstructed_activations.unsqueeze(0)
            print("Applied unsqueeze to add batch dimension")

        # Assert shapes match exactly
        assert (
            reconstructed_activations.shape == original_activations.shape
        ), f"Shape mismatch! reconstructed: {reconstructed_activations.shape}, original: {original_activations.shape}"

        print(f"Final shapes after adjustment:")
        print(f"Reconstructed activations shape: {reconstructed_activations.shape}")
        print(f"Original activations shape: {original_activations.shape}\n")

        # Calculate compression ratio
        compression_ratio = model.calculate_compression_ratio(compression_states)

        # Calculate loss
        loss, loss_components = loss_fn(
            logits,
            labels,
            compression_states,
            original_activations,
            reconstructed_activations,  # Now using properly decompressed activations
        )

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()

        return loss_components

    # Example usage with dummy data
    print("Training with custom compression-aware loss function:")
    print(
        "Note: This is a demonstration only. In practice, you would use a proper dataset."
    )

    # Create dummy input
    input_text = (
        "This is an example sentence to demonstrate compression-aware training."
    )
    input_ids = model.tokenizer.encode(input_text, return_tensors="pt")
    labels = input_ids.clone()  # For language modeling, labels are the same as inputs

    # Train for a few steps
    for step in range(3):
        print(f"\nStep {step+1}:")
        loss_components = train_step(input_ids, labels)

        print(f"  Loss: {loss_components['total_loss']:.4f}")
        print(f"  Compression Ratio: {loss_components['compression_ratio']:.2f}x")
        print(f"  Quality Metric: {loss_components['quality_metric']:.4f}")
        print(f"  CE Loss: {loss_components['ce_loss']:.4f}")
        print(f"  Compression Loss: {loss_components['compression_loss']:.4f}")
        print(f"  Residual Loss: {loss_components['residual_loss']:.4f}")

    print("\nTraining complete!")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Compression Demo")
    parser.add_argument(
        "--mode",
        type=str,
        default="demo",
        choices=["demo", "train"],
        help="Mode to run: 'demo' for demonstration, 'train' for training",
    )
    args = parser.parse_args()

    if args.mode == "demo":
        demo()
    elif args.mode == "train":
        train_compression_model()
