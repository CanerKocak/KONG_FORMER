# Enhanced Recursive Compression for Language Models

This repository implements an enhanced recursive compression approach for language models, focusing on efficient token representation through semantic similarity detection and clustering.

## Key Features

### 1. Differentiable Adaptive Thresholding
- Dynamically computes similarity thresholds based on content
- Uses a learnable modulator to adjust thresholds based on context
- Combines statistical thresholds with learned components
- Ensures a minimum base threshold as a safety floor

### 2. Attention-Weighted Clustering
- Leverages the model's own attention patterns to guide compression
- Combines embedding similarity with attention similarity
- Tokens that attend to similar contexts are considered functionally similar
- Configurable weighting between embedding and attention similarities

### 3. Contrastive Regularization
- Explicitly pushes dissimilar token clusters apart
- Uses a contrastive projection head to project tokens into a contrastive space
- Applies margin-based contrastive loss between cluster centroids
- Helps maintain semantic distinctions between different token groups

### 4. Emergency Parachute
- Monitors language modeling loss during training
- Automatically adjusts compression thresholds when quality degrades
- Increases threshold (less compression) when loss spikes
- Decreases threshold (more compression) when loss improves

### 5. Progressive Compression
- Applies different compression strategies based on layer depth
- Early layers use conservative thresholds to preserve information
- Middle layers use moderate thresholds
- Later layers use aggressive thresholds for maximum compression

## Implementation Details

### RecursiveCompressionLayer
The core compression layer that implements the following methods:
- `compute_adaptive_threshold`: Dynamically computes thresholds based on content
- `compute_similarity_matrix`: Computes similarity with optional attention weighting
- `contrastive_loss`: Enforces separation between dissimilar token clusters
- `compress`: Compresses hidden states using attention-weighted clustering
- `forward`: Handles dynamic compression with emergency parachute

### ProgressiveCompressionLayer
Extends RecursiveCompressionLayer with layer-position-based thresholds:
- Early layers (position < 0.3): Conservative threshold
- Middle layers (0.3 ≤ position < 0.7): Moderate threshold
- Late layers (position ≥ 0.7): Aggressive threshold

### CompressibleLanguageModel
Integrates compression layers into a language model:
- Supports progressive compression based on layer depth
- Uses attention scores to guide compression
- Tracks compression statistics during training
- Provides save/load functionality for trained models

## Usage

### Training
```bash
python train_compression.py --train_file sample_data.txt --base_model_name gpt2 --use_progressive_compression --attention_weight 0.3 --contrastive_margin 0.2
```

### Testing
```bash
python test_enhanced_compression.py --model_path ./model_output
```

## Monitoring and Visualization
The implementation includes comprehensive monitoring:
- Tracks thresholds, compression ratios, and loss values
- Generates visualizations of compression statistics
- Logs detailed information during training

## Results
The enhanced compression approach offers several benefits:
- Higher compression ratios without quality degradation
- More stable training through adaptive thresholding
- Better preservation of semantic information
- Improved generation quality through attention-guided compression

## Requirements
- PyTorch
- Transformers
- Matplotlib (for visualization)
- tqdm (for progress bars) 