# Recursive Neural Compression

This repository implements a state-of-the-art recursive neural compression model designed specifically for language model activations. Unlike traditional compression methods, this approach integrates compression directly into the language modeling objective, enabling dynamic, on-the-fly compression without significant degradation in generation quality.

## Quick Start with SmolLM-1.7B

For the best results with minimal resources, we recommend using the SmolLM-1.7B model:

```bash
# Train the compression model (only need a few iterations)
python train_enhanced_fixed.py --model_name HuggingFaceTB/SmolLM-1.7B --output_dir compressed_smollm --batch_size 1 --epochs 2 --max_length 64
```

After approximately 3 iterations (not full epochs), you can cancel the training with Ctrl+C. This is sufficient to initialize the compression layers.

Then launch the interactive GUI to experiment with your compressed model:

```bash
python compression_gui.py
```

In the GUI, load your model from the "compressed_smollm" directory and explore the real-time memory efficiency benefits!

## Core Components

- **`train_enhanced_fixed.py`**: Enhanced implementation with adaptive compression ratio limits.
- **`compression_gui.py`**: Interactive GUI for visualizing compression effects and memory usage.
- **`train_compression.py`**: Advanced implementation supporting end-to-end trainable compression.
- **`evaluate.py`**: Evaluation script providing detailed compression metrics.
- **`sample_data.txt`**: Example dataset for quick-start training and testing.

## Highlights

- **End-to-End Trainability**: Compression layers learn simultaneously with the language model, dynamically balancing compression and performance.
- **Adaptive Similarity Thresholding**: Real-time adjustment of clustering thresholds based on token similarity distributions.
- **Residual Information Preservation**: Maintains critical information via gated residuals, preserving generation coherence.
- **Contrastive Token Clustering**: Enforces distinct token clusters through contrastive learning objectives.
- **Comprehensive Metrics Tracking**: Detailed statistics for compression ratios, quality, and loss during training.

## Compression Strategies

The compression system integrates several innovative techniques:

- **Semantic Token Clustering**: Dynamically merges similar activations to reduce redundancy.
- **Token Importance Identification**: Prioritizes essential tokens, minimizing semantic loss.
- **Gated Residual Compression**: Adaptively retains critical activation differences for reconstruction accuracy.
- **Attention-Weighted Similarity**: Utilizes transformer attention mechanisms to guide token clustering decisions.
- **Emergency Quality Parachute**: Automatically reduces compression if generation quality drops.

## Installation

Install necessary dependencies:

```bash
pip install torch transformers numpy matplotlib scikit-learn
```

## Training

To train the model with default settings:

```bash
python run_training.py
```

### Available Arguments

- `--model_name`: HuggingFace model identifier (default: "gpt2")
- `--data_path`: Training data file (default: "sample_data.txt")
- `--output_dir`: Output directory for saving models (default: "compressed_model")
- `--batch_size`: Training batch size (default: 4)
- `--epochs`: Number of epochs (default: 3)
- `--learning_rate`: Optimizer learning rate (default: 5e-5)
- `--freeze_base`: Flag to freeze base model parameters
- `--compression_layers`: Number of compression layers (default: 1)
- `--similarity_threshold`: Base threshold for clustering (default: 0.85)
- `--use_residuals`: Flag to enable residual vectors
- `--adaptive_threshold`: Flag to enable adaptive similarity thresholding
- `--contrastive_weight`: Contrastive loss weight (default: 0.1)
- `--max_length`: Maximum token sequence length (default: 128)

### Example with Custom Parameters

```bash
python run_training.py \
    --model_name="gpt2-medium" \
    --batch_size=8 \
    --epochs=5 \
    --use_residuals \
    --adaptive_threshold
```

## Evaluation

Evaluate your trained model:

```bash
python evaluate.py --model_dir="compressed_model/final_model"
```

## Future Enhancements

- **Quantization Integration**: Further compress activations by combining clustering with quantization.
- **Multi-Layer Progressive Compression**: Vary compression intensity adaptively across transformer layers.
- **Task-Specific Fine-tuning**: Optimize compression strategies for specific downstream applications.
- **Model Distillation**: Use compressed models as efficient teachers for compact student networks.
- **Tokenizer-Level Adaptation**: Integrate adaptive BPE merging with neural compression for greater efficiency.

---

