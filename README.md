# Recursive Neural Compression

This project implements an end-to-end trainable neural compression model for language model activations. The approach integrates compression directly into the language modeling objective, allowing the model to learn how to effectively compress its internal representations while maintaining generation quality.

## Project Components

- `main.py`: Original implementation of post-hoc compression approach
- `train_compression.py`: End-to-end trainable compression model
- `run_training.py`: Script to run the training process
- `sample_data.txt`: Sample text data for training

## Key Features

1. **End-to-End Training**: Compression layers are trained jointly with language modeling objective
2. **Adaptive Thresholding**: Dynamically adjusts similarity thresholds based on token distributions
3. **Residual Compression**: Preserves important information using residual vectors
4. **Contrastive Learning**: Enhances token clustering through contrastive loss
5. **Compression Statistics**: Tracks and reports compression metrics during training

## Compression Techniques

The model implements several compression techniques:

- **Semantic Clustering**: Groups similar token activations to reduce redundancy
- **Token Importance**: Identifies and preserves semantically important tokens
- **Residual Vectors**: Stores differences between original tokens and centroids
- **Gated Residuals**: Learns which parts of residuals are important to keep
- **Adaptive Thresholds**: Adjusts compression based on input characteristics

## Running the Training

To train the model with default settings:

```bash
python run_training.py
```

### Command-line Arguments

- `--model_name`: Base model to use (default: "gpt2")
- `--data_path`: Path to training data (default: "sample_data.txt")
- `--output_dir`: Directory to save model (default: "compressed_model")
- `--batch_size`: Batch size for training (default: 4)
- `--epochs`: Number of epochs to train (default: 3)
- `--learning_rate`: Learning rate (default: 5e-5)
- `--freeze_base`: Freeze base model parameters (flag)
- `--compression_layers`: Number of compression layers (default: 1)
- `--similarity_threshold`: Similarity threshold for compression (default: 0.85)
- `--use_residuals`: Use residual vectors in compression (flag)
- `--adaptive_threshold`: Use adaptive thresholding (flag)
- `--contrastive_weight`: Weight for contrastive loss (default: 0.1)
- `--max_length`: Maximum sequence length (default: 128)

### Example with Custom Settings

```bash
python run_training.py --model_name="gpt2-medium" --batch_size=8 --epochs=5 --use_residuals --adaptive_threshold
```

## Requirements

- PyTorch
- Transformers
- NumPy
- Matplotlib (for visualization)
- scikit-learn (for clustering analysis)

Install dependencies:

```bash
pip install torch transformers numpy matplotlib scikit-learn
```

## Future Directions

1. **Quantization Integration**: Combine with quantization for further compression
2. **Multi-layer Compression**: Apply compression at different layers with varying intensities
3. **Task-specific Fine-tuning**: Optimize compression for specific downstream tasks
4. **Distillation**: Use compressed model as a teacher for smaller student models
5. **Adaptive BPE-level Merging**: Integrate with tokenizer-level compression 


```bash
python run_training.py --model_name gpt2 --data_path sample_data.txt --output_dir compressed_model --batch_size 4 --epochs 3 --learning_rate 5e-5 --compression_layers 2 --similarity_threshold 0.85 --use_residuals --adaptive_threshold --max_length 128 --eval_steps 50 --save_steps 100
```




