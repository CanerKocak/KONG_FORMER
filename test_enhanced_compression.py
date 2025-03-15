import torch
import argparse
from train_compression import CompressibleLanguageModel, generate_text


def test_compression_model(model_path, prompts=None, max_length=100):
    """Test the enhanced compression model."""
    print(f"Loading model from {model_path}")
    model = CompressibleLanguageModel.from_pretrained(model_path)

    # Move model to appropriate device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model.to(device)

    # Print model configuration
    print("\nModel Configuration:")
    print(f"Base model: {model.base_model.config._name_or_path}")
    print(f"Compression layers: {model.compression_layer_indices}")
    print(
        f"Progressive compression: {any(hasattr(layer, 'layer_position') for layer in model.compression_layers)}"
    )

    # Print compression layer thresholds
    print("\nCompression Layer Thresholds:")
    for i, layer in enumerate(model.compression_layers):
        if hasattr(layer, "layer_position"):
            print(
                f"Layer {i} (position {layer.layer_position:.2f}): threshold={layer.base_threshold:.4f}"
            )
        else:
            print(f"Layer {i}: threshold={layer.base_threshold:.4f}")

    # Test text generation
    if prompts is None:
        prompts = [
            "Artificial intelligence will",
            "The future of technology is",
            "In the next decade, humans will",
            "Recursive compression of neural networks",
            "The most efficient way to represent language is",
        ]

    print("\nGenerating text with enhanced compression:")
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")

        # Generate with compression enabled
        model.compression_enabled = True
        compressed_text = generate_text(model, prompt, max_length=max_length)
        print(f"With compression: {compressed_text}")

        # Generate with compression disabled
        for layer in model.compression_layers:
            layer.compression_enabled = False
        uncompressed_text = generate_text(model, prompt, max_length=max_length)
        print(f"Without compression: {uncompressed_text}")

        # Re-enable compression for next prompt
        for layer in model.compression_layers:
            layer.compression_enabled = True

    # Test attention-weighted compression
    print("\nTesting attention-weighted compression:")
    input_text = "The recursive compression of neural networks enables efficient processing of language by identifying redundant patterns in the activation space."
    input_ids = model.tokenizer(input_text, return_tensors="pt").input_ids.to(device)

    # Forward pass with attention
    outputs_with_attention = model(
        input_ids=input_ids,
        output_attentions=True,
    )

    # Forward pass without attention
    for layer in model.compression_layers:
        layer.attention_weight = 0.0
    outputs_without_attention = model(
        input_ids=input_ids,
        output_attentions=False,
    )

    # Restore attention weight
    for layer in model.compression_layers:
        layer.attention_weight = 0.3

    # Compare results
    print(
        f"Compression ratio with attention: {outputs_with_attention['compression_ratio']:.2f}x"
    )
    print(
        f"Compression ratio without attention: {outputs_without_attention['compression_ratio']:.2f}x"
    )

    print("\nTest completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test enhanced compression model")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model directory"
    )
    parser.add_argument(
        "--max_length", type=int, default=100, help="Maximum length for generated text"
    )
    parser.add_argument("--prompt", type=str, help="Custom prompt for text generation")

    args = parser.parse_args()

    # Use custom prompt if provided
    prompts = [args.prompt] if args.prompt else None

    test_compression_model(args.model_path, prompts, args.max_length)
