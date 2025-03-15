import torch
import argparse
import time
from train_compression import CompressibleLanguageModel


def generate_text_directly(model, prompt, max_length=100, temperature=0.7, top_p=0.9):
    """Generate text directly using the model's tokenizer and base model."""
    # Tokenize prompt
    inputs = model.tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(model.base_model.device)
    attention_mask = inputs.attention_mask.to(model.base_model.device)

    # Generate text
    with torch.no_grad():
        output = model.base_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=len(input_ids[0]) + max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=model.tokenizer.eos_token_id,
        )

    # Decode and return generated text
    generated_text = model.tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text


def test_generation(model_path, prompts=None, max_length=200):
    """Test text generation with the enhanced compression model."""
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

    # Default prompts if none provided
    if prompts is None:
        prompts = [
            "Recursive compression of neural networks works by",
            "The most efficient way to represent language is to",
            "Attention-weighted clustering in language models helps",
            "Progressive compression across transformer layers enables",
            "The future of efficient language models involves",
        ]

    print("\nGenerating text with the model:")
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")

        # Generate text
        start_time = time.time()
        generated_text = generate_text_directly(model, prompt, max_length=max_length)
        generation_time = time.time() - start_time

        print(f"Generated ({generation_time:.2f}s): {generated_text}")

    print("\nGeneration completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text with the model")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model directory"
    )
    parser.add_argument(
        "--max_length", type=int, default=200, help="Maximum length for generated text"
    )
    parser.add_argument("--prompt", type=str, help="Custom prompt for text generation")

    args = parser.parse_args()

    # Use custom prompt if provided
    prompts = [args.prompt] if args.prompt else None

    test_generation(args.model_path, prompts, args.max_length)
