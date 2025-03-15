import torch
import argparse
from train_compression import CompressibleLanguageModel


def generate_response(model, prompt, max_length=100, temperature=0.8, top_p=0.9):
    """Generate text from a model given a prompt."""
    device = model.base_model.device

    # Tokenize the prompt
    inputs = model.tokenizer(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(inputs.input_ids)

    # Generate text
    with torch.no_grad():
        output_ids = model.base_model.generate(
            inputs.input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )

    # Decode the generated text
    generated_text = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text


def main():
    parser = argparse.ArgumentParser(description="Ask a question to the models")
    parser.add_argument("--prompt", type=str, help="The prompt to give to the models")
    parser.add_argument(
        "--max_length", type=int, default=100, help="Maximum length of generated text"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Temperature for sampling"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.9, help="Top-p sampling parameter"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["original", "compressed", "both"],
        default="both",
        help="Which model to use: original, compressed, or both",
    )

    args = parser.parse_args()

    # Use CPU to avoid MPS compatibility issues
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load the requested models
    if args.model in ["original", "both"]:
        print("Loading original model...")
        model_original = CompressibleLanguageModel(base_model_name="gpt2").to(device)

    if args.model in ["compressed", "both"]:
        print("Loading compressed model...")
        model_compressed = CompressibleLanguageModel.load(
            "compressed_model/final_model"
        ).to(device)

    # If no prompt was provided, enter interactive mode
    if not args.prompt:
        print("\nEntering interactive mode. Type 'exit' to quit.")
        while True:
            prompt = input("\nEnter your prompt: ")
            if prompt.lower() == "exit":
                break

            if args.model in ["original", "both"]:
                print("\nOriginal GPT-2 response:")
                response = generate_response(
                    model_original,
                    prompt,
                    max_length=args.max_length,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                print(response)

            if args.model in ["compressed", "both"]:
                print("\nCompressed model response:")
                response = generate_response(
                    model_compressed,
                    prompt,
                    max_length=args.max_length,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                print(response)

            print("\n" + "-" * 80)
    else:
        # Generate responses for the provided prompt
        if args.model in ["original", "both"]:
            print("\nOriginal GPT-2 response:")
            response = generate_response(
                model_original,
                args.prompt,
                max_length=args.max_length,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            print(response)

        if args.model in ["compressed", "both"]:
            print("\nCompressed model response:")
            response = generate_response(
                model_compressed,
                args.prompt,
                max_length=args.max_length,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            print(response)


if __name__ == "__main__":
    main()
