import torch
from train_compression import CompressibleLanguageModel

# Explicitly use CPU to avoid MPS compatibility issues
device = torch.device("cpu")
print(f"Using device: {device}")

# Load models
print("Loading original model...")
model_original = CompressibleLanguageModel(base_model_name="gpt2").to(device)
print("Loading compressed model...")
model_compressed = CompressibleLanguageModel.load("compressed_model/final_model").to(
    device
)

# Generation parameters
prompts = [
    "Artificial intelligence will",
    "The future of technology is",
    "In the next decade, humans will",
    "The most important invention is",
    "Climate change will affect",
]
max_length = 100
temperature = 0.8
top_p = 0.9

# Generate text for each prompt
for i, prompt in enumerate(prompts):
    print(f"\n\n=== Prompt {i+1}: '{prompt}' ===")

    print("\nOriginal GPT-2 output:")
    with torch.no_grad():
        input_ids = model_original.tokenizer(prompt, return_tensors="pt").input_ids.to(
            device
        )
        attention_mask = torch.ones_like(input_ids)
        output_ids = model_original.base_model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )
        output_text = model_original.tokenizer.decode(
            output_ids[0], skip_special_tokens=True
        )
        print(output_text)

    print("\nCompressed model output:")
    with torch.no_grad():
        input_ids = model_compressed.tokenizer(
            prompt, return_tensors="pt"
        ).input_ids.to(device)
        attention_mask = torch.ones_like(input_ids)
        output_ids = model_compressed.base_model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )
        output_text = model_compressed.tokenizer.decode(
            output_ids[0], skip_special_tokens=True
        )
        print(output_text)

    print("\n" + "-" * 80)
