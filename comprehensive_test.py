import torch
import time
import psutil
import os
from train_compression import CompressibleLanguageModel


def get_memory_usage():
    """Get the current memory usage of the process in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


# Explicitly check for MPS first on Mac
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"Using MPS device for Apple Silicon")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA device")
else:
    device = torch.device("cpu")
    print(f"Using CPU device")

print(f"Device: {device}")

# Load models
print("Loading original model...")
model_original = CompressibleLanguageModel(base_model_name="gpt2").to(device)
print("Loading compressed model...")
model_compressed = CompressibleLanguageModel.load("compressed_model/final_model").to(
    device
)

# Test with different sequence lengths - reduced for faster completion
sequence_lengths = [32, 64, 128]
runs_per_test = 3

results = {
    "original": {"time": [], "memory": []},
    "compressed": {"time": [], "memory": []},
}

print("\n=== Performance Comparison ===")
print(
    f"{'Sequence Length':<15} {'Original Time':<15} {'Compressed Time':<15} {'Speedup':<10} {'Memory Diff %':<15}"
)
print("-" * 70)

for seq_len in sequence_lengths:
    # Create a longer input sequence
    prompt = "The future of artificial intelligence is " + " ".join(
        ["fascinating"] * (seq_len // 2)
    )
    input_ids = model_original.tokenizer(prompt, return_tensors="pt").input_ids.to(
        device
    )

    if input_ids.shape[1] > seq_len:
        input_ids = input_ids[:, :seq_len]

    # Measure original model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Warm-up
    with torch.no_grad():
        model_original(input_ids)

    # Memory before
    mem_before_orig = get_memory_usage()

    # Time measurement
    start_time = time.time()
    for _ in range(runs_per_test):
        with torch.no_grad():
            model_original(input_ids)
    orig_time = (time.time() - start_time) / runs_per_test

    # Memory after
    mem_after_orig = get_memory_usage()
    mem_diff_orig = mem_after_orig - mem_before_orig

    # Measure compressed model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Warm-up
    with torch.no_grad():
        model_compressed(input_ids)

    # Memory before
    mem_before_comp = get_memory_usage()

    # Time measurement
    start_time = time.time()
    for _ in range(runs_per_test):
        with torch.no_grad():
            model_compressed(input_ids)
    comp_time = (time.time() - start_time) / runs_per_test

    # Memory after
    mem_after_comp = get_memory_usage()
    mem_diff_comp = mem_after_comp - mem_before_comp

    # Calculate metrics
    speedup = orig_time / comp_time if comp_time > 0 else 0
    mem_diff_percent = (
        ((mem_diff_orig - mem_diff_comp) / mem_diff_orig) * 100
        if mem_diff_orig > 0
        else 0
    )

    # Store results
    results["original"]["time"].append(orig_time)
    results["original"]["memory"].append(mem_diff_orig)
    results["compressed"]["time"].append(comp_time)
    results["compressed"]["memory"].append(mem_diff_comp)

    # Print results
    print(
        f"{seq_len:<15} {orig_time:.4f}s{' '*9} {comp_time:.4f}s{' '*9} {speedup:.2f}x{' '*5} {mem_diff_percent:.2f}%"
    )

# Print summary
print("\n=== Summary ===")
avg_speedup = sum(
    [
        o / c if c > 0 else 0
        for o, c in zip(results["original"]["time"], results["compressed"]["time"])
    ]
) / len(sequence_lengths)
print(f"Average speedup: {avg_speedup:.2f}x")

# Parameter count comparison
orig_params = sum(p.numel() for p in model_original.parameters()) / 1e6
comp_params = sum(p.numel() for p in model_compressed.parameters()) / 1e6
param_reduction = ((orig_params - comp_params) / orig_params) * 100

print(f"Original model parameters: {orig_params:.2f} million")
print(f"Compressed model parameters: {comp_params:.2f} million")
print(f"Parameter reduction: {param_reduction:.2f}%")

# Generate text comparison
print("\n=== Text Generation Comparison ===")
prompt = "Artificial intelligence will"
max_length = 100
temperature = 0.8
top_p = 0.9

print(f"Prompt: '{prompt}'")
print("\nOriginal model output:")
with torch.no_grad():
    input_ids = model_original.tokenizer(prompt, return_tensors="pt").input_ids.to(
        device
    )
    output_ids = model_original.base_model.generate(
        input_ids,
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
    input_ids = model_compressed.tokenizer(prompt, return_tensors="pt").input_ids.to(
        device
    )
    output_ids = model_compressed.base_model.generate(
        input_ids,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
    )
    output_text = model_compressed.tokenizer.decode(
        output_ids[0], skip_special_tokens=True
    )
    print(output_text)
