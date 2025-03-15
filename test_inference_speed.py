import torch
import time
from train_compression import CompressibleLanguageModel

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

model_original = CompressibleLanguageModel(base_model_name="gpt2").to(device)
model_compressed = CompressibleLanguageModel.load("compressed_model/final_model").to(
    device
)

input_ids = model_original.tokenizer("Hello world", return_tensors="pt").input_ids.to(
    device
)


def measure_speed(model, input_ids, runs=10):
    model.eval()
    times = []
    for _ in range(runs):
        start = time.time()
        with torch.no_grad():
            model(input_ids)
        times.append(time.time() - start)
    return sum(times) / runs


# Warm-up run (first run is often slower)
print("Warming up models...")
with torch.no_grad():
    model_original(input_ids)
    model_compressed(input_ids)

print(
    f"Original model avg inference time: {measure_speed(model_original, input_ids):.4f}s"
)
print(
    f"Compressed model avg inference time: {measure_speed(model_compressed, input_ids):.4f}s"
)

# Calculate speedup
original_time = measure_speed(model_original, input_ids)
compressed_time = measure_speed(model_compressed, input_ids)
speedup = original_time / compressed_time if compressed_time > 0 else 0

print(f"Speedup factor: {speedup:.2f}x")
