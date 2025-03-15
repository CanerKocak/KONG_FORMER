import torch
from train_compression import CompressibleLanguageModel

model_original = CompressibleLanguageModel(base_model_name="gpt2")
model_compressed = CompressibleLanguageModel.load("compressed_model/final_model")


def count_params(model):
    return sum(p.numel() for p in model.parameters())


print(f"Original GPT-2 params: {count_params(model_original)/1e6:.2f} million")
print(f"Compressed model params: {count_params(model_compressed)/1e6:.2f} million")
