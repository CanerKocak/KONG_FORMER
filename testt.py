from train_compression import CompressibleLanguageModel, generate_text

# Load the model from checkpoint
try:
    model = CompressibleLanguageModel.from_pretrained("./compressed_model/final_model")
except FileNotFoundError:
    print("Error: Model checkpoint not found. Please check the path and try again.")
    exit(1)

# Generate text
prompt = "Artificial intelligence is"
generated = generate_text(model, prompt, max_length=100)
print(generated)
