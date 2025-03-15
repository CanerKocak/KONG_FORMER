import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import psutil
import time
from torch.utils.data import DataLoader, Dataset
from train_compression import CompressibleLanguageModel, train, generate_text


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.encodings = tokenizer(
            texts, truncation=True, padding="max_length", max_length=max_length
        )

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}


class ImprovedCompression:
    """
    Implements multiple compression improvement strategies:
    1. Layer importance analysis to select optimal layers
    2. Progressive compression with gradual increase
    3. Adaptive similarity thresholds
    4. Memory usage tracking
    5. Generation quality evaluation
    """

    def __init__(
        self,
        base_model_name="gpt2",
        train_file="sample_data.txt",
        output_dir="improved_compressed_model",
        initial_compression_factor=1,
        final_compression_factor=3,
        initial_similarity_threshold=0.9,
        final_similarity_threshold=0.8,
        position_weight=0.1,
        epochs=8,
        batch_size=4,
        learning_rate=5e-5,
        max_length=512,
        residual_compression_factor=2,
        use_residuals=True,
    ):
        self.base_model_name = base_model_name
        self.train_file = train_file
        self.output_dir = output_dir
        self.initial_compression_factor = initial_compression_factor
        self.final_compression_factor = final_compression_factor
        self.initial_similarity_threshold = initial_similarity_threshold
        self.final_similarity_threshold = final_similarity_threshold
        self.position_weight = position_weight
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.residual_compression_factor = residual_compression_factor
        self.use_residuals = use_residuals

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Load training data
        print(f"Loading training data from {train_file}")
        with open(train_file, "r", encoding="utf-8") as f:
            self.train_texts = [line.strip() for line in f]

        print(f"Loaded {len(self.train_texts)} training examples")

        # Load base model for analysis
        print(f"Loading base model {base_model_name} for analysis")
        self.base_model = CompressibleLanguageModel(base_model_name=base_model_name)

        # Set device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        print(f"Using device: {self.device}")
        self.base_model.to(self.device)

        # Evaluation prompts
        self.eval_prompts = [
            "Artificial intelligence will",
            "The future of technology is",
            "In the next decade, humans will",
            "The most important invention is",
            "Climate change will affect",
        ]

    def analyze_layer_importance(self):
        """Analyze which layers are most important for generation quality."""
        print("Analyzing layer importance...")

        # Get number of layers
        num_layers = self.base_model.num_layers
        print(f"Model has {num_layers} layers")

        # Initialize importance scores
        layer_importance = np.zeros(num_layers)

        # For each sample text
        for text in self.eval_prompts:
            print(f"Testing prompt: '{text}'")

            # Tokenize
            inputs = self.base_model.tokenizer(text, return_tensors="pt").to(
                self.device
            )
            attention_mask = torch.ones_like(inputs.input_ids)

            # Get baseline output (no masking)
            with torch.no_grad():
                baseline_outputs = self.base_model.base_model(
                    input_ids=inputs.input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
                baseline_logits = baseline_outputs.logits

            # For each layer
            for layer_idx in range(num_layers):
                # Create a hook to mask this layer's output
                def create_hook(layer_idx):
                    def hook(module, input, output):
                        # Apply small random noise to output
                        noise = torch.randn_like(output) * 0.1
                        return output + noise

                    return hook

                # Register hook
                hook = self.base_model.base_model.transformer.h[
                    layer_idx
                ].register_forward_hook(create_hook(layer_idx))

                # Get output with this layer masked
                with torch.no_grad():
                    masked_outputs = self.base_model.base_model(
                        input_ids=inputs.input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    masked_logits = masked_outputs.logits

                # Remove hook
                hook.remove()

                # Calculate difference in output distribution
                diff = F.kl_div(
                    F.log_softmax(masked_logits, dim=-1),
                    F.softmax(baseline_logits, dim=-1),
                    reduction="batchmean",
                )

                # Update importance score
                layer_importance[layer_idx] += diff.item()

        # Normalize importance scores
        layer_importance = layer_importance / len(self.eval_prompts)

        # Sort layers by importance
        sorted_indices = np.argsort(layer_importance)

        print("Layer importance analysis results:")
        for i, layer_idx in enumerate(sorted_indices):
            print(
                f"Rank {i+1}: Layer {layer_idx} (Score: {layer_importance[layer_idx]:.4f})"
            )

        # Select least important layers for compression
        least_important_layers = sorted_indices[:3].tolist()
        print(
            f"Selected least important layers for compression: {least_important_layers}"
        )

        return least_important_layers

    def measure_memory_usage(self, model, input_ids, runs=5):
        """Measure memory usage during inference."""
        process = psutil.Process(os.getpid())

        # Get baseline memory
        baseline_memory = process.memory_info().rss / (1024 * 1024)  # MB

        # Warm up
        with torch.no_grad():
            model(input_ids)

        # Measure memory during inference
        peak_memory = baseline_memory
        for _ in range(runs):
            # Clear cache if using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Run inference
            with torch.no_grad():
                model(input_ids)

            # Measure memory
            current_memory = process.memory_info().rss / (1024 * 1024)  # MB
            peak_memory = max(peak_memory, current_memory)

        # Calculate memory usage
        memory_usage = peak_memory - baseline_memory
        return memory_usage

    def measure_inference_speed(self, model, input_ids, runs=10):
        """Measure inference speed."""
        model.eval()

        # Warm up
        with torch.no_grad():
            model(input_ids)

        # Measure time
        start_time = time.time()
        for _ in range(runs):
            with torch.no_grad():
                model(input_ids)
        end_time = time.time()

        # Calculate average time
        avg_time = (end_time - start_time) / runs
        return avg_time

    def evaluate_generation_quality(
        self, model, prompts, max_length=100, temperature=0.8, top_p=0.9
    ):
        """Evaluate text generation quality."""
        results = []

        for prompt in prompts:
            # Generate text
            with torch.no_grad():
                input_ids = model.tokenizer(prompt, return_tensors="pt").input_ids.to(
                    self.device
                )
                attention_mask = torch.ones_like(input_ids)
                output_ids = model.base_model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                )
                output_text = model.tokenizer.decode(
                    output_ids[0], skip_special_tokens=True
                )

            # Calculate metrics
            prompt_len = len(prompt.split())
            output_len = len(output_text.split())
            new_content_len = output_len - prompt_len

            # Store results
            results.append(
                {
                    "prompt": prompt,
                    "output": output_text,
                    "prompt_len": prompt_len,
                    "output_len": output_len,
                    "new_content_len": new_content_len,
                }
            )

        # Calculate average metrics
        avg_output_len = sum(r["output_len"] for r in results) / len(results)
        avg_new_content_len = sum(r["new_content_len"] for r in results) / len(results)

        return {
            "results": results,
            "avg_output_len": avg_output_len,
            "avg_new_content_len": avg_new_content_len,
        }

    def train_progressively(self):
        """Train with progressive compression."""
        # Analyze layer importance
        compression_layer_indices = self.analyze_layer_importance()

        # Create initial model with minimal compression
        print("Creating initial model with minimal compression")
        model = CompressibleLanguageModel(
            base_model_name=self.base_model_name,
            compression_layer_indices=compression_layer_indices,
            compression_factor=self.initial_compression_factor,
            similarity_threshold=self.initial_similarity_threshold,
            position_weight=self.position_weight,
            residual_compression_factor=self.residual_compression_factor,
            use_residuals=self.use_residuals,
        ).to(self.device)

        # Create dataset
        train_dataset = TextDataset(
            self.train_texts, model.tokenizer, max_length=self.max_length
        )

        # Calculate compression and threshold schedules
        compression_steps = []
        threshold_steps = []

        for epoch in range(self.epochs):
            # Linear interpolation for compression factor
            compression_factor = self.initial_compression_factor + (
                (self.final_compression_factor - self.initial_compression_factor)
                * (epoch / (self.epochs - 1))
            )
            compression_steps.append(max(1, int(compression_factor)))

            # Linear interpolation for similarity threshold (decreasing)
            similarity_threshold = self.initial_similarity_threshold - (
                (self.initial_similarity_threshold - self.final_similarity_threshold)
                * (epoch / (self.epochs - 1))
            )
            threshold_steps.append(similarity_threshold)

        print(f"Progressive compression schedule: {compression_steps}")
        print(f"Progressive threshold schedule: {threshold_steps}")

        # Baseline evaluation
        print("\nBaseline evaluation (original model):")
        baseline_input = self.base_model.tokenizer(
            self.eval_prompts[0], return_tensors="pt"
        ).input_ids.to(self.device)
        baseline_memory = self.measure_memory_usage(self.base_model, baseline_input)
        baseline_speed = self.measure_inference_speed(self.base_model, baseline_input)
        baseline_quality = self.evaluate_generation_quality(
            self.base_model, self.eval_prompts
        )

        print(f"Memory usage: {baseline_memory:.2f} MB")
        print(f"Inference speed: {baseline_speed:.4f}s")
        print(f"Average output length: {baseline_quality['avg_output_len']:.2f} tokens")
        print(
            f"Average new content: {baseline_quality['avg_new_content_len']:.2f} tokens"
        )

        # Train with progressive compression
        for epoch, (compression_factor, similarity_threshold) in enumerate(
            zip(compression_steps, threshold_steps)
        ):
            print(f"\nEpoch {epoch+1}/{self.epochs}")
            print(f"Compression Factor: {compression_factor}")
            print(f"Similarity Threshold: {similarity_threshold:.4f}")

            # Update compression parameters for all layers
            for i, layer in enumerate(model.compression_layers):
                # Update the layer's parameters
                layer.compression_factor = compression_factor
                layer.similarity_threshold = similarity_threshold

                # Create a new compressor with the current compression factor if needed
                if (
                    hasattr(layer, "compressor")
                    and layer.compressor.weight.shape[0]
                    != layer.d_model // compression_factor
                ):
                    layer.compressor = nn.Linear(
                        layer.d_model, layer.d_model // compression_factor
                    ).to(self.device)

            # Train for one epoch
            model = train(
                model=model,
                train_dataset=train_dataset,
                batch_size=self.batch_size,
                epochs=1,
                learning_rate=self.learning_rate,
                output_dir=f"{self.output_dir}/epoch-{epoch+1}",
            )

            # Save checkpoint
            model.save(f"{self.output_dir}/checkpoint-{epoch+1}")

            # Evaluate
            print("\nEvaluation:")
            eval_input = model.tokenizer(
                self.eval_prompts[0], return_tensors="pt"
            ).input_ids.to(self.device)
            memory_usage = self.measure_memory_usage(model, eval_input)
            inference_speed = self.measure_inference_speed(model, eval_input)
            generation_quality = self.evaluate_generation_quality(
                model, self.eval_prompts
            )

            # Calculate improvements
            memory_reduction = (baseline_memory - memory_usage) / baseline_memory * 100
            speed_improvement = (
                (baseline_speed - inference_speed) / baseline_speed * 100
            )

            print(
                f"Memory usage: {memory_usage:.2f} MB ({memory_reduction:.2f}% reduction)"
            )
            print(
                f"Inference speed: {inference_speed:.4f}s ({speed_improvement:.2f}% improvement)"
            )
            print(
                f"Average output length: {generation_quality['avg_output_len']:.2f} tokens"
            )
            print(
                f"Average new content: {generation_quality['avg_new_content_len']:.2f} tokens"
            )

            # Print sample generation
            print("\nSample generation:")
            for i, result in enumerate(generation_quality["results"]):
                if i < 2:  # Print only first 2 examples to save space
                    print(f"Prompt: '{result['prompt']}'")
                    print(f"Output: '{result['output']}'")
                    print()

        # Save final model
        print("\nSaving final model")
        model.save(f"{self.output_dir}/final_model")

        # Final comprehensive evaluation
        print("\nFinal evaluation:")

        # Parameter count
        orig_params = sum(p.numel() for p in self.base_model.parameters()) / 1e6
        comp_params = sum(p.numel() for p in model.parameters()) / 1e6
        param_reduction = ((orig_params - comp_params) / orig_params) * 100

        print(f"Original model parameters: {orig_params:.2f} million")
        print(f"Compressed model parameters: {comp_params:.2f} million")
        print(f"Parameter reduction: {param_reduction:.2f}%")

        # Disk size
        os.system(
            f"du -sh {self.output_dir}/final_model > {self.output_dir}/disk_size.txt"
        )
        with open(f"{self.output_dir}/disk_size.txt", "r") as f:
            disk_size = f.read().strip()
        print(f"Disk size: {disk_size}")

        # Memory and speed
        final_memory = self.measure_memory_usage(model, eval_input)
        final_speed = self.measure_inference_speed(model, eval_input)
        memory_reduction = (baseline_memory - final_memory) / baseline_memory * 100
        speed_improvement = (baseline_speed - final_speed) / baseline_speed * 100

        print(
            f"Memory usage: {final_memory:.2f} MB ({memory_reduction:.2f}% reduction)"
        )
        print(
            f"Inference speed: {final_speed:.4f}s ({speed_improvement:.2f}% improvement)"
        )

        # Generation quality
        final_quality = self.evaluate_generation_quality(
            model, self.eval_prompts, max_length=150
        )
        print(f"Average output length: {final_quality['avg_output_len']:.2f} tokens")
        print(f"Average new content: {final_quality['avg_new_content_len']:.2f} tokens")

        # Print all generations
        print("\nFinal generations:")
        for result in final_quality["results"]:
            print(f"Prompt: '{result['prompt']}'")
            print(f"Output: '{result['output']}'")
            print()

        return model


# Run the improved compression
if __name__ == "__main__":
    compressor = ImprovedCompression(
        base_model_name="gpt2",
        train_file="sample_data.txt",
        output_dir="improved_compressed_model",
        initial_compression_factor=1,
        final_compression_factor=3,
        initial_similarity_threshold=0.9,
        final_similarity_threshold=0.8,
        position_weight=0.1,
        epochs=8,
        batch_size=4,
        learning_rate=5e-5,
        max_length=512,
        residual_compression_factor=2,
        use_residuals=True,
    )

    final_model = compressor.train_progressively()
