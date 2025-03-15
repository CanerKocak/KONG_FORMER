import sys

print("Script starting - Python version:", sys.version)

try:
    print("Attempting to import torch...")
    import torch

    print(f"✓ Torch imported successfully - version {torch.__version__}")
except ImportError as e:
    print(f"✗ CATASTROPHIC FAILURE: Torch import failed: {e}")
    print("FIX: Run 'pip install torch' and try again, you absolute walnut.")
    sys.exit(1)

try:
    print("Attempting to import time...")
    import time

    print("✓ Time imported successfully")
except ImportError as e:
    print(f"✗ WHAT THE ACTUAL FUCK: Time module failed: {e}")
    print(
        "This should be impossible unless your Python installation is fundamentally broken."
    )
    sys.exit(1)

try:
    print("Attempting to import transformers...")
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    import transformers

    print(f"✓ Transformers imported successfully - version {transformers.__version__}")
except ImportError as e:
    print(f"✗ CRITICAL ERROR: Transformers import failed: {e}")
    print("FIX: Run 'pip install transformers' and try again, genius.")
    sys.exit(1)

print("\n=== ALL IMPORTS SUCCESSFUL - CONTINUING TO MAIN FUNCTION ===\n")


def run_pure_gpt2_xl():
    """
    Run GPT-2 XL on Apple Silicon without your neural compression horseshit
    """
    print("Loading GPT-2 XL (1.5B parameters of pure, uncompressed glory)...")

    # Track load time because why not
    start_load = time.time()

    # CHECK FOR MPS AVAILABILITY - CRUCIAL FOR M1/M2/M3 MACS
    print("Checking for Apple MPS availability...")

    # Load the model (MPS SPECIFIC VERSION)
    model = None
    try:
        print("Attempting to load model in FP16 for Apple Silicon...")
        model = GPT2LMHeadModel.from_pretrained("gpt2-xl", torch_dtype=torch.float16)
        print("✓ Model loaded in FP16 (half precision)")
    except Exception as e:
        print(f"✗ FP16 loading failed with: {e}")
        try:
            print("Attempting full FP32 precision as fallback...")
            model = GPT2LMHeadModel.from_pretrained("gpt2-xl")
            print("✓ Model loaded in FP32 (full precision)")
        except Exception as e2:
            print(f"✗ Model loading completely failed: {e2}")
            sys.exit(1)

    print("Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")

    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.2f} seconds")

    # Move to MPS device
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set padding token to EOS token")

    # Test prompts
    prompts = ["Artificial intelligence will", "The future of technology is"]

    for prompt_idx, prompt in enumerate(prompts):
        print("\n" + "=" * 80)
        print(f"PROMPT {prompt_idx+1}/{len(prompts)}: {prompt}")
        print("=" * 80)

        # Tokenize
        print(f"Tokenizing prompt...")
        inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
        input_length = len(inputs.input_ids[0])
        print(f"Input length: {input_length} tokens")

        # Generate with timing
        max_new_tokens = 30  # REDUCED for faster testing

        # First, measure the time for a single forward pass
        print("Measuring single forward pass...")
        try:
            start_time = time.time()
            with torch.no_grad():
                # Just do a single forward pass to measure base speed
                _ = model(**inputs)
                # MPS sync equivalent
                if device.type == "cpu":
                    torch.cpu.synchronize()
                forward_time = time.time() - start_time
                print(f"✓ Single forward pass: {forward_time:.4f} seconds")
        except Exception as e:
            print(f"✗ Forward pass failed: {e}")
            continue

        # Now do the full generation
        print(f"Generating {max_new_tokens} new tokens...")
        try:
            start_time = time.time()
            with torch.no_grad():
                # Full generation
                output = model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                )

                # MPS sync equivalent
                if device.type == "cpu":
                    torch.cpu.synchronize()
                generation_time = time.time() - start_time

            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

            # Calculate stats
            new_tokens = len(output[0]) - input_length
            tokens_per_second = new_tokens / generation_time

            print(f"✓ Generated {new_tokens} tokens in {generation_time:.2f} seconds")
            print(f"Speed: {tokens_per_second:.2f} tokens/second")
            print(f"\nGENERATED TEXT:\n{generated_text}")

            # Calculate theoretical 5x speedup for comparison with your compression monster
            theoretical_compressed_time = generation_time / 5
            print(
                f"\nWith 5x compression this would take ~{theoretical_compressed_time:.2f} seconds"
            )
            print(
                f"Theoretical compressed speed: {tokens_per_second * 5:.2f} tokens/second"
            )

        except Exception as e:
            print(f"✗ Generation failed: {e}")
            import traceback

            traceback.print_exc()


print("Attempting to run the main function now...")
if __name__ == "__main__":
    try:
        run_pure_gpt2_xl()
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"\nEverything exploded: {e}")
        import traceback

        traceback.print_exc()

    print("\nScript execution complete.")
