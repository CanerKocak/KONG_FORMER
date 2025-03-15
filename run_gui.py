#!/usr/bin/env python3
"""
Run the Neural Compression Model GUI with a sample model.
This script provides a convenient way to start the GUI and automatically load a model.
"""

import os
import sys
import argparse
from compression_gui import app


def main():
    parser = argparse.ArgumentParser(description="Run the Neural Compression Model GUI")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="ultra_safe_model_v2",
        help="Path to the model directory to load automatically",
    )
    parser.add_argument(
        "--port", type=int, default=7860, help="Port to run the Gradio app on"
    )
    parser.add_argument(
        "--share", action="store_true", help="Create a public link for the interface"
    )

    args = parser.parse_args()

    # Check if model directory exists
    if not os.path.exists(args.model_dir):
        print(f"Warning: Model directory '{args.model_dir}' not found.")
        print("The GUI will still launch, but you'll need to manually load a model.")
    else:
        print(f"Starting GUI with model from: {args.model_dir}")
        print("The model will be loaded automatically when the GUI starts.")

    # Launch the app
    app.launch(
        server_port=args.port,
        share=args.share,
        inbrowser=True,
        server_name="0.0.0.0",  # Allow connections from other devices on the network
    )


if __name__ == "__main__":
    main()
