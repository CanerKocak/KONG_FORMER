# Neural Compression Model GUI

This graphical user interface (GUI) allows you to interact with trained neural compression models. The interface provides tools for text generation, parameter adjustment, and visualization of compression statistics.

## Features

- **Load Model**: Load a trained compression model from a directory
- **Generate Text**: Generate text with or without compression enabled
- **Compare Generation**: Compare text generation with and without compression, including speed metrics
- **Compression Parameters**: Adjust similarity threshold and maximum compression ratio
- **Visualizations**: View compression statistics and layer thresholds

## Installation

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Make sure you have a trained compression model available. You can use one of the provided models in the project or train your own using the training scripts.

## Usage

1. Run the GUI:
   ```
   python compression_gui.py
   ```

2. The interface will open in your default web browser.

3. Load a model by providing the path to the model directory and clicking "Load Model".

4. Navigate through the tabs to use different features:
   - **Generate Text**: Enter a prompt and adjust generation parameters
   - **Compare Generation**: Compare generation with and without compression
   - **Compression Parameters**: Adjust compression settings
   - **Visualizations**: View compression statistics and layer thresholds

## Example Prompts

Here are some example prompts you can use to test the model:

- "Recursive compression of neural networks works by"
- "The most efficient way to represent language is to"
- "Attention-weighted clustering in language models helps"
- "Progressive compression across transformer layers enables"
- "The future of efficient language models involves"

## Adjusting Compression Parameters

- **Similarity Threshold**: Controls how similar tokens need to be to be compressed together. Higher values (closer to 1.0) result in less compression but better quality.
- **Max Compression Ratio**: Sets the maximum allowed compression ratio. Higher values allow more aggressive compression but may reduce quality.

## Troubleshooting

- If you encounter errors loading a model, ensure the model directory contains all necessary files.
- For visualization issues, check that the model directory contains a compression_stats.json file.
- If text generation is slow, consider using a smaller model or reducing the max length parameter.

## Advanced Usage

You can modify the `compression_gui.py` file to add additional features or customize the interface. The GUI is built using Gradio, which provides a simple way to create web interfaces for machine learning models. 