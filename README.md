# Simple AirLLM Project

This project demonstrates using AirLLM to run large language models with optimized memory usage.

## Features

- Uses AirLLM's AutoModel to load Platypus2-70B-instruct
- Simple text generation example
- Automatic device detection (CUDA, MPS, or CPU)
- Compatible with Apple Silicon Macs

## Requirements

- Python 3.8+
- GPU recommended but not required:
  - NVIDIA GPU with CUDA support, or
  - Apple Silicon Mac with MPS support, or
  - CPU (slower performance)
- Sufficient disk space for model weights (~140GB for Platypus2-70B)

## Installation

1. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script:
```bash
source venv/bin/activate
python main.py
```

The script will:
1. Detect available hardware (CUDA/MPS/CPU)
2. Load the Platypus2-70B-instruct model
3. Process the input question: "What is the capital of United States?"
4. Generate a response with up to 20 new tokens
5. Print the output

## Configuration

You can modify these parameters in `main.py`:
- `MAX_LENGTH`: Maximum input token length (default: 128)
- `max_new_tokens`: Maximum tokens to generate (default: 20)
- Input text can be modified in the `input_text` list

## Notes

- The script automatically detects and uses the best available device
- First run will download the model weights (~140GB)
- AirLLM optimizes memory usage for large models by loading weights layer-by-layer
- On Apple Silicon Macs, the script will use MPS (Metal Performance Shaders) for GPU acceleration
- If no GPU is available, it will fall back to CPU (much slower)
