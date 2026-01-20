# Simple AirLLM Project

This project demonstrates using AirLLM to run large language models with optimized memory usage.

## Features

- Multiple model size options (from 1.1B to 70B parameters)
- Simple text generation examples
- Automatic device detection (CUDA, MPS, or CPU)
- Compatible with Apple Silicon Macs

## Available Scripts

| Script | Model | Size | Best For |
|--------|-------|------|----------|
| `main_tiny.py` | TinyLlama-1.1B | ~2.2GB | Quick testing, learning |
| `main_small.py` | Mistral-7B-Instruct | ~14GB | Good balance of speed/quality |
| `main.py` | Platypus2-70B | ~140GB | Best quality, slow download |

## Requirements

- Python 3.8+
- GPU recommended but not required:
  - NVIDIA GPU with CUDA support, or
  - Apple Silicon Mac with MPS support, or
  - CPU (slower performance)
- Sufficient disk space for model weights (see table above)

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

Choose a script based on your needs:

**Quick Testing (Recommended for first run):**
```bash
source venv/bin/activate
python main_tiny.py
```

**Good Performance:**
```bash
source venv/bin/activate
python main_small.py
```

**Best Quality:**
```bash
source venv/bin/activate
python main.py
```

Each script will:
1. Detect available hardware (CUDA/MPS/CPU)
2. Download and load the model (only on first run)
3. Process the input question: "What is the capital of United States?"
4. Generate a response
5. Print the output

## Configuration

You can modify these parameters in any script:
- `MAX_LENGTH`: Maximum input token length (default: 128)
- `max_new_tokens`: Maximum tokens to generate (20-50 depending on script)
- Input text can be modified in the `input_text` list
- `model_name`: Change to any compatible Hugging Face model

## Notes

- The script automatically detects and uses the best available device
- First run will download model weights (size varies by model)
- AirLLM optimizes memory usage for large models by loading weights layer-by-layer
- On Apple Silicon Macs, the script will use MPS (Metal Performance Shaders) for GPU acceleration
- If no GPU is available, it will fall back to CPU (much slower)
- Start with `main_tiny.py` for quick testing before trying larger models

## Model Comparison

- **TinyLlama-1.1B**: Fast download, good for testing, basic responses
- **Mistral-7B-Instruct**: Better quality responses, reasonable download time
- **Platypus2-70B**: Highest quality, very large download, requires significant resources
