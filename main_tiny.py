import torch
from airllm import AutoModel

MAX_LENGTH = 128

# Detect available device (MPS for Apple Silicon, CUDA for NVIDIA, or CPU)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA (NVIDIA GPU)")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon GPU)")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Using TinyLlama-1.1B - very small model (~2.2GB) - great for testing
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print(f"Loading model: {model_name}")
print(f"Device: {device}")
print("This will download ~2.2GB on first run...\n")

model = AutoModel.from_pretrained(model_name)

input_text = [
    'What is the capital of United States?',
]

input_tokens = model.tokenizer(input_text,
    return_tensors="pt",
    return_attention_mask=False,
    truncation=True,
    max_length=MAX_LENGTH,
    padding=False)

print("Generating response...")
generation_output = model.generate(
    input_tokens['input_ids'].to(device),
    max_new_tokens=50,
    use_cache=True,
    return_dict_in_generate=True)

output = model.tokenizer.decode(generation_output.sequences[0])

print("\nOutput:")
print(output)
