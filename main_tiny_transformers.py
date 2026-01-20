import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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

# Load tokenizer and model using transformers
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device.type != "cpu" else torch.float32,
    device_map="auto" if device.type == "cuda" else None
)

# Move model to device if not using device_map
if device.type != "cuda":
    model = model.to(device)

model.eval()

# Format input for chat model
input_text = "What is the capital of United States?"
print(f"Input: {input_text}\n")

# Tokenize input
input_tokens = tokenizer(input_text, return_tensors="pt").to(device)

print("Generating response...")
with torch.no_grad():
    generation_output = model.generate(
        **input_tokens,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

output = tokenizer.decode(generation_output[0], skip_special_tokens=True)

print("\nOutput:")
print(output)
