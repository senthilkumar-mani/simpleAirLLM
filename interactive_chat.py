#!/usr/bin/env python3
"""
Interactive Chat with TinyLlama
Type your questions and get responses in real-time.
Type 'exit', 'quit', or 'q' to stop.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("=" * 60)
print("Interactive Chat with TinyLlama")
print("=" * 60)
print("\nLoading model... (this may take a moment)")

# Setup device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("✓ Using CUDA (NVIDIA GPU)")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✓ Using MPS (Apple Silicon GPU)")
else:
    device = torch.device("cpu")
    print("✓ Using CPU")

# Load model
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device.type != "cpu" else torch.float32
).to(device)
model.eval()

print("\n✓ Model loaded successfully!")
print("\nYou can now chat with the AI.")
print("Type 'exit', 'quit', or 'q' to stop.\n")
print("=" * 60)

# Interactive loop
while True:
    try:
        question = input("\n👤 You: ")

        if question.strip().lower() in ['exit', 'quit', 'q', '']:
            print("\n👋 Goodbye!")
            break

        # Tokenize and generate
        inputs = tokenizer(question, return_tensors="pt").to(device)

        print("🤖 AI: ", end="", flush=True)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(response)

    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        break
    except Exception as e:
        print(f"\n❌ Error: {e}")
        continue
