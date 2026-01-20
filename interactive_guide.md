# How to Use AirLLM Interactively

## Method 1: Python Interactive Shell (REPL)

1. Activate your virtual environment and start Python:
```bash
source venv/bin/activate
python
```

2. Import libraries and load the model:
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Detect device
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# Load model and tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device.type != "cpu" else torch.float32
).to(device)

model.eval()
print("Model loaded!")
```

3. Now you can generate responses for any question:
```python
# Ask a question
question = "What is the capital of France?"
inputs = tokenizer(question, return_tensors="pt").to(device)

# Generate response
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

# Decode and print
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

4. Ask more questions without reloading the model:
```python
# Just change the question and run again
question = "What is 2 + 2?"
inputs = tokenizer(question, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

5. Exit Python:
```python
exit()
```

---

## Method 2: Create a Simple Interactive Script

Create a file that asks for input in a loop:

```python
# Save as interactive_chat.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading model...")

# Setup device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load model
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device.type != "cpu" else torch.float32
).to(device)
model.eval()

print("Model loaded! Type 'exit' to quit.\n")

# Interactive loop
while True:
    question = input("You: ")

    if question.lower() in ['exit', 'quit', 'q']:
        print("Goodbye!")
        break

    inputs = tokenizer(question, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nAI: {response}\n")
```

Run it:
```bash
source venv/bin/activate
python interactive_chat.py
```

---

## Method 3: Jupyter Notebook

1. Install Jupyter:
```bash
source venv/bin/activate
pip install jupyter
```

2. Start Jupyter:
```bash
jupyter notebook
```

3. Create a new notebook and run cells:

**Cell 1 - Load Model:**
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using: {device}")

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device.type != "cpu" else torch.float32
).to(device)
model.eval()
```

**Cell 2 - Helper Function:**
```python
def ask(question):
    inputs = tokenizer(question, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

**Cell 3 - Ask Questions:**
```python
print(ask("What is the capital of France?"))
```

**Cell 4 - Ask More:**
```python
print(ask("Tell me a joke"))
```

---

## Quick Commands Reference

**Activate environment:**
```bash
source venv/bin/activate
```

**Start Python REPL:**
```bash
python
```

**One-liner to test:**
```python
from transformers import pipeline; generator = pipeline('text-generation', model='TinyLlama/TinyLlama-1.1B-Chat-v1.0'); print(generator("What is AI?", max_length=50)[0]['generated_text'])
```

**Deactivate environment:**
```bash
deactivate
```

---

## Tips

- Model loading takes 10-30 seconds on first run (it's cached after that)
- Each generation takes 2-10 seconds depending on your hardware
- You can change `max_new_tokens` to control response length
- Lower `temperature` (0.1-0.5) = more focused responses
- Higher `temperature` (0.7-1.0) = more creative responses
