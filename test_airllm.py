
import time
import torch
from airllm import AutoModel

MODEL_NAME = "Qwen/Qwen3-8B"

print("1. CUDA available:", torch.cuda.is_available(), flush=True)
print("2. Loading AirLLM model...", flush=True)

start = time.time()
model = AutoModel.from_pretrained(MODEL_NAME)
print(f"3. Model loaded in {time.time() - start:.1f}s", flush=True)

inputs = model.tokenizer(
    ["Say hello in one short sentence."],
    return_tensors="pt",
    return_attention_mask=False,
    padding=False,
)

print("4. Input prepared.", flush=True)
print("5. Starting generation...", flush=True)

start = time.time()
output = model.generate(
    inputs["input_ids"].cuda(),
    max_new_tokens=5,
)

print(f"6. Generation finished in {time.time() - start:.1f}s", flush=True)
print(model.tokenizer.decode(output[0], skip_special_tokens=True))