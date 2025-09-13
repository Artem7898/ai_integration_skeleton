import os
from huggingface_hub import InferenceClient

client = InferenceClient(model="meta-llama/Meta-Llama-3-8B-Instruct", token=os.environ.get("HF_TOKEN"))

def hf_generate_json(prompt: str, max_new_tokens: int = 128) -> str:
    return client.text_generation(prompt, max_new_tokens=max_new_tokens, details=False)
