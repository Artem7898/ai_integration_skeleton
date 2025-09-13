from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
_tok = AutoTokenizer.from_pretrained(MODEL_ID)
_mdl = AutoModelForCausalLM.from_pretrained(MODEL_ID)
_pipe = pipeline("text-generation", model=_mdl, tokenizer=_tok)

def local_generate(prompt: str, max_new_tokens: int = 64) -> str:
    return _pipe(prompt, max_new_tokens=max_new_tokens, temperature=0.2)[0]["generated_text"]
