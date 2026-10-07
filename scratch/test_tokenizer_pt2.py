import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
msgs = [{"role": "user", "content": "Hello"}]

def encode_prompt_ids(tokenizer, msgs, max_len=None) -> list[int]:
    tensor = tokenizer.apply_chat_template(
        msgs,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True  # Force BatchEncoding to simulate the error
    )
    if isinstance(tensor, dict) and "input_ids" in tensor:
        ids = tensor["input_ids"][0].tolist()
    else:
        ids = tensor[0].tolist()
        
    if max_len is not None and len(ids) > max_len:
        ids = ids[:max_len]
    return ids

try:
    ids = encode_prompt_ids(tokenizer, msgs)
    print("Success ids length:", len(ids))
    print("Success ids:", ids)
except Exception as e:
    print("Error in encode_prompt_ids:", type(e), e)
