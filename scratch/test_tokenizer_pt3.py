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
        return_dict=True  # simulate
    )
    if hasattr(tensor, "input_ids"):
        ids = tensor.input_ids[0].tolist()
    elif hasattr(tensor, "tolist"):
        ids = tensor[0].tolist()
    elif isinstance(tensor, list) and len(tensor) > 0 and hasattr(tensor[0], "ids"):
        ids = list(tensor[0].ids)
    elif isinstance(tensor, list) and len(tensor) > 0 and isinstance(tensor[0], list):
        ids = list(tensor[0])
    else:
        ids = list(tensor)

    if max_len is not None and len(ids) > max_len:
        ids = ids[:max_len]
    return ids

try:
    ids = encode_prompt_ids(tokenizer, msgs)
    print("Success ids length:", len(ids))
    print("Success ids:", type(ids), type(ids[0]))
except Exception as e:
    print("Error in encode_prompt_ids:", type(e), e)
