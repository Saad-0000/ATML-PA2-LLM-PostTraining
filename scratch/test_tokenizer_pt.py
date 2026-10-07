import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
msgs = [{"role": "user", "content": "Hello"}]
encoded = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt")
print("Type of encoded:", type(encoded))
if hasattr(encoded, "shape"):
    print("Shape:", encoded.shape)
elif isinstance(encoded, list):
    if len(encoded) > 0:
        print("Type of encoded[0]:", type(encoded[0]))
    try:
        print("encoded[0].tolist():", encoded[0].tolist())
    except Exception as e:
        print("Error on tolist:", type(e), e)

def encode_prompt_ids(tokenizer, msgs, max_len=None) -> list[int]:
    tensor = tokenizer.apply_chat_template(
        msgs,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    ids = tensor[0].tolist()
    if max_len is not None and len(ids) > max_len:
        ids = ids[:max_len]
    return ids

try:
    ids = encode_prompt_ids(tokenizer, msgs)
    print("Success ids:", type(ids), type(ids[0]))
except Exception as e:
    print("Error in encode_prompt_ids:", type(e), e)
