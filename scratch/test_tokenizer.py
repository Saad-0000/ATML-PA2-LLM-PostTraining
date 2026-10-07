import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
msgs = [{"role": "user", "content": "Hello"}]
encoded = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
print("Type of encoded:", type(encoded))
if isinstance(encoded, list):
    if len(encoded) > 0:
        print("Type of encoded[0]:", type(encoded[0]))
        if hasattr(encoded[0], "ids"):
            print("ids:", encoded[0].ids)

def extract_input_ids(encoded) -> list[int]:
    if hasattr(encoded, "ids"):
        return list(encoded.ids)
    if isinstance(encoded, dict) and "input_ids" in encoded:
        val = encoded["input_ids"]
        if hasattr(val, "tolist"):
            val = val.tolist()
        if val and isinstance(val, list) and isinstance(val[0], list):
            val = val[0]
        return list(val)
    if hasattr(encoded, "tolist"):
        val = encoded.tolist()
        if val and isinstance(val, list) and isinstance(val[0], list):
            val = val[0]
        return list(val)
    if isinstance(encoded, list):
        if encoded and hasattr(encoded[0], "ids"):
            return list(encoded[0].ids)
        if encoded and isinstance(encoded[0], list):
            return list(encoded[0])
        return list(encoded)
    return list(encoded)

extracted = extract_input_ids(encoded)
print("Type of extracted:", type(extracted))
if len(extracted) > 0:
    print("Type of extracted[0]:", type(extracted[0]))

try:
    inp = torch.tensor([extracted], dtype=torch.long)
    print("Tensor shape:", inp.shape)
except Exception as e:
    print("Error:", type(e), e)
