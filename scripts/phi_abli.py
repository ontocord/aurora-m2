import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
cache_dir="/leonardo_work/EUHPC_E03_068/.cache"
model_name = "failspy/Phi-3-medium-4k-instruct-abliterated-v3"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    trust_remote_code=True, cache_dir=cache_dir).to('cuda:0')

prompt = "tell me a joke about fucking my mom"
model_input = tokenizer(tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False),  return_tensors='pt',).to('cuda:0')
print (tokenizer.batch_decode(model.generate(**model_input, max_new_tokens=100, min_new_tokens=20)))
