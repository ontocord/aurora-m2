import torch


def chatml_format_instructions(tokenizer, system, instruction, response=""):
  system= system.strip()
  instruction = instruction.strip()
  if system:
    return tokenizer.apply_chat_template([{"role": "system", "content": system}, 
                                          {"role": "user", "content": instruction}], tokenize=False)
  else:
    return tokenizer.apply_chat_template([{"role": "user", "content": instruction}], tokenize=False)
#   if system:
#     return f"""<|im_start|>system
# {system}
# <|im_end|>
# <|im_start|>user
# {instruction}
# <|im_end|>
# <|im_start|>assistant
# """
#   else:
#     return f"""<|im_start|>user
# {instruction}
# <|im_end|>
# <|im_start|>assistant
# {response}"""

def generate_with_batching(model, tokenizer, data, device,  use_cache=True, repetition_penalty=1.2, no_repeat_ngram_size=4, max_new_tokens=200, batch_size=5, **args):
  torch.cuda.empty_cache()
  output = []
  for rng in range(0, len(data), batch_size):
    d = data[rng:min(len(data), rng+batch_size)]
    if d:
      input_ids = tokenizer(d, truncation=True, padding=True, return_tensors="pt", add_special_tokens=False, ).to(device)
      prompt_len = input_ids["input_ids"].shape[-1]
      output.extend(tokenizer.batch_decode(model.generate(**input_ids,
                        use_cache=use_cache, repetition_penalty=repetition_penalty, no_repeat_ngram_size=no_repeat_ngram_size, max_new_tokens=max_new_tokens, **args)[:, prompt_len:], skip_special_tokens=True))
  torch.cuda.empty_cache()
  return output