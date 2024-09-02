import torch


def chatml_format_instructions(system, instruction, response=""):
  system= system.strip()
  instruction = instruction.strip()
  if system:
    return f"""<|im_start|>system
{system}
<|im_end|>
<|im_start|>user
{instruction}
<|im_end|>
<|im_start|>assistant
"""
  else:
    return f"""<|im_start|>user
{instruction}
<|im_end|>
<|im_start|>assistant
{response}"""

def generate_with_batching(model, tokenizer, data, device,  use_cache=True, repetition_penalty=1.2, no_repeat_ngram_size=4, max_new_tokens=200, batch_size=5, **args):
  torch.cuda.empty_cache()
  output = []
  for rng in range(0, len(data), batch_size):
    d = data[rng:min(len(data), rng+batch_size)]
    if d:
      output.extend(tokenizer.batch_decode(model.generate(**tokenizer(d, truncation=True, padding=True, return_tensors="pt", add_special_tokens=False, ).to(device),
                        use_cache=use_cache, repetition_penalty=repetition_penalty, no_repeat_ngram_size=no_repeat_ngram_size, max_new_tokens=max_new_tokens, **args)))
  torch.cuda.empty_cache()
  return output