# make sure you have the latest version of transfomers and install wget
from transformers import pipeline

from transformers import AutoTokenizer, AutoModelForCausalLM
try:
  if tokenizer is None: assert False
except:
  tokenizer = AutoTokenizer.from_pretrained("UCLA-AGI/Gemma-2-9B-It-SPPO-Iter3")
  model = AutoModelForCausalLM.from_pretrained("UCLA-AGI/Gemma-2-9B-It-SPPO-Iter3").half().cuda()

  pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

prompts = ["""Revise this story to make it compelling and more logical and detailed. Keep as much of the feelings and actions as possibe, but remove anything that doesn't make sense. Make the story at least 10 paragraphs. The story should unfold through the characters interactions, decisions, and the consequences of their actions. Aim to weave in common sense lessons and social cues. The narrative should cater to a diverse age group, including at least one dialogue and presenting both positive and negative outcomes. Do not start with classic sentences like "Once upon a time", be creative:""",
           """Revise this story to make it compelling and more logical and detailed. Keep as much of the feelings and actions as possibe, but remove anything that doesn't make sense. Make the story at least 10 paragraphs. Write as a real-life story shared by someone in a social media forum. The story should include:
- Niche interests or humor: dive into specific hobbies, interests, or humorous situations
- An unexpected plot twist or engaging conflict: introduce a relatable yet challenging situation or dilemma that the author faced.
- Reflection and insight: end with a resolution that offers a new understanding, a sense of community, or a personal revelation, much like the conclusions drawn in forum discussions.
Start the story right away. Do not start with sentences like  "Once upon a time" as this is a reddit post and not a novel, you should also avoid starting with classic sentences like "A few years ago" or "A few years back", be creative:""",
           """Revise this story to make it compelling and more logical and detailed. Keep as much of the feelings and actions as possibe, but remove anything that doesn't make sense. Make the story at least 10 paragraphs. Write the story in the style of real-life situations that people share in forums. The story needs to include a compelling and unexpected plot twist. Your narrative should resonate with the authenticity and personal touch found in forum discussions. Include relatable events and emotional depth. Do not start with classic sentences like "Once upon a time", "A few years back" or "A few montsh ago", be creative:""",
           """Revise this story to make it compelling and more logical and detailed. Keep as much of the feelings and actions as possibe, but remove anything that doesn't make sense. Make the story at least 10 paragraphs. The story should incorporate the following elements:
- Dialogue: the story must feature at least one meaningful dialogue that reveals character depth, advances the plot, or unravels a crucial piece of the mystery
- Interesting themes: explore themes resonant with a mature audience, such as moral ambiguity, existential queries, personal transformation, or the consequences of past actions.
Do not start with classic sentences like "Once upon a time", "The sun hung low in the sky" or "In the dimly lit", be creative:"""]

import wget, json, random, os

url = 'https://huggingface.co/datasets/ontocord/atomic_2024/resolve/main/data/atomic_stories.jsonl'
if not os.path.exists("atomic_stories.jsonl"):
  wget.download(url)
with open("generated_atomic_stories.jsonl", "w") as outfile:
  with open("atomic_stories.jsonl") as infile:
    for l in infile:
      dat = json.loads(l)
      text = dat['text']
      prompt = random.choice(prompts)
      messages = [
        {"role": "user", "content": f"{prompt}\n\n{text}"}
      ]
      output_text = pipe(messages, max_length=2048, min_length=512, use_cache=True)[0]['generated_text'][1]['content']
      outfile.write(json.dumps({'text': output_text, 'prompt': prompt, 'input': text, 'metadata': dat['metadata']})+"\n")
