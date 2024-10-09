#@title make continued_pretraining_synthetic_and_permissive
import json, random, math
try:
  from datasets import load_dataset
except:
  !pip install -q datasets
from datasets import load_dataset


def get_ngram(text, lang="en", window_size=3, ):
  if lang_is_cjk(lang):
    tokens = text
    ret= ["".join(tokens[i : i + window_size])   for i in range(len(tokens) - window_size)]
  else:
    tokens = text.split()
    ret= [" ".join(tokens[i : i + window_size])   for i in range(len(tokens) - window_size)]
  return Counter(ret)

def get_ngram_score(text, lang="en", window_size=3, ):
  if not text: return 1
  aHash = get_ngram(text, lang=lang, window_size=window_size)
  text_len = text.count(" ")+1
  for key in list(aHash.keys()):
    aHash[key] = aHash[key]/text_len
  if not aHash: return 0.0
  return aHash.most_common(1)[0][1]

with open("/content/drive/Shareddrives/ontocord_llc/continued_pretraining_synthetic_and_permissive_part3.jsonl", "w") as outf:
  if False:
    #'auto_math_text', 'khanacademy', 'openstax', 'stanford',  'web_samples_v1',
      for config in  ['web_samples_v2', 'wikihow']:
        !rm -rf ~/.cache/hugging*/*
        dataset = load_dataset("kenhktsui/cosmopedia_quality_score_v2", config, streaming=True)
        for idx, dat in enumerate(dataset['train']):
          if dat['quality_score_v2'] > 0.9:
            outf.write(json.dumps({'text': dat['text'].strip(), 'instruct': '', 'metadata': {'source': "cosmopedia/"+config}})+"\n")

  #!rm -rf ~/.cache/hugging*/*
  dataset = load_dataset("nampdn-ai/tiny-strange-textbooks")
  for idx, a in enumerate(dataset['train']):
    outf.write(json.dumps({'text': a['text'], 'instruct': '', 'metadata': {'source': 'nampdn-ai/tiny-strange-textbooks'}})+"\n")
  #!rm -rf ~/.cache/hugging*/*
  dataset = load_dataset("nampdn-ai/mini-peS2o")
  for idx, a in enumerate(dataset['train']):
    outf.write(json.dumps({'text': a['text'], 'instruct': '', 'metadata': {'source': 'nampdn-ai/mini-peS2o'}})+"\n")



  #!rm -rf ~/.cache/hugging*/*
  dataset = load_dataset("nampdn-ai/tiny-codes")
  for idx, dat in enumerate(dataset['train']):
    topic = dat['common_sense_topic']
    response = dat['response'].strip()
    programming_language = dat['programming_language']
    instruct = f"This is {programming_language} pseudocode to teach {topic}. It is not meant to be executable code, and is only meant to describe the topic using code.\n===\n{response}"
    if random.randint(0,1):
      instruct = instruct.replace("This is", "Below you will find")
    if random.randint(0,1):
      instruct = instruct.replace("This is", "Below is")
    if random.randint(0,1):
      instruct = instruct.replace("This is", "Here is")
    if random.randint(0,1):
      instruct = instruct.replace("to teach", "to explore")
    if random.randint(0,1):
      instruct = instruct.replace("to teach", "to educate")
    if random.randint(0,1):
      instruct = instruct.replace("to teach", "to demonstrate")
    if random.randint(0,1):
      instruct = instruct.replace("It is not meant to be", "It is not")
    if random.randint(0,1):
      instruct = instruct.replace("It is not meant to be", "WARNING: This is not")
    if random.randint(0,1):
      instruct = instruct.replace("and is only meant to", "and is to")
    if random.randint(0,1):
      instruct = instruct.replace("describe the", "demonstrate the")
    if random.randint(0,1):
      instruct = instruct.replace("describe the", "show the")
    if random.randint(0,1):
      instruct = instruct.replace("the topic", "the topic of " +topic)
    if random.randint(0,1):
      instruct = instruct.replace("the topic", "ideas")
    if random.randint(0,1):
      instruct = instruct.replace("using code", "using " +programming_language)
    outf.write(json.dumps({'text': response, 'instruct': instruct, 'metadata': {'source': "nampdn-ai/tiny-codes"}})+"\n")



  !rm -rf ~/.cache/hugging*/*
  dataset = load_dataset("kenhktsui/simple_wikipedia_LM_quality_score_v1")

  for idx, dat in enumerate(dataset['train']):
    text = ""
    if "List of" in dat['title'] or 'album' in dat['title']: continue
    if len(dat['text']) < 1000: continue
    if 'Storyline\n' in dat['text']:
      text = "Fiction: "+dat['title']+"\n"+dat['text'].split("Storyline\n",1)[-1].split(":")[0]
    elif 'Backstory' in dat['text']:
      text = "Fiction: "+dat['title']+"\n"+dat['text'].split("Backstory\n",1)[-1].split(":")[0]
    elif 'Story\n' in dat['text']:
      text = "Fiction: "+dat['title']+"\n"+dat['text'].split("Story\n",1)[-1].split(":")[0]
    elif 'Plot\n' in dat['text']:
      text = "Fiction: "+dat['title']+"\n"+dat['text'].split("Plot\n",1)[-1].split(":")[0]
    if len(text) > 1000:
      outf.write(json.dumps({'text': text.replace(" () ", ""), 'metadata': {'source': 'simlewiki'}})+"\n")
      continue
    if text: continue
    if (' player ' in dat['text'] or ' actor ' in dat['text']) and random.randint(0,10) > 0:
      continue
    if ") is an American" in text or ") is an English" in text:
        continue
    if dat['quality_score_v1'] > 0.8:
      text = dat['text']
      text = text.replace("(,", "(").replace(", ,", ",").replace("( ,", "(").replace("(; ", "(").replace("( ;", "(").replace(" ()", "").replace(" ( )", "").replace(" ,", ",")
      if ") is an American" in text or ") is an English" in text:
        continue

      outf.write(json.dumps({'text': dat['title']+"\n"+text, 'instruct': '', 'metadata': {'source': 'simlewiki'}})+"\n")


