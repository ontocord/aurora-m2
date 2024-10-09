#@title stack exchange no summaries
from typing import List
import re
try:
  from huggingface_hub import hf_hub_download
except:
  !pip install huggingface_hub
from huggingface_hub import hf_hub_download
try:
  import fasttext
except:
  !pip install fasttext
import fasttext
try:
  if textbook_model is None: assert False
except:
  textbook_model = fasttext.load_model(hf_hub_download("kenhktsui/llm-data-textbook-quality-fasttext-classifer-v1", "model.bin"))

try:
  if rpj_model is None: assert False
except:
  rpj_model = fasttext.load_model(hf_hub_download("ontocord/riverbed", "rj_model.bin"))


def predict(model, text, ltype):
  pred = model.predict(text.replace("\n", " ")[:min(1000, len(text))])
  pred2 = model.predict(text.replace("\n", " ")[-min(1000, len(text)):])
  label =  pred[0][0]
  label2 =  pred2[0][0]
  if ltype not in label and ltype not in label2:
    return max(1 - pred[1][0], 1 - pred2[1][0])
  if ltype not in label2:
    return 1 - pred[1][0]
  if ltype not in label:
    return 1 - pred2[1][0]
  return max(pred[1][0],pred2[1][0])


device = "cuda"
#from datasets import load_dataset
import random

EU_langs = ['bg',
 'hr',
 'cs',
 'da',
 'nl',
 'en',
 'et',
 'fi',
 'fr',
 'de',
 'el',
 'hu',
 'ga',
 'it',
 'lv',
 'lt',
 'mt',
 'pl',
 'pt',
 'ro',
 'sk',
 'sl',
 'es',
 'sv']




import json, random, glob
if True: # with open("/content/drive/Shareddrives/ontocord_llc/stack_excahnge_combined.jsonl", "w") as combined:
  with open("/content/drive/Shareddrives/ontocord_llc/stack_excahnge2.jsonl", "w") as outf:
    for file in glob.glob("/content/drive/Shareddrives/MDEL/dataset/M3_v0/en/red_pajama_stackexchange_part_*.jsonl"):
      print (file)
      with open(file) as infile:
        for idx, l in enumerate(infile):
          dat = json.loads(l)
          if float(dat['meta']['question_score']) < 5: continue
          text = dat['text']
          if len(text) < 1000: continue
          text = text.replace("...", ".").replace("..", ".").replace("\n\n\n*\n\n", "\n").replace(" :", ":").replace("\n\n", "\n").replace("\n*\n", "\n").replace("\n*\n", "").replace("\n\n\n", "\n\n").replace("\n\n\n\n", "\n\n").replace("\n\n\n\n\n", "\n\n").strip()
          text = " ".join(a for a in text.split(" ") if "http" not in a)
          instruction_answer = text.replace("Q: ", "").split("A: ")
          if len(instruction_answer) < 2: continue
          instruction = instruction_answer[0]
          answer = instruction_answer[1]
          score = predict(textbook_model, (instruction+" ")*5, "HIGH")
          score2 = predict(textbook_model, (answer+" ")*5, "HIGH")
          score_a = max ([score, score2])
          score = predict(rpj_model, instruction, "wiki")
          score2 = predict(rpj_model, answer, "wiki")
          score_b = max([score, score2])
          if score_a > 0.2 and score_b > 0.2 and (score_a + score_b)/2 > 0.3:
            print (score_a, score_b)
            dat['meta']['textbook_score'] = score_a
            dat['meta']['quality_score'] = score_b
            dat['text'] = text
            outf.write (json.dumps(dat)+"\n")
