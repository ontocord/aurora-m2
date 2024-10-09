#@title translate fineweb
try:
  import faker
except:
  !pip install -q faker wordnet sentencepiece

from google.colab import drive
import os
if not os.path.exists("/content/drive"):
  drive.mount('/content/drive')
from typing import List
import re
import json, os, random
try:
  import googletrans as gt
except:
  !pip3 install googletrans-python
import googletrans as gt
import random
import time


langs = ['bg',
 'hr',
 'cs',
 'da',
 'nl',
# 'en',
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
 'sv',
                    'vi', 'zh-CN', 'ar', 'ru', 'hi', 'ar', 'sw', 'ja', 'ko', 'id']

try:
  import kawa.ontology
except:
  !git clone https://github.com/ontocord/kawa

from collections import Counter
from kawa.ontology.ontology_builder_data import OntologyBuilderData
bd = OntologyBuilderData()
male_to_female_gender_swap = dict([(a, b) for a, b in bd.male_to_female_gender_swap.items() if a not in {"miss"}] + [(a[0].upper()+a[1:], b[0].upper()+b[1:]) for a, b in bd.male_to_female_gender_swap.items()])
female_to_male_gender_swap = dict([(a, b) for a, b in bd.female_to_male_gender_swap.items() if a not in {"miss"}] + [(a[0].upper()+a[1:], b[0].upper()+b[1:]) for a, b in bd.female_to_male_gender_swap.items()])

from kawa.ontology.ontology_manager import OntologyManager

!mkdir -p /content/drive/Shareddrives/ontocord_llc/safe_llm/fineweb-edu-multi/

def process_one(file):
  out_file = "/content/drive/Shareddrives/ontocord_llc/safe_llm/fineweb-edu-multi/"+file.split("/")[-1]
  if os.path.exists(out_file): return
  !touch $out_file
  with open(out_file, "w") as outf:

    tran_text = []
    tran_len = 0
    idx = -1
    for l in open(file):
      idx += 1
      dat = json.loads(l)
      dat['metadata']['lang'] = 'en'
      dat['metadata']['augmented'] = False
      dat['metadata']['original_en'] = ''
      text = dat['text']
      head = text[:100].lower()
      tail = text[-100:].lower()
      if "cc-by " in head or "cc-0 " in head or "cc-by-" in head or \
          "creative common" in head:
          if 'Flickr' in head or 'Flicker' in head or 'flickr' in head or 'flicker' in head: continue
      if "cc-by " in tail or "cc-0 " in tail or "cc-by-" in tail or \
          "creative common" in tail:
          if 'Flickr' in tail or 'Flicker' in tail or 'flickr' in tail or 'flicker' in tail: continue
      outf.write(json.dumps(dat)+"\n")

      if dat['metadata']['score'] > 1 and not dat['metadata']['nc']:
        prob_3 = 4
        prob_4 = 2
        len_text = len(dat['text'])
        if len_text > 1000 and len_text < 2500:
          prob_3 -= 1
          prob_4 -= 1
        if dat['metadata']['score'] == 2 and random.randint(0,100) == 0:
          tran_text.append (dat)
          tran_len += len(dat['text'])
        elif dat['metadata']['score'] == 3 and random.randint(0,prob_3) == 0:
          tran_text.append (dat)
          tran_len += len(dat['text'])
        elif dat['metadata']['score'] == 4 and random.randint(0,prob_4) == 0:
          tran_text.append (dat)
          tran_len += len(dat['text'])
        else:
          tran_text.append (dat)
          tran_len += len(dat['text'])

      if tran_len > 2000:
        from_text_arr = []
        for dat in tran_text:
          found_cap = 0
          for sent in dat['text'].split(". "):
            words = sent.split()[1:]
            if any(w for w in words if w[0] == w[0].upper() and w[0] not in "0123456789"):
              found_cap +=1
          text = text2 = dat['text'].strip()
          if found_cap <=  3 and not (" She " in dat['text'] or ' she ' in dat['text']):
            text2 = " ".join([t.replace(t.strip(",.;:\'\""), male_to_female_gender_swap.get(t.strip(",.;:\'\""),t.strip(",.;:\'\""))) for t in text2.split(" ")])
          text2 = text2.replace(" father or father ", " father ").replace(" mother or mother ", " mother ").replace(" hers or hers ", " hers ").replace(" her or her ", " her ").replace(" his or his ", " his ").replace(" woman or woman ", "woman ").replace(" man or man ", " man ")
          text2 = text2.replace(" father and father ", " father ").replace(" mother and mother ", " mother ").replace(" hers and hers ", " hers ").replace(" her and her ", " her ").replace(" his and his ", " his ").replace(" woman and woman ", "woman ").replace(" man and man ", " man ")
          if random.randint(0,1) and text2 != text:
            from_text_arr.append((".".join(text2[:1000].split(".")[:-1])+".") if len(text2) > 1000 else text2)
          else:
            from_text_arr.append((".".join(text[:1000].split(".")[:-1])+".") if len(text) > 1000 else text)
        from_text_arr = [a for a in from_text_arr if a.strip()]
        if not from_text_arr:
          tran_text = []
          train_len = 0
          continue
        from_text = "\n#\n".join(from_text_arr)
        lang = random.choice(langs)
        extra_time = 0
        if random.randint(0,5)==0: time.sleep(extra_time+random.randint(0,2))
        try:
          to_text = gt.translate(from_text, to_language=lang)
          if  len(to_text.strip()) < 10 : assert False
        except:
          try:
            time.sleep(extra_time+2+random.randint(1,4))
            to_text = gt.translate(from_text, to_language=lang)
            if len(to_text.strip()) < 10 :
              time.sleep(extra_time+5+random.randint(2,5))
              to_text = gt.translate(from_text, to_language=lang)
          except:
            print ('problem 1', lang)
            tran_text = []
            train_len = 0
            continue
        if len(to_text.strip()) < 10:
          print ('problem 2', lang)
          tran_text = []
          tran_len = 0
          continue
        to_text = to_text.split("\n#\n")
        if len(tran_text) != len(to_text):
          print ('problem 3', lang)
          tran_text = []
          tran_len = 0
          continue
        for (dat, txt, from_text) in zip(tran_text, to_text, from_text_arr):
          dat['metadata']['original_en'] = from_text
          dat['metadata']['lang'] = lang
          dat['metadata']['augmented'] = True
          dat['text'] = txt
          outf.write(json.dumps(dat)+"\n")

        tran_text = []
        tran_len = 0
import glob
from multiprocessing import Pool
files =  list(glob.glob("/content/drive/Shareddrives/ontocord_llc/safe_llm/fineweb-edu-2/*"))
random.shuffle(files)
with Pool(8) as p:
  p.map(process_one, files)
