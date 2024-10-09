#@title create wit_base
import json,yaml,sys,os
import pandas as pd
import locale
def getpreferredencoding(do_setlocale = True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding

try:
  import einops
except:
  #!pip install git+https://github.com/kernelmachine/transformers@openlm#egg=transformers
  !pip install -q bitsandbytes datasets  pyspellchecker peft accelerate
  !pip install -q pyspellchecker einops

from datasets import load_dataset

dataset = load_dataset("wikimedia/wit_base", streaming=True)
with open("/content/drive/Shareddrives/ontocord_llc/safe_llm/wit_base.jsonl", "w") as outf:
  reader = enumerate(dataset['train'])
  while True:
    try:
      idx, data = next(reader)
    except:
      continue
    d = data['wit_features']
    #for idx, d in enumerate():
    if 'page_url' in d:
      del d['page_url']
    if 'is_main_image' in d:
      del d['is_main_image']
    if 'page_changed_recently' in d:
      del d['page_changed_recently']
      #
    ab = [(('\n## '+ (a[0]+' :: ' + a[1] if (a[0] != a[1] and a[1]) else a[0])), ('' if not a[2] and not a[3] else ("\n - "+a[2]+"\n" if not a[3] else ("\n - "+a[3]+"\n" if not a[2] else ("\n - "+a[2]+'. ' + a[3] if a[2] != a[3] else "\n - "+a[2])+"\n"))))  for a in zip(d['page_title'], d['hierarchical_section_title'], d['caption_alt_text_description'], d['caption_title_and_reference_description'])  if a[0] or a[1] or a[2] or a[3]]
    basic = ""
    for a, b in ab:
      if b.strip(" -") not in a:
        c = a+b
        if c not in basic:
          basic = basic+""+ c
    basic = basic.strip().replace(" [SEP]", ".")
    if (basic):
      outf.write(json.dumps({'text': basic, 'instruct': '', 'metadata': {'source': 'wikimedia/wit_base/'+str(idx)}})+"\n")
      if len(basic) > 100 and len(d['page_title']) > 3:
        outf.write(json.dumps({'text': pd.DataFrame.from_dict(d).to_markdown().replace(" [SEP] ", ".     \\n"), 'instruct': '', 'metadata': {'source': 'wikimedia/wit_base/'+str(idx)}})+"\n")
