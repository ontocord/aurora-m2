#@title fineweb permissive - further filter
import os, json
import glob, tqdm, json
if not os.path.exists("/content/drive"):
  from google.colab import drive
  drive.mount('/content/drive')

!mkdir -p /content/drive/Shareddrives/ontocord_llc/safe_llm/fineweb-edu-2/

import glob
for file in glob.glob(f"/content/drive/Shareddrives/ontocord_llc/safe_llm/fineweb-permissive/*.jsonl"):
  with open(file.replace("fineweb-permissive", "fineweb-edu-2"), "w") as outf:
    rights = 0
    cc = 0
    for idx, l in enumerate(open(file)):
      dat = json.loads(l)
      text = dat['text'].lower()
      dat['metadata']['nc'] = False
      head = text[:100]
      tail = text[-100:]
      if dat['metadata']['cc']:
        if ('picture' in head or 'image' in head) and ("cc-by " in head or "cc-0 " in head or "cc-by-" in head or  "creative common" in head):
          continue
        if ('picture' in tail or 'image' in tail) and ("cc-by " in tail or "cc-0 " in tail or "cc-by-" in tail or  "creative common" in tail):
          continue
        if ("cc-by " in head or "cc-0 " in head or "cc-by-" in head or  "creative common" in head) and ('noncommercial' in head or 'non-commercial' in head or 'non commercial' in head):
          dat['metadata']['nc'] = True
        if ("cc-by " in tail or "cc-0 " in tail or "cc-by-" in tail or  "creative common" in tail) and ('noncommercial' in tail or 'non-commercial' in tail or 'non commercial' in tail):
          dat['metadata']['nc'] = True
        if "cc-by-nc" in text or "cc-by-nc-sa" in text:
          dat['metadata']['nc'] = True
      elif 'rights reserved' in head or  'rights reserved' in tail:
        continue
      elif '©' in head or  '©' in tail :
        continue
      outf.write(json.dumps(dat)+"\n")
