#@title fineweb permissive
import os
import glob, tqdm, json
if not os.path.exists("/content/drive"):
  from google.colab import drive
  drive.mount('/content/drive')
try:
  from datasets import load_dataset
except:
  !pip install -q datasets
from datasets import load_dataset

all_segments = ['CC-MAIN-2020-45', 'CC-MAIN-2017-47', 'CC-MAIN-2018-09', 'CC-MAIN-2017-13', 'CC-MAIN-2014-52', 'CC-MAIN-2017-43',
                'CC-MAIN-2016-36', 'CC-MAIN-2014-41', 'CC-MAIN-2018-39', 'CC-MAIN-2020-50', 'CC-MAIN-2017-26', 'CC-MAIN-2018-43',
                'CC-MAIN-2014-49', 'CC-MAIN-2023-23', 'CC-MAIN-2022-21', 'CC-MAIN-2017-04', 'CC-MAIN-2014-15', 'CC-MAIN-2015-18',
                 'CC-MAIN-2021-31', 'CC-MAIN-2019-18', 'CC-MAIN-2017-51', 'CC-MAIN-2019-22', 'CC-MAIN-2019-39', 'CC-MAIN-2016-07',
                'CC-MAIN-2018-30', 'CC-MAIN-2018-47', 'CC-MAIN-2021-17', 'CC-MAIN-2014-23', 'CC-MAIN-2015-35', 'CC-MAIN-2017-39',
                'CC-MAIN-2023-14', 'CC-MAIN-2019-30', 'CC-MAIN-2014-42', 'CC-MAIN-2021-04', 'CC-MAIN-2020-34', 'CC-MAIN-2017-17',
                 'CC-MAIN-2019-47', 'CC-MAIN-2020-24', 'CC-MAIN-2017-09', 'CC-MAIN-2018-51', 'CC-MAIN-2018-34', 'CC-MAIN-2013-48',
                'CC-MAIN-2018-05', 'CC-MAIN-2021-10', 'CC-MAIN-2023-40', 'CC-MAIN-2016-50', 'CC-MAIN-2021-21', 'CC-MAIN-2020-16',
                'CC-MAIN-2018-17', 'CC-MAIN-2019-51', 'CC-MAIN-2021-25', 'CC-MAIN-2020-29', 'CC-MAIN-2015-14', 'CC-MAIN-2018-26',
                 'CC-MAIN-2018-13', 'CC-MAIN-2021-49', 'CC-MAIN-2019-13', 'CC-MAIN-2021-39', 'CC-MAIN-2023-50', 'CC-MAIN-2015-22',
                'CC-MAIN-2024-10', 'CC-MAIN-2022-49', 'CC-MAIN-2019-09', 'CC-MAIN-2019-43', 'CC-MAIN-2019-26', 'CC-MAIN-2017-34',
                'CC-MAIN-2016-26', 'CC-MAIN-2014-35', 'CC-MAIN-2016-18', 'CC-MAIN-2023-06', 'CC-MAIN-2015-32', 'CC-MAIN-2016-30',
                 'CC-MAIN-2015-40', 'CC-MAIN-2015-11', 'CC-MAIN-2022-27', 'CC-MAIN-2019-04', 'CC-MAIN-2016-40', 'CC-MAIN-2014-10',
                'CC-MAIN-2019-35', 'CC-MAIN-2017-30', 'CC-MAIN-2016-44', 'CC-MAIN-2015-27', 'CC-MAIN-2022-05', 'CC-MAIN-2022-40',
                'CC-MAIN-2017-22', 'CC-MAIN-2021-43', 'CC-MAIN-2016-22', 'CC-MAIN-2015-48', 'CC-MAIN-2013-20', 'CC-MAIN-2022-33',
                'CC-MAIN-2020-05', 'CC-MAIN-2020-10', 'CC-MAIN-2015-06', 'CC-MAIN-2018-22', 'CC-MAIN-2020-40']


from multiprocessing import Process
import random
random.shuffle(all_segments)

!rm -rf /content/fineweb-gov-*/
!rm -rf /content/fineweb-edu/
!mkdir -p /content/drive/Shareddrives/ontocord_llc/safe_llm/fineweb-gov/

from datasets import load_dataset
import json

def filter(segment):
  #print (segment)
  %cd /content/
  dataset = load_dataset(f"fineweb-gov-{segment}", streaming=True)
  with open(f"/content/drive/Shareddrives/ontocord_llc/safe_llm/fineweb-gov/{segment}.jsonl", "w") as outf:
    for idx, dat in enumerate(dataset['train']):
      if idx % 100000 == 0:
        print (idx, segment)
      text = dat['text']
      head = text[:100].lower()
      tail = text[-100:].lower()
      if "cc-by " in head or "cc-by " in tail or "cc-0 " in head or "cc-0 " in tail or "cc-by-" in head or "cc-by-" in tail or \
        "creative common" in head or "creative common" in tail:
        dat['cc'] = True
      else:
        dat['cc'] = False
      if dat['cc']  or "europa.eu/" in dat['url'] or ".un/" in dat['url'] or ".int/" in dat['url'] or ".gov/" in dat['url'] or '.gov.' in dat['url'] or '.gouv.' in dat['url']:
        if "ymca.int" in dat['url']: continue
        outf.write(json.dumps(dat)+"\n")
  !rm -rf /content/fineweb-gov-$segment

import time
from multiprocessing import Process
p =[]
for segment in all_segments:
    file = f"/content/drive/Shareddrives/ontocord_llc/safe_llm/fineweb-gov/{segment}.jsonl"
    if os.path.exists(file):
      continue
    !touch $file
    %cd /content/
    !rm -rf /content/fineweb-edu/
    !GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
    %cd /content/fineweb-edu
    !mkdir -p /content/fineweb-gov-$segment/data
    !git lfs pull --include /data/$segment/*
    !mv data/$segment  /content/fineweb-gov-$segment/data
    !du -sh /content/fineweb-gov-$segment/data/*
    %cd /content/
    !rm -rf /content/fineweb-edu/
    p.append(Process(target=filter, args=(segment,)))
    p[-1].start()
    while True:
      is_alive = 0
      for proc in p:
        if not proc.is_alive():
          proc.join()
        else:
          is_alive += 1
      if is_alive < 3:
        break
      time.sleep(60)
for proc in p:
  if proc.is_alive():
    proc.join()
