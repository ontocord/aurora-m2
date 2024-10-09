#@title logic datasets
import glob
import random
import json
files = list(glob.glob("llm-logic/data/*/*/*/*"))
print (len(files))
file = random.choice(files)
print (file)
json.load(open(file))


#!git clone https://github.com/teacherpeterpan/Logic-LLM
dat = json.load(open("/content/Logic-LLM/data/AR-LSAT/train.json"))
print (len(dat))
for d in dat:
  #print (d.keys())
  print (d['context']+"\n"+d['question']+"\n"+"\n".join(d['options'])+"\nAnswer: "+d['answer'])
