#@title logic datasets
import glob
import random
import json
files = list(glob.glob("llm-logic/data/*/*/*/*"))
print (len(files))
file = random.choice(files)
print (file)
json.load(open(file))
