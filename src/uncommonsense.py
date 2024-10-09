#@title uncommonsense
try:
  from datasets import load_dataset
except:
  !pip install -q datasets
from datasets import load_dataset
try:
  if dataset is None: assert False
except:

  dataset = load_dataset("allenai/UNcommonsense")


import random
with open("/content/drive/Shareddrives/ontocord_llc/uncommonsense.jsonl", "") as outf:
  intro = "### Below are very short scenarios that show suprising or unusual outcomes with explanations why those outcomes might occur.\n\n"
  text2 = intro
  if random.randint(0,1):
    text2 = text2.replace("Below are", "Please find below")
  if random.randint(0,1):
    text2 = text2.replace("Below are", "Here are")
  if random.randint(0,1):
    text2 = text2.replace("very short scenarios", "stories")
  if random.randint(0,1):
    text2 = text2.replace("why those outcomes might occur", "why")
  sep = "\n***\n"
  for idx, dat in enumerate(dataset['train']):
    #print (dat)
    exp = [s for s in dat['human_explanations'] if "(" not in s]
    if not exp: continue
    choice = random.randint(0,4)
    if choice == 0:
      text = dat['context']+"\nSuprisingly, what might happen next is "+dat['outcome'].replace(" is ", " will become ")+"\nBecause "+" ".join(exp)
    elif choice == 1:
      text = dat['context']+" Instead, "+" ".join(exp).strip() + " So, "+dat['outcome'].replace("will want to", "will").replace("will have to", "will")
    elif choice == 2:
      text = dat['context']+" But do you know why " + dat['outcome'].replace(" is ", " will become ").strip(".")+"?"+ "\nAnswer: "+" ".join(exp) + "\nSo, "+dat['outcome']
    elif choice == 3:
      text = dat['context']+" But " + dat['outcome'].replace(" is ", " will become ")+" Why?"+ "\nAnswer: Possibly, "+" ".join(exp).strip() + " So, "+(("we conclude that "+ dat['outcome']) if " is " in dat['outcome'] else dat['outcome'])
    else:
      text = "Context: "+dat['context']+"\nUnexpected outcome: "+dat['outcome']+"\nPossible explanations: "+" ".join(exp)
    text = text.replace("sued special", "used special")
    text2 = text2 + sep + text
    if len(text2) > 10000:
      outf.write(json.dumps({'text': text2.strip(), 'instruct': '', 'metadata': {'source': "allenai/UNcommonsense"}})+"\n")
      text2 = intro
      if random.randint(0,1):
        text2 = text2.replace("Below are", "Please find below")
      if random.randint(0,1):
        text2 = text2.replace("Below are", "Here are")
      if random.randint(0,1):
        text2 = text2.replace("very short scenarios", "stories")
      if random.randint(0,1):
        text2 = text2.replace("why those outcomes might occur", "why")
  if sep in text2:
    outf.write(json.dumps({'text': text2.strip(), 'instruct': '', 'metadata': {'source': "allenai/UNcommonsense"}})+"\n")
