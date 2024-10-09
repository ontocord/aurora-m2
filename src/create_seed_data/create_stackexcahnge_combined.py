#@title create combined stackexchange
import json
currStats = {'apple': 320,
 'askubuntu': 348,
 'dba': 1021,
 'electronics': 1857,
 'ell': 3041,
 'english': 6409,
 'es': 2,
 'gaming': 815,
 'gis': 746,
 'magento': 70,
 'mathematica': 1262,
 'mathoverflow': 1543,
 'math': 12996,
 'meta': 927,
 'physics': 6313,
 'pt': 14,
 'ru': 83,
 'salesforce': 880,
 'scifi': 6956,
 'serverfault': 549,
 'sharepoint': 123,
 'softwareengineering': 2973,
 'stackoverflow': 13083,
 'stats': 3219,
 'superuser': 1120,
 'tex': 2358,
 'unix': 1863,
 'wordpress': 130}

prev_topic = ""
curr_text = ""
topicMapper = {'ell': 'English Language Learner', 'meta': 'Technical Support'}
cntHash = {}
with open("/content/drive/Shareddrives/ontocord_llc/stack_excahnge_combined.jsonl", "w") as combined:
  with open("/content/drive/Shareddrives/ontocord_llc/stack_excahnge2.jsonl") as info:
    for l in info:
      dat = json.loads(l)
      if float(dat['meta']['question_score']) < 5: continue
      if len(dat['text']) > 4000:
        dat['text'] = "A:".join(dat['text'][:4000].split("A:")[:-1])
      topic = dat['meta']['url'].split("//")[-1].split(".")[0]
      if topic in {'ell', 'english', 'math', 'mathoverflow', 'physics', 'stats', 'softwareengineering'}:
        if topic != prev_topic and prev_topic != "":
          if curr_text:
            if curr_text.startswith("<|endoftext|>"):
              curr_text = curr_text[len("<|endoftext|>"):].strip()
            combined.write(json.dumps({'text': curr_text, 'meta': {'source': dat['meta']['source'], 'topic': prev_topic}})+"\n")
          curr_text = dat['text']
        elif len(curr_text+dat['text']) > 4000:
          if curr_text:
            if curr_text.startswith("<|endoftext|>"):
              curr_text = curr_text[len("<|endoftext|>"):].strip()
            combined.write(json.dumps({'text': curr_text, 'meta': {'source': dat['meta']['source'], 'topic': prev_topic}})+"\n")
          curr_text = dat['text']
        else:
          curr_text += "<|endoftext|>"+ dat['text']

        cntHash[topic] = cntHash.get(topic,0) + 1
        prev_topic = topic
    if curr_text:
        if curr_text.startswith("<|endoftext|>"):
          curr_text = curr_text[len("<|endoftext|>"):].strip()
        combined.write(json.dumps({'text': curr_text, 'meta': {'source': dat['meta']['source'], 'topic': prev_topic}})+"\n")
  print (cntHash)
