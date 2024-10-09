import glob, json
shard = 0
curr_size = 0
outf = open("starcoder_cluster_"+str(shard)+".jsonl", "w")
for file in glob.glob("/content/drive/Shareddrives/ontocord_llc/aurora-m-cluster/code/*"):
  with open(file) as inf:
    cluster_id = file.split("/")[-1].replace(".jsonl", "")
    text = ""
    source = ""
    for l in inf:
      dat = json.loads(l)
      if len(text) > 20000:
        line = json.dumps({'text': text, 'metadata': {'source': source, 'cluster': cluster_id}})
        outf.write (line+"\n")
        curr_size += len(line)
        if curr_size > 5000000000:
          curr_size = 0
          outf.close()
          shard += 1 
          outf = open("starcoder_cluster_"+str(shard)+".jsonl", "w")
        text = ""
        source = ""
        
      if 'programming_language' in dat['meta'] and dat['meta']['programming_language'] == 'assembly': continue
      text2= dat['text']
      programming_lang = dat['meta']['programming_language'] if 'programming_language' in dat['meta'] else ('python' if 'python' in text2 or '\ndef ' in text2 else '')
      if not programming_lang and '###' in text2:
        programming_lang = 'markdown'
      if 'jupyter' in programming_lang:
        programming_lang = ''
      if source == "" and 'source' in dat['meta']:
        source = dat['meta']['source']
      elif 'source' in dat['meta'] and dat['meta']['source'] not in source:
        source = source+","+dat['meta']['source']
      if '```' not in text and  programming_lang:
          text2 = "```"+ programming_lang +"\n"+text2+"\n```"
      if not text:
          text = text2
      else:
        if text2.startswith("```"):
          text = text2 + "<|endoftext|>"+text
        else:
          text = text + "<|endoftext|>"+text2
    if text:
      line = json.dumps({'text': text, 'metadata': {'source': source, 'cluster': cluster_id}})
      outf.write (line+"\n")
        
      #print (file, dat['meta']['programming_language'] if 'programming_language' in dat['meta'] else '', dat)
      #break
