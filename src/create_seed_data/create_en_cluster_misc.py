
import glob, json
shard = 0
curr_size = 0
outf = open("en_cluster_with_tasks_"+str(shard)+".jsonl", "w")
for file in glob.glob("/content/drive/Shareddrives/ontocord_llc/aurora-en-cluster/en/*"):
  with open(file) as inf:
    cluster_id = file.split("/")[-1].replace(".jsonl", "")
    text = ""
    source = ""
    dat = None
    prev_dat = None
    for l in inf:
      prev_dat = dat
      dat = json.loads(l)
      if len(text) > 40000:
        if source:
          print (text.count("<|endoftext|>"))
          line = json.dumps({'text': text, 'metadata': {'source': source, 'cluster': cluster_id}})
          outf.write (line+"\n")
          curr_size += len(line)
          #print ( (source,  cluster_id, text[:100], prev_dat['meta'] if prev_dat else None,))
          #break
        if curr_size > 5000000000:
          curr_size = 0
          outf.close()
          shard += 1 
          outf = open("en_cluster_"+str(shard)+".jsonl", "w")
        text = ""
        source = ""
      source2 = ""
      if "APPLICATION_ID" in dat['meta']:
        source2 = "nih"
      if 'bibliographic_information' in dat['meta']:
        source2 = "uspto"
      if 'case_jurisdiction' in dat['meta']:
        source2 = "freelaw"
      if 'url' in dat['meta'] and 'wiki' in dat['meta']['url']:
        source2 = 'wikipedia'
      elif 'source' in dat['meta']:
        source2 = dat['meta']['source']
      if 'source' in dat['meta'] and ('red-pajama-cc' in dat['meta']['source'] or 'refinedweb' in dat['meta']['source']): continue  
      text2= dat['text']+"\n\n===\n\n"+random.choice(["Potential Tasks", "Exercises", "Questions", "Further Study"])+":\n"+dat['meta']['instruction'].replace("blog", random.choice(["blog", "article", "post", "text", "content"]))\
            .replace("following paragraph", random.choice(["following paragraph", "following", "this", "our discussion subject"]))\
            .replace("pragraph", random.choice(["paragraph", "text", "piece"]))
            

      if source2 == 'red-pajama-books':
        source2 = ""
        for year in range(1700, 1920):
          if str(year) in dat['meta']['name']:
            source2 = "red-pajama-books"
            break
        if not source2: 
          prev_dat = None
          continue
      if source == "" and source2:
        
        source = source2
      elif source2 and source2 not in source:
        source = source+","+source2
      if not text:
          text = text2
      else:
        if text2.startswith("```"):
          text = text2 + "<|endoftext|>"+text
        else:
          text = text + "<|endoftext|>"+text2
      if not source2:
        prev_dat = None
        #print (dat)
    if text and source:
      print ('!!', text.count("<|endoftext|>"))
      line = json.dumps({'text': text, 'metadata': {'source': source, 'cluster': cluster_id}})
      outf.write (line+"\n")
        
      #print (file, dat['meta']['programming_language'] if 'programming_language' in dat['meta'] else '', dat)
      #break
