#@title flatten youtube commons by id
import os, glob, json
import tqdm
from google.colab import drive
drive.mount('/content/drive')



aHash = {}

if True:
  !mkdir -p /content/drive/Shareddrives/ontocord_llc/safe_llm/youtube_commons_input2/
  #8074216
  cnt = -1

  for file in glob.glob("/content/drive/Shareddrives/ontocord_llc/safe_llm/youtube_commons_a*"):
    with open(file) as infile:
      for l in infile:
        cnt += 1
        dat = json.loads(l)
        lang = dat['metadata']['transcription_language']
        if dat['text'].count("[") > 1:
            continue
        if lang in {'zh', 'ko', 'ja'} and len(dat['text']) < 20: continue
        if lang not in {'zh', 'ko', 'ja'} and len(dat['text']) < 100: continue

        id = dat['metadata']['video_id']
        idx = id[:2].replace("-", "_")
        if idx  not in aHash:
          aHash[idx ] = open("/content/drive/Shareddrives/ontocord_llc/safe_llm/youtube_commons_input2/"+idx+"_out.jsonl", "a+")
        aHash[idx ].write(dat['metadata']['video_id']+"\t"+json.dumps(dat)+"\n")

  for id in aHash:
      aHash[id].close()
