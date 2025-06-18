import os
import multiprocessing
import glob, random
import subprocess
from pathlib import Path
from tqdm import tqdm
import json
import json, glob, os
import math
import multiprocessing, functools, json, glob
from flagged_words import *
from collections import Counter
import string
import argparse
from collections import defaultdict
import sys
import pyarrow.parquet as pq
import time, random
import json, os, glob, random
import multiprocessing
from multiprocessing import set_start_method
import os
import json, os, glob
from tqdm import tqdm
from typing import List
import glob, json
import re
from huggingface_hub import hf_hub_download
import fasttext
from multiprocessing import Pool
from shared import *


def process (arg):    
    global sentHash
    global model, tokenizer, device, args, num_devices
    global common_pile_sites, white_list_sites
    if common_pile_sites is None:
        common_pile_sites = set(json.load(open("common_pile_urls.json")))
        white_list_sites = set(json.load(open("white_list_urls.json")))
    
    
    file, device_no, args = arg
    ret = []
    paraphrase = []
    model, tokenizer = init_model(device_no, args)
    load_fasttext_models()    
    for l in open(file, "rb"):
        #print (num_devices)
        #line, device_no, args = arg
        try:
            data = json.loads(l)
        except:
            print ("problem loading line", l)
            continue
        if len(data['text']) < 200:
            continue
        data= dedup(data)
        if not data:
            continue
        fix_idx(data)            
        new_text = []
        new_meta = []
        upsampled=False
        if 'quality_classify' not in data or ('domain' in data and 'multilingual' in data['domain']):
            for text, metadata in  zip(data['text'].split("<|endoftext|>"), data['metadata']):
              is_hq = ('edu_int_score' in metadata and metadata['edu_int_score'] >= 3) or ('dcad' in metadata['source'] and 'keep' in metadata['source']) or 'mint-pdf' in metadata['source'] or 'MAGA' in metadata['source'] or 'cosmo' in metadata or 'nemo_high' in metadata['source']
              en_text = ""
              if metadata['lang'] == 'en':
                en_text = text
              elif metadata['lang'] != 'en' or 'europat' in metadata['source']:
                lang = langid.classify(text[100:600])[0]                  
                if lang == 'en':
                  en_text = text[100:600]
                elif langid.classify(text[-600:-100])[0] == 'en':
                  en_text = text[-600:-100]
                if metadata['lang'] == 'en':
                    metadata['lang'] = lang
              if not en_text and len(translation_models) <= 5:
                  text0 = text[:min(len(text), 1000)]
                  model, tokenizer = get_translation_model_tokenizer(metadata['lang'], "en")
                  if model:
                      text0 = ctranslate2_with_batching(model, tokenizer, [text0])[0]
                      metadata['en_text'] = en_text = text0
                      print ("translated", text0)
              if not en_text:
                  metadata['quality_classify'] = [0, ""]
                  metadata['rpj_score'] =  0
                  metadata['edu_score'] =  0
                  metadata['domain'] = "multilingual-"+lang
              else:
                  if is_hq:
                    aggregate, score, score3 = classify_and_qs(en_text)
                    metadata['quality_classify'] = list(aggregate)
                    metadata['rpj_score'] =  score3
                    metadata['edu_score'] =  score
                    metadata['quality_classify'][1] = metadata['quality_classify'][1].replace("Wikipedia-(en)", "Wikipedia")
                    metadata['domain'] = metadata['quality_classify'][1].split("-")[0]
                  else:
                    aggregate, score, score3 = classify_and_qs(en_text)
                    metadata['domain'] = aggregate[1].split("-")[0]          
                    do_special = True
                    if 'science' in metadata['domain'] or 'engine' in metadata['domain'] or 'ology' in metadata['domain'] or 'physi' in metadata['domain'] or "math" in metadata['domain']:
                        do_special=False
                    if aggregate[0] < 0.01 and do_special:
                        score1 = get_special_char_score(en_text)
                        if score1 > 0.1:
                            #print (("filtering low quality", aggregate[0], en_text))
                            continue
                        score2 = get_stopword_score(en_text)
                        if score2 < 0.05:
                            #print (("filtering low quality", aggregate[0], en_text))
                            continue
                    if aggregate[0] < 0.005:
                        #print (("filtering low quality", aggregate[0], en_text))
                        continue
                    metadata['quality_classify'] = list(aggregate)
                    metadata['rpj_score'] =  score3
                    metadata['edu_score'] =  score
                    metadata['quality_classify'][1] = metadata['quality_classify'][1].replace("Wikipedia-(en)", "Wikipedia")
                    if  metadata['quality_classify'][0] > 0.3:
                      is_hq = True

                  # do some additional cleanup
                  if "law" in metadata['source'] or "mint-pdf" in metadata['source'] or "kl3m" in metadata['source'] or 'fandom.com' in metadata['idx'] or 'wikia.com' in metadata['idx'] or "wikia.org" in metadata['idx'] or 'mail' in metadata['idx'] or "gutenberg" in metadata['idx']:
                    text = cleanup_raw_text(text, metadata['lang'], cleanup_sents=True)
                    if len(text) < 200:
                        continue
              domain = metadata['domain']
              domain = ("hq-" if is_hq else "")+domain
              metadata['domain'] = domain
              if  "<|" not in data['text'] and "<transcript" not in data['text'] and"<caption" not in data['text'] and "<image" not in data['text'] and "<audio" not in data['text'] and len(data['text']) > 15000 and data['lang'] not in {"zh", "ko", "ja"}:
                  # break up the data into parts
                  arr = data['text'].split(" ")
                  for rng in range(0, len(arr), 4000):
                      text2 = " ".join(arr[rng:min(len(arr),rng+4000)]).strip()
                      new_text.append(text2)
                      new_meta.append(metadata)
                  # upsample some
                  if  "hq-" in domain:
                      for rng in range(0, len(arr), 4000):
                          if random.randint(0,1)==0: continue
                          text2 = " ".join(arr[rng:min(len(arr),rng+4000)]).strip()                      
                          text3 =register_text_for_upsample(text2, metadata['lang'], add_trans_prob=0.1, sent_reorder_prob=0.1, sent_shuffle_prob=0.1)
                          if text3 != text2:
                              new_text.append(text3)
                              new_meta.append(metadata)
                          
              elif "<|" not in data['text'] and "<transcript" not in data['text'] and"<caption" not in data['text'] and "<image" not in data['text'] and "<audio" not in data['text'] and len(data['text']) > 4000 and data['lang'] in {"zh", "ko", "ja"}:
                  arr = list(data['text'])
                  # break up the data into parts                  
                  for rng in range(0, len(arr), 4000):
                      text2 = "".join(arr[rng:min(len(arr),rng+4000)]).strip()
                      new_text.append(text2)
                      new_meta.append(metadata)
                  # upsample some                      
                  if  "hq-" in domain:
                      for rng in range(0, len(arr), 4000):
                          if random.randint(0,1)==0: continue
                          text2 = "".join(arr[rng:min(len(arr),rng+4000)]).strip()                      
                          text3 = register_text_for_upsample(text2, metadata['lang'], do_augment=True)
                          upsampled=True                          
                          if text3 != text2:
                              new_text.append(text3)
                              new_meta.append(metadata)
                          
                      
              # double the hq data with agumented permuted text
              elif "hq-" in domain and  not ('math' in metadata['source'] or "{" in text or "\ndef " in text or "):" in text or  "${" in text):
                  new_text.append(text)
                  new_meta.append(metadata)
                  
                  text2 = augment_for_upsample(text, meta['lang'])
                  upsampled=True
                  if text != text2:
                      new_text.append(text2)
                      new_meta.append(metadata)
              else:
                  new_text.append(text)
                  new_meta.append(metadata)

            if new_meta: 
                data['metadata'] = new_meta
                data['text'] = "<|endoftext|>".join(new_text)
                
        data['lang'] = data['metadata'][0]['lang']
        metadata_list = data['metadata']
        domains = [metadata['domain'] for metadata in metadata_list if 'domain' in metadata]    
        data['domain'] = data.get('domain', domains[0] if domains else 'fictional')
        domain = data['domain']
        if any(metadata for metadata in metadata_list if 'finemath' in metadata['source']):
            if "hq-" in domain:
                domain = "hq-mathematics"
            else:
                domain = "mathematics"            
        elif "fandom.com" in data['idx'] or  "wikia.com" in data['idx'] or   "wikia.org" in data['idx'] or ("gutenberg.org" in data['idx'] and "wiki" not in data['idx']):
            if "hq-" in domain:
                domain = "hq-fictional-"+domain.replace("hq-","")
            else:
                domain = "fictional-"+domain
        else:
            domains = [metadata['domain'] for metadata in metadata_list if 'domain' in metadata]
            if domains and domain not in domains and domain.replace("hq-", "") not in domains:
                domains.sort(key=lambda a: len(a))
                if domains:
                    if len(domains) > 2 and "public" in domains[-1]:
                        domain = domains[-2]
                    else:
                        domain = domains[-1]
        if domain:
            data['domain'] = domain
        ret.append(data)
        # paraphrase things we know we won't strip b/c of permissions        
        if upsampled or  any(meta for meta in data['metadata'] if meta['idx'] in white_list_sites or meta['idx'] in common_pile_sites or meta['is_govt'] or 'curated' in meta['source'] or 'kl3m' in meta['source'] or 'common-pile' in meta['source']):
            paraphrase.append(data)
            if len(paraphrase)==400:
                generate_upsample(paraphrase)
                paraphrase = []
        if "hq-" in data['domain']:
            #double the data with some variations
             new_text = []
        new_meta = []
        
    generate_upsample(paraphrase)
    pid = str(get_rank())+str(os.getpid())
    os.system("mkdir -p "+ "/".join(file.replace(args.input_dir, args.input_dir+"/done/"+pid+"/").split("/")[:-1]))
    file3 = file.replace(args.input_dir, args.input_dir+"/done/"+pid+"/")
    os.system("mv "+file + " " +file3)            
    #with open(file.replace(args.input_dir, f"{args.output_dir}/rank_{args.rank}/").replace(".jsonl", ".done"), "w") as o: pass        
    return ret

if __name__ == "__main__":
    args = parse_args()
    args.input_dir = args.target_dir+"2/"
    args.output_dir = args.target_dir+"3/"
    subset= args.subset
    if subset:
        args.output_dir = args.output_dir.rstrip("/")+"_"+subset+"/"
        args.input_dir = args.input_dir.rstrip("/")+"_"+subset+"/"
    print (args)
    os.system("rm -rf "+args.output_dir+"/"+str(get_rank())+".rank_done")    
    if True: 
      args.all_files =[]
      args.all_files.extend(list(set(list(glob.glob(args.input_dir + '*.jsonl', recursive=True)) +  list(glob.glob(args.input_dir + '*/*.jsonl', recursive=True)) +  list(glob.glob(args.input_dir + '*/*/*.jsonl', recursive=True)))))
      args.all_files = [file for file in args.all_files if args.input_dir+"/done/" not in file]      
      args.all_files.sort()
      ws = get_world_size()
      rank = args.rank = get_rank()
      print ("starting rank", rank)
      args.all_files.sort()
      rank2files = {}
      j = -1
      for file in args.all_files:
          j += 1
          for k in range(ws):
              if j == k:
                  p = rank2files[k] = rank2files.get(k,[])
                  p.append(file)
                  if j == ws-1:
                      j = -1
                  break
      if rank in rank2files:
          os.system(f"mkdir -p {args.output_dir}/rank_{rank}")
          files = rank2files[rank]
          random.shuffle(files)
          files = [(file, i%num_devices, args) for i, file in enumerate(files)]
          domain2outf = {}
          if  not torch.cuda.is_available():
            load_fasttext_models()
          domain2shard = {}
          domain2size = {}
          for file in glob.glob(f"{args.output_dir}/rank_{rank}/*.jsonl"):
              shard = int(file.split("/")[-1].split("_")[-1].split("-")[-1].replace(".jsonl", ""))
              domain = "_".join(file.split("/")[-1].split("_")[:1])
              domain2shard[domain] = max(domain2shard.get(domain, 0), shard)
              domain2size[domain] = 0
          with multiprocessing.Pool(10 if num_devices <= 1 else num_devices) as pool: 
              for ret in pool.imap_unordered(process, files):
                  for data in ret:
                      if not data:
                          continue
                      data = dedup(data)
                      # only do this if you are multiprocessing!!
                      if not data: continue
                      domain = data['domain']
                      if not domain.strip():
                          domain = data['domain'] = "fictional"
                      if domain not in domain2outf:
                          shard = domain2shard[domain] = domain2shard.get(domain, 0)
                          domain2size[domain] = domain2size.get(domain, 0)
                          domain2outf[domain] = open(f"{args.output_dir}/rank_{rank}/{domain}-{shard}.jsonl", "a+")
                      outf = domain2outf[domain]
                      outf.write(json.dumps(data)+"\n")
                      if (domain2size[domain] +1) % 1000 == 0:
                          outf.close()                      
                          domain2size[domain] = 0
                          shard = domain2shard[domain] = domain2shard.get(domain, 0) + 1
                          domain2outf[domain] = open(f"{args.output_dir}/rank_{rank}/{domain}-{shard}.jsonl", "a+")
                      domain2size[domain] += 1
                      if random.randint(0,1000)==0:
                          outf.close()
                          shard = domain2shard[domain] = domain2shard.get(domain, 0)
                          domain2size[domain] = domain2size.get(domain, 0)
                          domain2outf[domain] = open(f"{args.output_dir}/rank_{rank}/{domain}-{shard}.jsonl", "a+")
                  
      wait_for_other_ranks(args.output_dir)
