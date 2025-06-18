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
import subprocess
from subprocess import TimeoutExpired
import json, os, glob, random
import multiprocessing
from multiprocessing import set_start_method
import os
import json, os, glob
from tqdm import tqdm
from cdifflib import CSequenceMatcher
from typing import List
import glob, json
import re
from huggingface_hub import hf_hub_download
import fasttext
from multiprocessing import Pool
from shared import *
#import translators as ts
import random
import time
import random
import time
import langid


def reduce (arg):
    global common_pile_sites, white_list_sites
    if common_pile_sites is None:
        common_pile_sites = set(json.load(open("common_pile_urls.json")))
        white_list_sites = set(json.load(open("white_list_urls.json")))
    
    global model, tokenizer, device, args, num_devices
    domain, device_no, args = arg
    model, tokenizer = init_model(device_no, args)
    files = [a for a in args.all_files if f"/{domain}.jsonl" in a]
    if not files: return []
    
    tmp_file = dedup_paraphrase_upsample_reduce(domain, files, args.output_dir, sent_reorder_prob=0.15, sent_upsample_prob=0.15, sent_shuffle_prob=0.15, do_augment=True)
    batch=[]
    if tmp_file:
        i = len(list(glob.glob(args.output_dir+f"/{domain}-*.jsonl")))
        prev_data = None
        prev_idx = ""
        batch = []
        for l in open(tmp_file, "rb"):
              try:
                  data= json.loads(l)
              except:
                  continue
              if not data['text'] or not data['metadata']: continue
              #break up long text to many shorter text
              if  "<|endoftext|>" in data['text'] and ((len(data['text']) > 15000 and data['lang'] in {"zh", "ko", "ja"}) or (len(data['text']) > 15000 and data['lang'] not in {"zh", "ko", "ja"})):
                  data_arr = []
                  arr = list(zip(data['text'].split("<|endoftext|>"), data['metadata']))
                  arr.sort(key=lambda a: len(a[0]))
                  prev_text = ""
                  prev_meta_list = []
                  for text, meta in arr:
                      if prev_text and len(prev_text) + len(text) >= 13000:
                          data2 = copy.deepcopy(data)
                          data['text'] = prev_text
                          data['metadata'] = prev_meta_list
                          prev_text = ""
                          prev_meta_list = []
                          data_arr.append(data2)
                      if prev_text:
                          prev_text += "<|endotext|>" + text
                      else:
                          prev_text = text
                      prev_meta_list.append(meta)
                  if prev_text:
                      data2 = copy.deepcopy(data)
                      data['text'] = prev_text
                      data['metadata'] = prev_meta_list
                      prev_text = ""
                      prev_meta_list = []
                      data_arr.append(data2)
              else:
                  data_arr = [data]
                  
              # now filter out non-permissive and collapse data at the same parent idx/urls
              for data in data_arr:
                  text_arr= []
                  meta_arr = []
                  old_text = data['text']

                  license_header_footer = data['metadata'][0]['license_header_footer']
                  for text, meta in zip(data['text'].split("<|endoftext|>"), data['metadata']):
                    if meta['lang'] not in {"zh", "ko", "ja"} and len(text) < 200:
                        #print ("too short")
                        continue
                    if meta['lang'] == 'en' and "hq-" not in data['domain'] and "math" not in data['domain']:
                        text2 = remove_citations(text)
                        if text2 != text:
                            text = text2
                    text_arr.append(text)
                    meta_arr.append(meta)
                    if not license_header_footer and meta['license_header_footer']:
                        license_header_footer = meta['license_header_footer']
                  data['text'] = "<|endoftext|>".join(text_arr)
                  data['metadata'] = meta_arr
                  data['license_header_footer'] = license_header_footer
                  if not data['text']:
                      #print (("empy text", old_text))
                      continue

                  # filter out all non-permissive data
                  is_permitted =  data['idx'] in common_pile_sites or data['idx'].replace("https://", "http://") in common_pile_sites \
                                      or data['idx'] in white_list_sites or data['idx'].replace("https://", "http://") in white_list_sites or \
                                      any(metadata for metadata in data['metadata'] if metadata['is_govt'] or metadata['idx'] in common_pile_sites or \
                                          metadata['idx'].replace("https://", "http://") in common_pile_sites or metadata['idx'] in white_list_sites or \
                                          metadata['idx'].replace("https://", "http://") in white_list_sites )
                  # one last test
                  if not is_permitted:
                      permitted, _ = is_idx_match(data)
                      if not permitted and not data['license_header_footer']:
                          continue

                  is_en = [ab[0] for ab in zip(data['text'].split("<|endoftext|>"), data['metadata']) if ab[1]['lang'] == 'en']
                  if not is_en:
                    text = data['text'].split("<|endoftext|>")[0]
                    if langid.classify(text[:min(len(text),100)])[0] == 'en':
                        data['lang'] = 'en'
                  if prev_data:
                        curr_idx = data['idx'].split("://",1)[-1].split("/")[0].split(".")
                        if len(curr_idx)> 2:
                            curr_idx = curr_idx[-2:]
                        curr_idx = ".".join(curr_idx)
                        if curr_idx == prev_idx and len(prev_data['text']+ data['text']) < 15000:
                            data['text'] += "<|endoftext|>" + prev_data['text']
                            data['metadata'].extend(prev_data['metadata'])
                        else:
                            if prev_data:
                                batch.append(prev_data)
                            prev_idx = ""
                            prev_data = None
                  prev_idx = data['idx'].split("://",1)[-1].split("/")[0].split(".")
                  if len(prev_idx)> 2:
                      prev_idx = prev_idx[-2:]
                  prev_idx = ".".join(prev_idx)
                  prev_data = data
            
        if prev_data:
            batch.append(prev_data)
            
        if batch:
          with open(args.output_dir+f"/{domain}-{i}.jsonl", "a+") as outf:
              for data in batch:
                  outf.write(json.dumps(prev_data)+"\n")
                  i+=1
                  batch = []
        os.system("rm "+tmp_file)
    
    # do some cleanup    
    files = [a for a in args.all_files if f"/{domain}.jsonl" in a]
    for file2 in files:
        os.system("mkdir -p "+ "/".join(file2.replace(args.input_dir, args.input_dir+"/done/").split("/")[:-1]))
        file3 = file2.replace(args.input_dir, args.input_dir+"/done/")
        os.system("mv "+file2 + " " +file3)            

    return domain

if __name__ == "__main__":
    args = parse_args()
    args.input_dir = args.target_dir+"3/"
    args.output_dir = args.target_dir+"4/"
    subset= args.subset
    if subset:
        args.output_dir = args.output_dir.rstrip("/")+"_"+subset+"/"
        args.input_dir = args.input_dir.rstrip("/")+"_"+subset+"/"    
    os.system(f"mkdir -p {args.output_dir}")
    print (args)
    os.system("rm -rf "+args.output_dir+"/"+str(get_rank())+".rank_done")        
    if True: #not os.path.exists(args.output_dir+"/"+str(get_rank())+".rank_done"):
        #load_fasttext_models()    
        args.all_files =[]
        args.all_files.extend(list(set(list(glob.glob(args.input_dir + '*.jsonl', recursive=True)) +  list(glob.glob(args.input_dir + '*/*.jsonl', recursive=True)) +  list(glob.glob(args.input_dir + '*/*/*.jsonl', recursive=True)))))
        args.all_files = [file for file in args.all_files if args.input_dir+"/done/" not in file]        
        all_domains = list(set([a.split("/")[-1].replace(".jsonl", "") for a in args.all_files]))    
        all_domains.sort()
        ws = get_world_size()
        rank = args.rank = get_rank()
        print ("starting rank", rank)
        rank2files = {}
        j = -1
        for file in all_domains:
            j += 1
            for k in range(ws):
                if j == k:
                    p = rank2files[k] = rank2files.get(k,[])
                    p.append(file)
                    if j == ws-1:
                        j = -1
                    break
        if rank in rank2files:
            domains = rank2files[rank]
            print ("files for rank", rank, domains)
            random.shuffle(domains)
            domains = [(domain, i%num_devices, args) for i, domain in enumerate(domains)]        
            with multiprocessing.Pool(10 if num_devices <= 1 else num_devices) as pool:    
                for domain in pool.imap_unordered(reduce, domains):
                    print ("done with "+ domain)
        wait_for_other_ranks(args.output_dir)
