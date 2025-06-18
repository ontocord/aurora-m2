import os
import multiprocessing
import glob, random
import subprocess
from pathlib import Path
from tqdm import tqdm
import json
import json, glob, os

import multiprocessing, functools, json, glob
from flagged_words import *


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
from shared import *


def reduce (arg):
    global model, tokenizer, device, args, num_devices
    shard, device_no, args = arg
    model, tokenizer = init_model(device_no, args)
    files = [a for a in args.all_files if f"/{shard}.jsonl" in a]
    tmp_file = dedup_paraphrase_upsample_reduce(shard, files, args.output_dir, add_trans_prob=0.1, sent_reorder_prob=0.1, sent_shuffle_prob=0.1)
    if tmp_file:
        # collapse by exact url match
        prev_data = None
        prev_idx= ""
        with open(args.output_dir+f"/{shard}.jsonl", "a+") as outf:
            for l in open(tmp_file, "rb"):
                l = l.strip()
                try:
                    data = json.loads(l)
                except:
                    continue
                lang = data['metadata'][0]['lang']
                if 'metadata' in data and type(data['metadata']) is str:
                    try:
                        data['metadata'] = json.loads(data['metadata'])
                    except:
                        pass
                if type(data['metadata']) is not list:
                   data['metadata']= [data['metadata']]
                #print (data)
                
                #data = cleanup_raw_text(data)
                if prev_data:
                    if data['idx'] == prev_idx:
                        if not data['text'] or data['text'].strip()[:100] in prev_data['text']:
                            if 'domain' in data and ('domain' not in prev_data['metadata'][-1] or "public" in prev_data['metadata'][-1]['domain']):
                                prev_data['metadata'][-1]['domain'] = data['domain']
                            continue
                        else:
                            if data['text']:
                                data['text'] += "<|endoftext|>" + prev_data['text']
                                data['metadata'].extend(prev_data['metadata'])
                            else:
                                data['text'] = prev_data['text']
                    else:
                        if prev_data  and not filter_copyright_and_content_issues(prev_data):
                            outf.write(json.dumps(prev_data)+"\n")
                        prev_idx = ""
                        prev_data = None
                prev_data = data
                prev_idx = prev_data['idx']
            if prev_data:
                if not filter_copyright_and_content_issues(prev_data):
                    outf.write(json.dumps(prev_data)+"\n")
                prev_data = None
        os.system("rm "+tmp_file)
    files = [a for a in args.all_files if f"/{shard}.jsonl" in a]
    pid = str(get_rank())+str(os.getpid())
    for file2 in files:
        os.system("mkdir -p "+ "/".join(file2.replace(args.input_dir, args.input_dir+"/done/"+pid+"/").split("/")[:-1]))
        file3 = file2.replace(args.input_dir, args.input_dir+"/done/"+pid+"/")
        os.system("mv "+file2 + " " +file3)            
        
    return shard

if __name__ == "__main__":
    args = parse_args()
    args.input_dir = args.target_dir+"1/"
    args.output_dir = args.target_dir+"2/"
    
    subset= args.subset
    if subset:
        args.output_dir = args.output_dir.rstrip("/")+"_"+subset+"/"
        args.input_dir = args.input_dir.rstrip("/")+"_"+subset+"/"            
    print (args)
    os.system(f"mkdir -p {args.output_dir}")
    args.all_files = []
    args.all_files.extend(list(set(list(glob.glob(args.input_dir + '*.jsonl', recursive=True)) +  list(glob.glob(args.input_dir + '*/*.jsonl', recursive=True)) +  list(glob.glob(args.input_dir + '*/*/*.jsonl', recursive=True)))))
    args.all_files = [file for file in args.all_files if args.input_dir+"/done/" not in file]
    os.system("rm -rf "+args.output_dir+"/"+str(get_rank())+".rank_done")
    if True: # not os.path.exists(args.output_dir+"/"+str(get_rank())+".rank_done"):

        ws = get_world_size()
        rank = args.rank = get_rank()
        all_shard = list(set([a.split("/")[-1].replace(".jsonl", "") for a in args.all_files]))
        all_shard.sort()
        rank2files = {}
        j = -1
        for file in all_shard:
            j += 1
            for k in range(ws):
                if j == k:
                    p = rank2files[k] = rank2files.get(k,[])
                    p.append(file)
                    if j == ws-1:
                        j = -1
                    break
        if rank in rank2files:
            shards = rank2files[rank]

            random.shuffle(shards)
            shards = [(shard, i%num_devices, args) for i, shard in enumerate(shards)]
            with multiprocessing.Pool(10 if num_devices <= 1 else num_devices) as pool:    
                for shard in pool.imap_unordered(reduce, shards):
                    print ("done with "+ shard)
        wait_for_other_ranks(args.output_dir)

