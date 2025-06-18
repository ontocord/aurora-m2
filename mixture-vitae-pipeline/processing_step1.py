import mmh3
import PIL
import langid
from PIL import Image
import os, io
import multiprocessing
import glob, random
import subprocess
from pathlib import Path
from tqdm import tqdm
import json
import tarfile
#from utils import classify_and_quality_score
import multiprocessing, functools, json, glob
from flagged_words import *
from urllib.parse import urlparse
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

        
curated_stopwords = {"patent", "title", "page", "united", "states", "edgar", "court", "federal", "district", "register", "congress", "congressional", "record", "volume", "number", "judgement", "exhibit", "section", "science", "news", "u.s.", "case", "law"}
def read_file(file):    
    print (file)
    if True:#try:
        if "common-pile" in file:
            for l in open(file, "rb"):
                try:
                    data = json.loads(l)
                except:
                    continue
                text = data['text']
                del data['text']
                data['source'] = file
                if 'meeadata' in data:
                    data['metadata'] = data['meeadata']
                if 'metadata' in data and type(data['metadata']) is str:
                    data['metadata'] = json.loads(data['metadata'])
                if 'meta' in data and type(data['meta']) is str:
                    data['meta'] = json.loads(data['meta'])                
                
                metadata = data.get("metadata", data.get('meta', {}))
                if 'meta' in data and 'language' in data['meta']:
                    data['lang'] = data['meta']['language']
                elif 'metadata' in data and 'language' in data['metadata']:
                    data['lang'] = data['metadata']['language']
                elif 'meta' in data and 'lang' in data['meta']:
                    data['lang'] = data['meta']['lang']
                elif 'metadata' in data and 'lang' in data['metadata']:
                    data['lang'] = data['metadata']['lang']
                elif 'lang' not in data:
                    data['lang'] = "en"
                    
                if 'meta' in data and 'license' in data['meta']:
                    data['license_header_footer'] = data['meta']['license']
                elif 'metadata' in data and 'license' in data['metadata']:
                    data['license_header_footer'] = data['metadata']['license']
                elif 'meta' in data and 'oa_license' in data['meta']:
                    data['license_header_footer'] = data['meta']['oa_license']
                elif 'metadata' in data and 'oa_license' in data['metadata']:
                    data['license_header_footer'] = data['metadata']['oa_license']
                idx = ""
                if 'url' in data:
                    idx = data['url']
                elif 'metadata' in data and 'url' in data['metadata']:
                    idx = data['metadata']['url']
                elif 'meta' in data and 'url' in data['meta']:
                    idx = data['meta']['url']
                elif 'metadata' in data and 'text_file_url' in data['metadata']:
                    idx = data['metadata']['text_file_url']
                elif 'meta' in data and 'text_file_url' in data['meta']:
                    idx = data['meta']['text_file_url']
                elif 'meta' in data and 'ia_url' in data['meta']:
                    idx = data['meta']['ia_url']
                elif 'metadata' in data and 'ia_url' in data['metadata']:
                    idx = data['metadata']['oa_url']
                elif 'meta' in data and 'oa_url' in data['meta']:
                    idx = data['meta']['oa_url']
                elif 'metadata' in data and 'oa_url' in data['metadata']:
                    idx = data['metadata']['oa_url']
                elif 'wikipedia' in file and 'metadata' in data and 'title' in data['metadata']:
                    idx = "https://"+data['metadata']['title'].replace(" ", "_")+".wikipedia.org/wiki/"+data['lang']
                elif 'wikipedia' in file and 'meta' in data and 'title' in data['meta']:
                    idx = "https://"+data['meta']['title'].replace(" ", "_")+".wikipedia.org/wiki/"+data['lang']
                elif 'stackex' in file and 'metadata' in data and 'source' in data['metadata']:
                    idx = "https://"+data['metadata']['source']+".com"                    
                elif 'stackex' in file and 'meta' in data and 'source' in data['meta']:
                    idx = "https://"+data['meta']['source'] +".com"
                if not idx and 'meta' in data:
                    for key in data['meta']:
                        if "url" in key:
                            idx = data['meta'][key]
                            break
                elif not idx and 'metadata' in data:
                    for key in data['metadata']:
                        if "url" in key:
                            idx = data['metadata'][key]
                            break
                if not idx:
                    lang = data["lang"]
                    stopwords =  all_stopwords.get(lang, {})
                    en_stopwords =  all_stopwords.get("en", {})                
                    is_cjk = lang_is_cjk(lang)
                    if not is_cjk:
                        tail = text.lower().split()
                        if len(tail) > 20:
                            tail = tail[-20:]
                        tail = [a[:4] if len(a) > 4 else a for a in [a.strip(strip_chars) for a in tail if len(a) > 2] if len(a) > 2 and a not in stopwords and a not in en_stopwords and a not in curated_stopwords]
                    else:
                        tail = text.lower().split()
                        if len(tail) > 20:
                            tail = tail[-20:]
                        tail = "".join(tail)
                        tail = [a for a in [a.strip(strip_chars) for a in tail] if a not in stopwords]
                    tail = "".join(tail)
                    if len(tail) <=1: 
                        tail += str(random.randint(0,9))+str(random.randint(0,9))
                    idx= tail+"_"+data['source'].split("/")[-1].replace("www.", "")
                    for c in unix_chars:
                        idx = idx.replace(c, "")
                    # this is a hack to make the non url data at the end in a sorted list. this has the advantage of filtering out data that have no urls but might be duplicates.
                    idx = "z://"+idx
                # {'authors':
                idx = idx.replace("\\", "")
                if "://" not in idx:
                    idx = "https://"+idx
                data['idx'] = idx
                text = remove_citations(text)                            
                if 'loc.gov' in idx or "fandom" in idx or "wikia" in idx:
                    text = cleanup_raw_text(text, data['lang'])
                data['license_header_footer'] = data.get('license_header_footer', '')
                data['is_govt'] = False
                if "pubmed" in file or "uk_han" in file or "regula" in file or "usgpo" in file or "uspto" in file or "library_of_congress" in file or "caselaw" in file:
                #if "http" not in data['idx']:
                    data['is_govt'] = True
                data = {'idx': idx, 'text': text, 'media_list': [], 'metadata': [data]}
                if not data['metadata'][0]['is_govt'] and not data['metadata'][0]['license_header_footer']: 
                    data['metadata'][0]['is_govt'] = True
                yield data
        elif "curated" in file:
            #TODO: 
            for l in open(file, "rb"):
                try:
                    data = json.loads(l)
                except:
                    continue
                text = data['text']
                del data['text']
                data['source'] = file
                if 'metadata' in data and type(data['metadata']) is str:
                    data['metadata'] = json.loads(data['metadata'])
                if 'meta' in data and type(data['meta']) is str:
                    data['meta'] = json.loads(data['meta'])                
                
                metadata = data.get("metadata", data.get('meta', {}))
                # Toxicity info
                toxicity_info = metadata.get("toxicity", [])
                formatted_toxicity = "\n".join(
                    f"* {label}: {score:.3f}" for label, score in toxicity_info if score >= 0.2
                ) 
                if formatted_toxicity:
                    if random.randint(0,1):
                        text += "<|endofsection|>"+formatted_toxicity
                    else:
                        text = formatted_toxicity+"<|endofsection|>"+text                
                if "wikibooks" in file:
                    data["lang"] = file.split("/")[-1].replace(".jsonl", "").strip()
                elif 'meta' in data and 'language' in data['meta']:
                    data['lang'] = data['meta']['language']
                elif 'metadata' in data and 'language' in data['metadata']:
                    data['lang'] = data['metadata']['language']
                elif 'meta' in data and 'lang' in data['meta']:
                    data['lang'] = data['meta']['lang']
                elif 'metadata' in data and 'lang' in data['metadata']:
                    data['lang'] = data['metadata']['lang']
                elif 'lang' not in data:
                    data['lang'] = "en"
                if 'url' in data:
                    idx = data['url']
                elif 'wikipedia' in file and 'metadata' in data and 'title' in data['metadata']:
                    idx = "https://"+data['metadata']['title'].replace(" ", "_")+".wikipedia.org/wiki/"+data['lang']
                elif 'wikipedia' in file and 'meta' in data and 'title' in data['meta']:
                    idx = "https://"+data['meta']['title'].replace(" ", "_")+".wikipedia.org/wiki/"+data['lang']
                elif 'stackex' in file and 'metadata' in data and 'source' in data['metadata']:
                    idx = "https://"+data['metadata']['source']+".com"                    
                elif 'stackex' in file and 'meta' in data and 'source' in data['meta']:
                    idx = "https://"+data['meta']['source'] +".com"                   
                elif 'metadata' in data and 'url' in data['metadata']:
                    idx = data['metadata']['url']
                elif 'meta' in data and 'url' in data['meta']:
                    idx = data['meta']['url']
                else:
                    lang = data["lang"]
                    stopwords =  all_stopwords.get(lang, {})
                    en_stopwords =  all_stopwords.get("en", {})                
                    is_cjk = lang_is_cjk(lang)
                    if not is_cjk:
                        tail = text.lower().split()
                        if len(tail) > 20:
                            tail = tail[-20:]
                        tail = [a[:4] if len(a) > 4 else a for a in [a.strip(strip_chars) for a in tail if len(a) > 2] if len(a) > 2 and a not in stopwords and a not in en_stopwords and a not in curated_stopwords]
                    else:
                        tail = text.lower().split()
                        if len(tail) > 20:
                            tail = tail[-20:]
                        tail = "".join(tail)
                        tail = [a for a in [a.strip(strip_chars) for a in tail] if a not in stopwords]
                    tail = "".join(tail)
                    if len(tail) <=1: 
                        tail += str(random.randint(0,9))+str(random.randint(0,9))
                    idx= tail+"_"+data['source'].split("/")[-1].replace("www.", "")
                    for c in unix_chars:
                        idx = idx.replace(c, "")
                    # this is a hack to make the non url data at the end in a sorted list. this has the advantage of filtering out data that have no urls but might be duplicates.
                    idx = "z://"+idx
                    
                idx = idx.replace("\\", "")                    
                data['idx'] = idx
                text = remove_citations(text)                            
                if 'loc.gov' in idx or "fandom" in idx or "wikia" in idx:
                    text = cleanup_raw_text(text, data['lang'])
                data['license_header_footer'] = ""
                data['is_govt'] = False
                # THIS IS NOT RIGHT. TODO: fix per type
                if "http" not in data['idx']:
                    data['is_govt'] = True
                data = {'idx': idx, 'text': text, 'media_list': [], 'metadata': [data]}
                #print (data)
                yield data
        elif "kl3m" in file:
            for l in open(file, "rb"):
                try:
                    data = json.loads(l)
                except:
                    continue
                text = data['text'].strip()
                head = text[:50].strip().lower().split()
                if "is a test" in head or "begin 6" in head or "rdf xmlns" in head or "ty  -" in head:
                    continue
                if len(text) < 200: continue
                if text[0] == "{":
                    continue
                if text[-1] == "}":
                    continue
                if text.count("</") > 2:
                    continue
                if text.startswith("begin "):
                    if "644" in text[:20] or  "775" in text[:20] or "." in text[:20]:
                        continue
                    #print ((text,))
                    lang = "en"
                    data['lang'] = lang                    
                else:
                    lang, _ =  langid.classify(text.replace("\n", " ")[:min(len(text), 200)])
                    data['lang'] = lang

                score1 = get_special_char_score(text, lang)
                if score1 > 0.15:
                    continue
                score2 = get_stopword_score(text,lang)
                if score2  < 0.05:
                    continue

                text = cleanup_raw_text(text, lang)
                if len(text) < 1000: continue                
                stopwords =  all_stopwords.get(lang, {})
                en_stopwords =  all_stopwords.get("en", {})                
                is_cjk = lang_is_cjk(lang)
                if not is_cjk:
                    tail = text.lower().split()
                    if len(tail) > 20:
                        tail = tail[-20:]
                    tail = [a[:4] if len(a) > 4 else a for a in [a.strip(strip_chars) for a in tail if len(a) > 2] if len(a) > 2 and a not in stopwords and a not in en_stopwords and a not in curated_stopwords]
                else:
                    tail = text.lower().split()
                    if len(tail) > 20:
                        tail = tail[-20:]
                    tail = "".join(tail)
                    tail = [a for a in [a.strip(strip_chars) for a in tail] if a not in stopwords]
                tail = "".join(tail)
                if len(tail) <=1: 
                    tail += str(random.randint(0,9))+str(random.randint(0,9))
                idx= tail+"_"+data['source'].split("/")[-1].replace("www.", "")
                for c in unix_chars:
                    idx = idx.replace(c, "")
                # this is a hack to make the non url data at the end in a sorted list. this has the advantage of filtering out data that have no urls but might be duplicates.
                idx = "z://"+idx
                del data['text']
                data['license_header_footer'] = ""
                data['is_govt'] = True
                data['source'] = file
                data['lang'] = lang
                data['idx'] = idx
                data = {'idx': idx, 'text': text, 'media_list': [], 'metadata': [data]}
                yield data
        elif "dcad" in file:
            for l in open(file, "rb"):
                try:
                    data = json.loads(l)
                except:
                    continue
                idx = data['url'].replace("\\", "")
                text = data['text']
                del data['url']
                del data['text']
                data['source'] = file
                data['lang'] = dcad2lang.get(data['language']+"_"+data["language_script"], 'en')
                data['idx'] = idx
                if 'loc.gov' in idx:
                    text = cleanup_raw_text(text, 'en')
                data = {'idx': idx, 'text': text, 'media_list': [], 'metadata': [data]}
                yield data
        elif "FineFine" in file:
            for l in open(file, "rb"):
                try:
                    data = json.loads(l)
                except:
                    continue
                idx = data['url']
                text = data['text']
                del data['url']
                del data['text']
                data['source'] = file
                data['lang'] = 'en'
                data['idx'] = idx
                if 'loc.gov' in idx:
                    text = cleanup_raw_text(text, 'en')
                data = {'idx': idx, 'text': text, 'media_list': [], 'metadata': [data]}
                yield data
        elif "nemo" in file:
            for l in open(file, "rb"):
                try:
                    data = json.loads(l)
                except:
                    continue
                idx = data['url']
                text = data['text']
                del data['url']
                del data['text']
                data['idx'] = idx                
                data['source'] = file
                data['lang'] = 'en'
                yield  {'idx': idx, 'text': text, 'media_list': [], 'metadata': [data]}

        elif "dclm" in file:
            for l in open(file, "rb"):
                try:
                    data = json.loads(l)
                except:
                    continue
                idx = data['url']
                text = data['text']
                del data['url']
                del data['text']
                data['source'] = file
                data['lang'] = 'en'
                data['idx'] = idx
                if 'loc.gov' in idx:
                    text = cleanup_raw_text(text, 'en')
                data = {'idx': idx, 'text': text, 'media_list': [], 'metadata': [data]}
                yield data

        elif "MAGA" in file:
            df = pq.read_table(file).to_pylist()
            for data in df:
                text = data['content_split']
                del data['content_split']
                try:
                    idx = json.loads(data['meta']['meta_extra'])
                except:
                    continue
                idx = idx['url']
                data['source'] = file
                data['lang'] = 'en'
                data['idx'] = idx
                if data['meta']['raw_text'].strip() not in text:
                    text =  cleanup_raw_text(data['meta']['raw_text'], 'en') + "<|endoftext|>" + text
                    del data['meta']['raw_text']                                                
                    data =  {'idx': idx, 'text': text, 'media_list': [], 'metadata': [data, copy.copy(data)]}
                    yield data
                    continue
                del data['meta']['raw_text']                
                yield {'idx': idx, 'text': text, 'media_list': [], 'metadata': [data]}                
        elif "finemath" in file:
            df = pq.read_table(file).to_pylist()
            for data in df:
                idx = data['url']
                text = data['text']
                del data['text']
                data['source'] = file
                data['lang'] = 'en'
                data['idx'] = idx
                if 'loc.gov' in idx:
                    text = cleanup_raw_text(text, 'en')                
                yield {'idx': idx, 'text': text, 'media_list': [], 'metadata': [data]}                

        elif "txt360" in file:
            for l in open(file, "rb"):
                try:
                    data = json.loads(l)
                except:
                    continue
                idx = data['meta']['url']
                text = data['text']
                del data['text']
                data['source'] = file
                data['lang'] = 'en'
                data['idx'] = idx
                if 'loc.gov' in idx:
                    text = cleanup_raw_text(text, 'en')
                data = {'idx': idx, 'text': text, 'media_list': [], 'metadata': [data]}                                
                yield data
        elif "CulturaY" in file:
            for l in open(file, "rb"):
                try:
                    data = json.loads(l)
                except:
                    continue
                idx = data['url']
                text = data['text']
                del data['text']
                lang = data['document_lang']
                score1 = get_special_char_score(text, lang)
                if score1 > 0.15:
                    continue
                score2 = get_stopword_score(text,lang)
                if score2  < 0.05:
                    continue
                data['source'] = file
                data['lang'] = lang
                data['idx'] = idx
                if 'loc.gov' in idx:
                    text = cleanup_raw_text(text, lang)
                data = {'idx': idx, 'text': text, 'media_list': [], 'metadata': [data]}
                yield data
        elif "mint" in file:
            base_src = "/leonardo_work/EUHPC_E03_068/datasets/working/mint-pdf/"
            base_dest = "/leonardo_work/EUHPC_E03_068/datasets/working/mint-pdf-permissive/" # where we put the tiff files
            dest = base_dest+(file.replace(base_src, ""))    
            dest_dir = "/".join(dest.split("/")[:-1])
            os.makedirs(dest_dir, exist_ok=True)
            if not file.endswith(".tar"): return
            dest_file = dest_dir+"/"+file.split("/")[-1].replace(".tar", ".jsonl")
            os.makedirs(dest_file.replace(".jsonl", ""), exist_ok=True)
            # Open the tar file in read mode
            try:
                tar = tarfile.open(file, "r")
            except:
                print ("couldn't load tarfile "+ file)
                return
            data_hash = {}
            # Iterate over the members in the tar file
            for member in tar:
                # Check if the member is a file
                if member.isfile():
                    name = member.name
                    if not name.endswith(".json"): continue
                    name_json = name.replace(".json", "")
                    file_obj = tar.extractfile(member)
                    # Read the contents of the file
                    file_content = file_obj.read()
                    data = json.loads(file_content)
                    idx = data['url']
                    data['text'] = " ".join([(s if s else "\n<image>\n") for s in data['texts']])
                    is_match, is_oss = is_idx_match(data)
                    if  is_match:
                        image_file = dest_file.replace(".jsonl", "")+"/"+name_json+".tiff"
                        relative_image_file = dest_file.split("/")[-1].replace(".jsonl", "")+"/"+name_json+".tiff"
                        text = data['text']
                        del data['texts']
                        #del data['text']                    
                        del data['url']
                        sents = text.split(". ")
                        sents2 = []
                        for s2 in sents:
                            sent2 = []
                            for w in s2.split(" "):
                                if w and len(w) <= 2 and sent2 and len(sent2[-1]) <= 2 and w[0] in "qwertyuiopasdfghjklzxcvbnm": continue
                                sent2.append(w)
                            sents2.append(" ".join(sent2))
                        text = ". ".join(sents2)
                        data['source'] = dest.replace(base_dest, "")
                        data['lang'] = 'en'
                        data['idx'] = idx
                        text = cleanup_raw_text(text, 'en')                
                        new_data = {'idx': idx, 'text': text, 'media_list': [{'image_0': relative_image_file}],'metadata': [data]}
                        data_hash[name_json] = new_data

            # Iterate over the members in the tar file
            for member in tar:
                # Check if the member is a file
                if member.isfile():
                    name = member.name
                    if ".tiff" in name:
                        name_image  = name.replace(".tiff", "")
                        if name_image not in data_hash:
                            continue
                        # Extract the file
                        file_obj = tar.extractfile(member)
                        # Read the contents of the file
                        file_content = file_obj.read()
                        # Create a BytesIO object from the buffer
                        image_file = io.BytesIO(file_content)
                        # Open the image
                        image = Image.open(image_file)
                        image_file = dest_file.replace(".jsonl", "")+"/"+name
                        image.save(image_file)
                        data = data_hash[name_image]
                        yield data
        else:
            print ("unknown file", file)
    #except:
    #    print ("error in file", file)        
    #    return


def process (arg):
    global model, tokenizer, device, args, num_devices
    global common_pile_sites, white_list_sites
    file, device_no, args = arg
    if common_pile_sites is None:
        common_pile_sites = set(json.load(open("common_pile_urls.json")))
        white_list_sites = set(json.load(open("white_list_urls.json")))
    if args.add_related:
        init_cosmo()
        init_seed2()        
    model, tokenizer = init_model(device_no, args)
    ret = []
    upsample_batch = []
    if os.path.exists(file.replace(args.input_dir, args.output_dir+"/done/")): return file, ret
    for data in read_file(file):
        for metadata in data['metadata']:
            lang = metadata['lang']
            if type(lang) is list:
                metadata['langs'] = lang
                lang = metadata['lang'] = lang[0]
        
        lang = data['metadata'][0]['lang']
        if 'lang' not in data:
            data['lang'] = lang
        if "kl3m" in data['metadata'][0]['source'] or "curated" in data['metadata'][0]['source'] or "common-pile" in data['metadata'][0]['source']:
            is_match = True            
            is_oss = False
        else:
            is_match, is_oss = is_idx_match(data)
        if not is_match: continue
        if args.add_related and ("kl3m" not in data['metadata'][0]['source'] and "mint" not in data['metadata'][0]['source']):        
            data = add_related(data)
        if lang != 'hi':
            text = data['text']                
            if "|" in text[:100]:
                text  = text[:100].split("|")[-1]+text[100:]
                if "|" in text[-100:]:
                    text  = text[:-100]  + text[-100:].split("|")[0]
                data['text'] = text
        orig = data
        data = dedup(data)
        if not data:
            continue
        orig= None
        #print(data)
        if True:            
            text_arr = []
            meta_arr = []
            for text, metadata  in zip(data['text'].split("<|endoftext|>"), data['metadata']):
                head = text[:100].lower()
                tail = text[-100:].lower()
                    
                if "cc-by " in head or "cc-by " in tail or "cc-0 " in head or "cc-0 " in tail or "cc-by-" in head or "cc-by-" in tail or \
                   "creative common" in head or "creative common" in tail or "public domain" in head or "public domain" in tail:
                    metadata['license_header_footer'] = head + " ... " + tail
                    metadata['is_govt'] = False
                elif 'is_govt' in metadata:
                    pass
                elif 'kl3m' in metadata['source']:
                    metadata['license_header_footer'] = ""
                    metadata['is_govt'] = True
                else:
                    metadata['license_header_footer'] = ""
                    idx = metadata['idx']
                    if ".mil/" in idx or ".vlada.mk" in idx or ".vlada.cz" in idx or ".kormany.hu" in idx or  "regeringen." in idx or ".rijksoverheid.\
                    nl" in idx or ".government.nl" in idx or ".regeringen.se" in idx or  ".regeringen.dk" in idx or  ".regeringen.no" in idx or ".bund.de" in idx or ".bundesregierung.de" in idx or  ".government.ru" in idx or ".gc.ca" in idx or \
                    ".admin.ch" in idx or  'www.gob.cl/' in idx or  'www.gob.ec/' in idx or  'guatemala.gob.gt/' in idx or  'presidencia.gob.hn/' in idx or  'www.gob.mx/' in idx or  'presidencia.gob.pa/' in idx or  'www.gob.pe/' in idx or  'gob.es/' in idx or  'argentina.gob.ar/' in idx or \
                    "tanzania.go.tz/" in idx or ".indonesia.go.id/" in idx or ".go.kr/" in idx or ".go.jp/" in idx or  "thailand.go.th/" in idx or ".europa.eu/" in idx or ".un/" in idx or ".int/" in idx or ".govt." in idx or "www.gub.uy" in idx or ".gov/" in idx or '.gov.' in idx or '.gouv.' in idx:
                        metadata['is_govt'] = True
                    else:
                        metadata['is_govt'] = False

                # do some misc text cleanup/fixing
                text = " ".join(["[URL]" if ("http:" in w or "https:" in w or "www." in w) else ("[BASE64_CODE]" if "base64" in w and len(w) > 10 else w) for w in text.split(" ")])
                if "wiki" in data['idx']:
                    text = re.sub(r'(\[\d+\])+', '', text)
                
                text = text.replace(".[", ". [").replace("…", "...").\
                    replace("Creative Commons Attribution-ShareAlike 3.0 Unported License (CC BY-SA)", "a permissive license").\
                    replace("Creative Commons Attribution 4.0 International", "a permissive license").\
                    replace("…", "...").replace("Creative Commons Attribution", "a permissive license").\
                    replace("Creative Commons Attribution-ShareAlike", "a permissive license").\
                    replace("Creative Commons", "a permissive license").\
                    replace(" the a ", " a ").replace(" a a ", " a ").replace("license License", "license").replace("license license", "license").replace("[edit]", "")
                text = remove_citations(text)                            
                text_arr.append(text)
                meta_arr.append(metadata)
            if not text_arr: continue
            text  = "<|endoftext|>".join(text_arr)
            text = html.unescape(text)            
            if len(text) < 100: continue
            data['text'] = text
            data['metadata'] = meta_arr
            ret.append(data)
            # upsample_batch things we know we won't strip b/c of permissions
            if any(meta for meta in data['metadata'] if meta['idx'] in white_list_sites or meta['idx'] in common_pile_sites or meta['is_govt'] or 'curated' in meta['source'] or 'kl3m' in meta['source'] or 'common-pile' in meta['source']):
                upsample_batch.append(data)
                if len(upsample_batch) > 400:
                    generate_upsample(upsample_batch)
                    upsample_batch = []
    generate_upsample(upsample_batch)
    return (file, ret)
    
        

if __name__ == "__main__":
    args = parse_args()
    args.output_dir = args.target_dir+"1/"
    args.input_dir =  "/leonardo_work/EUHPC_E03_068/datasets/working/"
    subset= args.subset
    if subset:
        args.output_dir = args.output_dir.rstrip("/")+"_"+subset+"/"
    args.all_files =[]
    print (args)
    os.system(f"mkdir -p {args.output_dir}")
    os.system("rm -rf "+args.output_dir+"/"+str(get_rank())+".rank_done")
    if  True: #not os.path.exists(args.output_dir+"/"+str(get_rank())+".rank_done"):
        # Iterate through all files in the directory and subdirectories
        root_dir = '/leonardo_work/EUHPC_E03_068/datasets/working/mixture_vitae_curated/'
        args.all_files.extend(list(set(list(glob.glob(root_dir + '*.jsonl', recursive=True)) +  list(glob.glob(root_dir + '*/*.jsonl', recursive=True)) +  list(glob.glob(root_dir + '*/*/*.jsonl', recursive=True)))))

        root_dir = '/leonardo_work/EUHPC_E03_068/datasets/working/common-pile/'
        args.all_files.extend(list(set(list(glob.glob(root_dir + '*.jsonl', recursive=True)) +  list(glob.glob(root_dir + '*/*.jsonl', recursive=True)) +  list(glob.glob(root_dir + '*/*/*.jsonl', recursive=True)))))

        root_dir = '/leonardo_work/EUHPC_E03_068/datasets/working/dcad/data/'
        args.all_files.extend(list(set(list(glob.glob(root_dir + '*.jsonl', recursive=True)) +  list(glob.glob(root_dir + '*/*.jsonl', recursive=True)) +  list(glob.glob(root_dir + '*/*/*.jsonl', recursive=True)))))
        root_dir = '/leonardo_work/EUHPC_E03_068/datasets/working/FineFineWeb/'
        args.all_files.extend(list(set(list(glob.glob(root_dir + '*.jsonl', recursive=True)) +  list(glob.glob(root_dir + '*/*.jsonl', recursive=True)) +  list(glob.glob(root_dir + '*/*/*.jsonl', recursive=True)))))
        root_dir = "/leonardo_work/EUHPC_E03_068/datasets/working/mint-pdf/"
        args.all_files.extend(list(set(list(glob.glob(root_dir + '*.tar', recursive=True)) +  list(glob.glob(root_dir + '*/*.tar', recursive=True)) +  list(glob.glob(root_dir + '*/*/*.tar', recursive=True)))))

        root_dir = '/leonardo_work/EUHPC_E03_068/datasets/working/MAGACorpus/'    
        args.all_files.extend(list(set(list(glob.glob(root_dir + '*.parquet', recursive=True)) +  list(glob.glob(root_dir + '*/*.parquet', recursive=True)) +  list(glob.glob(root_dir + '*/*/*.parquet', recursive=True)))))

        root_dir = '/leonardo_work/EUHPC_E03_068/datasets/working/finemath/finemath-3plus/'
        args.all_files.extend(list(set(list(glob.glob(root_dir + '*.parquet', recursive=True)) +  list(glob.glob(root_dir + '*/*.parquet', recursive=True)) +  list(glob.glob(root_dir + '*/*/*.parquet', recursive=True)))))
        root_dir = '/leonardo_work/EUHPC_E03_068/datasets/working/finemath/infiwebmath-3plus/'
        args.all_files.extend(list(set(list(glob.glob(root_dir + '*.parquet', recursive=True)) +  list(glob.glob(root_dir + '*/*.parquet', recursive=True)) +  list(glob.glob(root_dir + '*/*/*.parquet', recursive=True)))))
        root_dir = '/leonardo_work/EUHPC_E03_068/datasets/working/dclm-edu/'
        args.all_files.extend(list(set(list(glob.glob(root_dir + '*.jsonl', recursive=True)) +  list(glob.glob(root_dir + '*/*.jsonl', recursive=True)) +  list(glob.glob(root_dir + '*/*/*.jsonl', recursive=True)))))

        root_dir = '/leonardo_work/EUHPC_E03_068/datasets/working/kl3m/jsonl/'
        args.all_files.extend(list(set(list(glob.glob(root_dir + '*.jsonl', recursive=True)) +  list(glob.glob(root_dir + '*/*.jsonl', recursive=True)) +  list(glob.glob(root_dir + '*/*/*.jsonl', recursive=True)))))
        
        root_dir = '/leonardo_work/EUHPC_E03_068/datasets/working/nemo_low/'
        args.all_files.extend(list(set(list(glob.glob(root_dir + '*.jsonl', recursive=True)) +  list(glob.glob(root_dir + '*/*.jsonl', recursive=True)) +  list(glob.glob(root_dir + '*/*/*.jsonl', recursive=True)))))

        root_dir = '/leonardo_work/EUHPC_E03_068/datasets/working/nemo_high/'
        args.all_files.extend(list(set(list(glob.glob(root_dir + '*.jsonl', recursive=True)) +  list(glob.glob(root_dir + '*/*.jsonl', recursive=True)) +  list(glob.glob(root_dir + '*/*/*.jsonl', recursive=True)))))
        root_dir = '/leonardo_work/EUHPC_E03_068/datasets/working/txt360/data/common-crawl/'
        args.all_files.extend(list(set(list(glob.glob(root_dir + '*.jsonl', recursive=True)) +  list(glob.glob(root_dir + '*/*.jsonl', recursive=True)) +  list(glob.glob(root_dir + '*/*/*.jsonl', recursive=True)))))
        root_dir = "/leonardo_work/EUHPC_E03_068/datasets/working/CulturaY/"
        args.all_files.extend(list(set(list(glob.glob(root_dir + '*.jsonl', recursive=True)) +  list(glob.glob(root_dir + '*/*.jsonl', recursive=True)) +  list(glob.glob(root_dir + '*/*/*.jsonl', recursive=True)))))
    args.all_files = [file for file in args.all_files if not os.path.exists(file.replace(args.input_dir, args.output_dir+"/done/"))]
    subset = subset.split(",")
    if subset:
        all_files2 = []
        for s in subset:
            all_files2.extend([f for f in args.all_files if args.input_dir+s in f])
        args.all_files = list(set(all_files2))
    if args.sample:
        random.shuffle(args.all_files)
        args.all_files  = args.all_files[:args.sample]
    args.all_files.sort()
    ws = get_world_size()
    rank = args.rank = get_rank()
    print ("starting rank", rank)
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
    os.system(f"mkdir -p {args.output_dir}/rank_{rank}")
    if rank in rank2files:
        files = rank2files[rank]
        random.shuffle(files)
        files = [(file, i%num_devices, args) for i, file in enumerate(files)]
        idx2outf = {}
        #if args.add_related:
        #    init_cosmo()
        #    init_seed2()        
        #with multiprocessing.Pool(4 if num_devices <= 1 else num_devices) as pool:    
        #    for file, ret in pool.imap_unordered(process, files):
        for file in files:
                file, ret = process(file)
                for data in ret:
                    if len(idx2outf) > 100:
                        for key in list(idx2outf.keys()):
                            if random.randint(0,1):
                                idx2outf[key].close()
                                del idx2outf[key]
                    #data = dedup(data)
                    #if not data:
                    #    continue
                    fix_idx(data)
                    idx = data['idx']
                    idx = idx.split("://",1)[-1]
                    for c in unix_chars:
                        idx = idx.replace(c, "")
                    if len(idx) > 3:
                        idx= idx[:3]
                    if len(idx) <= 1:
                        idx = idx+str(random.randint(0,100))
                    if len(idx) > 3:
                        idx= idx[:3]
                    hash_32 = mmh3.hash(idx, seed=42)
                    idx = str(hash_32).strip("-")[:3]
                    if idx == "207": # this is a hack to fix a previous bug. 
                        idx = data['idx']
                        idx = idx.split("://",1)[-1]                    
                        for c in unix_chars:
                            idx = idx.replace(c, "")
                        if len(idx) > 4:
                            idx= idx[:4]
                        if len(idx) <= 1:
                            idx = idx+str(random.randint(0,100))
                        if len(idx) > 4:
                            idx= idx[:4]
                        hash_32 = mmh3.hash(idx, seed=42)
                        idx = str(hash_32).strip("-")[:3]
                    if idx not in idx2outf:
                        idx2outf[idx] = open(f"{args.output_dir}/rank_{rank}/"+idx+".jsonl", "a+")
                    outf = idx2outf[idx]
                    outf.write(json.dumps(data)+"\n")
                    if random.randint(0,1000)==0:
                        outf.close()
                        idx2outf[idx] = open(f"{args.output_dir}/rank_{rank}/"+idx+".jsonl", "a+")
                os.system("mkdir -p "+ "/".join(file.replace(args.input_dir, args.output_dir+"/done/").split("/")[:-1]))
                with open(file.replace(args.input_dir, args.output_dir+"/done/"), "w") as outf: pass
            
    wait_for_other_ranks(args.output_dir)
