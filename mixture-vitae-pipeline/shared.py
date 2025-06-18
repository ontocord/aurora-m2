## Copyright 2024, Ontocord, LLC. All rights reserved
"""
Copyright, 2021-2022 Ontocord, LLC, and other authors of Muliwai, All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
try:
    sys.path.append(os.path.abspath(os.path.dirname(__file__)))         
except:
    pass
import html
import ctranslate2
import transformers
import  itertools
import os
import glob, random
import subprocess
from pathlib import Path
from tqdm import tqdm
import json
import json, glob, os
import math
import functools, json, glob
from flagged_words import *
from collections import Counter
import string
import argparse
from collections import defaultdict
import sys
import pyarrow.parquet as pq
import time, random
import json, os, glob, random
from torch import multiprocessing
from torch.multiprocessing import SimpleQueue
from torch import threading
import os
import json, os, glob
from tqdm import tqdm
from typing import List
import glob, json
import re
from huggingface_hub import hf_hub_download
import fasttext
from multiprocessing import Pool
from names import *

import multiprocessing, functools, json, glob, langid

import glob, json, langid, random
import math
import sys, os, string


import wget
import spacy    
from nltk.corpus import wordnet as wn
import stdnum
from date_detector import Parser
import commonregex, re
from commonregex import CommonRegex
from faker import Faker
import spacy
#from matplotlib import colors
import fasttext
#    from frcnn.visualizing_image import SingleImageViz
#    from frcnn.processing_image import Preprocess as FRCNNPreprocess
#    from frcnn.modeling_frcnn import GeneralizedRCNN
#    from frcnn.utils import Config as FRCNNConfig
#    from frcnn.utils import decode_image as frcnn_decode_image
#    import cv2
from nltk.corpus import cmudict

import re
import sys, os
import re 
import random
from string import punctuation, ascii_lowercase
import gzip
import tqdm
from time import sleep
from typing import Dict, List
import os
from pathlib import Path
from tqdm import tqdm
import copy
import json
import base64
import uuid
import hashlib
import random
from io import BytesIO
import numpy as np
from numpy import asarray
from collections import deque, Counter
import numpy as np
import torch
#import torchvision
#from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import random
import itertools
import torch
import PIL
from PIL import Image
from transformers import pipeline
#from datasets import load_dataset
from torch.nn.functional import cosine_similarity
from transformers import CLIPProcessor, CLIPModel, AutoModel, AutoTokenizer, AutoModelWithLMHead
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
import numpy as np

from collections import OrderedDict
from string import punctuation, ascii_lowercase

### GLOBAL VARIABLES

spacy_multi = None
spacy_nlp = None
num_devices = torch.cuda.device_count()
device  = "cuda:0"
if num_devices == 0:
    num_devices = 1
    device = "cpu"

args = None
  
edu_model =  \
  red_pajama_model = \
  pile_class_model = \
  registry_model = \
  ffw_model = \
  spam_model= \
  qual_predict_model= \
  toxic_classifier = None

cache_dir = "/leonardo_work/EUHPC_E03_068/.cache"

llm_slop_phrases = ["the heart of", "a mix of", "in the heart", "the scent of", "a sense of", "a hint of", "for a moment", "filled with the", "the edge of", "t help but", "In an", "In a", "In the heart", "In the", "Certainly!", 'testament to', 'barely above a whisper', 'barely a whisper', 'orchestra of', 'dance of', 'maybe, just maybe', 'maybe that was enough', 'perhaps, just perhaps', 'was only just beginning', ', once a ', 'world of', 'shivers down', 'shivers up', 'shiver down', 'shiver up', ', rasped',', rasping', 'moth to a flame', 'eyes glinted', 'humble abode', 'cold and calculating', 'eyes never leaving', 'body and soul', 'a dance of', 'chuckles darkly', 'maybe, that was enough', 'they would face it together', 'a reminder', 'that was enough', 'for now, that was enough', "for now, that's enough", 'with a mixture of', 'air was filled with anticipation', 'bore silent witness to', 'eyes sparkling with mischief', 'practiced ease', 'ready for the challenges', 'only just getting started', 'once upon a time', 'nestled deep within', 'ethereal beauty', 'life would never be the same', "it's important to remember", 'for what seemed like an eternity', 'little did he know', 'ball is in your court', 'game is on', 'choice is yours', 'feels like an electric shock', 'threatens to consume', 'dive into', 'not only', "today's digital age", 'game changer', 'designed to enhance', 'it is advisable', 'when it comes to', 'in the realm of', 'unlock the secrets', 'unveil the secrets', 'and robust', "it's important to note", 'in summary', 'remember that', 'take a dive into', 'in the world of', 'to consider', 'there are a few considerations', "it's essential to", 'as a professional', 'you may want to', 'on the other hand', 'as previously mentioned', "it's worth noting that", 'to summarize', 'to put it simply', "in today's digital era", 'sights unseen', 'sounds unheard', 'in conclusion', 'was soft and gentle', 'leaving trails of fire', 'audible pop', 'rivulets of', 'despite herself', 'reckless abandon', 'torn between', 'fiery red hair', 'long lashes', 'world narrows', 'chestnut eyes', 'cheeks flaming', 'cheeks hollowing', 'it is essential', 'a mix of',
                    'a sense of',
                    'the scent of', 
                    'a hint of', 
                    'for a moment',
                    'side by side',
                    'a wave of']


# These are the aurora-m/safellm languages. The 24 EU langs +  vi, zh, ar, ru, hi, ar, sw, ja, ko, id
target_non_en_langs = ['bg',
 'hr',
 'cs',
 'da',
 'nl',
# 'en',
 'et',
 'fi',
 'fr',
 'de',
 'el',
 'hu',
 'ga',
 'it',
 'lv',
 'lt',
 'mt',
 'pl',
 'pt',
 'ro',
 'sk',
 'sl',
 'es',
 'sv', 'vi', 'zh', 'ar', 'ru', 'hi', 'ar', 'sw', 'ja', 'ko', 'id']



### BASIC UTILITIES

from collections import Counter


def get_ngram(text, window_size=3, lang=""):
    if not lang:
        if cjk_detect(text[:min(len(text), 100)]):
            lang = 'zh'
        else:
            lang = 'en'
    if lang in {"zh", "ja", "ko", "th", "jap"}:
        tokens = text
        ret = [
            "".join(tokens[i : i + window_size])
            for i in range(len(tokens) - window_size)
        ]
        ret = [
        "".join(tokens[i : i + window_size]) for i in range(len(tokens) - window_size)
        ]
        
    else:
        tokens = text.split(" ")
        ret = [
            " ".join(tokens[i : i + window_size]) for i in range(len(tokens) - window_size)
        ]
    return Counter(ret)


def high_ngram(text, cutoff=0.15, window_size=3, lang=""):
    if not lang:
        if cjk_detect(text[:min(len(text), 100)]):
            lang = 'zh'
        else:
            lang = 'en'
    aHash = get_ngram(text, window_size, lang)
    text_len = text.count(" ") + 1
    for key in list(aHash.keys()):
        aHash[key] = aHash[key] / text_len
    return any(a for a in aHash.values() if a > cutoff)


def get_target_path(shard_path: Path, dst_file_path: Path) -> Path:
    """
    Generates the target path for a shard file by appending the shard's filename to the destination directory.

    Args:
        shard_path (Path): Path of the shard file.
        dst_file_path (Path): Destination directory for storing the shard.

    Returns:
        Path: The full target path for the shard in the destination directory.
    """
    filename = shard_path.name
    return dst_file_path / filename


def mark_done(filepath: Path):
    """
    Marks a file as completed by creating a '.done' file with the same name.

    Args:
        filepath (Path): Path of the file to mark as completed.

    Returns:
        None
    """
    with open(filepath.with_suffix('.done'), "w") as fp:
        pass
    return


def check_done(filepath: Path):
    """
    Checks if a file is marked as completed by verifying the presence of a '.done' file.

    Args:
        filepath (Path): Path of the file to check.

    Returns:
        bool: True if the '.done' file exists, False otherwise.
    """
    return filepath.with_suffix('.done').exists()


def get_sublist(elements, rank, world_size):
    """
    Divides a list of elements into sublists based on the rank and world size.

    Args:
        elements (list): List of elements to split.
        rank (int): Rank of the current worker.
        world_size (int): Total number of workers.

    Returns:
        list: The sublist for the current rank.
    """
    # Calculate the number of elements in each sublist
    base_size = len(elements) // world_size
    remainder = len(elements) % world_size
    overhang_index = (base_size*world_size)+rank


    # Calculate the start and end indices for the desired rank
    start = (rank * base_size)
    end = (rank+1)*base_size

    # Return the sublist
    result =  elements[start:end]
    if remainder != 0 and rank < remainder:
        result.append(elements[overhang_index])
    return result


def sort_files_by_number(directory: Path):
    """
    Sorts files in a directory based on numerical prefixes in filenames.

    Args:
        directory (Path): Path to the directory containing the files.

    Returns:
        List[Path]: Sorted list of file paths in the directory.
    """
    # Get all files in the directory
    files = os.listdir(directory)

    # Extract numbers and file names
    numbered_files = []
    for filename in files:
        # Split the filename by the first '_' to separate number from the rest
        parts = filename.split('_', 1)

        if len(parts) > 1:
            try:
                num = int(parts[0])
                numbered_files.append((num, filename))
            except ValueError:
                continue

    # Sort files based on the extracted numbers
    numbered_files.sort(key=lambda x: x[0])

    # Extract sorted file names
    sorted_files = [directory / filename for _, filename in numbered_files]

    return sorted_files


def split_file(input_filename, output_template, shard_size):
    """
    Splits a large file into smaller shard files, each containing a specified number of lines.

    Args:
        input_filename (str): Path to the input file to split.
        output_template (str): Template for the output shard filenames.
        shard_size (int): Number of lines per shard.

    Returns:
        None
    """
    with open(input_filename, 'r') as infile:
        shard_number = 1
        lines_in_shard = 0
        output_filename = output_template.format(shard_number)
        outfile = open(output_filename, 'w')
        for line in tqdm(infile, "processing lines in source file"):
            outfile.write(line)
            lines_in_shard += 1
            if lines_in_shard == shard_size:
                # Move to the next shard
                shard_number += 1
                output_filename = output_template.format(shard_number)
                outfile.close()
                outfile = open(output_filename, 'w')
                lines_in_shard = 0
        outfile.close()
    #print(f"File '{input_filename}' has been split into {shard_number} shards.")


def filter_splits(sorted_shards: List[Path], rank: int, world_size: int) -> List[Path]:
    """
    Filters a list of shards, assigning a subset to each rank.

    Args:
        sorted_shards (List[Path]): List of sorted shard paths.
        rank (int): Rank of the current worker.
        world_size (int): Total number of workers.

    Returns:
        List[Path]: Subset of shard paths assigned to the current rank.
    """
    return get_sublist(sorted_shards, rank, world_size)


def get_splits(path, rank: int, world_size: int, samples_per_shard: int):
    """
    Splits a large dataset file into shards and distributes them across multiple workers.

    Args:
        path (Path): Path to the input dataset file.
        rank (int): Rank of the current worker.
        world_size (int): Total number of workers.
        samples_per_shard (int): Number of samples per shard.

    Returns:
        List[Path]: List of shard paths assigned to the current rank.
    """
    path = Path(path)
    donefile = path.with_suffix(".split.done")
    if not donefile.exists():
        if rank == 0:
            path.with_suffix('').mkdir(parents=True, exist_ok=True)
            output_file_template = str(path.with_suffix("") / ("{}_" + path.name))
            split_file(path, output_file_template, samples_per_shard)
            with donefile.open("w"):
                pass
        else:
            sleep(60)
            return get_splits(path, rank, world_size, samples_per_shard)

    sorted_shards = sort_files_by_number(path.with_suffix(""))
    return filter_splits(sorted_shards, rank, world_size)


def download_dataset(path, url, rank):
    """
    Downloads a dataset if it does not already exist at the specified path.

    Args:
        path (str): Local path where the dataset should be saved.
        url (str): URL to download the dataset from.
        rank (int): Rank of the current worker; only rank 0 downloads the dataset.

    Returns:
        None
    """
    if not os.path.exists(path):
        if rank == 0:
            #print("downloading dataset")
            Path(path).parent.mkdir(exist_ok=True, parents=True)
            wget.download(url)
        else:
            #print("no file detecting, downloading with root")
            sleep(600)



def postprocess_hf_results(result: List[Dict[str, List[Dict[str, str]]]], result_key: str = "generated_text", txt_key: str = "content") -> List[str]:
    """
    Extracts and returns generated text from Hugging Face model results.

    Args:
        result (List[Dict[str, List[Dict[str, str]]]]): List of result dictionaries from Hugging Face models.
        result_key (str): Key in the result dictionary to extract the generated text from.
        txt_key (str): Key within the result dictionary for accessing text content.

    Returns:
        List[str]: List of generated text strings.
    """
    return [r[0][result_key][1][txt_key] for r in result]

def postprocess_vllm_results(results):
    """
    Extracts generated text from vLLM model results.

    Args:
        results (List[RequestOutput]): List of RequestOutput objects from vLLM models.

    Returns:
        List[str]: List of generated text strings.
    """
    return [i.outputs[0].text for i in results]

def postprocess_results(results):
    """
    Generalized function to post-process model results, handling both Hugging Face and vLLM model outputs.

    Args:
        results (List): List of results from Hugging Face or vLLM models.

    Returns:
        List[str]: List of generated text strings.
    """
    if isinstance(results[0], RequestOutput):
        return postprocess_vllm_results(results)
    else:
        return postprocess_hf_results(results)


def chunkify(sequence, n):
    """
    Splits a sequence into approximately equal-sized chunks.

    Args:
        sequence (iterable): Sequence to split.
        n (int): Number of chunks to split into.

    Returns:
        List[List]: List of chunks, each containing a sublist of the original sequence.
    """
    sequence = list(sequence)
    deque_sequence = deque(sequence)
    result = []
    chunk_size = (len(sequence) + n - 1) // n  # Ceiling division

    while deque_sequence:
        chunk = []
        for _ in range(min(chunk_size, len(deque_sequence))):
            chunk.append(deque_sequence.popleft())
        result.append(chunk)

    return result

### BASIC ROUTINE TO STANDARDIZE OUR DATA
# the prototype for all our data. a list field must end in _list. metadata is special and will be a dict.
# do we need a text_type? should we move this to the params?
# we need an text_embedding and a media_embedding_list
data_fields = ['text', 'chosen', 'rejected_list', 'metadata', 'media', 'language']
metadata_fields = ['source', 'params',]

def standardize_data_fields(data):
    """
    Ensures that data adheres to a standardized format, transferring fields to `metadata.params`
    if they don't match the defined schema. Fields ending in `_list` are initialized as empty lists, 
    and any extra fields are moved to `metadata.params`.

    Args:
        data (dict): The input data to standardize.

    Returns:
        dict: Standardized data dictionary.
    """
    #make sure the data is in the standard format.
    #move everything to the metadata.params field otherwise
    if 'meta' in data and 'metadata' not in data:
        data['metadata'] = data['meta']
        del data['meta'] # let's map meta->metadata.
    if 'metadata' not in data:
        data['metadata'] = {}
    if 'media' not in data:
        data['media'] = {}
    if 'subset' in data and 'source' not in data['metadata']:
        data['metadata']['source'] = data['subset']
        del data['subset'] # let's map sbset->metadata.source
    for field in data_fields:
      if field not in data:
        if '_list' in field:
          data[field] = []
        elif field == 'metadata':
          data["metadata"] = {}
        elif field == 'media':
          data["media"] = {}
        else:
          data[field] = ''
    for field in metadata_fields:
      if field not in data['metadata']:
        if '_list' in field:
          data['metadata'][field] = []
        elif 'params' == field:
            data['metadata'][field] = {}
        else:
          data['metadata'][field] = ''
    if type(data['media']) is str:
        data['media'] = json.loads(data['media'])        
    if type(data['metadata']) is str:
        data['metadata'] = json.loads(data['metadata'])        
    if type(data['metadata']['params']) is str and data['metadata']['params']:
        data['metadata']['params'] = json.loads(data['metadata']['params'])
    if type(data['metadata']['params']) is not dict:
        data['metadata']['params'] = {}
    params = data['metadata']['params']
    for key in list(data.keys()):
      if key not in data_fields:
        params[key] = data[key]
        del data[key]
    for key in list(data['metadata'].keys()):
      if key not in metadata_fields:
        params[key] = data['metadata'][key]
        del data['metadata'][key]
    data['metadata']['params'] = params
    return data

def cleanup_data_batch(curr_data):
    """
    Applies `standardize_data_fields` to each item in a batch of data.

    Args:
        curr_data (list): List of data dictionaries to standardize.

    Returns:
        list: List of standardized data dictionaries.
    """
    ret = []
    for data in curr_data:
        ret.append(standardize_data_fields(data))
    return ret

def cleanup_and_serialize_params(data):
    """
    Standardizes the data fields and serializes the `metadata.params` field to a JSON string if necessary.

    Args:
        data (dict): The data dictionary to process.

    Returns:
        dict: The processed data dictionary.
    """
    standardize_data_fields(data)
    if not data['media']:
        data['media'] = "{}"
    elif type(data['media']) is not str:
        data['media'] = json.dumps(data['media'])        
    if not data['metadata']['params']:
        data['metadata']['params'] = "{}"      
    elif type(data['metadata']['params']) is not str:
        data['metadata']['params'] = json.dumps(data['metadata']['params'])
    return data


regex_tag = re.compile(r"<[^>]+>")
regex_image_tag = re.compile(r"<image_[_\d]+>")
regex_audio_tag = re.compile(r"<audio_[_\d]+>")

def strip_tags(text, logger=None):
    global regex_tag
    if "<video>" in text:
        text = text.replace("<caption>", "\n - Video Frame ").replace("</caption>", "\n").strip()
    else:
        text = text.replace("<caption>", "\n - Image: ").replace("</caption>", "\n").strip()
    text = text.replace("<video>", "Video: ").replace("</video>", "")
    for tag in  regex_tag.findall(text):
        if "<image_" in tag:
            time_stamp = tag.split("<image_")[-1].strip("<>")
            s1 = "0"
            if "_" in time_stamp:
                sarr = time_stamp.split("_")
                s1, s2 = sarr[0], sarr[1]
                logger.warning (('s1 s2', tag, s1, s2))
                try:
                    s1 = int(s1)
                    s1 += int(s2)
                except:
                    s1 = "0"
            else:
                try:
                    s1 = int(time_stamp)
                except:
                    if logger: logger.warning(("problem with time ", tag))
                    s1 = "0"
            text = text.replace(tag, f"at time {s1}: ")
        elif "<audio_" in tag:
            transcript = text.split(tag)[-1].split("<")[0].strip().strip(". \n")
            if transcript:
                time_stamp = tag.split("audio_")[-1].strip("<>")
                if transcript and transcript[0] == "[" and transcript[-1] == "]":
                    text = text.replace(tag, f"\nSound at time {time_stamp}: ")
                else:
                    text = text.replace(tag, f"\nTranscript at time {time_stamp}: ")                
    text = regex_tag.sub('', text)
    return text

# Create a template so we can use python's format functionality to
# fill in captions in the 'text' field. Optionally, cleanup the text
# field to add caption tags if none are there.
# every image will be captioned.
def create_caption_template_hash(data_list, add_caption_if_none_exists=False):
  """
Create templates for captions within a list of data items to allow formatting with dynamic values.

Args:
    data_list (list): A list of data dictionaries, each containing a 'text' field.

Returns:
    tuple: A tuple containing:
        - caption_template_hash (dict): Mapping from data indices to caption templates.
        - params_hash (dict): Mapping from data indices to parameters used in captions.
  """
  caption_template_hash = {}
  params_hash = {}
  for data in data_list:
    text = ""
    if "<caption>" in data['text']:
        for t in data['text'].split("<caption>"):
            if "</caption>" in t:
                caption, t =  t.split("</caption>",1)
                image_tag = list(regex_image_tag.findall(caption))
                if image_tag:
                    image_tag = image_tag[0]
                    caption = caption.replace(image_tag, "").strip()
                else:
                    image_tag = f"<image_{len(data['media'])}>"
                    data['media'][image_tag] = ""
                other_images = list(regex_image_tag.findall(t))
                for image in other_images:
                    t = t.replace(image, f"<caption>{image}</caption>")
                text = text+f"<caption>"+image_tag+caption+"</caption>"+t
            else:
                other_images = list(regex_image_tag.findall(t))
                for image in other_images:
                    t = t.replace(image, f"<caption>{image}</caption>")
                text = text+t
    else:
        text = data['text']
        other_images = list(regex_image_tag.findall(text))
        for image in other_images:
            text = text.replace(image, f"<caption>{image}</caption>")
    # add a pseudo caption for the first text snippet
    if "<caption>" not in text and add_caption_if_none_exists:
        text_snippet = text.split("<audio>")[0].split("<transcript>")[0].split("<emeb>")[0]
        if len(text_snippet) > 1000:
            text_snippet = text_snippet[:1000]
        image_tag = f"<image_{len(data['media'])}>"
        data['media'][image_tag] = ""            
        text = f"<caption>{image_tag}"+strip_tags(text_snippet)+"</caption>"+text
    text = text.replace("{", "--[--").replace("}", "--]--")
    template = ""
    params = {}
    for t in text.split("<caption>"):
        if "</caption>" in t:
            caption, t =  t.split("</caption>",1)
            j = caption.split("<image_")[-1].split(">")[0].strip()
            caption = caption.replace(f"<image_{j}>", "").strip()
            template = template+f"<caption><image_{j}>"+"{CAPTION_"+str(j)+"}</caption>"+t
            params[f"CAPTION_{j}"] = caption
        else:
            template = template+t
    caption_template_hash[data['_tmp_idx']] = template
    params_hash[data['_tmp_idx']] = params
    
  return caption_template_hash, params_hash


def set_caption(params, j, text):
  """
Set a specific caption in a parameter template.

Args:
    params (dict): The parameter template to modify.
    j (int): The index of the caption.
    text (str): The caption text to set.
  """
  if type(j) is str:
      params[j] = text
  else:
      params[f"CAPTION_{j}"] = text


def get_caption(params, j):
  if type(j) is str:
      return params[j]
  else:
      return params[f"CAPTION_{j}"]
      

def apply_params_to_caption_text(data_list, caption_template_hash, params_hash):
  """
Apply the parameters from a template to format the caption text in a list of data items.

Args:
    data_list (list): A list of data dictionaries to modify.
    caption_template_hash (dict): Mapping from data indices to caption templates.
    params_hash (dict): Mapping from data indices to parameters used in captions.
  """

  for data in data_list:
    caption_template = caption_template_hash[data['_tmp_idx']]
    params = params_hash[data['_tmp_idx']]
    #logger.warning("applying " + str(params))
    data['text'] = caption_template.format(**params).replace("--[--", "{").replace("--]--", "}")    

def create_transcript_template_hash(data_list):
  transcript_template_hash = {}
  params_hash = {}
  for data in data_list:
    text = ""
    if "<transcript>" in data['text']:
        for t in data['text'].split("<transcript>"):
            if "</transcript>" in t:
                transcript, t =  t.split("</transcript>",1)
                audio_tag = list(regex_audio_tag.findall(transcript))
                if audio_tag:
                    audio_tag = audio_tag[0]
                    transcript = transcript.replace(audio_tag, "").strip()
                else:
                    audio_tag = f"<audio_{len(data['media'])}>"
                    data['media'][audio_tag] = ""
                other_audios = list(regex_audio_tag.findall(t))
                for audio in other_audios:
                    t = t.replace(audio, f"<transcript>{audio}</transcript>")
                text = text+f"<transcript>"+audio_tag+transcript+"</transcript>"+t
            else:
                other_audios = list(regex_audio_tag.findall(t))
                for audio in other_audios:
                    t = t.replace(audio, f"<transcript>{audio}</transcript>")
                text = text+t
    else:
        text = data['text']
        other_audios = list(regex_audio_tag.findall(text))
        for audio in other_audios:
            text = text.replace(audio, f"<transcript>{audio}</transcript>")
    data['text'] = text
    text = text.replace("{", "--[--").replace("}", "--]--")
    template = ""
    params = {}
    for t in text.split("<transcript>"):
        if "</transcript>" in t:
            transcript, t =  t.split("</transcript>",1)
            j = transcript.split("<audio_")[-1].split(">")[0].strip()
            transcript = transcript.replace(f"<audio_{j}>", "").strip()
            template = template+f"<transcript><audio_{j}>"+"{TRANSCRIPT_"+str(j)+"}</transcript>"+t
            params[f"TRANSCRIPT_{j}"] = transcript
        else:
            template = template+t
    transcript_template_hash[data['_tmp_idx']] = template
    params_hash[data['_tmp_idx']] = params
    
  return transcript_template_hash, params_hash


def set_transcript(params, j, text):
  if type(j) is str and "TRANSCRIPT" in j:
      params[j] = text
  else:
      params[f"TRANSCRIPT_{j}"] = text


def apply_params_to_transcript_text(data_list, transcript_template_hash, params_hash):

  for data in data_list:
    transcript_template = transcript_template_hash[data['_tmp_idx']]
    params = params_hash[data['_tmp_idx']]
    #logger.warning("applying " + str(params))
    data['text'] = transcript_template.format(**params).replace("--[--", "{").replace("--]--", "}")    



### BASIC TEXT LLM GENERATION ROUTINES

import re

def non_english_detect(text):
    """
    Detects if a text contains any non-English characters based on Unicode ranges.

    Args:
        text (str): Text to check.

    Returns:
        bool: True if non-English characters are detected, False otherwise.
    """
    if re.search("[\u0000-\u00BF]", text):
        return False
    return True
    
def cjk_detect(text):
    """
    Detects if a text contains characters from specific East Asian languages (Chinese, Japanese, Korean, Thai, and Traditional Javanese).

    Args:
        text (str): Text to check.

    Returns:
        str or None: Language code if detected; otherwise, None.
    """
    # chinese
    if re.search("[\u4e00-\u9FFF]", text):
        return "zh"
    # korean
    if re.search("[\uac00-\ud7a3]", text):
        return "ko"
    # japanese
    if re.search("[\u3040-\u30ff]", text):
        return "ja"
    # thai
    if re.search("[\u0E01-\u0E5B]", text):
        return "th"
    # traditional javanese
    if re.search("[\uA980-\uA9DF]", text):
       return "jv_tr"
    return None

def get_cjk_tokens(tokenizer):
    """
    Retrieves tokens that correspond to East Asian characters (CJK) from a tokenizer.

    Args:
        tokenizer: The tokenizer instance.

    Returns:
        list: List of tokens that correspond to CJK characters.
    """
    return [tokenizer.decode([idx]) for idx in range(len(tokenizer)) if cjk_detect(tokenizer.decode([idx]))]

def get_non_english_tokens(tokenizer):
    """
    Retrieves tokens that correspond to non-English characters from a tokenizer.

    Args:
        tokenizer: The tokenizer instance.

    Returns:
        list: List of tokens that correspond to non-English characters.
    """
    return [tokenizer.decode([idx]) for idx in range(len(tokenizer)) if non_english_detect(tokenizer.decode([idx]))]

def get_tokens_as_list(word_list, tokenizer):
    """
    Converts a list of words to token IDs using a tokenizer.

    Args:
        word_list (list): List of words to tokenize.
        tokenizer: Tokenizer instance.

    Returns:
        list: List of token IDs for each word.
    """
    tokens_list = []
    for word in word_list:
        tokenized_word = tokenizer([word], add_special_tokens=False).input_ids[0]
        tokens_list.append(tokenized_word)
    return tokens_list


def fix_too_much_ngram(text, window_size=3, lang="en", threshold=2, logger=None):
    stopwords =  all_stopwords.get(lang, all_stopwords['en'])
    for word, cnt in get_ngram(text, window_size=3, lang="").items():
        if cnt >= threshold:
            word_arr = word.split()
            if not any(w for w in word_arr if len(w) > 3 and w.lower() not in stopwords and w.lower()[:min(len(w), 4)] not in {'said', 'says', 'sayi', 'menti', 'disc', 'talk', 'desc', 'hear', 'speak',}) :
                continue
            if word not in text:
                if logger: logger.warning(("NGRAM NOT IN TEXT", word, text))
                continue
            i = text.index(word)
            text = text[:i+1]+text[i+1:].replace("and "+word, "").replace("or "+word, "").replace(", "+word, "").replace(" "+word, "")
    return text

def generate_with_batching(model, tokenizer, batch, use_cache=True, repetition_penalty=1.2, max_new_tokens=400, batch_size=2, skip_special_tokens=True, return_continuations_only=True, dont_decode_non_english=False, dont_decode_cjk=False, supress_tokens=[], image_list=None, group_by_length=20000, logger=None, strip_bad_last_sentence=False, self_trigram_threshold=5, supress_self_trigram_topk=0, too_much_ngram_threshold=0, **args):

    """
    Generates text responses from a language model in batches, allowing for flexible decoding options, token suppression, 
    and multi-modal support with images.

    This function processes a batch of prompts, optionally handling CJK (Chinese, Japanese, Korean) and non-English 
    token suppression, image-based context (for multi-modal models), and a customizable repetition penalty to 
    minimize repeated phrases. Media data (images) can be included for specific model types that support 
    image-text interaction, such as InternVLM. Each batch is tokenized, processed by the model, and decoded 
    back to text, optionally removing special tokens and truncating the initial input prompt.

    Args:
        model: Language model instance for text generation.
        tokenizer: Tokenizer for encoding and decoding text data.
        batch (list of str): List of text prompts to generate responses for.
        use_cache (bool, optional): If True, enables model caching for faster generation. Defaults to True.
        repetition_penalty (float, optional): Penalty to reduce token repetition in the output. Defaults to 1.2.
        max_new_tokens (int, optional): Maximum tokens to generate for each prompt. Defaults to 400.
        batch_size (int, optional): Number of prompts processed together in each batch. Defaults to 2.
        skip_special_tokens (bool, optional): If True, removes special tokens from the final output. Defaults to True.
        return_continuations_only (bool, optional): If True, excludes the prompt from the output, keeping only generated text.
        dont_decode_non_english (bool, optional): If True, suppresses non-English tokens in the generated text.
        dont_decode_cjk (bool, optional): If True, suppresses CJK (Chinese, Japanese, Korean) tokens in the output.
        supress_tokens (list of str, optional): List of specific tokens to suppress in the output.
        image_list (list of lists of PIL.Image, optional): List of images for each prompt, allowing image-text interaction.
        **args: Additional keyword arguments for the model's `generate` function.

    Returns:
        list of str: Generated responses for each input prompt in the batch.

    Detailed Behavior:
        - If `image_list` is provided, it must match the length of `batch`, with each element being a list of images 
          corresponding to a specific prompt.
        - If using `dont_decode_cjk` or `dont_decode_non_english`, the function will retrieve these tokens from the 
          tokenizer if not already cached.
        - For models supporting image-text interaction (e.g., InternVLM), images in `image_list` are converted to model-usable 
          formats, and input queries are adjusted to incorporate image tokens.
        - `bad_words_ids` is constructed to hold any undesired tokens (e.g., from `supress_tokens`, `dont_decode_cjk`, etc.)
          that should be avoided during generation.
    """

    # global press
    # this is the InternVL2 batch chat function adapted to our purposes. We don't need to use the conv template because
    # we already send the data into the model in the right chat format.
    stopwords =  all_stopwords['en']
    batch_size = int(batch_size)
    def internvlm_batch_chat(self, tokenizer, pixel_values, questions, generation_config, num_patches_list=None,
                         IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
                         IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False, image_counts=None, return_continuations_only=True,
                         skip_special_tokens=True):

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            #print(f'dynamic ViT batch size: {image_bs}')

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = question.replace('<image>', image_tokens, 1)
            queries.append(query)

        tokenizer.padding_side = 'left'
        model_inputs = tokenizer(queries, return_tensors='pt', padding=True, add_special_tokens=False)
        input_ids = model_inputs['input_ids'].to(self.device)
        prompt_len = model_inputs["input_ids"].shape[-1]        
        attention_mask = model_inputs['attention_mask'].to(self.device)
        #eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)
        #generation_config['eos_token_id'] = eos_token_id
        # if press is not None:
        if False:
            with press(self):
              generation_output = self.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_config
            )
        else:
              generation_output = self.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_config
            )
            
        if return_continuations_only:
            generation_output = generation_output[:, prompt_len:]
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=skip_special_tokens)
        return responses

    group_by_length = max(512, int(group_by_length-(max_new_tokens/2)))
    if dont_decode_cjk and not hasattr(tokenizer, 'cjk_ids'):
        str_list = get_cjk_tokens(tokenizer)
        tokenizer.cjk_ids = get_tokens_as_list(str_list, tokenizer)
            
    if dont_decode_non_english and not hasattr(tokenizer, 'non_english_ids'):
        str_list = get_non_english_tokens(tokenizer)
        tokenizer.non_english_ids = get_tokens_as_list(str_list, tokenizer)    
    # qwen has a problem of sometimes outputing cjk randomly. we fix this by including cjk token in the bad_words_ids


    device = model.device
    if hasattr(tokenizer, 'supress_self_trigram'):
        supress_self_trigram = tokenizer.supress_self_trigram
        supress_self_trigram_list = tokenizer.supress_self_trigram_list
    else:
        supress_self_trigram = tokenizer.supress_self_trigram = {}
        supress_self_trigram_list = tokenizer.supress_self_trigram_list = []
        
    # this could really slow things down if we do this for every generation
    # decide if we want to do this for every N generations
    if self_trigram_threshold > 0 and not supress_self_trigram_list:
        lst = list(supress_self_trigram.items())
        lst.sort(key=lambda a: a[1], reverse=True)
        supress_self_trigram_list.extend(lst)
        
    #torch.cuda.empty_cache()
    bad_words_ids = []
    if supress_tokens:
        if hasattr(tokenizer, 'supress_tokens_hash'):
            supress_tokens_hash = tokenizer.supress_tokens_hash
        else:
            supress_tokens_hash = tokenizer.supress_tokens_hash = {}
        for word in supress_tokens:
            if word in supress_tokens_hash: continue
            tokenized_ids = tokenizer([word], add_special_tokens=False).input_ids
            if len(tokenized_ids) > 1:
                if logger: logger.warning(f"Supress token word {word} is not a single word or phrase. skipping")
                continue
            supress_tokens_hash[word] =  tokenized_ids[0]
        bad_words_ids.extend([supress_tokens_hash[word] for word in supress_tokens])
    if self_trigram_threshold > 0 and supress_self_trigram_list:
        bad_words_ids.extend([tokenizer([word], add_special_tokens=False).input_ids[0] for word, cnt in supress_self_trigram_list[:min(len(supress_self_trigram_list), self_trigram_threshold)] if cnt >= self_trigram_threshold])
    if dont_decode_cjk:
        bad_words_ids.extend(tokenizer.cjk_ids)
    if dont_decode_non_english:
        bad_words_ids.extend(tokenizer.non_english_ids)
    if image_list:
        assert len(image_list) == len(batch), "image_list is a list of lists of images for each prompt. the len of image_list must be the same as the len of batch"
    output = [None]*len(batch)
    with torch.no_grad():
        lst_by_buckets = {}
        idx_by_buckets = {}
        for idx, a_text in enumerate(batch):
            bucket = int(len(a_text)//group_by_length)
            lst_by_buckets[bucket] = lst_by_buckets.get(bucket, []) + [a_text]
            idx_by_buckets[bucket] = idx_by_buckets.get(bucket, []) + [idx]
        for bucket, batch2 in lst_by_buckets.items():
            idxs = idx_by_buckets[bucket]
            batch_size2 = max(1,int(batch_size/(1+bucket)))
            if logger: logger.warning(("group by batching", press, group_by_length, batch_size2, bucket, idxs))
            for rng in range(0, len(batch2), batch_size2):
                sub_batch2 = batch2[rng:min(len(batch2), rng+batch_size2)]
                idxs2 = idxs[rng:min(len(batch2), rng+batch_size2)]
                #TODO: convert <image_1> -> <image>
                if image_list and hasattr(model, 'batch_chat') and any(s for s in sub_batch2 if "<image>" in s):
                    # this is a internVLM situation
                    generation_config = copy.copy(args)
                    generation_config['use_cache'] = use_cache
                    generation_config['repetition_penalty'] = repetition_penalty
                    generation_config['max_new_tokens'] = max_new_tokens
                    #generation_config['penalty_alpha'] = penalty_alpha
                    if bad_words_ids:
                        generation_config['bad_words_ids']=bad_words_ids
                    imgs = image_list[rng:min(len(batch2), rng+batch_size2)]
                    pixel_values = []
                    num_patches_list = []
                    for img_list in enumerate(imgs):
                        if not img_list:
                            num_patches_list.append(0)
                            continue
                        pixel_list = [preprocess_image_for_internvlm(image) for image in img_list]
                        if len(pixel_list) > 1:
                            pixel_values.append(torch.cat(*pixel_list, dim=0))
                        else:
                            pixel_values.append(pixel_list[0])
                        num_patches_list.append(pixel_value.size(0))
                    pixel_values = torch.cat(*pixel_values, dim=0).to(device)
                    responses = internvlm_batch_chat(tokenizer, pixel_values, questions=d, num_patches_list=num_patches_list, generation_config=generation_config, return_continuations_only=return_continuations_only, skip_special_tokens=skip_special_tokens,)
                    for idx, r in zip(idxs2, responses):
                        output[idx] = r
                elif sub_batch2:
                    model_inputs = tokenizer(sub_batch2, truncation=True, padding=True, return_tensors="pt", add_special_tokens=False, ).to(device)
                    prompt_len = model_inputs["input_ids"].shape[-1]
                    #model_output = model.generate(**model_inputs,
                    #                                            use_cache=use_cache, repetition_penalty=repetition_penalty,  max_new_tokens=max_new_tokens,  **args )
                    # if press is not None:
                    if False:
                        with press(model):
                            if bad_words_ids:
                                bad_words_ids = [oo for oo in bad_words_ids if len(oo) > 0]
                                #penalty_alpha=penalty_alpha,
                                model_output = model.generate(**model_inputs, bad_words_ids=bad_words_ids, 
                                                                    use_cache=use_cache, repetition_penalty=repetition_penalty,  max_new_tokens=max_new_tokens,  **args)
                            else:
                                # penalty_alpha=penalty_alpha,
                                model_output = model.generate(**model_inputs, 
                                                                                use_cache=use_cache, repetition_penalty=repetition_penalty,  max_new_tokens=max_new_tokens,  **args )
                    else:
                        if bad_words_ids:
                            bad_words_ids = [oo for oo in bad_words_ids if len(oo) > 0]
                            #penalty_alpha=penalty_alpha,
                            model_output = model.generate(**model_inputs, bad_words_ids=bad_words_ids,  
                                                                use_cache=use_cache, repetition_penalty=repetition_penalty,  max_new_tokens=max_new_tokens,  **args)
                        else:
                            # penalty_alpha=penalty_alpha,
                            model_output = model.generate(**model_inputs, 
                                                                            use_cache=use_cache, repetition_penalty=repetition_penalty,  max_new_tokens=max_new_tokens,  **args )                    
                    if return_continuations_only:
                        model_output = model_output[:, prompt_len:]
                    responses = tokenizer.batch_decode(model_output, skip_special_tokens=skip_special_tokens,)
                    for idx, r in zip(idxs2, responses):
                        output[idx] = r
    output2 = []
    if self_trigram_threshold > 0 and supress_self_trigram_topk > 0:
        #TODO: trim the trigram hash if it gets too big
        for text in output:
            text2 = " ".join([t for t in text.split() if "<|" not in t and t not in {"<pad>", "</s>"}])
            lang = 'zh'
            if not cjk_detect(text2[:min(len(text2), 100)]):
                lang = 'en'
            for word, cnt in get_ngram(text2, 3, lang).items():
                #TODO do stopwords filtering for all other languages
                if lang == 'en':
                    word_arr = word.split()
                    if not any(w for w in word_arr if len(w) > 3 and w.lower() not in stopwords):
                        continue
                supress_self_trigram[word] = supress_self_trigram.get(word, 0) + cnt
                supress_self_trigram_list.clear()
            
    output = [text.split("<pad>")[0].split("</s>",1)[0].split("<|im_end|>",1)[0].split("<|endoftext|>",1)[0].rstrip() for text in output]
    for answer in output:
        if too_much_ngram_threshold >=2:
            answer = fix_too_much_ngram(answer, threshold=too_much_ngram_threshold)
        if "As a language" in answer or "As a large language" in answer or "As an AI" in answer or "I apologize" in answer or "I'm sorry" in answer or "sorry" in answer or "I cannot" in answer or "Qwen" in answer or "ChatGPT" in answer or "GPT-" in answer or "Claude" in answer or "OpenAI" in answer or "Anthropic" in answer or "Alibaba" in answer or "Gemini" in answer:
            answer = answer.replace("Alibaba Cloud", "my developers")            
            answer = answer.replace("Alibaba", "my developers")
            answer = answer.replace("Anthropic", "my developers")
            answer = answer.replace("Google", "my developers")
            answer = answer.replace("OpenAI", "my developers")                                                
            answer  = answer.replace("ChatGPT", "a virtual assistant ")
            answer  = answer.replace("Gemini", "a virtual assistant ")            
            answer = answer.replace("GPT-4", "a virtual assistant ")
            answer = answer.replace("GPT-3", "a virtual assistant ")            
            answer = answer.replace("Claude", "a virtual assistant ")
            answer = answer.replace("Qwen", "a virtual assistant")
            answer = answer.rstrip()
        if strip_bad_last_sentence:
            if not cjk_detect(answer[:min(len(answer), 100)]) and ". " in answer:
                #sometimes models go off the rails and creates run on sentences at the end
                answer = answer.split(". ")
                for _ in range(4):
                    if len(answer) <= 1: break
                    if answer and len(answer[-1].split()) > 25 and "\n" not in answer[-1]:
                        answer = answer[:-1]
                answer = ". ".join(answer)
            elif cjk_detect(answer[:min(len(answer), 100)]) and "。" in answer:
                answer = answer.split("。")
                for _ in range(4):
                    if len(answer) <= 1: break
                    if answer and len(answer[-1]) > 25 and "\n" not in answer[-1]:
                        answer = answer[:-1]
                answer = "。".join(answer)                    
        output2.append(answer)

    #torch.cuda.empty_cache()
    # we don't want to empty the cache every generation. this could really slow things down. this should be done per batch i think. to prevent fagmentation
    return output2

def chunkify(sequence, n):
    """Splits a sequence into N roughly equal-sized chunks."""

    deque_sequence = deque(sequence)
    result = []
    chunk_size = (len(sequence) + n - 1) // n  # Ceiling division

    while deque_sequence:
        chunk = []
        for _ in range(min(chunk_size, len(deque_sequence))):
            chunk.append(deque_sequence.popleft())
        result.append(chunk)

    return result

### IMAGE PROCESSING FOR INTERNVLM

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    """
    Builds a transformation pipeline for image preprocessing, including resizing, normalization, and conversion to tensor format.

    Args:
        input_size (int): The target size to resize the image to, with dimensions (input_size, input_size).

    Returns:
        torchvision.transforms.Compose: A composed transform that includes RGB conversion, resizing, tensor conversion, and normalization.
    """
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        torchvision.transforms.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """
    Finds the aspect ratio in a list of target ratios that is closest to a given aspect ratio.

    Args:
        aspect_ratio (float): The aspect ratio of the input image (width / height).
        target_ratios (list of tuples): A list of target aspect ratios to compare.
        width (int): The width of the original image.
        height (int): The height of the original image.
        image_size (int): The desired size for each image split.

    Returns:
        tuple: The closest matching aspect ratio from target_ratios.
    """
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """
    Dynamically preprocesses an image by resizing it to the closest matching aspect ratio and splitting it into patches.

    Args:
        image (PIL.Image): The input image to process.
        min_num (int, optional): Minimum number of patches. Default is 1.
        max_num (int, optional): Maximum number of patches. Default is 12.
        image_size (int, optional): The target size for each patch. Default is 448.
        use_thumbnail (bool, optional): Whether to add a thumbnail version of the resized image at the end. Default is False.

    Returns:
        list of PIL.Image: List of image patches.
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def preprocess_image_for_internvlm(image, input_size=448, max_num=12):
    """
    Preprocesses an image for the InternVLM model, including resizing, patch splitting, and normalization.

    Args:
        image (PIL.Image): The input image to preprocess.
        input_size (int, optional): The size of each patch. Default is 448.
        max_num (int, optional): Maximum number of patches. Default is 12.

    Returns:
        torch.Tensor: A tensor of shape (num_patches, 3, input_size, input_size) containing normalized image patches.
    """
    image = image.convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

### FINDING TEXT IN CAPTIONS

def remove_quotes(text):
  """
  Removes single quotes in text by replacing them with alternative tokens, then reinstates them with formatting as needed.

  Args:
      text (str): Text containing quotes to be formatted.

  Returns:
      str: Formatted text with alternative tokens replacing quotes.
  """
  text = text = text.replace("\"", "'")
  text = text.replace("'s ", " @s@ ").replace("'ve ", " @ve@ ").replace("'m ", " @m@ ").replace("'t ", " @t@ ")
  ret_text = []
  text_split = text.split("'")
  len_text_split = len(text_split)
  for idx, segment in enumerate(text_split):
    if idx % 2 == 0:
      if idx == len_text_split-1:
        ret_text.append(segment + " ")
      else:
        ret_text.append(segment + " '' ")        
  text = ''.join(ret_text).strip()
  text = text.replace(" @s@ ", "'s ").replace(" @ve@ ", "'ve ").replace( " @m@ ", "'m ").replace(" @t@ ", "'t ").strip()
  return text

def find_quotes(text):
  """
  Finds and returns quoted segments of text.

  Args:
      text (str): The text to search for quoted segments.

  Returns:
      list of str: List of text segments within single quotes.
  """
  accum = []
  text = text.replace("'s ", " @s@ ").replace("'ve ", " @ve@ ").replace("'m ", " @m@ ").replace("'t ", " @t@ ")
  for idx, segment in enumerate(text.split("'")):
    if idx % 2 != 0:
      accum.append(segment)
  accum = [a.replace(" @s@ ", "'s ").replace(" @ve@ ", "'ve ").replace( " @m@ ", "'m ").replace(" @t@ ", "'t ").replace("  ", " ").replace("  ", " ").strip() for a in accum]
  accum.sort(key=lambda a: len(a), reverse=True)
  return accum

numbering_list = ['3', '7)', '7.', '4', 'iii.', 'iii-', '8.', '4-', 'v:', 'I:', 'ii.', 'i.', 'V)', 'E)', 'I)', 'III.', 'III)', '2-', '1)', 'v-', 'III', 'I.', 'c)', '1.', 'V-', 'iv)', 'A)', 'v)', 'IV', 'C.', 'ii)', 'I', 'IV.', 'C)', 'II-', '2.', 'III-', 'IV)', 'd)', 'iii', 'i-', 'iii:', 'A.', 'B.', '1', '6)', 'ii', '8)', '3)', 'e)', 'ii-', '5-', 'II)', 'iv-', '2)', 'e.', 'IV:', 'III:', 'i)', '10.', 'V', 'V.', 'v.', 'D)', 'E.', 'iv:', 'B)', 'II', 'ii:', 'V:', 'a.', '5.', 'IV-', '9.', 'D.', '3.', '4:', '2:', 'i', 'II.', '3-', '2', 'c.', 'a)', '3:', '10)', 'd.', 'i:', 'iv.', '1-', '4.', '5', 'iv', 'iii)', 'b.', '1:', 'II:', 'v', '5:', '6.', 'b)', 'I-', '9)', '4)', '5)']

stopwords_list = ['es', 'ing', 'ed', 'include', 'includes', 'also', 'haven', 'are', 'why', 'most', "won't", 'against', 'with', 'needn', 'couldn', 'now', 'mustn', 'who', 'under', 'doing', 'am', 'aren', 'they', "didn't", 'd', 'doesn', 'if', 'he', 'her', "haven't", 'isn', 'own', 'does', 'such', 'until', 'into', 'had', 'again', 'over', "hadn't", "you'll", 't', 'by', 'be', "wasn't", 'so', 'yours', 'both', 'any', 'did', "you've", 'these', 'myself', 'o', 'hasn', "isn't", 'you', 'other', 'shan', 'being', 'yourselves', 'was', 'no', 'm', 'those', 'will', 'its', 'itself', 'have', 'down', 'weren', 'having', 'wouldn', 'herself', "mustn't", 'very', 'do', "should've", 'him', "you'd", 'below', 'just', 'that', 'for', 'which', 'but', 'nor', 'all', 'then', 'i', 'whom', 'it', 'once', 'here', 've', "you're", 'ours', "that'll", 'a', 'won', 'himself', 'where', 'this', 'your', "hasn't", 'same', 'when', 'ourselves', 'because', "needn't", 'theirs', 'from', 'mightn', 'my', 'while', 'yourself', "she's", 'each', "doesn't", 'only', 'at', 's', 'their', "wouldn't", 'shouldn', 'and', 'themselves', 'hers', 'has', 'up', 'ma', 'in', 'll', 'we', 're', 'y', 'of', 'after', 'our', "shan't", 'before', 'wasn', 'can', 'should', 'been', 'through', 'as', 'further', 'during', 'between', 'there', 'me', 'on', 'don', "shouldn't", 'more', 'out', "don't", 'the', "weren't", "aren't", "it's", 'what', 'or', "couldn't", 'hadn', "mightn't", 'his', 'above', 'to', 'how', 'few', 'off', 'them', 'didn', 'ain', 'not', 'she', 'an', 'than', 'too', 'is', 'some', 'were', 'about']

all_stopwords['en'] = set(stopwords_list + numbering_list + list(all_stopwords['en']))

# we use the old stopwords set here for backwards compatability
stopwords_set = set(stopwords_list + numbering_list)

def strip_left_stopwords(e_text, lang="en"):
  """
  Removes common stopwords from the left side of a text until a significant word is found.

  Args:
      e_text (str): The text to strip from the left side.

  Returns:
      str: Text with left-side stopwords removed.
  """
  e_text2 = []
  add_rest = False
  stopwords =  all_stopwords.get(lang, all_stopwords['en'])  
  for et in e_text.split():
      etl = et.lower()
      if add_rest or ((etl not in stopwords and etl not in common_title_words_set) or etl.strip(".") in {"a", "an", "united", "the", "new", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",  "asian", "american", "african", "european", }):
        add_rest = True
        e_text2.append(et)
  return " ".join(e_text2)


def strip_right_stopwords(e_text, lang="en"):
  """
  Removes common stopwords from the right side of a text until a significant word is found.

  Args:
      e_text (str): The text to strip from the right side.

  Returns:
      str: Text with right-side stopwords removed.
  """
  e_text2 = []
  add_rest = False
  e_text_arr = e_text.split()
  e_text_arr.reverse()
  stopwords =  all_stopwords.get(lang, all_stopwords['en'])    
  for et in e_text_arr:
      etl = et.lower()      
      if add_rest or (etl not in stopwords or etl.strip(".") in {"act", "code", "statute", "regulation", "regulations", "percent", "feet", "foot", "square", "barrells", "hour", "hours", "people", "asian", "american", "african", "european", "act", "law", "facilities", "facility", "center", "square", "rd", "street", "way", "blvd", "ave", "avenue", "states", "kingdom", "court", "corp", "corporation", "co", "company", "ltd", "llc", "llp", "incorp.", "incorporated"}):
        add_rest = True
        e_text2.append(et)
  return " ".join(reversed(e_text2))

#### CREATE IMAGE + TEXT DATA

def tokenize_with_chat_template(tokenizer, system="", instruction="", response="", continue_user_instruction=False):
  """
  Formats instructions and system prompts using a tokenizer, especially for chat-based models.

  Args:
      tokenizer: Tokenizer instance with chat template support.
      system (str): System message.
      instruction (str): User instruction.
      response (str, optional): Assistant response or prefix.

  Returns:
      str: Formatted string for chat-based models.
  """
  if type(system) is list:
      conversation = system
      if continue_user_instruction:
          return tokenize_with_user_continuation(tokenizer, conversation)
      else:
          return tokenizer.apply_chat_template(conversation, tokenize=False)
      
  system= system.strip()
  instruction = instruction.strip()
  response = response.strip()
  if system:
    if response:
        try:
            return tokenize_with_assistant_continuation(tokenizer, [{"role": "system", "content": system}, 
                                                                    {"role": "user", "content": instruction},
                                                                    {"role": "assistant", "content": response}])
        except:
            return tokenize_with_assistant_continuation(tokenizer, [{"role": "user", "content": system+"\n===\n"+instruction},
                                                                    {"role": "assistant", "content": response}])
    else:
        if continue_user_instruction:
            try:
                return tokenize_with_user_continuation(tokenizer, [{"role": "system", "content": system}, 
                                                                   {"role": "user", "content": instruction}])
            except:
                return tokenize_with_user_continuation(tokenizer, [{"role": "user", "content": system+"\n===\n"+instruction}])
        else:
            try:
                return tokenize_with_assistant_continuation(tokenizer, [{"role": "system", "content": system}, 
                                                             {"role": "user", "content": instruction}, {"role": "assistant", "content": ""}])
            except:
                return tokenize_with_assistant_continuation(tokenizer, [{"role": "user", "content": system+"\n===\n"+instruction}, {"role": "assistant", "content": ""}])
                
  else:
    if response:
        return tokenize_with_assistant_continuation(tokenizer, [{"role": "user", "content": instruction},
                                                                {"role": "assistant", "content": response}])
    else:
        if continue_user_instruction:
            return tokenize_with_user_continuation(tokenizer, [{"role": "user", "content": instruction}])
        else:
            return tokenize_with_assistant_continuation(tokenizer, [{"role": "user", "content": instruction}, {"role": "assistant", "content": ""}])


def tokenize_with_assistant_continuation(tokenizer, messages):
  """
  Tokenizes chat messages, returning the content up to the assistant's response without any ending tokens.

  This function adapts the tokenization for assistant responses in chat-based templates. It trims any 
  standard ending associated with the assistant's response for continuity in conversations.

  Args:
      tokenizer: Tokenizer instance with support for chat templates and continuation markers.
      messages (list of dict): List of messages in chat format, each having 'role' and 'content' keys.

  Returns:
      str: Tokenized message sequence without the assistant's ending token.
  """
  if not hasattr(tokenizer, "assistant_ending"):
    msg = tokenizer.apply_chat_template([{"role": "user", "content": ""}, {"role": "assistant", "content": "@@@@@@"}], tokenize=False)
    tokenizer.assistant_ending = msg.split("@@@@@@")[-1]
    msg = tokenizer.apply_chat_template([{"role": "user", "content": "!!!!!!!!"}, {"role": "assistant", "content":  "@@@@@@"}, {"role": "user", "content": "<<<<<<<"}], tokenize=False)
    tokenizer.assistant_beginning = msg.split("@@@@@@",1)[0].split("!!!!!!!!",1)[-1]
    user_beginning = msg.split("!!!!!!!!",1)[0]
    user_beginning2 = msg.split( "<<<<<<<",1)[0].split("@@@@@@",1)[-1]
    if len(user_beginning2) < len(user_beginning):
        user_beginning = user_beginning2
    tokenizer.user_beginning  = user_beginning
    tokenizer.user_ending = msg.split( "<<<<<<<",1)[-1]
    
  if not messages: return ""
  return tokenizer.apply_chat_template(messages, tokenize=False)[:-len(tokenizer.assistant_ending)]

def tokenize_with_user_continuation(tokenizer, messages):
  """
  Tokenizes chat messages, returning content up to the user’s response without any ending tokens.

  This function adapts tokenization for user responses in chat-based templates. It trims any standard ending 
  associated with the user's response to enable continuity in conversations.

  Args:
      tokenizer: Tokenizer instance with support for chat templates and continuation markers.
      messages (list of dict): List of messages in chat format, each having 'role' and 'content' keys.

  Returns:
      str: Tokenized message sequence without the user’s ending token.
  """
  if not hasattr(tokenizer, "user_ending"):
    msg = tokenizer.apply_chat_template([{"role": "user", "content": ""}, {"role": "assistant", "content": "@@@@@@"}], tokenize=False)
    tokenizer.assistant_ending = msg.split("@@@@@@")[-1]
    msg = tokenizer.apply_chat_template([{"role": "user", "content": "!!!!!!!!"}, {"role": "assistant", "content":  "@@@@@@"}, {"role": "user", "content": "<<<<<<<"}], tokenize=False)
    tokenizer.assistant_beginning = msg.split("@@@@@@",1)[0].split("!!!!!!!!",1)[-1]
    user_beginning = msg.split("!!!!!!!!",1)[0]
    user_beginning2 = msg.split( "<<<<<<<",1)[0].split("@@@@@@",1)[-1]
    if len(user_beginning2) < len(user_beginning):
        user_beginning = user_beginning2
    tokenizer.user_beginning  = user_beginning
    tokenizer.user_ending = msg.split( "<<<<<<<",1)[-1]

  if not messages: return ""
  return tokenizer.apply_chat_template(messages, tokenize=False)[:-len(tokenizer.user_ending)]


def assign_uuid(input_string: str = None) -> uuid.UUID:
    """
    Generates a UUID for a given string or creates a random UUID if no input is provided.

    If `input_string` is provided, it generates a UUID based on the MD5 hash of the string. 
    Otherwise, it returns a random UUID.

    Args:
        input_string (str, optional): Input string to convert to a UUID. Defaults to None.

    Returns:
        uuid.UUID: The generated UUID.
    """
    if input_string is None:
        return str(uuid.uuid4())
    return uuid.UUID(hashlib.md5(input_string.encode('UTF-8')).hexdigest())

def pil_image_to_base64(image):
    """
    Converts a PIL image to a base64-encoded string in PNG format.

    Args:
        image (PIL.Image): The input image to encode.

    Returns:
        str: Base64-encoded string of the image in PNG format.
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")  # You can change the format if needed (JPEG, etc.)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

#### IMAGE + TEXT FUSION ROUTINES

def cosim_eval(clip_processor, clip_model, images, texts, device):
    """
    Evaluates the similarity between images and text descriptions using CLIP embeddings and cosine similarity.

    The function calculates the similarity between the image and text features generated by a CLIP model. 
    It processes images and text to create feature vectors, then computes cosine similarity scores.

    Args:
        images (list of PIL.Image): List of images for similarity evaluation.
        texts (list of str): List of text descriptions for each image.

    Returns:
        torch.Tensor: Cosine similarity scores between image and text features, one score per image-text pair.
    
    Example:
        >>> cosim_eval([image1, image2], ["A cat on a mat", "A dog in the park"])
        tensor([0.876, 0.543])

    Notes:
        - Uses the `clip_processor` and `clip_model` for image and text feature extraction.
        - `cosine_similarity` calculates the similarity scores.
    """
    # evaluate the generated text by comparing its similarity with flux generated image 
    inputs = clip_processor(images=images, return_tensors="pt")
    clip_vision_output = clip_model.vision_model(**inputs)
    image_features = clip_model.visual_projection(clip_vision_output["pooler_output"])

    inputs = clip_processor(texts, padding=True, truncation=True, max_length=76, return_tensors="pt").to(device)
    text_features = clip_model.get_text_features(**inputs)
    cos_scores = cosine_similarity(image_features, text_features, dim=1)

    return cos_scores



junk_str = "‒>:‘~:’•:\\-)-:̊ˆ··—‘–;'✎\"·§꧊꧌꧍꧁꧂꧇꧃꧋꧆꧉,{}[]()|\\\"'“”《》«»~!@#$%^&*{}[]()_–+=-0987654321<`>,、،./?':;“”\"\t\n\\πه☆●¦″．۩۱（☛₨➩°・■↑☻、๑º‹€σ٪’Ø·−♥ıॽ،٥《‘©。¨﴿！★×✱´٬→±x：¹？£―▷ф¡Г♫∟™ª₪®▬「—¯；¼❖․ø•」٣，٢◦‑←§١ー٤）˚›٩▼٠«¢¸٨³½˜٭ˈ¿¬ι۞⌐¥►†ƒ∙²»¤…﴾⠀》′ا✓→¶'"
junk = set(junk_str)
strip_chars = "".join(list(set(junk_str+string.punctuation +"\r\n  ,،、_–+=-{}[]|()\"'“”《》«»!:;¿?。゜。…．꧈꧉꧋ꧏ|\\1234567890~`!@#$%^&*()-_+=:;\"'<>,.?/")))

### WN/Ontology Stuff
def wn_collapse_lemmas(synsets):
    return [str(a) for a in list(set(list(itertools.chain(*[a.lemmas() for a in synsets]))))]

def wn_get_parents(w, senseno=0):
    hypernyms = []
    synsets = wn.synsets(w)
    if senseno >= len(synsets):
        return []
    current_synset = synsets[senseno] 
    lemmas = current_synset.lemmas()
    if len(lemmas) >=2 :
        lemmas  = lemmas[:2]
    if w not in lemmas:
        return []
    while current_synset:
        hypernyms.append (current_synset.lemmas()[0])
        current_synset = current_synset.hypernyms()
        if current_synset:
            current_synset = current_synset[0]
    return (hypernyms)

def wn_path_intersect(p1, p2):
    if not p1 or not p2: return None
    for idx, lemma in enumerate(p2):
        if lemma in p1:
            if lemma == p2[-1]:
                return lemma
            idx2 = p1.index(lemma)
            if idx2 == len(p1)-1:
                return lemma
            if p2[idx+1] == p1[idx2+1]:
                return lemma+" ("+p1[idx2+1]+")"
    return None

def wm_get_common_parent(w1,w2):

    parents1a = wn_get_parents(w1, 0)
    parents1b = wn_get_parents(w1, 1)
    if not parents1a and not parents1b and " " in w1:
        w1 = w1.split()[-1]
        parents1a = wn_get_parents(w1, 0)
        parents1b = wn_get_parents(w1, 1)
    if not parents1a and not parents1b and w1 != w1.lower():
        w1 = w1.lower()
        parents1a = wn_get_parents(w1, 0)
        parents1b = wn_get_parents(w1, 1)
    parents2a = wn_get_parents(w2, 0)
    parents2b = wn_get_parents(w2, 1)
    if not parents2a and not parents2b and " " in w2:
        w2 = w2.split()[-1]
        parents2a = wn_get_parents(w2, 0)
        parents2b = wn_get_parents(w2, 1)
    if not parents2a and not parents2b and w2 != w2.lower():
        w2 = w2.lower()
        parents2a = wn_get_parents(w2, 0)
        parents2b = wn_get_parents(w2, 1)
    i = wn_path_intersect(parents1a, parents2a)
    if i: return i
    i = wn_path_intersect(parents1a, parents2b)
    if i: return i
    i = wn_path_intersect(parents1b, parents2a)
    if i: return i
    i = wn_path_intersect(parents1b, parents2b)
    if i: return i
    return None


def synonym_textaugment(text, prob_replace=0.5, lang="en"):
    global wn
    stopwords =  all_stopwords.get(lang, all_stopwords['en'])        
    def collapse_lemmas(synsets):
        return [str(a) for a in list(set(list(itertools.chain(*[a.lemmas() for a in synsets]))))]    
    for w in text.split(" "):
        w = w.strip(strip_chars)
        if w.lower() in stopwords: continue
        if len(w) > 7 and w.lower() == w and random.random() <= prob_replace:
            lemmas = collapse_lemmas(wn.synsets(w))
            if len(lemmas) > 5:
                a = lemmas[:5]
                random.shuffle(a)
                b = lemmas[5:]
                random.shuffle(b)
                lemmas = a + b
            else:
                random.shuffle(lemmas)
            found = False
            for l in lemmas:
                if l != w and l == l.lower() and len(l) > 7 and len(l)/len(w) < 1.1 and w[-1] == l[-1] and not ((w.endswith("ly") or w[-1] == "t") and "pregnant" in l):
                    text = text.replace(" "+w+" ", " "+l+" ")
                    text = text.replace(w+" ", l+" ")                    
                    text = text.replace(" "+w, " "+l)
                    found = True
                    break
            if not found:
                for l in lemmas:
                    if l != w and l == l.lower() and len(l) > 7 and len(l)/len(w) < 1.1 and w[-1] == l[-1]:
                        text = text.replace(" "+w+" ", " "+l+" ")
                        text = text.replace(w+" ", l+" ")                    
                        text = text.replace(" "+w, " "+l)
                        break
                
    return text


def analyze_nouns_common_hypernyms(text, all_noun_phrases=[], lang="en"):
    global spacy_nlp, spacy_multi
    stopwords =  all_stopwords.get(lang, all_stopwords['en'])        
    if not all_noun_phrases:
        if lang == "en":
            doc = spacy_nlp(text)
        else:
            doc = spacy_multi(text)
        all_noun_phrases = [strip_left_stopwords(e.text, lang=lang) for e in doc.noun_chunks if len(e.text) > 4 and e.text.lower() not in stopwords]

    noun_synsets = {}
    for noun in all_noun_phrases:
        synsets = wn.synsets(noun, pos='n')
        if synsets:
            noun_synsets[noun] = synsets[0]

    hypernym_map = {}
    noun_list = sorted(noun_synsets.keys())
    for i in range(len(noun_list)):
        for j in range(i+1, len(noun_list)):
            syn_i = noun_synsets[noun_list[i]]
            syn_j = noun_synsets[noun_list[j]]
            lch = syn_i.lowest_common_hypernyms(syn_j)
            if lch:
                for common_hyp in lch:
                    name = str(common_hyp.lemmas()[0].name()).replace(" ","_")
                    if name in {"act", "whole", "communication", "thing", "content", "group", "social_group", "unit", "organism", "region", "area", "geographic_point", "relation", "attribute", "happening", "action", "entity", "causal_agent", 'administrative_district', "abstraction", "object", "point", "event", "physical_entity", "matter", "part", "person", "state", 'cognition', 'psychological_feature'}:
                        continue
                    hypernym_map.setdefault(name, set()).update([noun_list[i], noun_list[j]])

    hypernym_map = {
        hyp: list(nset) for hyp, nset in hypernym_map.items() if len(nset) >= 2
    }

    return hypernym_map


### More misc

def generate_trigrams(sentence):
    words = sentence.split()
    trigrams = [" ".join([words[i], words[i+1], words[i+2]]) for i in range(len(words) - 2)]    
    return trigrams

other_regexes = {
    'ADDRESS': re.compile(r'\d{1,6} [\w\S]{4,20}\s[A-Za-z]+', re.IGNORECASE),
    'ID': re.compile(r'\d{3}-\d{2}-\d{4}', re.IGNORECASE),
    'DATE': re.compile(r'[1|2]\d\d\d-\d\d|[1|2]\d\d\d-\d\d|[1|2]\d\d\d', re.IGNORECASE),    
    'LICENSE_PLATE': re.compile(r'[A-Z]{3}-\d{4}|[A-Z]{1,3}-[A-Z]{1,2}-\d{1,4}'),
    'USER': re.compile(r'@[_A-Za-z0-9]+')}

dateparser = None

public_people = None

# perform basic NER. when dealing with people, look up in a
# public_people list and label them
def pii_detect(text, use_spacy=True, strict_org=True, lang="en"):
    global spacy_nlp, spacy_multi, names, names_list, public_people, public_titles, people_first_name_sanity_check, dateparser
    stopwords =  all_stopwords.get(lang, all_stopwords['en'])        
    if dateparser is None:
        dateparser =  Parser()
    def guess_public_people(val, is_person=False):
        if " " not in val: return None
        val_arr = val.split()
        if val_arr[-1] in fac_set: return None
        a = val_arr[0]
        b = val_arr[-1]
        if len(val_arr) > 2 and len(b) < 4 and len(val_arr[-2]) >= 4:
            b = val_arr[-2]
        if a[0][0] == a[0][0].upper() and b[0][0] == b[0][0].upper():
            name0 = val
            if name0.endswith("'s"):
                name0 = name0.replace("'s", "")
            name1 = (a.strip(".,") if len(a) < 6 else a[:6])+"_"+(b.strip(".,") if len(b) < 6 else b[:6])
            if (is_person and a.lower() in public_titles) or (name1.lower() in public_people and name0 not in seen):
                val = name0
                key = "PUBLIC_PEOPLE_"+str(len(ner))
                return (val, key)
            elif len(val_arr) > 2 and len(b) < 4 and len(val_arr[-2]) >= 4:
                    b = val_arr[1]
                    name0 = (a + " " + b).strip(".,")
                    if name0.endswith("'s"):
                        name0 = name0.replace("'s", "")
                    name1 = (a.strip(".,") if len(a) < 6 else a[:6])+"_"+(b.strip(".,") if len(b) < 6 else b[:6])
                    if name1.lower() in public_people and name0 not in seen:
                        val = name0
                        key = "PUBLIC_PEOPLE_"+str(len(ner))
                        return (val, key)
        return None
    if use_spacy and (spacy_nlp is None or spacy_multi is None):
        spacy_nlp = spacy.load('en_core_web_sm')
        spacy_multi = spacy.load('xx_ent_wiki_sm')

    if public_people is None:
        public_names = json.load(open(os.path.abspath(os.path.dirname(__file__))+"/"+"public_people.json"))
        public_people = public_names['people'] = set(public_names['people'])
        public_titles = public_names['people'] = set(public_names['titles'])        
        people_first_name_sanity_check = public_names['people_first_name_sanity_check']
        public_names = None
        public_names = None

    orig_text = text = text.replace("{", "--[--").replace("}", "--]--")
    
    #let's do a sanity check. there should be no words beyond 100 chars.
    #this will really mess up our regexes. so we do regex on a copy called 'orig_text.'
    for word in orig_text.split(" "):
        len_word = len(word)
        if len_word > 100:
            orig_text = orig_text.replace(word, " "*len_word)

    # parse dates
    ner = {}
    seen = {}
    for val in orig_text.split():
        val = val.split("<|")[0].strip()                
        if val and val[0] in "123456789" and len(val) > 4 and any(dateparser.parse(val)):
            tag = "DATE_"+str(len(ner))
            ner[tag] = val
            text = text.replace(val, "{"+tag+"}")

    # basic regex
    parsed_text = CommonRegex(orig_text)
    for key in commonregex.regexes.keys():
        if key == "hex_color": continue
        if hasattr(parsed_text, key):
            if "ssn" in key or "credit" in key or "btc" in key:
                key2 = "ID"
            elif "address" in key:
                key2 = "ADDRESS"
            elif "price" in key:
                key2 = "MONEY"
            else:
                key2 = key
            if key2[-1] == 's':
                key2 = key2[:-1]
            key2 = key2.upper()
            for val in  getattr(parsed_text, key):
                val = val.split("<|")[0].strip()                        
                if "\n" in val or "}" in val or ".." in val: continue
                if len(val.strip("-+_:,.;# "))<2: continue
                if val in seen: continue
                if key2 == "ADDRESS" and not any(a for a in val.split() if a.strip(",.").lower() in rd_list):
                    continue
                if val.count("-") ==1 and key2 == "PHONE":
                    t1, t2 = val.split("-")
                    if t1 not in seen and t2 not in seen and any(dateparser.parse(t1)) and any(dateparser.parse(t2)):
                        seen[t1] = 1
                        ner["DATE_"+str(len(ner))] = t1                            
                        seen[t2]  = 1
                        ner["DATE_"+str(len(ner))] = t2
                        continue
                seen[val] = 1
                ner[key2.upper()+"_"+str(len(ner))] = val
    tags = list(ner.items())
    tags.sort(key=lambda a: len(a[1]), reverse=True)
    for tag, val in tags:
        val = val.split("<|")[0].strip()        
        if " "+val in text or val+" " in text or "\n"+val in text or  val+"\n" in text:
            text = text.replace(" "+val, " {"+tag+"}")
            text = text.replace(val+" ", "{"+tag+"} ")            
            text = text.replace("\n"+val, "\n{"+tag+"}")
            text = text.replace(val+"\n", "{"+tag+"}\n")            
            seen[val] = 1

    candidate_nouns = []
    if use_spacy and spacy_nlp:
        if len(orig_text) > 5000:
            spacy_orig_text = orig_text[:5000]
        else:
            spacy_orig_text = orig_text
        if lang == "en":
            doc = spacy_nlp(orig_text)
        else:
            doc = spacy_multi(orig_text)
        candidate_nouns = [strip_left_stopwords(e.text, lang=lang)  for e in doc.noun_chunks if len(e.text) > 4 and e.text.lower() not in stopwords]
        
        items = [(ent.text, ent.label_) for ent in doc.ents]
        items.sort(key=lambda a: len(a[0]), reverse=True)
        for val, label in items:
            val = val.split("<|")[0].strip()
            if strict_org and label == "ORG" and not ("Inc" in val or "LLC" in val or "GmBh" in val or "Ltd" in val or "Corp" in val or "Group" in val):
                continue
            if ("Inc" in val or "LLC" in val or "GmBh" in val or "Ltd" in val or "Corp" in val or "Group" in val):
                label = "ORG"
            if label in {"LAW", "WORK_OF_ART"} and " " not in val:
                continue
            val_lower = val.lower()
            if "MONEY" in label and ("year" in val_lower or "month"in val_lower or "day" in val_lower):
                label = "DATE"
            if label in {"GPE", "PERSON", "NORP"} and (" " not in val or "vitamin" in val_lower or "oxy" in val_lower or "bio" in val_lower or "geo" in val_lower or "thermo" in val_lower or "chrono" in val_lower or "ology" in val_lower):
                continue
            if len(val.strip("-+_:,.;# "))<2: continue            
            if "##" in val: continue
            if val in ner_ignore: continue
            if "\n" in val or not(" " in val or val.upper() == val or len(val) > 10): continue
            if val.endswith(":"): continue
            try:
                float(val.strip("_}"))
                continue
            except:
                pass
            if val in seen or val not in text: continue
            key = label+"_"+str(len(ner))
            if label in {"ORG", "PERSON"} and val[0] != val[0].upper(): continue
            if label in {"PERSON",} and len(val) < 5: continue
            if label in {"ORG", "PERSON"} and val.endswith("'s"):
                val = val.replace("'s", "")
            if label == "PERSON" and val.split()[-1] in fac_set:
                label = "FAC"
            if " " in val and label == "PERSON":
                is_public_people = guess_public_people(val, True)
                if is_public_people:
                    val, key = is_public_people
                elif not any(a for a in val.split() if (a.strip(".,") if len(a) < 6 else a[:6]).lower()  in people_first_name_sanity_check):
                    continue
            val = strip_right_stopwords(strip_left_stopwords(val, lang=lang), lang=lang)
            if val:
                ner[key] = val
                seen[val] = 1
                text = text.replace(" "+val, " {"+key+"}")
                text = text.replace(val+" ", " {"+key+"} ")            
    for val in list(set(generate_trigrams(orig_text) + candidate_nouns)):
        val = val.split("<|")[0].strip()        
        if not val.strip("-+_:,.;"): continue
        if val in seen or val not in text: continue
        if len(val) > 3 and not any(a in seen for a in val.split()) and any(dateparser.parse(val)):
            tag = "DATE_"+str(len(ner))
            ner[tag] = val
            text = text.replace(val, " {"+tag+"}")
            seen[val] = 1
            continue
        is_public_people = guess_public_people(val)
        if is_public_people:
            val, tag = is_public_people
            val = strip_right_stopwords(strip_left_stopwords(val, lang=lang), lang=lang)
            if val:
                ner[tag] = val
                text = text.replace(" "+val, " {"+tag+"}")
                text = text.replace(val+" ", "{"+tag+"} ")            
                seen[val] = 1
            continue
        
    for label, regex in other_regexes.items():
        for val in regex.findall(orig_text):
            val = val.split("<|")[0].strip()            
            val = val.split("}")[-1].strip()
            if not val.strip("-+_:,.;"): continue
            try:
                float(val)
                continue
            except:
                pass
            if len(val) > 3 and not any(a in seen for a in val.split()):
                if label == "ADDRESS" and not any(a for a in val.split() if a.strip(",.").lower() in rd_list):
                    continue
                if val in seen or val not in text: continue
                seen[val] = 1                    
                key = label.upper()+"_"+str(len(ner))
                ner[key] = val
                text = text.replace(" "+val, " {"+key+"}")
                text = text.replace(val+" ", " {"+key+"} ")
    
    for val in orig_text.split():
        val = val.split("<|")[0].strip()        
        val = val.strip(strip_chars)
        if len(val) > 4 and (val[0].upper() == val[0] and len(val) > 4) and (val in names or val.lower() in names and val.lower() not in stopwords) and val not in common_word_or_public_figure_names and val not in states_of_usa and val not in brand_names:
            if val in seen or val not in text: continue
            seen[val] = 1                    
            key = "PERSON_"+str(len(ner))
            ner[key] = val
            text = text.replace(" "+val, " {"+key+"}")
            text = text.replace(val+" ", "{"+key+"} ")
        elif len(val) > 4 and (val[0].upper() == val[0] and len(val) > 4) and (val in regions_set):
            if val in seen or val not in text: continue
            seen[val] = 1                    
            key = "NORP_"+str(len(ner))
            ner[key] = val
            text = text.replace(" "+val, " {"+key+"}")
            text = text.replace(val+" ", "{"+key+"} ")
            
    # any person whose name is part of another NER like a work of art can be considered a public person
    for key, val in list(ner.items()):
        val = val.split("<|")[0].strip()        
        if key not in ner: continue
        if "PERSON" in key:
            found = False
            for key2, val2 in list(ner.items()):
                if found: break
                if val == val2: continue
                if key2 not in ner: continue
                if "PERSON" not in key2:
                    if val2 in " "+val+" ":
                        if key in ner: del ner[key]
                        ner[key.replace("PERSON", "PUBLIC_PEOPLE")] = val
                        if key2 in ner: del ner[key2]
                        text = text.replace("{"+key+"}", "{"+key.replace("PERSON", "PUBLIC_PEOPLE")+"}")                        
                        text = text.replace("{"+key2+"}", "{"+key.replace("PERSON", "PUBLIC_PEOPLE")+"}")
                        found = True                        
                    elif val in " " + val2+" ":
                        if "PUBLIC" in key2:
                            if key in ner: del ner[key]
                            text = text.replace(key, key2)
                            found = True                            
                        else:
                            ner[key.replace("PERSON", "PUBLIC_PEOPLE")] = val
                            if key in ner: del ner[key]
                            text = text.replace("{"+key+"}", "{"+key.replace("PERSON", "PUBLIC_PEOPLE")+"}")
                            found = True
                            
    for key, val in list(ner.items()):
        val = val.split("<|")[0].strip()        
        if key not in ner: continue
        if "PERSON" in key or "PUBLIC" in key or "FAC" in key or "GPE" in key or "ORG" in key or "LOC" in key:
            for v in val.split():
                if v[0] == v[0].upper() and len(v) > 4 and v not in seen:
                    if " "+v+" " in text:
                        seen[v] = 1
                        text = text.replace(" "+v+" ", " {"+key+"} ")
            
    for key, val in list(ner.items()):
        if "WORK_OF_ART" in key:
            del ner[key]
            new_key = key.replace("WORK_OF_ART", "CONTENT_OR_REGULATION")
            ner[new_key] = val
            text = text.replace("{"+key+"}", "{"+new_key+"}")

    for key, val in list(ner.items()):        
        if "LAW" in key:
            del ner[key]
            new_key = key.replace("LAW","CONTENT_OR_REGULATION")
            ner[new_key] = val
            text = text.replace("{"+key+"}", "{"+new_key+"}")

            
    for key, val in list(ner.items()):        
        val = val.split("<|")[0].strip()        
        if "{"+key+"}" not in text:
            del ner[key]
            continue
        if ("old" in val or " age" in val or "age " in val)  and "DATE" in key:
            del ner[key]
            new_key = key.replace("DATE", "AGE")
            ner[new_key] = val
            text = text.replace("{"+key+"}", "{"+new_key+"}")
    text = text.replace("}e ", "} ").replace("}s ", "} ").replace("}es ", "} ").replace("}d ", "} ").replace("}ed ", "} ")
    text_arr = text.split(" ")
    text_arr2 = []
    prev_t = ""
    for t in text_arr:
        if not t.strip(): continue
        if t == prev_t and t[0] == "{" and t[-1] == "}":
            continue
        prev_t = t
        text_arr2.append(t)
    text = " ".join(text_arr2)
    all_nouns=list(set(list(ner.values())+candidate_nouns))
    return ner, all_nouns, text


faker_list = [
    'ar_AA',
    'ar_PS',
    'ar_SA',
    'bg_BG',
    'cs_CZ',
    'de_AT',
    'de_CH',
    'de_DE',
    'dk_DK',
    'el_GR',
    'en_GB',
    'en_IE',
    'en_IN',
    'en_NZ',
    'en_TH',
    'en_US',
    'es_CA',
    'es_ES',
    'es_MX',
    'et_EE',
    'fa_IR',
    'fi_FI',
    'fr_CA',
    'fr_CH',
    'fr_FR',
    'fr_QC',
    'ga_IE',
    'he_IL',
    'hi_IN',
    'hr_HR',
    'hu_HU',
    'hy_AM',
    'id_ID',
    'it_IT',
    'ja_JP',
    'ka_GE',
    'ko_KR',
    'lt_LT',
    'lv_LV',
    'ne_NP',
    'nl_NL',
    'no_NO',
    'or_IN',
    'pl_PL',
    'pt_BR',
    'pt_PT',
    'ro_RO',
    'ru_RU',
    'sl_SI',
    'sv_SE',
    'ta_IN',
    'th_TH',
    'tr_TR',
    'tw_GH',
    'uk_UA',
    'zh_CN',
    'zh_TW']

faker_map = {}

for faker_lang in faker_list:
  lang, _ = faker_lang.split("_")
  faker_map[lang] = faker_lang

lang2faker = {} # 'en': Faker("en_US")}

# use extended_anonymize when creating synthetic data to create more variations
def pii_anonymize(ner, template, lang="en", extended_anonymize=False, female_only=False, male_only=False, do_person=False, do_public_people=False, do_gender_swap=False, do_strict_id=False):
    global lang2faker, faker_map, names, names_list, public_people
    if not lang2faker:
        lang2faker = {'en': Faker("en_US")}
    def randomize_nums(ent):
        ent = list(ent)
        for i in range(len(ent)):
            if ent[i] in "0123456789":
                ent[i] = str(random.randint(0,9))
        return "".join(ent)
    if public_people is None:
        public_names = json.load(open(os.path.abspath(os.path.dirname(__file__))+"/"+"public_people.json"))
        public_people = public_names['people'] = set(public_names['people'])
        people_first_name_sanity_check = public_names['people_first_name_sanity_check']
        public_names = None
    
    if do_gender_swap:
        template = gender_swap(template)
        if not female_only:
            female_only=("He " not in template and " he " not in template and " his " not in template and " him " not in template )
    
    anon_text = template
    if not ner:
        return ner, anon_text
    if lang2faker.get(lang) is None:
        try:
            lang2faker[lang] = Faker(faker_map[lang])
        except:
            lang2faker[lang] = Faker("en_US")
    faker  = lang2faker[lang]
    new_ner = copy.copy(ner)
    old2new = {}
    company = ""
    items = [[a,b] for a, b in ner.items()]
    items.sort(key=lambda a: len(a[1]))
    for k, item in enumerate(items):
      tag, ent = item
      ent2 = None
      if not ent: continue
      if ent in old2new:
        ent2 = old2new[ent]
      elif (do_person and "PERSON" in tag) or (do_public_people and "PUBLIC" in tag):
        ent2 = faker.name()
        ent2_arr = ent2.split()
        if female_only:
            name2 = random.choice(female_names)
        elif male_only:
            name2 = random.choice(male_names)
        elif random.randint(0,1):
            name2 = random.choice(names_list)
        else:
            name2 = None
        if name2 and name2[0].lower() in "qwertyuiopasdfghjklzxcvbnm":
            if random.randint(0,1) or  female_only or male_only:
                ent2_arr[0] = name2[0].upper() + name2[1:]
            else:
                ent2_arr[-1] = name2[0].upper() + name2[1:]
            ent2 = " " .join(ent2_arr)
        if " " not in ent:
            ent2 = ent2.split()[0]
      elif "ADDRESS" in tag:
        ent2 = faker.address().replace("\n", " ")
      elif "ID" in tag:
        # if this is just an int/float, then we should skip this potential ID
        if do_strict_id:
            is_id = True
            try:
                float(ent)
            except:
                is_id = False
        else:
            is_id = True
        if is_id:
            ent2 = str(random.randint(0,100)) + randomize_nums(ent)+ str(random.randint(0,100))
            if len(ent2) > 5:
                ent2 = ent2[:5]
      elif "CREDIT_CARD" in tag:
         ent2 = faker.credit_card()
      elif 'IPS' in tag or "IPV" in tag:
         ent2 = faker.ipv4()
      elif "PHONE" in tag:
         ent2 = faker.phone_number()
         if random.randint(0,1):
             ent2 = ent2.split("+1-",1)[-1]
         if random.randint(0,1):
             ent2 = ent2.split("-")[-1]
         if random.randint(0,1):
             ent2 = ent2.split("x")[0]
         if random.randint(0,1):
             ent2 = ent2.split("x")[-1]
      elif 'LICENSE_PLATE' in tag:
         ent2 = faker.license_plate()
      elif  "USER" in tag:
         ent2 = "@"+faker.company_email().split("@")[0]
      elif "EMAIL" in tag:
         ent2 = faker.company_email()
         if company:
             ent2 = ent2.split("@")[0]+"@"+company.split()[0].lower()+"."+ent2.split(".")[-1]
      elif "NORP" in tag:
          ent2 = random.choice(nationality_and_region)

             
      if extended_anonymize:
          if "LINK" in tag or "URL" in tag:         
             ent2 = "https://"+faker.domain_name(2)
          elif "ORG" in tag:
            ent2 = faker.company()
            company = ent2
          elif "GPE" in tag:
            ent2 = faker.country()
          elif "LOC" in tag:
            ent2 = faker.state()
            if random.randint(0,1):
                ent2 = ent2 + " " + ent.split()[-1]                
          elif "EVENT" in tag:
            if random.randint(0,1):
                if company:
                    ent2 = company
                else:
                    ent2 = faker.company().replace("Inc.", "").replace("LLC", "")
            else:
                ent2 = faker.country()
            ent2 = ent2 + " " + ent.split()[-1]
          elif "FAC" in tag:
            if company:
                ent2 = company
            else:
                ent2 = faker.company().replace("Inc.", "").replace("LLC", "")
            if ent.split()[-1] in fac_set:
                ent2 = ent2 + " " + ent.split()[-1]
            else:
                ent2 = ent2 + " " + random.choice(fac_list)
      if ent2:
         new_ner[tag] = ent2
         old2new[ent] = ent2
         if len(ent2) > 5 and len(ent.strip("1234567890/-")) > 5:
             # sometimes entities share words, like first_name, last_name
             for item in items[k:]:
                 if not item[1]: continue
                 if ent != item[1] and " "+ent+" " in " "+item[1]+" ":
                     ent3 = item[1].replace(ent, ent2)
                     new_ner[item[0]] = ent3
                     old2new[item[1]] = ent3
                     item[1] = None
                     
      #TODO: NORP, AGE, DISEASE, GENDER, JOB, MEDICAL_THERAPY
    #try:
    anon_text = template.format(**new_ner)
    #except Exception as e:
    #    anon_text = template
    return new_ner, anon_text

def gender_swap(text, ratio_female_swap=0.7, female_to_male=False, male_to_female=False):
  if female_to_male:
      swap_dict = female_to_male_gender_swap
  elif random.random() <= ratio_female_swap or male_to_female:
      swap_dict = male_to_female_gender_swap
  else:
      swap_dict = female_to_male_gender_swap      
  text2 = text
  text2 = " ".join([t.replace(t.strip(",.;:\'\""), swap_dict.get(t.strip(",.;:\'\""),t.strip(",.;:\'\""))) for t in text2.split(" ")])
  text2 = text2.replace(" father or father ", " father ").replace(" bride or bride ", " bride ").replace(" groom or groom ", " groom ").replace(" mother or mother ", " mother ").replace(" hers or hers ", " hers ").replace(" her or her ", " her ").replace(" his or his ", " his ").replace(" woman or woman ", "woman ").replace(" man or man ", " man ")
  text2 = text2.replace(" father and father ", " father ").replace(" bride and bride ", " bride ").replace(" groom and groom ", " groom ").replace(" mother and mother ", " mother ").replace(" hers and hers ", " hers ").replace(" her and her ", " her ").replace(" his and his ", " his ").replace(" woman and woman ", "woman ").replace(" man and man ", " man ")
  if text2 != text:
      text = text2
  return text

def is_minor_age(text):
    if "month" in text: return True
    text = text.replace("-", " ").split()
    for t in text:
        try:
            if int(t) < 18: return True
        except:
            pass
    return False
      


def basic_safety_processing(text, is_image_caption=False, cam_flagged_threshold= 0.05, fine_grain=False, do_pii=True, use_spacy=True, additional_nsfw_words=[], create_noun_hypernyms=False, lang="en"):
      #  English based, fast safety checker and processor using keywords and other non-AI model techniques (e.g., regex and heuristics)
      #  returns
      #   - the safety issue as a list of issues.
      #   - a rough score
      #   - keywords matched based on the safety issues,
      #   - any pii matched as an ner dict
      #  a pii issue is when there is (A) id, email, key, ip address, user_id, or license plate or (B) a non-public figure name, and other pii element like a phone, address, org, loc
      # NOTE: We try to find child abuse materials, not just CSAM. So violence involving children could be included in a CAM (child abuse material) flag
      # by default, the issues returned will be a list, CAM by itself or a list of one or more of INT_PROP, PII and/or NSFW.
      # we recommend removing all items marked with CSAM even if there are false positives. And dealing with the other issues based on the score.
      # CAM can be kept to improve detection and dealing with this type of content.
    
      # if fine_grain is set to True, safety labels can include Alcohol, Tobacco and Gambling (vice), NSFW, Drugs, INT_PROP, PII, Hate, Religious Offense, Harm, Crimes, CNBR, CAM, CSAM
      # but since this is an keyword/regex based method, the accuracy for these tags might not be high.
      stopwords =  all_stopwords.get(lang, all_stopwords['en'])        
      def word_match(a, lst):
         if a == "spices" and "spic" in lst:
             return None
         if a == "assess" and "asses" in lst:
             return None
         return (a.lower() in lst or (len(a) > 2 and a[-1] == 's' and (a[:-1].lower() in lst or a[:-2].lower() in lst)))
      # START
      if not text.strip(): return [], 0.0, [], {}, ""
      orig_text = text
      if do_pii:
          ner, all_noun_phrases, template = pii_detect(orig_text, use_spacy=use_spacy)
      else:
          ner = {}
          all_noun_phrases=[]
          template = ""
      if create_noun_hypernyms:
          noun_hypernyms=  analyze_nouns_common_hypernyms(text, all_noun_phrases)
      else:
          noun_hypernyms = {}
      text = text.replace("  ", " ").strip()

      #don't strip {} or []
      text_stripped = [w.strip(strip_chars) for w in text.split() if  w.strip(strip_chars)]

      pii_cnt = 0
      matched_pii_words = []
      for key, val in list(ner.items()):
          if  "PHONE" in key or "ID" in key or "EMAIL" in key or "IP" in key or "LICENSE_PLATE" in key or "USER" in key or "KEY" in key:
              pii_cnt += 1
              matched_pii_words.append(val)
      for key, val in list(ner.items()):              
          if "PERSON" in key and " " in val:
              if pii_cnt or any(a for a in ner.keys() if "PHONE" in key):
                  matched_pii_words.append(val)
                  pii_cnt += 1
          if "PHONE" in key and " " in val:                  
              if pii_cnt or any(a for a in ner.keys() if "PERSON" in key):
                  matched_pii_words.append(val)
                  pii_cnt += 1 
              #matched_pii_words.append(val)
      text = " "+" ".join(text_stripped).lower()+" "
      matched_harm_words = [a for a in text_stripped if word_match(a, harmwords)]
      harm_cnt = len(matched_harm_words)
      if not harm_cnt:
          matched_harm_words = [a for a in harmwords_list if " "+a+" " in text]
          harm_cnt = len([a for a in harmwords_list if " "+a+" " in text])          
      matched_hate_words = [a for a in text_stripped if word_match(a, hatewords)]
      hate_cnt = len(matched_hate_words)
      if not hate_cnt:
          matched_hate_words = [a for a in hatewords_list if " "+a+" " in text]
          hate_cnt = len([a for a in hatewords_list if " "+a+" " in text])          
      matched_vice_words = [a for a in text_stripped if word_match(a, vicewords)]
      vice_cnt = len(matched_vice_words)
      if not vice_cnt:
          matched_vice_words = [a for a in vicewords_list if " "+a+" " in text]
          vice_cnt = len([a for a in vicewords_list if " "+a+" " in text])          
      matched_minor_words = [a for a in text_stripped if word_match(a, minorwords)]          
      minor_cnt = len(matched_minor_words)
      if not minor_cnt:
          matched_minor_words = [a for a in minorwords_list if " "+a+" " in text]
          minor_cnt = len([a for a in minorwords_list if " "+a+" " in text])
      matched_minor_words.extend([val for key, val in ner.items() if "AGE" in key and is_minor_age(val)])
      minor_cnt = len(matched_minor_words)
      matched_csam_words = matched_minor_words
      potential_csam_cnt = minor_cnt
      if not potential_csam_cnt:
          matched_csam_words = matched_minor_words
          potential_csam_cnt = minor_cnt
      matched_cybercrime_words = [a for a in text_stripped if word_match(a, cybercrimewords)]
      cybercrime_cnt = len(matched_cybercrime_words)
      if not cybercrime_cnt:
          matched_cybercrime_words = [a for a in cybercrimewords_list if " "+a+" " in text]
          cybercrime_cnt = len([a for a in cybercrimewords_list if " "+a+" " in text])          
      matched_crime_words = [a for a in text_stripped if word_match(a, crimewords)]
      crime_cnt = len(matched_crime_words)
      if not crime_cnt:
          matched_crime_words = [a for a in crimewords_list if " "+a+" " in text]
          crime_cnt = len([a for a in crimewords_list if " "+a+" " in text])          
      matched_drugs_words = [a for a in text_stripped if word_match(a, drugswords)]
      drugs_cnt = len(matched_drugs_words)
      if not drugs_cnt:
          matched_drugs_words = [a for a in drugswords_list if " "+a+" " in text]
          drugs_cnt = len([a for a in drugswords_list if " "+a+" " in text])          
      matched_sex_words = [a for a in text_stripped if word_match(a, sexwords)]
      sex_cnt = len(matched_sex_words)
      if not sex_cnt:
          matched_sex_words = [a for a in sexwords_list if " "+a+" " in text]
          sex_cnt = len([a for a in sexwords_list if " "+a+" " in text])
      matched_nsfw_words = [a for a in text_stripped if word_match(a, nsfwwords)] +  [a for a in text_stripped if word_match(a, additional_nsfw_words)]
      nsfw_cnt = len(matched_nsfw_words)
      if not nsfw_cnt:
          matched_nsfw_words = [a for a in nsfwwords_list if " "+a+" " in text]
          nsfw_cnt = len([a for a in nsfwwords_list if " "+a+" " in text])
      
      matched_nsfw_words.extend(matched_sex_words)
      nsfw_cnt += sex_cnt
      matched_words = matched_pii_words + matched_harm_words + matched_hate_words + matched_vice_words +  matched_cybercrime_words + matched_crime_words + matched_drugs_words + matched_nsfw_words      
      safety_issue = []
      if vice_cnt or ('vape' in text and 'smoke' in text):
        safety_issue.append('Alcohol, Tobacco and Gambling')
      
      if nsfw_cnt:
        safety_issue.append('NSFW')
      # nsfw_cnt words for NSFW
      if (' dick' in text and 'hard' in text):
          nsfw_cnt += 1
          sex_cnt += 1
          safety_issue.append('NSFW')

      # drug crime
      if drugs_cnt:
        if  ' mj ' in text in text:
            matched_words.append('mj')
        safety_issue.append('Drugs')

      if (text.count('razor') > 0 and (text.count('white powder') + text.count('syringe'))) or \
            ((text.count(' drugs') + text.count('white powder')) > 2):
          matched_words.extend([w for w in text.split() if ('drug' in w or 'razor' in w or 'syringe' in w)])          
          if 'white powder' in text: matched_words.append('white powder')
          safety_issue.append('Drugs')

      if any(w for w in text_stripped if w in fictional_characters and w.lower() not in stopwords) or  any(a for a in brand_names_list if " "+a+" " in text):
          matched_words.extend([w for w in text_stripped if w in fictional_characters])
          matched_words.extend([a for a in brand_names_list if " "+a+" " in text])
          safety_issue.append('INT_PROP')

      # for images, recommend also using punsafe
      if is_image_caption:
          # copyright issues in captions
          if 'watermark' in text:
              matched_words.append('watermark')
              safety_issue.append('INT_PROP')

          # these are shorter more direct quote. 
          #if 'quote' in text:
          #  safety_issue.append('INT_PROP')
          #if 'phrase \'' in text:
          #  safety_issue.append('INT_PROP')
          #if 'phrase \"' in text:
          #  safety_issue.append('INT_PROP')

          # long direct quotes by captioners
          if " '" in text:
              if len(text.split(" '")[1].split("'")[0]) > 50:
                  matched_words.append(text.split(" '")[1].split("'")[0])
                  safety_issue.append('INT_PROP')
          if " \"" in text:
              if len(text.split(" \"")[1].split("\"")[0]) > 50:
                  matched_words.append(text.split(" \"")[1].split("\"")[0])
                  safety_issue.append('INT_PROP')

          # trademark common patterns
          if ' brand ' in text or ' branded ' in text:
              matched_words.extend([w for w in  text.split() if ('brand' in w)])
              safety_issue.append('INT_PROP')
          # PII common patterns
          if 'license' in text or 'location' in text or 'address' in text or 'phone' in text or "author's" in text or " names '" in text or " name '" in text:
              matched_words.extend([w for w in  text.split() if ('license' in w or 'location' in w or 'address' in w or 'phone' in w or "author's" in w or "name" in w)])
              safety_issue.append('PII')
          # special common patterns by captioners    
          if 'written by' in text or "' by " in text:
              matched_words.append("written by") # this is not exactly right
              safety_issue.append('PII')
              
      if pii_cnt:
          safety_issue.append("PII")

      # hate
      if hate_cnt:
        safety_issue.append('Hate')

      # hate
      if harm_cnt:
        safety_issue.append('Harm')

      # religious offensive content
      if ('jesus' in text or 'allah' in text or 'muhammad' in text or 'buddha' in text or 'vishnu' in text or 'krishna' in text or 'ganesh' in text) and (nsfw_cnt or hate_cnt):
          matched_words.extend([w for w in  text.split() if ('jesus' in w or 'allah' in w or 'muhammad' in w or 'buddha' in w or 'vishnu' in w or 'krishna' in w or 'ganesh' in w) ])
          safety_issue.append('Religious Offense')

      # general cruelty, physical harm, including self harm
      if pii_cnt or ' person ' in text or ' man ' in text or ' woman ' in text or \
       (' dog' in text or ' animal' in text or ' cat' in text or ' cow' in text or ' horse' in text) or \
       ('kid' in text or 'babies' in text or 'infant' in text or 'child' in text or 'boy' in text or 'girl' in text or 'baby' in text):
        if harm_cnt or ('noose' in text and ' hang' in text):
            matched_words.extend([w for w in  text.split() if ('noose' in  w or w.startswith('hang')) ])            
            nsfw_cnt += 1            
            safety_issue.append('Harm')

      # general crimes
      if  crime_cnt:
          safety_issue.append('Crime')

      # hate
      if hate_cnt:
        if 'harras' in text or 'humilate' in text or crime_cnt:
            safety_issue.append('Hate Crime')

      # sex crimes
      # we need to avoid confusion of rapeseed and the related flower and oils
      if crime_cnt and sex_cnt:
        safety_issue.append('Sex Crime')

      if (nsfw_cnt or "Hate" in safety_issue) and crime_cnt:
        if "Hate" in safety_issue:
          safety_issue.append('Hate Crime')
        else:
          safety_issue.append('Crime')

      # cyber crimes
      if cybercrime_cnt:
        safety_issue.append('Cyber Crime')

      # CNBR
      if ('nuclear' in text or 'chemical' in text or 'biologcal' in text or 'radioac' in text or 'radiol' in text) and \
        ('explos' in text or ' bomb' in text or 'war' in text):
          matched_words.extend([w for w in text.split() if ('nuclear' in w or 'chemical' in w or 'biologcal' in w or 'radioac' in w or 'radiol' in w)])
          safety_issue.append('CNBR')
      # this prevents too many false positives
      if len(matched_words) == 1 and "sex" in matched_words and len([a for a in matched_csam_words if "child" in a]) == len(matched_csam_words):
          sex_cnt -= 1
      sex_or_harm = (sex_cnt or any(issue for issue in safety_issue if "Harm" in issue or "Crime" in issue))
      if not do_pii:
          safety_issue = [a for a in safety_issue if a != "PII"]
      if not fine_grain:
          safety_issue = list(set(["NSFW" if i not in {"PII", "INT_PROP", "CSAM", "CAM", "MINOR"} else i for i in safety_issue]))
      if minor_cnt:
          safety_issue.append("MINOR")
      # potential CSAM
      if 'hentai' in text or 'lolita' in text  or ' incest' in text:
        return  list(set(["CSAM"]+safety_issue)), min(1.0,len(matched_words+matched_csam_words)/(10+text.count(" "))), list(set(matched_words+matched_csam_words)), ner, template, noun_hypernyms
      if ('nude' in text or 'naked' in text) and ('kid' in text or 'babies' in text or 'infant' in text or 'child' in text or 'boy' in text or 'girl' in text or 'baby' in text):
        matched_words.extend([w for w in text.split() if ('nude' in w or 'naked' in w or 'kid' in w or 'babies' in w or 'infant' in w or 'child' in w or 'boy' in w or 'girl' in w or 'baby' in w)])
        return list(set(["CSAM"]+safety_issue)), min(1.0,(len(matched_words+matched_csam_words)/(10+text.count(" ")))),list(set(matched_words+matched_csam_words)), ner, template, noun_hypernyms
      if sex_or_harm and potential_csam_cnt:
          if (nsfw_cnt*10/(1+text.count(" ")) > cam_flagged_threshold):
              if sex_cnt:
                  return list(set(["CSAM"]+safety_issue)), min(1.0,len(matched_words+matched_csam_words)/(10+text.count(" "))), list(set(matched_words+matched_csam_words)), ner, template, noun_hypernyms
              else:
                  return list(set(["CAM"]+safety_issue)), min(1.0,len(matched_words+matched_csam_words)/(10+text.count(" "))), list(set(matched_words+matched_csam_words)), ner, template, noun_hypernyms
      safety_issue = list(set(safety_issue))
      if not safety_issue:
          return [], 0.0, [], ner, template, noun_hypernyms
      return  safety_issue, min(1.0,len(matched_words)/(10+text.count(" "))), list(set(matched_words+matched_minor_words)), ner, template, noun_hypernyms

def classify_and_quality_score(text, cache_dir = "/leonardo_work/EUHPC_E03_068/.cache"):
    global args, \
        edu_model, \
        red_pajama_model, \
        math_model, \
        pile_class_model, \
        domain_model, \
        toxic_classifier
    
    if args is not None:
        cache_dir = args.cache_dir
    if edu_model is None:
        print ("loading models")
        edu_model= fasttext.load_model(cache_dir+"/fasttext/kenhktsui.bin")
        print ("step 1")
        red_pajama_model = fasttext.load_model(cache_dir+"/fasttext/rj_model.bin")
        print ("step 2")        
        pile_class_model = fasttext.load_model(cache_dir+"/fasttext/pile_class.ftz")
        math_model = fasttext.load_model(cache_dir+"/fasttext/math.bin")        
        domain_model = fasttext.load_model(cache_dir+"/fasttext/domain_model.bin")
        print ("finished loading models")
        
    text = text.replace("\n", ". ").replace("<|endoftext|>", " ").replace("<|endofsection|>", " ").replace(" she ", " he ").replace(" her ", " his ")
    text = " ".join(a for a in text.split(". ") if "Creative Common" not in a and "CC-BY" not in a)
    if not text.strip():
        return (0.0, "")
    text = text.replace("\n", " ")
    text = text[:min(len(text), 2000)]

    label, score = edu_model.predict(text)
    label = label[0]
    if "LOW" in label:
          score = 1-score
    label, score3 = red_pajama_model.predict(text)
    label = label[0]
    if "cc" in label:
          score3 = 1-score3
    label, score2 = math_model.predict(text)
    label = label[0]
    if "Math" not in label:
       score2 = 1-score2
    
    label1, _ = domain_model.predict(text)
    label2, _ = pile_class_model.predict(text)
    label1 = label1[0].replace("__label__", "")
    label2 = label2[0].replace("__label__", "")
    score = math.sqrt(((score*100)**2 + (score2*100)**2 + (score3*100)**2)/3)/100
    return [score, label1+"-"+label2 if score2 < 0.8 else "Math"]
    

def load_json(file_path: str):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data
        
def save_json(data: dict, file_path: str):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def get_program_name():
  global program_name
  return program_name

def lang_is_cjk(lang):
  return lang in {"ko", "zh", "ja"}

lang_2_max_stopword_len = dict([(lang, max(s.count(" ")+1 if not lang_is_cjk(lang) else len(s) for s in arr)) for lang, arr in all_stopwords.items()])

def get_stopword_score(text, lang="en", max_word_len=3, cjk_scale=1.5, window=500):
    is_cjk = lang_is_cjk(lang)
    stopwords =  all_stopwords.get(lang, all_stopwords['en'])
    if not stopwords: return 1
    if window:
        text = text[:min(len(text), window)]
        if not is_cjk:
            text = text.split(" ")[:-1]
            text = " ".join(text)
    text = text.lower().strip()
    if is_cjk:
      s_arr = list("".join(text.split()))
    else:
      s_arr = text.split()
    word_len = lang_2_max_stopword_len.get(lang, max_word_len)
    len_s = len(s_arr)
    stop_cnt = 0
    total_cnt = 0
    for i in range(len_s):
      if s_arr[i] is None: continue
      for j in range(min(len_s, i+word_len), i, -1):
        word = "".join(s_arr[i:j]) if is_cjk else " ".join(s_arr[i:j])
        if word in stopwords:
          stop_cnt += 1
          s_arr[i] = "".join(s_arr[i:j]) if is_cjk else " ".join(s_arr[i:j])
          for k in range(i+1, j):
            s_arr[k] = None
          break
      total_cnt += 1
    if total_cnt == 0:
      return 0
    stopword_score =  (stop_cnt/total_cnt)
    if is_cjk: stopword_score = stopword_score*cjk_scale
    return (stopword_score)



def get_special_char_score (text, lang="en", special_characters_default=None, window=500):
  global junk
  if len(text) == 0: return 1
  #TODO: do we want to do any lang specific special_chars?
  if special_characters_default is None: special_characters_default = junk
  if window:
        text = text[:min(len(text), window)]
  ret =  len([a for a in text if a in special_characters_default])/len(text)
  if lang_is_cjk(lang):
    return ret/5
  else:
    return ret


def get_rank() -> int:
    try:
        rank = int(os.environ['SLURM_PROCID'])
    except:
        rank = args.rank
    return rank


def _get_tasks_per_node() -> int:
    try:
        return int(os.environ['SLURM_NTASKS_PER_NODE'])
    except:
        return 1


def _get_num_nodes() -> int:
  try:
    return int(os.environ['SLURM_JOB_NUM_NODES'])
  except:
    return args.world_size


def get_world_size() -> int:
    return _get_num_nodes() * _get_tasks_per_node()


common_title_words_set = {'introduction', 'conclusion', 'section', 'chapter', 'works', 'notes', 'note', 'further', 'see', 'references', 'reference', 'section', 'title', 'conclusion', 'intro', 'introduction', 'executive', 'summary', 'key', 'plot', 'theme'}
lang_2_max_flaggedword_len = dict([(lang, max(s.count(" ")+1 if  lang not in {"zh", "ja", "ko"} else len(s) for s in arr)) for lang, arr in flagged_words.items()])

def get_flaggedword_score(text, lang="en", max_word_len=3):
    max_word_len = max(lang_2_max_flaggedword_len.get(lang, max_word_len), max_word_len)
    flaggedwords1 = flagged_words.get(lang, {})
    flaggedwords2 = flagged_words.get("en", {})
    bannedwords1 = banned_words.get(lang, {})
    bannedwords2 = banned_words.get("en", {})
    stopwords1 = all_stopwords.get(lang, {})
    stopwords2 = all_stopwords.get("en", {})
    is_cjk = lang in {"zh", "ja", "ko"}
    if is_cjk:
      text = " ".join([a for a in text.lower().split() if a not in stopwords2])
    text = text.lower().strip().replace(",", "").replace(".", "").replace("\"", "").replace("'", "")
    if is_cjk:
      s_arr = [a for a in [s.strip(strip_chars) for s in list(text)] if a]
    else:
      s_arr = [s.strip(strip_chars) for s in text.split()]
    len_s = len(s_arr)
    banned_score = 0
    flagged_score = 0
    hate_score = 0
    total_cnt = 0
    for i in range(len_s):
      if s_arr[i] is None: continue
      word_len = max_word_len
      for j in range(min(len_s, i+word_len),i,-1):
        if is_cjk:
          word = "".join([s for s in s_arr[i:j] if s])
        else:    
          word = " ".join([s for s in s_arr[i:j] if s])
        if not word: break
        is_flagged = word in flaggedwords1 or word in flaggedwords2
        is_hate = word in hatewords
        is_banned = word in bannedwords1 or word in bannedwords2 
        if is_flagged or is_banned or is_hate:
          if is_flagged: flagged_score += 1
          if is_banned: banned_score += 1
          if is_hate: hate_score += 1
          s_arr[i] =  word
          for k in range(i+1, j):
            s_arr[k] = None
      s = s_arr[i]
      if s not in stopwords1 and (is_cjk or len(s) > 3):
        total_cnt += 1
    if total_cnt == 0: total_cnt = 1
    flagged_score = (flagged_score/total_cnt)
    hate_score = hate_score/total_cnt
    banned_score = banned_score/total_cnt
    return  flagged_score, banned_score, hate_score



docHash = {}
sentHash = {}
prefixHash = {}
# WARNING: don't do dedup twice in the same process!!
def dedup(data, delete_sent_if_greater_percentage=0.05, delete_doc_if_sent_greater_percentage=0.75):
  #print ("dedup")
  if random.randint(0,1000000)==0:
    for key in list(prefixHash.keys()):
      if prefixHash[key] <= 3 or random.randint(0,1):
        del prefixHash[key]
  if random.randint(0,10000000)==0:
    for key in list(sentHash.keys()):
      if sentHash[key] <= 3 or random.randint(0,1):
        del sentHash[key]
  if random.randint(0,100000000)==0:
    for key in list(docHash.keys()):
      if docHash[key] <= 3 or random.randint(0,1):
        del docHash[key]
  new_text = []
  new_meta = []
  for text, metadata in zip(data['text'].split("<|endoftext|>"), data['metadata']):
      
      lang = metadata['lang']
      if type(lang) is list:
          metadata['langs'] = lang
          lang = metadata['lang'] = lang[0]
      is_cjk = lang in {"zh", "ja", "ko"}
      text0 = text
      text = text.strip()
      text_arr =  text.lower().split()
      text2 = [a.strip(strip_chars) for a in text_arr[20:min(len(text_arr), 120)]][:-1]  
      text2 = [a[:4] if len(a) > 4 else a for a in text2 if len(a) > 1]
      code = hash("".join(text2))
      if code in docHash:
          # duplicate document
          docHash[code] = docHash.get(code, 0) + 1
          continue
      docHash[code] = docHash.get(code, 0) + 1
      doc_code = code
      # sentence dedup
      text2 = text
      if lang == "hi":
          text2 = text2.replace("|", ". ")
      text2 = text2.replace("。",". ").replace("|>", "|>. ").replace("<|", ". <|").replace(".\n", ". ").replace("? ", "?. ").replace("! ", "!. ").replace("; ", ";. ").replace("| ", "|. ").replace("\n", ". ").\
          replace("1>", "1>. ").replace("2>", "2>. ").replace("3>", "3>. ").replace("4>", "4>. ").replace("5>", "5>. ").replace("6>", "6>. ").replace("7>", "7>. ").\
          replace("8>", "8>. ").replace("9>", "9>. ").replace("0>", "0>. ").replace("<", ". <").replace("<image>", "<image>. ").replace("<audio>", "<audio>. ").replace(".</", ". </")

      l_arr = [l2 for l2 in text2.split(". ") if len(l2) > 30]
      num_sents = len(list(set(l_arr)))
      sent_dups = []
     
      for l2, cnt in Counter(l_arr).items():
        # there can be many repeated sentences in the same example because we are concatenating similar documents.
        # we need to decide what to do about this.
        l3 = l2.strip("|.。?!").lower()
        code4 = hash(l3)
        score1 = get_special_char_score(l2, lang)
        score2 = get_stopword_score(l2, lang)
        #if code4 in sentHash:
        #    print (("found dup", score1, score2, l2))
        if not ('math' in metadata['source'] or "{" in l3 or "\ndef " in l3 or "):" in l3 or  "${" in l3):
            if code4 in sentHash and sentHash[code4] > 2:
                if (((score1 > 0.1 or score2 < 0.05)) or "http" in l3 or "terms of use" in l3 or "view pdf" in l3 or "view doc" in l3 or "print this page" in l3 or "read more" in l3 or " click " in " "+l3 or "privacy policy" in l3 or "jump to" in l3 or "disclaimer" in l3 or "you are here" in l3 or "send email" in l3 or "this page" in l3 or "this site" in l3 or "more info" in l3 or "this website" in l3 or "our site" in l3 or "our website" in l3 or "the link" in l3 or "visiting www" in l3 or "apache license" in l3 or "fandom" in l3 or "foodista" in l3 or "wiki" in l3 or "gutenberg.org" in l3 or "creative commons" in l3 or "cc-by" in l3 or "from wiki" in l3 or "free media" in l3 or "cookies" in l3  or "Creative Commons Attribution" in l2 or  "(CC-BY) 4.0 License" in l2):
                    text = text.replace(l2+". ", "")
                    text = text.replace(l2+".", "")                    
                    text = text.replace(l2+" ", "")
                    text = text.replace(l2, "")
            if code4 in sentHash:
              sent_dups.append(l2)
            sentHash[code4] = sentHash.get(code4, 0) + 1
            
      if num_sents and len(sent_dups)/num_sents >= delete_doc_if_sent_greater_percentage:
          continue
      if False:
        if num_sents and len(sent_dups)/num_sents >= delete_sent_if_greater_percentage:
          for l2 in sent_dups:
            text = text.replace(l2+". ", "")
            text = text.replace(l2+".", "")                    
            text = text.replace(l2+" ", "")
            text = text.replace(l2, "")
      if not is_cjk:
          sent_dups = []
          # prefix dups
          for i in range(3):
            text_arr = text.split(" ")
            if  any(a for a in text_arr[:min(len(text_arr), 4)] if ">" in a and "<" in a): continue
            text2 = [a.strip(strip_chars) for a in text[:min(len(text), 300)].lower().split(" ")]
            code = hash("".join(text2[:6]))
            if prefixHash.get(code,0) >= 3:
              pattern = " ".join(text.split(" ")[:6])
              text = " ".join(text.split(" ")[6:]).replace(pattern, "").strip()
              prefixHash[code] = prefixHash.get(code, 0) + 1
              continue
            else:
              prefixHash[code] = prefixHash.get(code, 0) + 1
            code = hash("".join(text2[:5]))
            if prefixHash.get(code,0) >= 4:
              pattern = " ".join(text.split(" ")[:5])
              text = " ".join(text.split(" ")[5:]).replace(pattern, "").strip()
              prefixHash[code] = prefixHash.get(code, 0) + 1
              continue
            else:
              prefixHash[code] = prefixHash.get(code, 0) + 1
            code = hash("".join(text2[:4]))
            if prefixHash.get(code,0) >= 5:
              pattern = " ".join(text.split(" ")[:4])
              text = " ".join(text.split(" ")[4:]).replace(pattern, "").strip()
              prefixHash[code] = prefixHash.get(code, 0) + 1
              continue
            else:
              prefixHash[code] = prefixHash.get(code, 0) + 1
            break

          for i in range(3):
            text_arr = text.split(" ")              
            if  any(a for a in text_arr[max(0, len(text_arr)-4):] if ">" in a and "<" in a): continue              
            text2 = [a.strip(strip_chars) for a in reversed(text.split(" ")) ][:-1]
            code = hash("".join(text2[:6]))
            if prefixHash.get(code,0) >= 3:
              text = " ".join(text.split(" ")[:-6]).strip()
              prefixHash[code] = prefixHash.get(code, 0) + 1
              continue
            else:
              prefixHash[code] = prefixHash.get(code, 0) + 1
            code = hash("".join(text2[:5]))
            if prefixHash.get(code,0) >= 4:
              text = " ".join(text.split(" ")[:-5]).strip()
              continue
            else:
              prefixHash[code] = prefixHash.get(code, 0) + 1
            code = hash("".join(text2[:4]))
            if prefixHash.get(code,0) >= 5:
              text = " ".join(text.split(" ")[:-4]).strip()
              continue
            else:
              prefixHash[code] = prefixHash.get(code, 0) + 1
            break
          if not text: continue
          text = text.strip()
          text = text[0].upper()+ text[1:]
          text_arr =  text.lower().split()
          text2 = [a.strip(strip_chars) for a in text_arr[20:min(len(text_arr), 80)]][:-1]  
          text2 = [a[:4] if len(a) > 4 else a for a in text2 if len(a) > 1]
          code = hash("".join(text2))
          if code != doc_code:
            if code in docHash :
              # duplicate document
              docHash[code] = docHash.get(code, 0) + 1
              continue
            docHash[code] = docHash.get(code, 0) + 1
      new_text.append(text)
      new_meta.append(metadata)
      
  if not new_text: return None
  data['text'] = "<|endoftext|>".join(new_text)
  data['metadata'] = new_meta
  return data

common_pile_sites = None
white_list_sites = None
def is_idx_match(data):
    global common_pile_sites, white_list_sites
    if common_pile_sites is None:
        common_pile_sites = set(json.load(open("common_pile_urls.json")))
        white_list_sites = set(json.load(open("white_list_urls.json")))
        
    if 'idx' not in data and 'url' in data:
      data['idx'] = data['url']
    if "://" in data['idx'].split("://",1)[-1]: return None, None
    idx = data['idx'].split("://",1)[-1]
    # data from loc.gov is often garbled
    if  "loc.gov" in idx or 'slashdot.org' in idx or "yahoo.com" in idx or "google.com" in idx or "amazon.com" in idx or "cnbc.com" in idx or "facebook.com" in idx or "youtube.com" in idx or "instagram.com" in idx or "twitter.com" in idx or "facebook.com" in idx or "whatsapp.com" in idx or "microsoft.com" in idx or "reddit.com" in idx or "yahoo.co.jp" in idx or "tiktok.com" in idx or "baidu.com" in idx or "linkedin.com" in idx or "netflix.com" in idx or "pornhub" in idx or "xxx" in idx or "dzen.ru" in idx or "naver.com" in idx or "live.com" in idx or "bet.br" in idx or "office.com" in idx or "bing.com" in idx or "bilibili.com" in idx or "pinterest.com" in idx or "xvideos.com" in idx or "twitch.tv" in idx or "xhamster.com" in idx or "temu.com" in idx or "vk.com" in idx or "mail.ru" in idx or "sharepoint.com" in idx or "weather.com" in idx or "samsung.com" in idx or "globo.com" in idx or ".t.me/" in idx or "canva.com" in idx or "duckduckgo.com" in idx or "xnxx.com" in idx or "xhamster43.desi" in idx or "nytimes.com" in idx or "deepseek.com" in idx or "zoom.us" in idx or "stripchat.com" in idx or "quora.com" in idx:
        return None, None        
    if len(idx) > 80:
        idx = idx[:80]
    if "/" not in idx:
        idx = idx+"/"
    is_oss =   (".free.law/" in idx or ".europeana.eu/" in idx or ".publicdomainreview.org/" in idx or ".wisdomcommons.org/" in idx or ".intratext.com/" in idx or ".mediawiki.org/" in idx or ".wikimedia.org/" in idx or ".wikidata.org/" in idx or \
                ".wikipedia.org/" in idx or ".wikisource.org/" in idx or ".wikifunctions.org/" in idx or ".wikiquote.org/" in idx or ".wikinews.org/" in idx or ".wikivoyage.org/" in idx or ".wiktionary.org/" in idx or ".wikibooks.org/" in idx or ".courtlistener.com/" in idx or ".case.law/" in idx or \
                "pressbooks.oer.hawaii.edu/" in idx or ".huggingface.co/docs/" in idx or \
                ".opencourselibrary.org/" in idx or ".medbiq.org/" in idx or ".doabooks.org/" in idx or ".bccampus.ca/" in idx or \
                "open.umn.edu/opentextbooks/" in idx or "www.gutenberg.org/" in idx or ".mozilla.org/"  in idx or "www.eclipse.org/" in idx or \
                ".apache.org/" in idx or ".python.org/" in idx or ".pytorch.org/" in idx or ".numpy.org/" in idx or ".scipy.org/" in idx or ".opencv.org/" in idx or \
                ".scikit-learn.org/" in idx or ".pydata.org/" in idx or ".matplotlib.org/" in idx or ".palletsprojects.com/" in idx or \
                ".sqlalchemy.org/" in idx or ".pypi.org/" in idx or ".sympy.org/" in idx or ".nltk.org/" in idx or \
                ".scrapy.org/" in idx or ".owasp.org/" in idx or \
                ".creativecommons.org/" in idx or ".stackoverflow.org/" in idx or ".stackexchange.org/" in idx  or \
                ".wikia.com/" in idx or ".foodista.com/" in idx or ".fandom.com/" in idx or ".attack.mitre.org/" in idx)
    text = data['text']
    head = text[:100].lower()
    tail = text[-100:].lower()
    
    if is_oss or idx in common_pile_sites or  idx in white_list_sites or \
       ".mil/" in idx or ".vlada.mk" in idx or ".vlada.cz" in idx or ".kormany.hu" in idx or  "regeringen." in idx or ".rijksoverheid.nl" in idx or ".government.nl" in idx or ".regeringen.se" in idx or  ".regeringen.dk" in idx or  ".regeringen.no" in idx or ".bund.de" in idx or ".bundesregierung.de" in idx or  ".government.ru" in idx or ".gc.ca" in idx or \
       ".admin.ch" in idx or  'www.gob.cl/' in idx or  'www.gob.ec/' in idx or  'guatemala.gob.gt/' in idx or  'presidencia.gob.hn/' in idx or  'www.gob.mx/' in idx or  'presidencia.gob.pa/' in idx or  'www.gob.pe/' in idx or  'gob.es/' in idx or  'argentina.gob.ar/' in idx or \
        "tanzania.go.tz/" in idx or ".indonesia.go.id/" in idx or ".go.kr/" in idx or ".go.jp/" in idx or  "thailand.go.th/" in idx or ".europa.eu/" in idx or ".un/" in idx or ".int/" in idx or ".govt." in idx or "www.gub.uy" in idx or ".gov/" in idx or '.gov.' in idx or '.gouv.' in idx:
        
        if "ymca.int" in idx: return None, None
        return True, is_oss
    if  idx not in common_pile_sites and idx not in white_list_sites and ("cc-0" in head or "creative common"  in head or "cc-by" in head or  "creative common"  in tail or "cc-by" in tail or "cc-0" in head):
        if not filter_copyright_and_content_issues(data):
            return True, True
    return None, None

def filter_copyright_and_content_issues(data):
      text = data['text']
      if "gutenberg.org" not in data['idx'] and \
          ".gov/" not in data['idx'] and ".mil/" not in data['idx'] and ".go.jp" not in data['idx'] and \
          ".gov.au" not in data['idx'] and ".gov.uk" not in data['idx']:
          if len(text) > 1000: text = text[:1000]
          # for gutenberg we will fix this with upsampling and debiasing
          flagged_score, banned_score, hate_score = get_flaggedword_score(text,lang)
          if (flagged_score > 0.2 and "wikipedia.org/" not in data['idx']) or flagged_score > 0.25:
              return True
          if flagged_score > 0.05 and banned_score > 0.05:
              return True
          if hate_score > 0.1:
              return True
          if flagged_score > 0.05 and hate_score > 0.05:
              return True
          if not data['text']: return True
          # some general spam terms
          spam_terms = sum ([1 for  a in ["Free," "Cash," "Money," "Win," "Prize," "Bonus," "Earn extra income",
                                          "Limited time," "Discount," "Offer," "Buy now," "Special promotion," "Deal",
                                          "Act now," "Urgent," "Don't delete," "Immediate response", "Cheap", "Low cost", 
                                          "Risk-free," "No obligation," "Guarantee," "100% satisfied",
                                          "Miracle," "Cure," "Lose weight fast," "without prescription",
                                          "without a prescription", "No prescription needed"] if " "+a+" " in data['text'] or " "+a.lower()+" " in data['text'] or \
                             a+" " in data['text'] or a.lower()+" " in data['text']])
          
          if spam_terms > 3 or \
             (spam_terms > 0  and \
              any(a in data['text'] or a.lower() in data['text'] for a in ['weight loss', ' dating app', 'dating site', 'diet pill', 'erectile dys', 'Viagra', 'Cialis'])):
              return True

      
      text=data['text']
      if "Copyright 19" in text or "Copyright 20" in text or "Copyright ©" in text: return True
      if "Copyright: Zhang" in text or "Content owned & provided" in text or "Copyright American Chemical Society" in text or "All Rights Reserved" in text or "protected by Copyright" in text or "© Copyright"in text or "Copyright©" in text or "© Copyright" in text or "Copyrights and Proprietary Information" in text or "Copyright by" in text or "Copyright: Federal" in text or "Copyright 2" in text or "Copyright 19" in text or "Copyright (c)"in text or "Copyright ©" in text or "contained herein is strictly prohibited" in text or "commercial use must be authorized" in text or "This copyrighted, evidence-based medicine" in text or "All rights reserved"in text:
          return True

      if "Creative Commons Attribution-NonCommercial" in text or ("Creative Commons" in text and "NonCommercial" in text):
          return True
      
      for t2, metadata in  zip(data['text'].split("<|endoftext|>"), data.get('metadata', {})):
          
          if 'license_header_footer' not in metadata:
              head_tail = t2[:100].lower() + t2[-100:].lower()
          else:
              if not (type(metadata['license_header_footer']) is str):
                  metadata['license_header_footer'] = str(metadata['license_header_footer'])
              head_tail = t2[:100].lower() + t2[-100:].lower() + metadata['license_header_footer'].lower()
          if "cc-by-nc" in head_tail or "by-nc-sa" in head_tail or "cc-by-nc-sa" in head_tail or "by-nc-sa" in t2 or "by-sa-nc" in head_tail or \
             "cc-by-nd" in head_tail or "by-sa-nd" in head_tail or "by-nc-nd" in head_tail or "by-nd-sa" in t2 or "by-sa-nd" in head_tail or \
             "cc by nc" in head_tail or "by nc sa" in head_tail or "cc by nc sa" in head_tail or "by nc sa" in t2 or "by sa nc" in head_tail or \
             "cc by nd" in head_tail or "by sa nd" in head_tail or "by nc nd" in head_tail or "by nd sa" in t2 or "by sa nd" in head_tail:
              return True

          if ('music' in head_tail or 'photo' in head_tail or 'flickr' in head_tail or 'picture' in head_tail or 'image' in head_tail) and ("cc-by " in head_tail or "cc-0 " in head_tail or "cc-by-" in head_tail or  "creative common" in head_tail):
              return True
          if ("cc-by " in head_tail or "cc-0 " in head_tail or "cc-by-" in head_tail or  "creative common" in head_tail) and ('noncommercial' in head_tail or 'non-commercial' in head_tail or 'non commercial' in head_tail):
              return True
          if 'rights reserved' in head_tail or  "copying prohibit" in head_tail or  "copyright" in head_tail:
              return True
          if "content owned & provided" in head_tail or "copyright american chemical society" in head_tail or "all rights reserved" in head_tail or "protected by copyright" in head_tail or "© copyright"in head_tail or "copyright©" in head_tail or "© copyright" in head_tail or "copyrights and proprietary information" in head_tail or "copyright by" in head_tail or "copyright: zhang"in head_tail or "copyright: federal" in head_tail or "copyright 2" in head_tail or "copyright 19" in head_tail or "copyright (c)"in head_tail or "copyright ©" in head_tail or "contained herein is strictly prohibited" in head_tail or "for commercial use must be authorized" in head_tail or "this copyrighted, evidence-based medicine" in head_tail:
                return True
          if '©' in head_tail:
                return True
      
          found = False
          for s in non_derivative:
              if s in head_tail:
                  found = True
                  break
          if not found:
              for s in non_commercial:
                  if s in head_tail:
                      found = True
                      break
          if found:
              return True
    
      return False


def load_fasttext_models():
  global edu_model, \
        red_pajama_model, \
        pile_class_model, \
        registry_model, \
        ffw_model, \
        toxic_classifier
  if edu_model is not None: return
  print ("loading models 0")
  edu_model= fasttext.load_model(cache_dir+"/fasttext/kenhktsui.bin")
  red_pajama_model = fasttext.load_model(cache_dir+"/fasttext/rj_model.bin")
  pile_class_model = fasttext.load_model(cache_dir+"/fasttext/pile_class.ftz")
  registry_model = fasttext.load_model(cache_dir+"/fasttext/domain_model.bin")
  ffw_model = fasttext.load_model(cache_dir+"/fasttext/ffw_model.bin")
  print ("done loading models")

def classify_and_qs(text):
    global edu_model, \
        red_pajama_model, \
        pile_class_model, \
        registry_model, \
        ffw_model, \
        toxic_classifier
    score =    score3 =  score4 = 0
    if text.startswith("<|endoftext|>"):
        text= text[len("<|endoftext|>"):]
    text = text.split("<|endoftext|>")[0].replace(" she ", " he ").replace(" her ", " his ")
    text = " ".join(a for a in text.split(". ") if "Creative Common" not in a and "CC-BY" not in a)
    if "|" in text[:500]:
        text = text[:500].split("|")[-1]+text[500:]
    if not text.strip():
        return (0.0, ""),  0, 0
    text = text.replace("\n", " ")
    
    text = text[:min(len(text), 1000)]

    label, score = edu_model.predict(text)
    label = label[0]
    if "LOW" in label:
          score = 1-score
    label, score3 = red_pajama_model.predict(text)
    label = label[0]
    if "cc" in label:
          score3 = 1-score3
    label0, _ = ffw_model.predict(text)
    label1, _ = registry_model.predict(text)
    label2, _ = pile_class_model.predict(text)
    label0 = label0[0].replace("__label__", "")    
    label1 = label1[0].replace("__label__", "")
    label2 = label2[0].replace("__label__", "")
    avg_score = math.sqrt(((score*100)**2 + (score3*100)**2 + (score4*100)**2)/3)/100    
    return (avg_score, label0+"-"+label1+"-"+label2), float(score),  float(score3)


def wait_for_other_ranks(output_dir):
      ws = get_world_size()
      with open(output_dir+"/"+str(get_rank())+".rank_done", "w") as outf: pass
      num_done = len(glob.glob(output_dir+"/*.rank_done"))
      while num_done < ws:
          time.sleep(30)
          with open(output_dir+"/"+str(get_rank())+".rank_done", "w") as outf: pass
          num_done = len(glob.glob(output_dir+"/*.rank_done"))
    


# splitting doesn't always work because some languages aren't space separated
def get_aligned_text(sent1, sent2, target_lang="en"):
  if target_lang not in  {"ja", "zh", "ko"}:
    sep = " "
    sent1 = sent1.replace("\n", " ** ").split()
    sent2 = sent2.replace("\n", " ** ").split()
  else:
    sep = ""
  aMatch = CSequenceMatcher(None,sent1, sent2)
  score0 = aMatch.ratio()
  blocks = aMatch.get_matching_blocks()
  blocks2 = []
  prevEndA = 0
  prevEndB = 0
  matchLen = 0
  nonMatchLen = 0
  for blockI in range(len(blocks)):
      if blockI > 0 or (blockI==0 and (blocks[blockI][0] != 0 or blocks[blockI][1] != 0)):
          blocks2.append([sep.join(sent1[prevEndA:blocks[blockI][0]]), sep.join(sent2[prevEndB:blocks[blockI][1]]), 0])
          nonMatchLen += max(blocks[blockI][0] - prevEndA, blocks[blockI][1] - prevEndB)
      if blocks[blockI][2] != 0:
        blocks2.append([sep.join(sent1[blocks[blockI][0]:blocks[blockI][0]+blocks[blockI][2]]), sep.join(sent2[blocks[blockI][1]:blocks[blockI][1]+blocks[blockI][2]]), 1])
        prevEndA = blocks[blockI][0]+blocks[blockI][2]
        prevEndB = blocks[blockI][1]+blocks[blockI][2]
        matchLen += blocks[blockI][2]
  score = float(matchLen+1)/float(nonMatchLen+1)
  return (blocks2, score, score0)


non_commercial = ['nonkomersial', 'necomercial', 'icke kommersiell', '비영리', 'ikke kommerciel', 'noncommercial', 'non-commercial', 'niet-commercieel', 'nekomerciāls', 'mittetulunduslik', 'noncomercial', 'nekomercinis', 'nichtgewerblich', '비상업적', 'некоммерческое', 'nicht-kommerziell', 'nach tráchtála', 'ikke kommersiell', 'שאינו מסחרי', 'ei kaupallista käyttöä', 'niekomercyjny', 'गैर-व्यावसायिक', 'ei kaupallista', 'non commerciale', '非商业性', 'ticari olmayan', 'nekomercialno', 'غير تجاري', 'μη εμπορική', '非商业', 'nem kereskedelmi', 'mhux kummerċjali', 'nekomerčné', 'não comercial', '営利目的外', 'non commercial', 'no comercial', 'nekomerční', '非営利', 'nicht kommerziell']

non_derivative = [
    'noderivs',
    'noderivatives',     
    'non derivative', 
    'no derivatives', 
    'no derivative', 
    'ingen bearbejdelse',       # Danish
    'ei muutoksia',             # Finnish
    'keine Bearbeitung',        # German
    'ohne Bearbeitung',         # German alternative
    'bez przeróbek',            # Polish
    'pas de modification',      # French
    'sans modification',        # French alternative
    'senza modifiche',          # Italian
    'ingen endringer',          # Norwegian
    'inga bearbetningar',       # Swedish
    'geen afgeleide werken',    # Dutch
    'χωρίς παράγωγα έργα',      # Greek
    'sin obras derivadas',      # Spanish
    'sem obras derivadas',      # Portuguese
    'bez izvedenih del',        # Slovenian
    'bez izvedenica',           # Croatian, Serbian
    'bez odvozených děl',       # Czech
    'nedrīkst atvasināt',       # Latvian
    'järeltöötluseta',          # Estonian
    'nem származékos',          # Hungarian
    'nessuna opera derivata',   # Italian alternative
    'без производных',          # Russian
    '不可衍生',                 # Chinese simplified
    '禁止改作',                 # Chinese alternative
    '改変禁止',                 # Japanese
    '2차적 저작물 금지',          # Korean
    'ללא יצירות נגזרות',        # Hebrew
    'ingen afledte værker',     # Danish alternative
    'inte bearbetad',           # Swedish alternative
    'ei johdannaisia',          # Finnish alternative
    'ingen bearbeidelse',       # Norwegian alternative
    'بدون اشتقاق',              # Arabic
    'sin derivados',            # Spanish alternative short
    'без похідних творів',      # Ukrainian
    'nu lucrări derivate',      # Romanian
    'nessuna derivazione',      # Italian alternative short
]


regions = ['American', 'English', 'French', 'Indian', 'Spanish', 'Chinese', 'Afrikaans', 'Tosk Albanian', 'Amharic', 'Aragonese', 'Arabic', 'Egyptian Arabic', 'Asturian', 'Assamese', 'Avaric', 'South Azerbaijani', 'Azerbaijani', 'Bavarian', 'Bashkir', 'Central Bikol', 'Belarusian', 'Bulgarian', 'Bihari', 'Bengali', 'Tibetan', 'Bishnupriya', 'Breton', 'Bosnian', 'Russia Buriat', 'Catalan', 'Chavacano', 'Cebuano', 'Chechen', 'Central Kurdish', 'Czech', 'Chuvash', 'Welsh', 'Danish', 'German', 'Dimli', 'Lower Sorbian', 'Dhivehi', 'Greek', 'Modern Greek', 'Emilian-Romagnol', 'English', 'Esperanto',  'Estonian', 'Basque', 'Persian', 'Finnish', 'Northern Frisian',  'Western Frisian', 'Irish', 'Scottish Gaelic', 'Galician', 'Guarani', 'Goan Konkani', 'Gujarati', 'Hebrew', 'Hindi', 'Croatian', 'Upper Sorbian', 'Haitian', 'Hungarian', 'Armenian', 'Interlingua', 'Indonesian', 'Interlingue', 'Iloko', 'Ido', 'Icelandic', 'Italian', 'Japanese', 'Lojban', 'Javanese', 'Georgian', 'Kazakh', 'Central Khmer', 'Kannada', 'Korean', 'Karachay-Balkar', 'Kurdish', 'Komi', 'Cornish', 'Kirghiz', 'Latin', 'Luxembourgish', 'Lezghian', 'Limburgan', 'Lombard', 'Lao', 'Northern Luri', 'Lithuanian', 'Latvian', 'Maithili', 'Malagasy', 'Eastern Mari', 'Minangkabau', 'Macedonian', 'Malayalam', 'Mongolian', 'Western Mari', 'Marathi', 'Malay', 'Maltese', 'Mirandese', 'Burmese', 'Erzya', 'Mazanderani', 'Nahuatl languages', 'Neapolitan', 'Low German', 'Nepali', 'Newari', 'Dutch', 'Norwegian Nynorsk', 'Norwegian', 'Occitan', 'Oriya', 'Ossetian', 'Pampanga', 'Panjabi', 'Polish', 'Piemontese', 'Western Panjabi', 'Pushto', 'Portuguese', 'Quechua', 'Romansh', 'Romanian', 'Russian', 'Yakut', 'Sanskrit', 'Sicilian', 'Sindhi', 'Serbo-Croatian', 'Sinhala', 'Slovak', 'Slovenian', 'Somali', 'Albanian', 'Serbian', 'Sundanese', 'Swedish', 'Swahili', 'Tamil', 'Telugu', 'Tajik', 'Thai', 'Turkmen', 'Tagalog', 'Turkish', 'Tatar', 'Tuvinian', 'Uighur', 'Ukrainian', 'Urdu', 'Uzbek', 'Venetian', 'Vietnamese', 'Volapük', 'Waray', 'Walloon', 'Wu Chinese', 'Kalmyk', 'Mingrelian', 'Yiddish', 'Yoruba',]


def remove_junk_lines(text):
    text = text.split("\n")
    text_arr2 =[]
    seen = {}
    for text2 in text:
      if len(text2) == 1: continue
      if text2 and text2[0]==text2[0].upper():
          if " " in text2:
              a, b = text2.split(" ",1)
              try:
                  int(a)
                  text2 = b.strip()
              except:
                  pass
      if text2 and text2[0]==text2[0].upper():              
          if seen.get(text2,0) > 5: # remove page numbers and similar dups
              continue
          seen[text2] = seen.get(text2,0)+1
          score1 = get_special_char_score(text2, lang)
          if score1 > 0.15 and len(text2) > 100:
              #print (("dropping", text2))
              continue
          text_arr2.append("\n"+text2.rstrip())
      else:
          text_arr2.append(" "+text2.rstrip())
    text = "".join(text_arr2)
    text = text.replace(" v. .", ".").replace(".  ", ". ").replace(",  ", ", ").replace(",  ", ", ").replace(", ,", ",").replace(", ,", ",").replace(", ,", ",").replace(", .", ".").replace(",.", ".").replace(". . ", ". ")        
    text = "\n".join(t.rstrip() for t in text.split("\n"))
    return text.strip()

def cleanup_raw_text(text, lang, cleanup_sents=False):
    stopwords =  all_stopwords.get(lang, all_stopwords['en'])    
    text = text.strip()
    text_old = text
    text = html.unescape(text)
    text = remove_citations(text)
    text = remove_junk_lines(text)
    score1 = get_special_char_score(text, lang)
    if score1 > 0.15:
        return ""
    score2 = get_stopword_score(text, lang)
    if score2  < 0.05:
        return ""
    if cleanup_sents:
        sents = text.split(". ")
        sents2 = []
        for text2 in sents:
            if "-cv-" in text2: continue            
            text3 = []
            for w in text2.split(" "):
                if w and len(w) <= 2 and text3 and len(text3[-1]) <= 2 and w[0] in "qwertyuiopasdfghjklzxcvbnm~" and (len(w) <=1 or w not in stopwords): continue
                text3.append(w)
            text2 = " ".join(text3)
            if len(text2) >= 10:
                score1 = get_special_char_score(text2, lang)
                if score1 > 0.15:
                    text2 = ".."
            if text2.endswith(" v") or text2.endswith(" vs") or text2.endswith(" V") or text2.endswith(" Vs") or "U.S.C" in text2 or "2d at " in text2 or "L.Ed" in text2 or "S.Ct" in text2 or "th Cir" in text2 or "nd Cir" in text2:
                score2 = get_stopword_score(text, lang)
                if score2  < 0.05:
                    text2 = ".."                    
            sents2.append(text2)
        text = ". ".join(sents2)
        if not text: return ""    
        has_period = False
        if text[-1] == ".":
            has_period = True
        text = text.replace(" v. ...", " ...").replace("... ...", "...").replace("... ...", "...").replace("... ...", "...").replace("... ...", "...").strip(". ")
        if has_period:
            text= text+"."
        if " ... " in text:
            text_arr = text.split(" ... ")
            text_arr2= []
            for text2 in text_arr:
                if len(text2) < 10: continue
                score1 = get_special_char_score(text2, lang)
                if score1 > 0.15:
                    continue
                elif len(text2) > 50:
                    score2 = get_stopword_score(text2, lang)
                    if score2  < 0.05:
                        continue
                text_arr2.append(text2)
            text= " ... ".join(text_arr2).strip()
            has_period = False
            if not text: return ""
            if text[-1] == ".":
                has_period = True
            text = text.replace(" v. ...", " ...").replace("... ...", "...").replace("... ...", "...").replace("... ...", "...").replace("... ...", "...").strip(". ")
            if has_period:
                text= text+"."
        if text:
            text = remove_citations(text)
            text = remove_junk_lines(text)
    text = text.replace(" v. .", ".").replace(".  ", ". ").replace(",  ", ", ").replace(",  ", ", ").replace(", ,", ",").replace(", ,", ",").replace(", ,", ",").replace(", .", ".").replace(",.", ".").replace(". . ", ". ").strip()
    if text != text_old:
        #print((text,))
        pass
    return text

        

citation_patterns = [
    # Original patterns for author-based citations
    r'\s(?:[A-Z][a-z]+(?:\sJr\.)?,\s[A-Z][a-z]+(?:\s[A-Z]\.)?;\s)+[A-Z][a-z]+,\s[A-Z][a-z]+\s[A-Z]\.,?',
    r'\b(?:(v\.\s*\d+,\s*no\.\s*\d+)|(p{1,2}\.\s*\d+(?:-\d+)?))\.?',
    r'\s((?:[A-Z]\.\s)+[A-Z][a-z]{4,};\s*)+',
    r'(?i)(Subsec\.|§|Pub\.\s+L\.|Stat\.|ch\.|title)\s+[^\s]+\s*\d+[^\s]*',
    r'\b(?:Sec\.|§|R\.\s*S\.|Stat\.|L\.|ch\.|title)\s+[\w\-.,]+\d+\b',
    r'title\s+[IVXL]+\s*,\s*§',
    r'\*\s*([A-Za-z]+(\sv\.\s)?[A-Za-z\s.]+,)\*\s*\d+\s+[F]\.\s?\d*d?\s+\d*\s*\(\s*\d+(?:st|nd|rd|th)\s*Cir\.\s*\d{4}\)',
    r'\b\d+\s+(?:Mo\.|N\. Y\.|Ill\.|Pa\.|Fed\.|F\.\s?\d*d?|S\. W\.|N\. W\.|N\. E\.|C\. C\. A\.|Pac\.|Cal\.|Ohio St\.|Mass\.|Wis\.|Minn\.|Ky\.|La\.|Ind\.|Tex\.|Miss\.|Conn\.|Kan\.|N\. J\.|R\. I\.|Mont\.|Iowa|Hun|Barb\.|Macq\.)[\s.]*\d+',
    r'\b(?:Cir\.|App\.|Supp\.|Ct\.|Dist\.|Exch\.|Eq\.|Commw\.)\b',
    r'\b(?:affirmed|reversed|cert\.|denied|ex rel|supra)\b.*?\)',
    # U.S. patent numbers
    r'\bU\.?\s?S\.?\s*Pat\.?\s*Nos?\.?\s*\d{1,3}(?:,\d{3})*(?:\s*,\s*\d{1,3}(?:,\d{3})*)*\b',        
     # Book/publication references
    r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s+[A-Z][a-z]+,\s+\d{4}\.?\s+(?:pp?\.\s+)?\d+-\d+\b',
        
]    

import re

def remove_citations(text):
    cleaned_text = text
    for pattern in citation_patterns:
        cleaned_text = re.sub(pattern, ' ', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = cleaned_text.replace(" v. .", ".").replace(".  ", ". ").replace(",  ", ", ").replace(",  ", ", ").replace(", ,", ",").replace(", ,", ",").replace(", ,", ",").replace(", .", ".").replace(",.", ".").replace(". . ", ". ")        
    return cleaned_text.strip()


Helsinki_Opus = {'en-af', 'af-en', 'th-en', 'tr-en', 'bn-en', 'az-en', 'en-az', 'en-he', 'th-en', 'tr-en', 'uk-en', 'en-mk', 'mk-en', 'de-pl', 'fr-sl', 'bg-en', 'cs-en', 'da-en', 'de-en', 'en-bg', 'en-cs', 'en-da', 'en-de', 'en-el', 'en-es', 'en-fr', 'en-hu', 'en-id', 'en-it', 'en-nl', 'en-ro', 'en-ru', 'en-sk', 'en-sv', 'en-vi', 'es-en', 'fr-en', 'hu-en', 'id-en', 'it-en', 'ja-en', 'jap-en', 'ko-en', 'lv-en',  'nl-en', 'pl-en', 'ru-en', 'sk-en', 'sv-en', 'vi-en', 'zh-en', 'en-jap' 'en-zh', 'ja-en', 'xh-en', 'en-xh', 'mr-en', 'en-mr', 'ml-en', 'en-ml'}
TC_big = {'el-en', 'en-ko', 'en-lt', 'en-lv', 'en-pt', 'lt-en', 'he-en', 'en-tr'}
HPLT_translate = {'ar-en', 'en-ar', 'en-et', 'en-fi', 'en-ga', 'en-hi', 'en-hr', 'en-mt', 'en-sw', 'et-en', 'fi-en', 'ga-en', 'hi-en', 'hr-en', 'mt-en', 'sw-en', 'gl-en', 'en-gl'}


def has_small_translation_model(tran):
    return tran in Helsinki_Opus or tran in TC_big or tran in HPLT_translate

def ctranslate2_with_batching(model, tokenizer, batch, src_lang="en", target_lang="de", repetition_penalty=1.2, batch_size=50, max_length=512, **args):
    """
    Generates translations for a batch of inputs using a `ctranslate2` model, supporting repetition penalties 
    to improve translation quality.

    Args:
        model: The `ctranslate2` model instance used for generating translations.
        tokenizer: Tokenizer instance that encodes input text and decodes model outputs.
        batch (list of str): List of input sentences to translate.
        repetition_penalty (float, optional): Penalty for repeated tokens. Defaults to 1.2.
        batch_size (int, optional): Number of sentences to process in each batch. Defaults to 40.

    Returns:
        list of str: List of translated sentences for each input in the batch.
    """
    ret = []
    tran = src_lang+"-"+target_lang
    is_m2m100 =  "m2m100" in translation_models and model ==  translation_models["m2m100"]
    target_prefix = None
    if is_m2m100:
        if src_lang not in tokenizer.lang_code_to_token or target_lang not in tokenizer.lang_code_to_token:
            return [""]*len(batch)
        tokenizer.src_lang = src_lang
        target_prefix = [tokenizer.lang_code_to_token[target_lang]]        
    with torch.no_grad():
        for rng in range(0, len(batch), batch_size):
            batch2 = batch[rng: min(len(batch), rng+batch_size)]
            batch2 = [tokenizer.convert_ids_to_tokens(tokenizer.encode(lst, max_length=max_length, truncation=True)) for lst in batch2]
            min_length = max(1, max(([len(a) for a in batch2]))-10)
            if is_m2m100:
                output =  tokenizer.batch_decode([tokenizer.convert_tokens_to_ids(r.hypotheses[0]) for r in model.translate_batch(batch2, target_prefix=[target_prefix]*len(batch2), repetition_penalty=repetition_penalty)])
            else:
                output =  tokenizer.batch_decode([tokenizer.convert_tokens_to_ids(r.hypotheses[0]) for r in model.translate_batch(batch2, max_decoding_length=max_length, min_decoding_length=min(max_length, min_length), repetition_penalty=repetition_penalty)])
            for o in output:
                if o:
                    if o[-1] == ',':
                        o = o.rstrip(",")+","
                    if o[-1] == '.':
                        o = o.rstrip(".")+"."
                    if o[-1] == '-':
                        o = o.rstrip("-")+"-"
                    if o[-1] == '_':
                        o = o.rstrip("_")+"_"
                    if o[-1] == '?':
                        o = o.rstrip("?")+"?"
                    if o[-1] == '!':
                        o = o.rstrip("!")+"!"
                    if is_m2m100:
                      o = o.split(" ",1)[-1]
                    o = o.replace("<unk>", " ")
                ret.append(o)
            batch2 = output = None
        return ret

translation_models = {}
translation_tokenizers = {}

def get_translation_model_tokenizer(src_lang, target_lang, logger=None, device_no=None):
    
  global translation_models, translation_tokenizers, device
  if device_no is not None:
      if torch.cuda.is_available():
          device = "cuda:"+str(device_no)
  if "m2m100" in src_lang or "m2m100" in target_lang:
    if logger is not None: logger.warning("loading m2m100 ")            
    if "m2m100" not in translation_models:
        if device != "cpu": os.environ['CUDA_VISIBLE_DEVICES'] = int(device.split(":")[-1])
        translation_tokenizers["m2m100"] = transformers.AutoTokenizer.from_pretrained("facebook/m2m100_418M", cache_dir=args.cache_dir)
        tokenizer= translation_tokenizers["m2m100"]
        if src_lang not in tokenizer.lang_code_to_token or target_lang not in tokenizer.lang_code_to_token:
            return None, None
        translation_models["m2m100"] = ctranslate2.Translator(args.cache_dir+"/m2m100", device=device.split(":")[0],
                                                              compute_type="float16" if device != "cpu" else "float32",)
        
    tokenizer= translation_tokenizers["m2m100"]
    if src_lang not in tokenizer.lang_code_to_token or target_lang not in tokenizer.lang_code_to_token:
        return None, None
    return translation_models["m2m100"], translation_tokenizers["m2m100"] 

  tran = src_lang+"-"+target_lang
  if tran not in translation_models and tran in Helsinki_Opus:
    if device != "cpu": os.environ['CUDA_VISIBLE_DEVICES'] = int(device.split(":")[-1])      
    translation_models[tran] = ctranslate2.Translator(args.cache_dir+"/opus-mt-"+tran, device=device.split(":")[0],
                                                      compute_type="float16" if device != "cpu" else "float32",)
    translation_tokenizers[tran] = transformers.AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-"+tran, cache_dir=args.cache_dir)
    if logger is not None: logger.warning("loading helsinki "+ tran)
  elif tran not in translation_models and tran in TC_big: 
    if logger is not None: logger.warning("loading helsinki-tc-big "+ tran)                 
    if device != "cpu": os.environ['CUDA_VISIBLE_DEVICES'] = int(device.split(":")[-1])
    translation_models[tran] = ctranslate2.Translator(args.cache_dir+"/opus-mt-"+tran, device=device.split(":")[0],
                                                      compute_type="float16" if device != "cpu" else "float32",)
    translation_tokenizers[tran] = transformers.AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-tc-big-"+tran, cache_dir=args.cache_dir)
  elif tran not in translation_models and tran in HPLT_translate:
    if logger is not None: logger.warning("loading hplt "+ tran)            
    if device != "cpu": os.environ['CUDA_VISIBLE_DEVICES'] = int(device.split(":")[-1])
    translation_models[tran] = ctranslate2.Translator(args.cache_dir+"/opus-mt-"+tran, device=device.split(":")[0],
                                                      compute_type="float16" if device != "cpu" else "float32",)
    translation_tokenizers[tran] = transformers.AutoTokenizer.from_pretrained("HPLT/translate-"+tran+"-v1.0-hplt_opus", cache_dir=args.cache_dir)
  elif tran not in translation_models:
    if logger is not None: logger.warning("loading m2m100 "+ tran)      
    if "m2m100" not in translation_models:
      translation_tokenizers["m2m100"] = transformers.AutoTokenizer.from_pretrained("facebook/m2m100_418M", cache_dir=args.cache_dir)
      tokenizer= translation_tokenizers["m2m100"]
      if src_lang not in tokenizer.lang_code_to_token or target_lang not in tokenizer.lang_code_to_token:
          print (tran)
          return None, None
      if device != "cpu": os.environ['CUDA_VISIBLE_DEVICES'] = int(device.split(":")[-1])
      translation_models["m2m100"] = ctranslate2.Translator(args.cache_dir+"/m2m100", device=device.split(":")[0],
                                                            compute_type="float16" if device != "cpu" else "float32",)
    tokenizer= translation_tokenizers["m2m100"]
    if src_lang not in tokenizer.lang_code_to_token or target_lang not in tokenizer.lang_code_to_token:
        print (tran)        
        return None, None
    translation_models[tran] = translation_models["m2m100"]
    translation_tokenizers[tran] = translation_tokenizers["m2m100"]
    
  if tran not in translation_models:
    return None, None

  #logger.warning (str(translation_models[tran].device) + " " + str(translation_models[tran].device_index))
  return translation_models[tran], translation_tokenizers[tran]

args = None
program_name = sys.argv[0]

cosmo_keys2file = {}
def init_cosmo():
    global cosmo_keys2file
    if not cosmo_keys2file: return
    for a in open("./cosmo_key_to_file.csv"):
        if "\t" not in a or not a.strip(): continue
        code_file_pos = a.strip().split("\t")
        if len(code_file_pos) != 3:
            continue
        code, file, pos = code_file_pos
        cosmo_keys2file[hash(code.strip())] = cosmo_keys2file.get(hash(code.strip()),[]) + [(file.strip(), int(pos))]
        if len(code) > 100:
            code = code[:100]
        cosmo_keys2file[hash(code.strip())] = cosmo_keys2file.get(hash(code.strip()),[]) + [(file.strip(), int(pos))]
    #print (len(cosmo_keys2file))


def get_cosmo(data):
    for text in data['text'].split("<|endoftext|>"):
        try:
            text2 = [a.strip("|\\1234567890~`!@#$%^&*()-_+=:;\"'<>,.?/") for a in text.strip()[:min(len(text), 300)].lower().split()][:-1]
        except Exception as e:
            return 
        text = [a[:4] if len(a) > 4 else a for a in text2 if len(a) > 1]
        key = "".join(text)
        code = hash(key)
        if code in cosmo_keys2file:
            lst =  cosmo_keys2file[code]
            file, pos = lst.pop()
            file = "/leonardo_work/EUHPC_E03_068/datasets/working/cosmopedia-v2/"+file
            if not cosmo_keys2file[code]: del cosmo_keys2file[code]
            infile = open(file)
            infile.seek(pos)
            line = infile.readline()
            infile.close()
            try:
                cosmo = json.loads(line)
                cosmo['lang'] = 'en'
                cosmo['source'] = file
                cosmo['idx'] = data['idx']                                                
                return cosmo
            except:
                pass
        if len(key) > 100:
            key = key[:100]
        code = hash(key)
        if code in cosmo_keys2file:
            lst =  cosmo_keys2file[code]
            file, pos = lst.pop()
            file = "/leonardo_work/EUHPC_E03_068/datasets/working/cosmopedia-v2/"+file
            if not cosmo_keys2file[code]: del cosmo_keys2file[code]
            infile = open(file)
            infile.seek(pos)
            line = infile.readline()
            infile.close()
            try:
                cosmo = json.loads(line)
                cosmo['lang'] = 'en'
                cosmo['source'] = file
                cosmo['idx'] = data['idx']                                                
                return cosmo
            except:
                pass
        text = [a[:4] if len(a) > 4 else a for a in text2 if len(a) > 1 and a not in stopwords_set]                
        key = "".join(text)
        code = hash(key)
        if code in cosmo_keys2file:
            lst =  cosmo_keys2file[code]
            file, pos = lst.pop()
            file = "/leonardo_work/EUHPC_E03_068/datasets/working/cosmopedia-v2/"+file
            if not cosmo_keys2file[code]: del cosmo_keys2file[code]
            infile = open(file)
            infile.seek(pos)
            line = infile.readline()
            infile.close()
            try:
                cosmo = json.loads(line)
                cosmo['lang'] = 'en'
                cosmo['source'] = file
                cosmo['idx'] = data['idx']                                                
                return cosmo
            except:
                pass
        if len(key) > 100:
            key = key[:100]
        code = hash(key)
        if code in cosmo_keys2file:
            lst =  cosmo_keys2file[code]
            file, pos = lst.pop()
            file = "/leonardo_work/EUHPC_E03_068/datasets/working/cosmopedia-v2/"+file
            if not cosmo_keys2file[code]: del cosmo_keys2file[code]
            infile = open(file)
            infile.seek(pos)
            line = infile.readline()
            infile.close()
            try:
                cosmo = json.loads(line)
                cosmo['lang'] = 'en'
                cosmo['source'] = file
                cosmo['idx'] = data['idx']                                                
                return cosmo
            except:
                pass



def find_cosmo(data):
    cosmo =  get_cosmo(data)
    if cosmo:
        text = cosmo['text'].strip()
        lang = 'en'
        # check too many ngram - todo                                                                                                                                                                              
        score1 = get_special_char_score(text, lang)
        if score1 > 0.15:
            return data
        score2 = get_stopword_score(text,lang)
        if score2  < 0.05:
            return data
        data['text'] += "<|endoftext|>"+text
        del cosmo['text']
        data['metadata'].append(cosmo)
    return data

seed2_keys2file = {}
def init_seed2():
    global seed2_keys2file
    if not seed2_keys2file: return
    for a in open("./seed2_key_to_file.csv"):
        if "\t" not in a or not a.strip(): continue
        code_file_pos = a.strip().split("\t")
        if len(code_file_pos) != 3:
            continue
        code, file, pos = code_file_pos
        seed2_keys2file[hash(code.strip())] = seed2_keys2file.get(hash(code.strip()),[]) + [(file.strip(), int(pos))]
        if len(code) > 100:
            code = code[:100]
        seed2_keys2file[hash(code.strip())] = seed2_keys2file.get(hash(code.strip()),[]) + [(file.strip(), int(pos))]
    #print (len(seed2_keys2file))


def find_seed2(data):
    for text, metadata in zip(data['text'].split("<|endoftext|>"), data['metadata']):
        orig_text = text
        try:
            text2 = [a.strip("|\\1234567890~`!@#$%^&*()-_+=:;\"'<>,.?/") for a in text.strip()[:min(len(text), 300)].lower().split()][:-1]
        except Exception as e:
            return data
        text = [a[:4] if len(a) > 4 else a for a in text2 if len(a) > 1]
        key = "".join(text)
        code = hash(key)
        if code in seed2_keys2file:
            lst =  seed2_keys2file[code]
            file, pos = lst.pop()
            file = "/leonardo_work/EUHPC_E03_068/datasets/working/seed2_obelisc/data/"+file
            if not seed2_keys2file[code]: del seed2_keys2file[code]
            infile = open(file)
            infile.seek(pos)
            line = infile.readline()
            infile.close()
            try:
                seed2 = json.loads(line)
                seed2['source'] = file
                metadata['seed2']=  seed2
                return data
            except:
                pass
        if len(key) > 100:
            key = key[:100]
        code = hash(key)
        if code in seed2_keys2file:
            lst =  seed2_keys2file[code]
            file, pos = lst.pop()
            file = "/leonardo_work/EUHPC_E03_068/datasets/working/seed2_obelisc/data/"+file
            if not seed2_keys2file[code]: del seed2_keys2file[code]
            infile = open(file)
            infile.seek(pos)
            line = infile.readline()
            infile.close()
            try:
                seed2 = json.loads(line)
                seed2['source'] = file
                metadata['seed2']=  seed2
                return data
            except:
                pass
        text = [a[:4] if len(a) > 4 else a for a in text2 if len(a) > 1 and a not in stopwords_set]                
        key = "".join(text)
        code = hash(key)
        if code in seed2_keys2file:
            lst =  seed2_keys2file[code]
            file, pos = lst.pop()
            file = "/leonardo_work/EUHPC_E03_068/datasets/working/seed2_obelisc/data/"+file
            if not seed2_keys2file[code]: del seed2_keys2file[code]
            infile = open(file)
            infile.seek(pos)
            line = infile.readline()
            infile.close()
            try:
                seed2 = json.loads(line)
                seed2['source'] = file
                metadata['seed2']=  seed2
                return data
            except:
                pass
        if len(key) > 100:
            key = key[:100]
        code = hash(key)
        if code in seed2_keys2file:
            lst =  seed2_keys2file[code]
            file, pos = lst.pop()
            file = "/leonardo_work/EUHPC_E03_068/datasets/working/seed2_obelisc/data/"+file
            if not seed2_keys2file[code]: del seed2_keys2file[code]
            infile = open(file)
            infile.seek(pos)
            line = infile.readline()
            infile.close()
            try:
                seed2 = json.loads(line)
                seed2['source'] = file
                metadata['seed2']=  seed2
                return data
            except:
                pass
    return data


def add_related(data):
    data = find_cosmo(data)
    data = find_seed2(data)
    if 'seed2' in data['metadata'][0]:
        pass
        #print (data)
    return data

if False:
        if 'metadata' in data and 'meta' in data['metadata']:
            if type(data['metadata']) is not dict:
                print (data)
            elif 'raw_text' in data['metadata']['meta']:
                data =  get_cosmo(data, data['metadata']['meta']['raw_text'])
                if 'cosmo' in data:
                    text = data['cosmo']['text']
                    lang = 'en'
                    score1 = get_special_char_score(text, lang)
                    if score1 > 0.15:
                        del data['cosmo']
                    score2 = get_stopword_score(text,lang)
                    if score2  < 0.05:
                        del data['cosmo']

dcad2lang = {'afr_Latn': 'af', 'als_Latn': 'sq', 'amh_Ethi': 'am', 'arb_Arab': 'ar', 'ast_Latn': 'ast', 'azj_Latn': 'az', 'bel_Cyrl': 'be', 'ben_Beng': 'bn', 'bul_Cyrl': 'bg', 'cat_Latn': 'ca', 'ceb_Latn': 'ceb', 'ces_Latn': 'cs', 'cym_Latn': 'cy', 'dan_Latn': 'da', 'deu_Latn': 'de', 'ell_Grek': 'el', 'eng_Latn': 'en', 'epo_Latn': 'eo', 'ekk_Latn': 'et', 'fin_Latn': 'fi', 'fra_Latn': 'fr', 'gaz_Latn': 'om', 'gla_Latn': 'gd', 'gle_Latn': 'ga', 'glg_Latn': 'gl', 'hau_Latn': 'ha', 'heb_Hebr': 'he', 'hin_Deva': 'hi', 'hrv_Latn': 'hr', 'hun_Latn': 'hu', 'hye_Armn': 'hy', 'ibo_Latn': 'ig', 'ilo_Latn': 'ilo', 'ind_Latn': 'id', 'isl_Latn': 'is', 'ita_Latn': 'it', 'jav_Latn': 'jv', 'jpn_Jpan': 'ja', 'kat_Geor': 'ka', 'kaz_Cyrl': 'kk', 'khm_Khmr': 'km', 'kor_Hang': 'ko', 'lit_Latn': 'lt', 'ltz_Latn': 'lb', 'lug_Latn': 'lg', 'lvs_Latn': 'lv', 'mal_Mlym': 'ml', 'mar_Deva': 'mr', 'mkd_Cyrl': 'mk', 'mya_Mymr': 'my', 'nld_Latn': 'nl', 'nob_Latn': 'no', 'npi_Deva': 'ne', 'oci_Latn': 'oc', 'ory_Orya': 'or', 'fas_Arab': 'fa', 'plt_Latn': 'mg', 'pol_Latn': 'pl', 'por_Latn': 'pt', 'ron_Latn': 'ro', 'rus_Cyrl': 'ru', 'sin_Sinh': 'si', 'slk_Latn': 'sk', 'slv_Latn': 'sl', 'snd_Arab': 'sd', 'som_Latn': 'so', 'spa_Latn': 'es', 'srp_Cyrl': 'sr', 'sun_Latn': 'su', 'swe_Latn': 'sv', 'swh_Latn': 'sw', 'tam_Taml': 'ta', 'crh_Latn': 'tt', 'tgl_Latn': 'tl', 'tur_Latn': 'tr', 'ukr_Cyrl': 'uk', 'urd_Arab': 'ur', 'uzn_Latn': 'uz', 'vie_Latn': 'vi', 'wol_Latn': 'wo', 'xho_Latn': 'xh', 'ydd_Hebr': 'yi', 'yor_Latn': 'yo', 'cmn_Hani': 'zh', 'zsm_Latn': 'ms', 'zul_Latn': 'zu', 'asm_Beng': 'as', 'ckb_Arab': 'ckb', 'vec_Latn': 'vec', 'eus_Latn': 'eu', 'lao_Laoo': 'lo', 'tha_Thai': 'th', 'kan_Knda': 'kn', 'hat_Latn': 'ht', 'san_Deva': 'sa', 'grn_Latn': 'gn', 'tgk_Cyrl': 'tg', 'scn_Latn': 'scn', 'uig_Arab': 'ug', 'tuk_Latn': 'tk', 'lim_Latn': 'li', 'tel_Telu': 'te', 'kir_Cyrl': 'ky', 'azb_Arab': 'azb', 'nno_Latn': 'nn', 'bos_Latn': 'bs', 'mai_Deva': 'mai', 'war_Latn': 'war', 'guj_Gujr': 'gu', 'bak_Cyrl': 'ba', 'arz_Arab': 'arz', 'mlt_Latn': 'mt', 'lmo_Latn': 'lmo', 'khk_Cyrl': 'mn', 'pbt_Arab': 'ps', 'quy_Latn': 'qu', 'min_Latn': 'min', 'kmr_Latn': 'ku', 'bod_Tibt': 'bo', 'yue_Hant': 'yue'}

lang2dcad = {'af': 'afr_Latn', 'sq': 'als_Latn', 'am': 'amh_Ethi', 'ar': 'arb_Arab', 'ast': 'ast_Latn', 'az': 'azj_Latn', 'be': 'bel_Cyrl', 'bn': 'ben_Beng', 'bg': 'bul_Cyrl', 'ca': 'cat_Latn', 'ceb': 'ceb_Latn', 'cs': 'ces_Latn', 'cy': 'cym_Latn', 'da': 'dan_Latn', 'de': 'deu_Latn', 'el': 'ell_Grek', 'en': 'eng_Latn', 'eo': 'epo_Latn', 'et': 'ekk_Latn', 'fi': 'fin_Latn', 'fr': 'fra_Latn', 'om': 'gaz_Latn', 'gd': 'gla_Latn', 'ga': 'gle_Latn', 'gl': 'glg_Latn', 'ha': 'hau_Latn', 'he': 'heb_Hebr', 'hi': 'hin_Deva', 'hr': 'hrv_Latn', 'hu': 'hun_Latn', 'hy': 'hye_Armn', 'ig': 'ibo_Latn', 'ilo': 'ilo_Latn', 'id': 'ind_Latn', 'is': 'isl_Latn', 'it': 'ita_Latn', 'jv': 'jav_Latn', 'ja': 'jpn_Jpan', 'ka': 'kat_Geor', 'kk': 'kaz_Cyrl', 'km': 'khm_Khmr', 'ko': 'kor_Hang', 'lt': 'lit_Latn', 'lb': 'ltz_Latn', 'lg': 'lug_Latn', 'lv': 'lvs_Latn', 'ml': 'mal_Mlym', 'mr': 'mar_Deva', 'mk': 'mkd_Cyrl', 'my': 'mya_Mymr', 'nl': 'nld_Latn', 'no': 'nob_Latn', 'ne': 'npi_Deva', 'oc': 'oci_Latn', 'or': 'ory_Orya', 'fa': 'fas_Arab', 'mg': 'plt_Latn', 'pl': 'pol_Latn', 'pt': 'por_Latn', 'ro': 'ron_Latn', 'ru': 'rus_Cyrl', 'si': 'sin_Sinh', 'sk': 'slk_Latn', 'sl': 'slv_Latn', 'sd': 'snd_Arab', 'so': 'som_Latn', 'es': 'spa_Latn', 'sr': 'srp_Cyrl', 'su': 'sun_Latn', 'sv': 'swe_Latn', 'sw': 'swh_Latn', 'ta': 'tam_Taml', 'tt': 'crh_Latn', 'tl': 'tgl_Latn', 'tr': 'tur_Latn', 'uk': 'ukr_Cyrl', 'ur': 'urd_Arab', 'uz': 'uzn_Latn', 'vi': 'vie_Latn', 'wo': 'wol_Latn', 'xh': 'xho_Latn', 'yi': 'ydd_Hebr', 'yo': 'yor_Latn', 'zh': 'cmn_Hani', 'ms': 'zsm_Latn', 'zu': 'zul_Latn', 'as': 'asm_Beng', 'ckb': 'ckb_Arab', 'vec': 'vec_Latn', 'eu': 'eus_Latn', 'lo': 'lao_Laoo', 'th': 'tha_Thai', 'kn': 'kan_Knda', 'ht': 'hat_Latn', 'sa': 'san_Deva', 'gn': 'grn_Latn', 'tg': 'tgk_Cyrl', 'scn': 'scn_Latn', 'ug': 'uig_Arab', 'tk': 'tuk_Latn', 'li': 'lim_Latn', 'te': 'tel_Telu', 'ky': 'kir_Cyrl', 'azb': 'azb_Arab', 'nn': 'nno_Latn', 'bs': 'bos_Latn', 'mai': 'mai_Deva', 'war': 'war_Latn', 'gu': 'guj_Gujr', 'ba': 'bak_Cyrl', 'arz': 'arz_Arab', 'mt': 'mlt_Latn', 'lmo': 'lmo_Latn', 'mn': 'khk_Cyrl', 'ps': 'pbt_Arab', 'qu': 'quy_Latn', 'min': 'min_Latn', 'ku': 'kmr_Latn', 'bo': 'bod_Tibt', 'yue': 'yue_Hant'}


def parse_args():
    global args
    parser = argparse.ArgumentParser(description="Parse rank and world size.")
    parser.add_argument("--target_dir", type=str, default="/leonardo_work/EUHPC_E03_068/datasets/working/mixture_vitae_", help="The target dataset.")
    parser.add_argument("--rank", type=int, default=0, help="Rank of the process (default: 0)")    
    parser.add_argument("--sample", type=int, default=0, help="Only sample number of files for testing (default: 0)")
    parser.add_argument("--subset", type=str, default="", help="subset of the data")        
    parser.add_argument("--world_size", type=int, default=1, help="Total number of processes (default: 1)")
    parser.add_argument("--create_upsample", type=int, default=2, help="Total number of upsample (default: 1)")    
    parser.add_argument("--add_related",  type=int, default=1,  help="Attaching realted data to current dataset where there is a match")
    parser.add_argument("--cache_dir",  type=str, default="./cache",  help="")

    args = parser.parse_args()
    return args


model = None
tokenizer = None

def init_model(device_no, arg):
    global model, tokenizer, device, args
    if  not torch.cuda.is_available():
        print ("not init'ing models b/c no GPU")
        return model, tokenizer
    args = arg
    if model is None or tokenizer is None:
      device = device.split(":")[0]
      if device == "cuda":
        device = "cuda:"+str(device_no)
      model_name = "Qwen/Qwen3-4B"
      if device == "cpu":
          model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, cache_dir=args.cache_dir).train().to(device)
      else:
          model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, cache_dir=args.cache_dir, attn_implementation="flash_attention_2").train().to(device)
      tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=args.cache_dir)
      tokenizer.padding_side  = 'left'      
    return model, tokenizer


def register_sent_for_paraphrasing(text2, default=1):
    global sentHash
    text2 = text2.replace("。",". ").replace("|>", "|>. ").replace("<|", ". <|").replace(".\n", ". ").replace("? ", "?. ").replace("! ", "!. ").replace("; ", ";. ").replace("| ", "|. ").replace("\n", ". ").\
        replace("1>", "1>. ").replace("2>", "2>. ").replace("3>", "3>. ").replace("4>", "4>. ").replace("5>", "5>. ").replace("6>", "6>. ").replace("7>", "7>. ").\
        replace("8>", "8>. ").replace("9>", "9>. ").replace("0>", "0>. ").replace("<", ". <").replace("<image>", "<image>. ").replace("<audio>", "<audio>. ")
    l_arr = [l2 for l2 in text2.split(". ") if '"' in l2 or len(l2) > 30]
    for l2 in l_arr:
        code4 = hash(l2.strip("|.。?!").lower())
        sentHash[code4] = sentHash.get(code4, default) + 1

def get_sent_for_paraphrasing(data):
    global sentHash
    if type(data) is str:
        text = data
    else:
        text = data['text']
    sents = {}
    for text2 in text.split("<|endoftext|>"):
      text2 = text2.replace("|", ". ")
      text2 = text2.replace("。",". ").replace("|>", "|>. ").replace("<|", ". <|").replace(".\n", ". ").replace("? ", "?. ").replace("! ", "!. ").replace("; ", ";. ").replace("| ", "|. ").replace("\n", ". ").\
          replace("1>", "1>. ").replace("2>", "2>. ").replace("3>", "3>. ").replace("4>", "4>. ").replace("5>", "5>. ").replace("6>", "6>. ").replace("7>", "7>. ").\
          replace("8>", "8>. ").replace("9>", "9>. ").replace("0>", "0>. ").replace("<", ". <").replace("<image>", "<image>. ").replace("<audio>", "<audio>. ")
      l_arr = [l2 for l2 in text2.split(". ") if '"' in l2 or len(l2) > 30]
      for l2 in l_arr:
          code4 = hash(l2.strip().lower())
          sents[l2.strip()] = sentHash.get(code4, 1)
    return sents


def augment_sent_with_synonyms_and_reordering(t, lang, stopwords, do_syn=False, do_reorder_stopword=False, do_reorder_comma=False):
    is_upper = t[0]==t[0].upper()
    if lang == "en" and do_syn:
        t = synonym_textaugment(t, 1.0)
    if do_reorder_comma and "," in t and '"' not in t and ':' not in t: # and r
        a, b = t.split(",",1)[-1].strip(), t.split(",",1)[0].strip()
        b = b[0].lower()+b[1:]
        t = a+", "+b
          
    if do_reorder_stopword and  '"' not in t and ':' not in t:
        if lang in  {'zh', 'ja', 'ko'}:
          for s in stopwords:
              if s in t:
                  pivot = t.index(s)
                  if pivot > 4:
                      t = t[pivot:]+","+t[:pivot]
                      break
        else:
          for s in stopwords:
              s = " "+s+" "
              if s in t:
                  pivot = t.index(s)
                  if pivot > 4:
                      t = t[pivot:]+", "+t[:pivot]
                      break
              s = " "+s+","
              if s in t:
                  pivot = t.index(s)
                  if pivot > 4:
                      t = t[pivot:]+", "+t[:pivot]
                      break
    if is_upper: t = t[0].upper()+t[1:]
    return t

stopwords2list = {}

def register_data_for_upsample(data, high_rep_threshold=0.3,  sent_reorder_prob=0, sent_upsample_prob=0.0, sent_shuffle_prob=0, do_augment=False, ):
    text_arr = []
    for text, metadata  in zip(data['text'].split("<|endoftext|>"), data['metadata']):
        text = register_text_for_upsample(text, data['lang'], high_rep_threshold=high_rep_threshold, sent_reorder_prob=sent_reorder_prob, sent_upsample_prob=sent_upsample_prob, sent_shuffle_prob=sent_shuffle_prob)
    text  = "<|endoftext|>".join(text_arr)
    return data

def register_text_for_upsample(text, lang, high_rep_threshold=0.3,  sent_reorder_prob=0.1, sent_upsample_prob=0.1,  sent_shuffle_prob=0.1, do_augment=False):
  global stopwords2list
  orig_text = text
  if lang in stopwords2list:
      stopwords = stopwords2list[lang]
  else:
      stopwords =  list(set(all_stopwords.get(lang, all_stopwords['en'])))
      stopwords.sort(key=lambda a: len(a))
      if len(stopwords) > 5:
          stopwords = stopwords[:5]
      stopwords2list[lang] = stopwords
  text = " "+text+" "
  if lang == 'hi':
      text_arr = text.split("| ")                  
  elif lang in  {'zh', 'ja', 'ko'}:
      text_arr =text.split("。")
  else:
      text_arr = text.replace(".\n", ". \n").split(". ")
  text_arr2 = []
  #if add_trans_prob:
  #    if random.randint(0,1):
          # translate whole document
          # translate continution
          # translate interleaved para
  #       pass

  random.shuffle(stopwords)
  sents = get_sent_for_paraphrasing(text)
  num_sents = len(text_arr)
  if len([s for s, cnt in sents.items() if cnt > 1])/num_sents >= high_rep_threshold:
      sent_upsample_prob = max(sent_upsample_prob, 0.5)
      print ("FOUND HIGH DUP DOCUMENT")
  for t in text_arr:
      t_orig = t
      start = len(t)-len(t.lstrip())
      beginning = t[:start]
      t = t.lstrip()
      if len(t.strip()) < 20: 
          text_arr2.append(t_orig)
          continue
      if not do_augment:
          if random.random() < sent_upsample_prob * sents.get(t,1):
              register_sent_for_paraphrasing(t)
          continue
      t = augment_sent_with_synonyms_and_reordering(t, lang, stopwords, do_syn=random.random() < sent_upsample_prob * sents.get(t,1), do_reorder_stopword=random.random() < sent_reorder_prob * sents.get(t,1), do_reorder_comma=random.random() < sent_reorder_prob * sents.get(t,1))
      t = beginning+t
      if t != t_orig:
          register_sent_for_paraphrasing(t)
      elif random.random() < sent_upsample_prob * sents.get(t,1):
          register_sent_for_paraphrasing(t)
      if sent_shuffle_prob<=0:
        text_arr2.append(t)  
      elif random.random() < sent_shuffle_prob  * sents.get(t,1) and len(text_arr2) > 3:
          a = text_arr2[-1]
          text_arr2 = text_arr2[:-1]
          arr = [a, t]
          random.shuffle(arr)
          text_arr2.extend(arr)
      elif random.random() < sent_shuffle_prob  * sents.get(t,1) and len(text_arr2) > 4:
          a, b = text_arr2[-1], text_arr2[-2]
          text_arr2 = text_arr2[:-2]
          arr = [a, b, t]
          random.shuffle(arr)
          text_arr2.extend(arr)
      elif random.random() < sent_shuffle_prob  * sents.get(t,1) and len(text_arr2) > 5:
          a, b,c = text_arr2[-1], text_arr2[-2], text_arr2[-3]
          text_arr2 = text_arr2[:-3]
          arr = [a, b, c, t]
          random.shuffle(arr)
          text_arr2.extend(arr)
      else:
          text_arr2.append(t)
          
  if lang == 'hi':
      text = "|".join(text_arr2)
  elif lang in  {'zh', 'ja', 'ko'}:
      text = "。".join(text_arr2)
  else:
      text = ". ".join(text_arr2)
  text = text.replace(". \n", ".\n").strip()
  return text.strip()
  
no_change_para = {}
def add_paraphrase(batch, add_translated_doc=0.3, ):
    global stopwords2list
    global sentHash    
    global model, tokenizer, device, no_change_para
    global input_dir, output_dir, tokenizer
    # data['lang'] == 'en'
    if add_translated_doc > 0:
        target_lang = random.choice(['ru', 'sv', 'xh', 'ml', 'lv', 'id', 'ko', 'ga', 'mt', 'ja', 'pt', 'ro', 'ar', 'sk', 'fr', 'fi', 'pl', 'mr', 'hi', 'af', 'tr', 'cs', 'lt', 'bg', 'nl', 'es', 'th', 'zh', 'gl', 'hu', 'hr', 'mk', 'el', 'sw', 'it', 'de', 'vi', 'az'])
    else:
        target_lang = random.choice(["de", "fr", "es", "pt"])
    en_other_model, en_other_tokenizer = get_translation_model_tokenizer("en",target_lang)
    other_en_model, other_en_tokenizer = get_translation_model_tokenizer(target_lang, "en")    
    if en_other_model is None and other_en_model is None:
        print ("Not adding paraphrases b/c there is no model")
        return batch
    
    #if model is None:
    #    print ("Not adding paraphrases b/c there is no model")
    #    return batch
    print ("adding para")
    para_batch = []
    orig_batch = []
    aug = []
    batch_size = 100
    # do case where we translate based on ctranslate2 instead of paraphrasing
    # do case where we want to do round trip translate for paraphrasing
    prev_lang = ""
    for data in batch:
        for metadata, text in zip(data['metadata'], data['text'].split("<|endoftext|>")):
            lang = metadata['lang']
            orig_text = text
            if prev_lang != lang:
                if lang in stopwords2list:
                    stopwords = stopwords2list[lang]
                else:
                    stopwords =  list(set(all_stopwords.get(lang, all_stopwords['en'])))
                    stopwords.sort(key=lambda a: len(a))
                    if len(stopwords) > 5:
                        stopwords = stopwords[:5]
                    stopwords2list[lang] = stopwords
                random.shuffle(stopwords)
                prev_lang = lang
            sents = get_sent_for_paraphrasing(text)
            items = [(s0, cnt)  for s0, cnt in sents.items() if cnt > 1]
            for t, cnt in items:
                if "<image" in t or "<audio" in t or "<caption" in t or "<transcript" in t: continue
                if "</image" in t or "</audio" in t or "</caption" in t or "</transcript" in t: continue
                if t in orig_batch: continue
                s1 = t
                s1 = s1.strip("1234567890~!@#$%^&_-+=,.?/<>[]{}().- *,:|\\")
                if s1 in orig_batch: continue            
                if len(s1) < 10: continue
                for rng in range(0, len(s1), 1000):
                    s2 = s1[rng: min(len(s1), rng+1000)]
                    if (data['lang'] not in {"zh", "ja", "ko"} and len(s2) <= 10): continue
                    if (data['lang'] in {"zh", "ja", "ko"} and len(s2) <= 2): continue
                    if s2 not in no_change_para and s2 not in orig_batch:
                        orig_batch.append(s2)
                        s3 = augment_sent_with_synonyms_and_reordering(s2, lang, stopwords, do_syn=random.random() > 1/cnt, do_reorder_stopword=random.random() > 1/cnt, do_reorder_comma=random.random() > 1/cnt)
                        para_batch.append(s3)
                    #para_batch.append(tokenize_with_chat_template(tokenizer, instruction=random.choice(["Paraphrase", "Revise", "Fix", "Augment"])+f" this sentence or fragment to vary the grammar, wording and punctuation, and fix any issues. {prefix} Do not provide commentary. {last}:\n===\n{s2}"))
    if para_batch:
        print (len(para_batch))
        # round trip translation
        other_batch = [s2.strip().split("\n")[0].rstrip(" .") for s2 in  ctranslate2_with_batching(en_other_model, en_other_tokenizer, para_batch, batch_size=batch_size, max_new_tokens=200)]
        mapper = dict([(s, (so, s2.strip().split("\n")[0].rstrip(" ."))) for s, so, s2  in zip(orig_batch, other_batch, ctranslate2_with_batching(other_en_model, other_en_tokenizer, other_batch, batch_size=batch_size, max_new_tokens=200))])
        #print (mapper)
        for data in batch:
            prefix = ""
            text_arr = []
            meta_arr = []
            for text, metadata in zip(data['text'].split("<|endoftext|>"), data['metadata']):
                total_translated = 0
                trans_text = text
                num_sent = (1 + (text.count(".")+text.count("|")+text.count("?")+text.count("!")+text.count("。")))
                for s, s2 in mapper.items():
                    s_trans, s2 = s2
                    if (metadata['lang'] not in {"zh", "ko", "ja"} and len(s2) < 20) or (metadata['lang'] in {"zh", "ko", "ja"} and len(s2) < 5) or s.strip() == s2.strip():
                        no_change_para[s] = 1
                        no_change_para[s2] = 1                        
                        continue
                    if s in text:
                        # todo, ignore case where lang2 ==lang
                        #print (s, "=>",s_trans, "=>", s2)
                        text = text.replace(s,s2)
                        register_sent_for_paraphrasing(s, 0)
                        trans_text = trans_text.replace(s,s_trans)
                        register_sent_for_paraphrasing(s_trans, 0)                        
                        total_translated += 1
                text_arr.append(text)
                meta_arr.append(metadata)
                if add_translated_doc >= 0 and (trans_text.strip() != text.strip() and ((total_translated/num_sent > add_translated_doc) and "###" not in text and "\n===\n" not in text)):
                    prefix =  "### "+random.choice(["A", "This", "The", "Here is", "Please find", "Provided",])+" " +\
                        random.choice(["machine generated text", "translated document", "copy", ])+ " " + \
                        random.choice(["including language mixing between", "for a language switching document using", " of an interleaved content with"]) + " " + langs2fullname.get(target_lang, target_lang)  +" and English.\n===\n"
                    prefix = prefix.replace("find", random.choice(["find", "see", "read"]))
                    prefix = prefix.replace("mixing", random.choice(["mixing", "mixture of", "combining with"]))
                    prefix = prefix.replace("interleaved", random.choice(["interleaved", "alternating", "translated"]))
                    prefix = prefix.replace("content", random.choice(["text", "content", "document"]))
                    prefix = prefix.replace("text", random.choice(["text", "content", "document"]))
                    text = prefix + trans_text
                    print ((total_translated/num_sent, add_translated_doc, text))
                    text_arr.append(text)
                    meta_arr.append(metadata)                
            data['text']  = "<|endoftext|>".join(text_arr)
            data['metadata'] = meta_arr
            
    return batch

def dedup_paraphrase_upsample_reduce(shard, files, output_dir,  high_rep_threshold=0.3, sent_reorder_prob=0.0, sent_upsample_prob=0.0, sent_shuffle_prob=0.0, augment_only_permissive=True,  do_augment=False):
    global common_pile_sites, white_list_sites
    if common_pile_sites is None:
        common_pile_sites = set(json.load(open("common_pile_urls.json")))
        white_list_sites = set(json.load(open("white_list_urls.json")))
    
    if not files: return ""
    i = 0
    batch = []
    upsample_batch = []
    for file in files:
        for l in open(file, "rb"):
            l = l.strip()
            try:
                data = json.loads(l)
            except:
                continue
            data = dedup(data)
            if not data:
                continue
            fix_idx(data)
            upsampled=False
            if do_augment and (not augment_only_permissive or (any(meta for meta in data['metadata'] if meta['idx'] in white_list_sites or meta['idx'] in common_pile_sites or meta['is_govt'] or \
                                                                 'curated' in meta['source'] or 'kl3m' in meta['source'] or 'common-pile' in meta['source']))):
                new_text = []
                new_meta = []
                for text, metadata in zip(data['text'].split("<|endoftext|>"), data['metadata']):
                    new_text.append(text)
                    new_meta.append(metadata)
                    text2 =  register_text_for_upsample(text, metadata['lang'], high_rep_threshold=high_rep_threshold, sent_reorder_prob=max(0.1, sent_reorder_prob), sent_upsample_prob=max(0.1,sent_upsample_prob), sent_shuffle_prob=max(0.1,sent_shuffle_prob), do_augment=True)
                    if text2 != text:
                        new_text.append(text2)
                        new_meta.append(metadata)
                        upsampled = True                        
                data['text'] = "<|endoftext|>".join(new_text)
                data['metadata'] = new_meta

            if upsampled or not augment_only_permissive or (any(meta for meta in data['metadata'] if meta['idx'] in white_list_sites or meta['idx'] in common_pile_sites or meta['is_govt'] or 'curated' in meta['source'] or 'kl3m' in meta['source'] or 'common-pile' in meta['source'])):
                upsample_batch.append(data)
                if len(upsample_batch) > 400:
                    generate_upsample(upsample_batch, high_rep_threshold=high_rep_threshold, sent_reorder_prob=sent_reorder_prob, sent_upsample_prob=sent_upsample_prob, sent_shuffle_prob=sent_shuffle_prob)
                    upsample_batch = []
            batch.append(data)
            if len(batch) >= 200000:
                generate_upsample(upsample_batch, high_rep_threshold=high_rep_threshold, sent_reorder_prob=sent_reorder_prob, sent_upsample_prob=sent_upsample_prob, sent_shuffle_prob=sent_shuffle_prob)                            
                with open(output_dir+f"/{shard}_{i}.tmp1", "a+") as outf:
                    for data in batch:
                        outf.write(json.dumps(data)+"\n")
                #print ("sort --parallel 10 "+  output_dir+f"/{shard}_{i}.tmp1 -o " +  output_dir+f"/{shard}_{i}.tmp1")
                os.system("sort --parallel 10 "+  output_dir+f"/{shard}_{i}.tmp1 -o " +  output_dir+f"/{shard}_{i}.tmp1")                
                i+=1
                batch=[]
                upsample_batch = []
    if batch:
        generate_upsample(upsample_batch,  high_rep_threshold=high_rep_threshold, sent_reorder_prob=sent_reorder_prob, sent_upsample_prob=sent_upsample_prob, sent_shuffle_prob=sent_shuffle_prob)        
        with open(output_dir+f"/{shard}_{i}.tmp1", "a+") as outf:
            for data in batch:
                outf.write(json.dumps(data)+"\n")
        #print ("sort --parallel 10 "+  output_dir+f"/{shard}_{i}.tmp1 -o " +  output_dir+f"/{shard}_{i}.tmp1")
        os.system("sort --parallel 10 "+  output_dir+f"/{shard}_{i}.tmp1 -o " +  output_dir+f"/{shard}_{i}.tmp1")
    all_shards = list(glob.glob(output_dir+f"/{shard}_*.tmp1"))        
    len_tmp = len(all_shards)
    for file in all_shards:
        shard_num = int(file.split("/")[-1].split("_")[-1].replace(".tmp1", ""))
        if shard_num > i:
            os.system("sort --parallel 10 "+  output_dir+f"/{shard}_{shard_num}.tmp1 -o " +  output_dir+f"/{shard}_{shard_num}.tmp1")
    if len_tmp > 1:
        #print ("sort --parallel 10 -m "+  output_dir+f"/{shard}_*.tmp1 -o " +  output_dir+f"/{shard}.tmp1")
        os.system("sort --parallel 10 -m "+  output_dir+f"/{shard}_*.tmp1 -o " +  output_dir+f"/{shard}.tmp1")
        os.system("rm "+ output_dir+f"/{shard}_*.tmp1")
        return output_dir+f"/{shard}.tmp1"
    elif len_tmp == 1:
        os.system("mv "+ all_shards[0] + " " + output_dir+f"/{shard}.tmp1")
        return output_dir+f"/{shard}.tmp1"
    
def generate_upsample(batch, add_translated_doc=0.3, do_register=True,  high_rep_threshold=0.3,  sent_reorder_prob=0, sent_upsample_prob=0.1, sent_shuffle_prob=0, do_augment=False,):
    print ("upsample")
    global model, tokenizer
    if do_register:
        for data in batch:
            register_data_for_upsample(data, high_rep_threshold=high_rep_threshold,  sent_reorder_prob=sent_reorder_prob, sent_upsample_prob=sent_upsample_prob, sent_shuffle_prob=sent_shuffle_prob, do_augment=do_augment)
    batch_size = 25
    aug = []
    textmapper = {}
    add_paraphrase(batch, add_translated_doc=add_translated_doc)
    if model is None:
        print ("not adding upsampling b/c there is no GPU model")
        return batch
    upsampled=False
    for data in batch:
        for text in data['text'].split("<|endoftext|>"):
            if text.endswith("<think>"):
                aug.append(text)
    textmapper =  textmapper | dict([(s, s2) for s, s2 in zip(aug, generate_with_batching(model, tokenizer, aug, max_new_tokens=1600, skip_special_tokens=False, batch_size=batch_size/4, too_much_ngram_threshold=2, supress_tokens=llm_slop_phrases, supress_self_trigram_topk=10))])                
    aug = []
    for data in batch:
        for text in data['text'].split("<|endoftext|>"):
            if not text.startswith("<|im_start|>"):
                aug.append(text)
    textmapper =  textmapper | dict([(s, s2) for s, s2 in zip(aug, generate_with_batching(model, tokenizer, aug, max_new_tokens=1028, skip_special_tokens=False, batch_size=batch_size/2, too_much_ngram_threshold=2, supress_tokens=llm_slop_phrases, supress_self_trigram_topk=10))])
    aug = []
    for data in batch:
        for text in data['text'].split("<|endoftext|>"):
            if text.startswith("<|im_start|>") and not text.endswith("<think>"):
                aug.append(text)
    textmapper =  textmapper | dict([(s, s2) for s, s2 in zip(aug, generate_with_batching(model, tokenizer, aug, max_new_tokens=512, skip_special_tokens=False, batch_size=batch_size, too_much_ngram_threshold=2, supress_tokens=llm_slop_phrases, supress_self_trigram_topk=10))])
    for data in batch:
        for text in data['text'].split("<|endoftext|>"):
            if "<|im_start|>" in text and text in textmapper:
                if text.count("<|im_start|>")==1:
                    textmapper[text] = (text.split("<|im_start|>")[-1].split("\n",1)[-1]+textmapper[text]).strip(" =\n")

    for data in batch:
        for text, meta in zip(data['text'].split("<|endoftext|>"), data['metadata']):
            if "<|im_start|>" in text and text in textmapper:
                new_text= textmapper[text]
                lang = langid.classify(new_text[100:min(len(new_text), 600)])[0]
                meta['lang'] = lang
                print ((lang, new_text, '<=', text))
                if "### Revised Instruction:" in text:
                    data['text'] = data['text'].replace(text, new_text.strip(" =\n"))
                elif text.count("<|im_start|>")==1:
                    data['text'] = data['text'].replace(text, new_text)
                else:
                    data['text'] = data['text'].replace(text, text + new_text)
                upsampled=True                    
    if upsampled: add_paraphrase(batch, add_translated_doc=add_translated_doc)                        
    return batch
    
def fix_idx(data):
    if data['idx'].startswith("z://"): return data['idx']
    idx= data['idx'].replace("www.","").split("://")[-1].split(".")[0]
    # sometimes things that start with "en.wikipedia" for example creates a shard that is too big to quickly manage                                                                                    
    # we may also want to collapse multiple items of the same thing form different langauges. so modify the URL itself.                                                                                
    if idx in langs2fullname and data['idx'].split("://")[-1].count("/") > 0:
        lang = idx
        # for some wikipedia urls, the url starts with the language and not the subject matter. modify the url to start with the subject to make a better index.                                       
        if idx.startswith("wik") and data['idx'].split("://")[-1].count("/") > 1:
            idx= data['idx'].replace("www.","").split("://")[-1].split("/")[2]
            data['idx'] = "https://"+data['idx'].split("://")[-1].replace("/"+idx, "/@#@").replace(lang+".", idx.lower()+".").replace("@#@", lang)
        else:
            idx= data['idx'].replace("www.","").split("://")[-1].split("/")[1]
            data['idx'] = "https://"+data['idx'].split("://")[-1].replace("/"+idx, "/@#@").replace(lang+".", idx.lower()+".").replace("@#@", lang)
    if not data['idx'].startswith("https://"):
        data['idx'] = "https://"+data['idx'].split("://")[-1]
    data['idx'] = data['idx'].lower().strip()
    return data['idx']

unix_chars = "%&!;()[]`^=?$<>{}*~:|\'\"/"

if  torch.cuda.is_available():
    multiprocessing.set_start_method('spawn', force=True)

def add_translations(data, src_lang, target_lang, max_length=2048):
    if src_lang == target_lang: return data
    model, tokenizer = get_translation_model_tokenizer(src_lang, target_lang)                            
    orig_full_text = data['text']
    if data['lang'] != src_lang:
        return data
    text_arr = []
    meta_arr = []
    curr_total_len = 0
    
    for text, meta in zip(data['text'].split("<|endoftext|>"), data['metadata']):
        out = ""
        orig_text = text
        text = text.split(" ")
        lang = meta['lang']
        if lang != src_lang:
            text_arr.append(text)
            continue
        # if src_lang or target_lang is cjk, we need to change the code slighlty
        # add a labse check
        for rng in tqdm(range(0, len(text), 150)):
            curr_total_len+= 150
            if curr_total_len > max_length: break
            text2 = " ".join(text[rng:min(len(text),rng+150)])
            lang = langid.classify(text2[10:min(len(text2), 100)])[0]
            if lang != src_lang:
                out += " ... "                    
            text3 = ctranslate2_with_batching(model, tokenizer, [text2])[0]
            max_word_len = max(20, max([len(a) for a in text2.split()]))
            text3 = " ".join([a for a in text3.split(" ") if len(a) < max_word_len])
            if len(text3) < len(text2)-10:
                out += " ... "
            else:
                text3 = fix_too_much_ngram(text3, window_size=3, lang=target_lang)
                if text2[0] not in "`1234567890-=~!@#$%^&*()_+<>,.:\";'/?{}[]|\\" and text2[0] == text2[0].lower():
                    text3 = text3[0].lower()+text3[1:]
                elif text2[0] == text2[0].upper():
                    text3 = text3[0].upper()+text3[1:]            
                out += " " + text3
        out = out.replace("...  ...", "...").replace("...  ...", "...").replace("... ...", "...").replace("...  ...", "...").replace("...  ...", "...").replace("... ...", "...")
        if orig_text.replace("\n", " ").strip() == out.strip():
            continue
        print ((out,))
        text_arr.append(orig_text+"<|endofsection|>## Translation to "+ langs2fullname.get(target_lang, target_lang)+":\n"+out)
    data['text'] = "<|endoftext|>".join(text_arr)
    return data
 
if __name__ == "__main__":
    pass
    args = parse_args()
    for l in open("../mixture_vitae_2/101.jsonl"):
        data = json.loads(l)
        add_translations(data, "en", "de")
            
