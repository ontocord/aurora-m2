from time import sleep
from typing import Dict, List
import os
import wget
from pathlib import Path
from tqdm import tqdm
import copy
import json
import spacy
import base64
import uuid
import hashlib
import random
from io import BytesIO
import numpy as np
from numpy import asarray
from collections import deque
import numpy as np
import torch
import torchvision
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import random

import torch
import PIL
from PIL import Image
from transformers import pipeline
from datasets import load_dataset
from torch.nn.functional import cosine_similarity
from transformers import CLIPProcessor, CLIPModel, AutoModel, AutoTokenizer, AutoModelWithLMHead
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from src.frcnn.visualizing_image import SingleImageViz
from src.frcnn.processing_image import Preprocess as FRCNNPreprocess
from src.frcnn.modeling_frcnn import GeneralizedRCNN
from src.frcnn.utils import Config as FRCNNConfig
from src.frcnn.utils import decode_image as frcnn_decode_image



import cv2
import numpy as np
from matplotlib import colors
from collections import OrderedDict


try:
    from vllm import RequestOutput
except:
    pass


### GLOBAL VARIABLES

digits_to_words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
                  'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen',
                  'nineteen', 'twenty']


hsv_color_ranges = {
    "red": [(0, 50, 50), (10, 255, 255)],  # Expanded to include more shades of red
    "red_alt": [(170, 50, 50), (180, 255, 255)],  # Wraparound red for hue near 0/180
    "orange": [(10, 50, 50), (25, 255, 255)],  # Expanded for different shades of orange
    "yellow": [(25, 50, 50), (35, 255, 255)],  # Expanded yellow range
    "lime green": [(35, 50, 50), (70, 255, 255)],  # Expanded bright green (lime) range
    "green": [(35, 50, 50), (85, 255, 255)],  # Combined lime and green ranges
    "cyan": [(80, 50, 50), (95, 255, 255)],  # Broadened cyan range
    "blue": [(80, 50, 50), (140, 255, 255)],  # Combined cyan, blue, and indigo ranges
    "indigo": [(115, 50, 50), (140, 255, 255)],  # Broadened range for indigo (between blue and violet)
    "purple": [(130, 50, 50), (160, 255, 255)],  # Expanded to include various shades of purple
    "pink": [(140, 50, 50), (170, 255, 255)],  # Combined pink and magenta ranges
    "magenta": [(140, 50, 50), (170, 255, 255)],  # Similar to pink, but more intense
    "brown": [(10, 50, 20), (20, 255, 200)],  # Broadened range for brown
    "black": [(0, 0, 0), (180, 255, 50)],  # Broadened black range
    "white": [(0, 0, 230), (180, 30, 255)],  # Adjusted to capture all shades of white
    "gray": [(0, 0, 40), (180, 20, 255)]  # Combined dark gray, gray, and light gray ranges
}

color_table_bgr = [('red', (0, 0, 255)),
      ('green', (0, 255, 0)),
      ('blue', (255, 0, 0)),
      ('white', (255, 255, 255)),
      ('black', (0, 0, 0)),
      ('gray', (211, 211, 211)),
      ('yellow', (0, 255, 255)),
      ('blue', (255, 255, 0)),  # Initially was blue (now cyan)
      ('pink', (255, 0, 255)),
      ('orange', (0, 165, 255)),
      ('purple', (128, 0, 128)),
      ('pink', (203, 192, 255)),
      ('brown', (42, 42, 165)),
      ('purple', (238, 130, 238)),  # Same as original
      ('purple', (130, 0, 75))]  # Indigo purple

light_colors_bgr = {
    "white": [255, 255, 255],
    "pink": [200, 200, 255],
    "yellow": [0, 255, 255],
    "blue": [230, 216, 173],
    "brown": [140, 230, 240],
    "green": [144, 238, 144],
    "coral": [193, 182, 255]
}

light_colors_bgr_keys = list(light_colors_bgr.keys())

dark_colors_bgr = {
    "black": [0, 0, 0],
    "gray": [105, 105, 105],
    "blue": [139, 0, 0],  # Dark red as blue
    "red": [0, 0, 139],  # Dark blue as red
    "green": [0, 100, 0],
}

dark_colors_bgr_keys = list(dark_colors_bgr.keys())

numbering_list = ['3', '7)', '7.', '4', 'iii.', 'iii-', '8.', '4-', 'v:', 'I:', 'ii.', 'i.', 'V)', 'E)', 'I)', 'III.', 'III)', '2-', '1)', 'v-', 'III', 'I.', 'c)', '1.', 'V-', 'iv)', 'A)', 'v)', 'IV', 'C.', 'ii)', 'I', 'IV.', 'C)', 'II-', '2.', 'III-', 'IV)', 'd)', 'iii', 'i-', 'iii:', 'A.', 'B.', '1', '6)', 'ii', '8)', '3)', 'e)', 'ii-', '5-', 'II)', 'iv-', '2)', 'e.', 'IV:', 'III:', 'i)', '10.', 'V', 'V.', 'v.', 'D)', 'E.', 'iv:', 'B)', 'II', 'ii:', 'V:', 'a.', '5.', 'IV-', '9.', 'D.', '3.', '4:', '2:', 'i', 'II.', '3-', '2', 'c.', 'a)', '3:', '10)', 'd.', 'i:', 'iv.', '1-', '4.', '5', 'iv', 'iii)', 'b.', '1:', 'II:', 'v', '5:', '6.', 'b)', 'I-', '9)', '4)', '5)']
stopwords_list = ['es', 'ing', 'ed', 'include', 'includes', 'also', 'haven', 'are', 'why', 'most', "won't", 'against', 'with', 'needn', 'couldn', 'now', 'mustn', 'who', 'under', 'doing', 'am', 'aren', 'they', "didn't", 'd', 'doesn', 'if', 'he', 'her', "haven't", 'isn', 'own', 'does', 'such', 'until', 'into', 'had', 'again', 'over', "hadn't", "you'll", 't', 'by', 'be', "wasn't", 'so', 'yours', 'both', 'any', 'did', "you've", 'these', 'myself', 'o', 'hasn', "isn't", 'you', 'other', 'shan', 'being', 'yourselves', 'was', 'no', 'm', 'those', 'will', 'its', 'itself', 'have', 'down', 'weren', 'having', 'wouldn', 'herself', "mustn't", 'very', 'do', "should've", 'him', "you'd", 'below', 'just', 'that', 'for', 'which', 'but', 'nor', 'all', 'then', 'i', 'whom', 'it', 'once', 'here', 've', "you're", 'ours', "that'll", 'a', 'won', 'himself', 'where', 'this', 'your', "hasn't", 'same', 'when', 'ourselves', 'because', "needn't", 'theirs', 'from', 'mightn', 'my', 'while', 'yourself', "she's", 'each', "doesn't", 'only', 'at', 's', 'their', "wouldn't", 'shouldn', 'and', 'themselves', 'hers', 'has', 'up', 'ma', 'in', 'll', 'we', 're', 'y', 'of', 'after', 'our', "shan't", 'before', 'wasn', 'can', 'should', 'been', 'through', 'as', 'further', 'during', 'between', 'there', 'me', 'on', 'don', "shouldn't", 'more', 'out', "don't", 'the', "weren't", "aren't", "it's", 'what', 'or', "couldn't", 'hadn', "mightn't", 'his', 'above', 'to', 'how', 'few', 'off', 'them', 'didn', 'ain', 'not', 'she', 'an', 'than', 'too', 'is', 'some', 'were', 'about']

common_title_words_set = {'introduction', 'conclusion', 'section', 'chapter', 'works', 'notes', 'note', 'further', 'see', 'references', 'reference', 'section', 'title', 'conclusion', 'intro', 'introduction', 'executive', 'summary', 'key', 'plot', 'theme'}
stopwords_set = set(stopwords_list + numbering_list)

default_sides = ["top", "top", "top", "top", "top",
                 "bottom", "bottom", "bottom", "bottom",
                 "left", "left",
                 "right", "right",
                 "upper left", "lower left",
                 "upper right", "lower right", "center"]
all_sides  = ["top", "bottom", "lower left", "upper left", "lower right", "upper right", "left", "right", "center", ]

base_colors = [ 'orange', 'cyan',
 'yellow',
 'lime green',
 'green',
 'blue',
 'indigo',
 'purple',
 'pink',
 'magenta',
 'brown',
 'black',
 'white',
 'gray']

discuss_phrases = [
    "document containing", "translate", "named", "states", "reads", "translating",
    "naming", "stating", "reading", "explanation", "labeled", "label", "calls for",
    "advertise", "advertising", "title", "titled", "information", "info", "explaining",
    "mentioned", "explained", "described", "mention", "explain", "describe",
    "emphasiz", "emphasize", "emphasized", "details the", "detailing the", "noting",
    "discuss", "discussed", "discussing", "quotes", "quotation", "speaks about",
    "talks about", "communicates", "message", "description", "paragraph", "sentence",
    "reference to", "referred to", "defining", "clarifies", "clarified", "informs",
    "presents", "presents details of", "recounts", "narrates", "elaborates on",
    "details", "highlighting that", "highlights in text", "shows in writing", "depicts in writing",
    "in words", "textual explanation", "verbal description", "summary of",
    "report on", "documented", "corresponds to", "mentions", "statement",
    "articulated", "provides details", "addresses", "suggests", "indicates",
    "written account", "lecture", "written depiction", "tells about", "annotated",
    "remarks", "notes", "defines", "specifies", "proposes", "conveys", "outlines",
    "clarifying", "summarizing", "documenting", "footnote", "annotation",
    "analyzes", "breaks down", "examines", "the passage", "the text indicates",
    "reports", "concludes", "observes", "elucidates", "delves into", "references",
    "interprets", "glossary", "analyzing", "refers to", "overview", "expounds on",
    "written explanation", "verbalizes", "further details", "outlines the key points",
    "restates", "contextualizes", "assesses", "reflects on", "summarized",
    "reviews", "offers insights", "an investigation of", "evaluates", "opinion",
    "sheds light on", "supports the idea", "expresses", "inscribed", "inscribing",
]

discuss_phrases.sort(key=lambda a: len(a), reverse=True)
text_mentioning_phrases = [
    "states", "stating", "reads", "reading", "words", "written", "text", "entitled", "titled", "title", "font", "caption",
    "subtitles", "heading", "label", "wording", "written word",
    "print", "typing", "typography", "annotations", "inscription", "motto",
    "slogan",  "written description", "subheading",
    "chapter", "line of text", "dialogue",  "font size",
    "printed", "words on", "tagline", "message written", "footnote", "header",
    "watermark", "quotation marks", "headline", "byline", "text formatting",
    "bullet points", "italicized", "bolded", "text placement",
    "footer", "annotation", "inline text", "typeface", "typed", "phrase",
    "textual", "quote marks",  "signage", "document title", "label text"
]

text_mentioning_phrases.sort(key=lambda a: len(a), reverse=True)

# this corresponds to the TurkuNLP registries. Add this when we know the type of registry a particular text is.
styles = ["Lyrical", "Spoken", "Interview", "Interactive Discussion", "Narrative", "News Report", "Sports Report", "Narrative Blog", "How-to", "Recipe", "Informational Description",
         "Encyclopedia Article", "Research Article", "Descriptive Article", "FAQ", "Opinion", "Review", "Opinion Blog",
         "Denominational Religious Blog or Sermon", "Informational Persuasion", "Sales Pitch", "News and Opinon Blog or Editoral", ]

length = ["Long", "Short", "Medium", "One Paragraph", "Two Paragraph", "Five Paragraph", "1000 words", "10 words", "100 words"]

professions = [
    "Engineer",
    "Doctor",
    "Nurse",
    "Teacher",
    "Software Developer",
    "Data Scientist",
    "Lawyer",
    "Pharmacist",
    "Researcher",
    "Accountant",
    "Architect",
    "Chef",
    "Dentist",
    "Journalist",
    "Pilot",
    "Photographer",
    "Police Officer",
    "Veterinarian",
    "Writer",
    "Painter",
    "Musician",
    "Athlete",
    "Actor",
    "Psychologist",
    "Carpenter",
    "Electrician",
    "Plumber",
    "Social Worker",
    "Farmer",
    "Mechanic"
]

# Use this if there is no life-skill involved (e.g., non-how-to videos)
tasks_template_list = [
    "Critical Thinking",
    "Problem Solving",
    "Communication",
    "Teamwork",
    "Adaptability",
    "Time Management",
    "Organization",
    "Creativity",
    "Emotional Intelligence",
    "Leadership",
    "Self-Motivation",
    "Stress Management",
    "Decision Making",
    "Assertiveness",
    "Resilience",
    "Empathy",
    "Negotiation",
    "Conflict Resolution",
    "Budgeting",
    "Computer Literacy",
    "Foreign Language",
    "Cultural Awareness",
    "Networking",
    "Personal Hygiene",
    "Cooking",
    "First Aid",
    "Document Drafting",
    "Purchasing",
    "Selling",
    "Risk Management",
]

### BASIC UTILITIES
def get_target_path(shard_path: Path, dst_file_path: Path) -> Path:
    filename = shard_path.name
    return dst_file_path / filename


def mark_done(filepath: Path):
    with open(filepath.with_suffix('.done'), "w") as fp:
        pass
    return


def check_done(filepath: Path):
    return filepath.with_suffix('.done').exists()


def get_sublist(elements, rank, world_size):
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
    print(f"File '{input_filename}' has been split into {shard_number} shards.")


def filter_splits(sorted_shards: List[Path], rank: int, world_size: int) -> List[Path]:
    return get_sublist(sorted_shards, rank, world_size)


def get_splits(path, rank: int, world_size: int, samples_per_shard: int):
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
    if not os.path.exists(path):
        if rank == 0:
            print("downloading dataset")
            Path(path).parent.mkdir(exist_ok=True, parents=True)
            wget.download(url)
        else:
            print("no file detecting, downloading with root")
            sleep(600)



def postprocess_hf_results(result: List[Dict[str, List[Dict[str, str]]]], result_key: str = "generated_text", txt_key: str = "content") -> List[str]:
    return [r[0][result_key][1][txt_key] for r in result]

def postprocess_vllm_results(results):
    return [i.outputs[0].text for i in results]

def postprocess_results(results):
    if isinstance(results[0], RequestOutput):
        return postprocess_vllm_results(results)
    else:
        return postprocess_hf_results(results)


def chunkify(sequence, n):
    """Splits a sequence into N roughly equal-sized chunks."""
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
data_fields = ['text', 'text_type', 'chosen', 'rejected_list', 'media_list', 'media_caption_scores_list', 'media_coordinates_list', 'media_types_list', 'metadata']
metadata_fields = ['source', 'params']

def standardize_data_fields(data):
    #make sure the data is in the standard format.
    #move everything to the metadata.params field otherwise
    if 'meta' in data and 'metadata' not in data:
        data['metadata'] = data['meta']
        del data['meta'] # let's map meta->metadata.
    if 'metadata' not in data:
        data['metadata'] = {}
    if 'subset' in data and 'source' not in data['metadata']:
        data['metadata']['source'] = data['subset']
        del data['subset'] # let's map sbset->metadata.source
    for field in data_fields:
      if field not in data:
        if '_list' in field:
          data[field] = []
        elif field == 'metadata':
          data["metadata"] = {}
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
    ret = []
    for data in curr_data:
        ret.append(standardize_data_fields(data))
    return ret

def cleanup_and_serialize_params(data):
    standardize_data_fields(data)    
    if not data['metadata']['params']:
        data['metadata']['params'] = "{}"      
    elif type(data['metadata']['params']) is not str:
        data['metadata']['params'] = json.dumps(data['metadata']['params'])
    return data

### BASIC TEXT LLM GENERATION ROUTINES

import re

def non_english_detect(text):
    if re.search("[\u0000-\u00BF]", text):
        return False
    return True
    
def cjk_detect(text):
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
    return [tokenizer.decode([idx]) for idx in range(len(tokenizer)) if cjk_detect(tokenizer.decode([idx]))]

def get_non_english_tokens(tokenizer):
    return [tokenizer.decode([idx]) for idx in range(len(tokenizer)) if non_english_detect(tokenizer.decode([idx]))]

def chatml_format_instructions_old(system, instruction, response=""):
  system= system.strip()
  instruction = instruction.strip()
  if system:
    return f"""<|im_start|>system
{system}
<|im_end|>
<|im_start|>user
{instruction}
<|im_end|>
<|im_start|>assistant
"""
  else:
    return f"""<|im_start|>user
{instruction}
<|im_end|>
<|im_start|>assistant
{response}"""

def generate_with_batching_old(model, tokenizer, data, device,  use_cache=True, repetition_penalty=1.2, no_repeat_ngram_size=4, max_new_tokens=200, batch_size=5, **args):
  torch.cuda.empty_cache()
  output = []
  for rng in range(0, len(data), batch_size):
    d = data[rng:min(len(data), rng+batch_size)]
    if d:
      output.extend(tokenizer.batch_decode(model.generate(**tokenizer(d, truncation=True, padding=True, return_tensors="pt", add_special_tokens=False, ).to(device),
                        use_cache=use_cache, repetition_penalty=repetition_penalty, no_repeat_ngram_size=no_repeat_ngram_size, max_new_tokens=max_new_tokens, **args)))
  torch.cuda.empty_cache()
  return output

# formats strings to chat_template accepted by a LLM. 
def chatml_format_instructions(tokenizer, system, instruction, response=""):
  system= system.strip()
  instruction = instruction.strip()
  if system:
    return tokenizer.apply_chat_template([{"role": "system", "content": system}, 
                                          {"role": "user", "content": instruction}], tokenize=False)
  else:
    return tokenizer.apply_chat_template([{"role": "user", "content": instruction}], tokenize=False)
#   if system:
#     return f"""<|im_start|>system
# {system}
# <|im_end|>
# <|im_start|>user
# {instruction}
# <|im_end|>
# <|im_start|>assistant
# """
#   else:
#     return f"""<|im_start|>user
# {instruction}
# <|im_end|>
# <|im_start|>assistant
# {response}"""


def get_tokens_as_list(word_list, tokenizer):
    "Converts a sequence of words into a list of tokens"
    tokens_list = []
    for word in word_list:
        tokenized_word = tokenizer([word], add_special_tokens=False).input_ids[0]
        tokens_list.append(tokenized_word)
    return tokens_list

    
# generate output from a batch of inputs
def generate_with_batching(model, tokenizer, prompts, use_cache=True, repetition_penalty=1.2,  max_new_tokens=400, batch_size=2, skip_special_tokens=True, return_continuations_only=True, dont_decode_non_english=False, dont_decode_cjk=False, supress_tokens=[], media_list=None, **args):

  # this is the InternVL2 batch chat function adapted to our purposes. We don't need to use the conv template because
  # we already send the data into the model in the right chat format.
  def internvlm_batch_chat(self, tokenizer, pixel_values, questions, generation_config, num_patches_list=None,
                         IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
                         IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False, image_counts=None, return_continuations_only=True,
                         skip_special_tokens=True):

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

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
  
  if dont_decode_cjk and not hasattr(tokenizer, 'cjk_ids'):
      str_list = get_cjk_tokens(tokenizer)
      tokenizer.cjk_ids = get_tokens_as_list(str_list, tokenizer)
          
  if dont_decode_non_english and not hasattr(tokenizer, 'non_english_ids'):
      str_list = get_non_english_tokens(tokenizer)
      tokenizer.non_english_ids = get_tokens_as_list(str_list, tokenizer)    
  # qwen has a problem of sometimes outputing cjk randomly. we fix this by including cjk token in the bad_words_ids

  device = model.device
  #torch.cuda.empty_cache()
  output = []
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
            logger.warning(f"Supress token word {word} is not a single token. skipping")
            continue
        supress_tokens_hash[word] =  tokenized_ids[0]
    bad_words_ids.extend([supress_tokens_hash[word] for word in supress_tokens])
  if dont_decode_cjk:
      bad_words_ids.extend(tokenizer.bad_words_ids)
  if dont_decode_non_english:
      bad_words_ids.extend(tokenizer.non_english_ids)
  if media_list:
      assert len(media_list) == len(prompts), "media_list is a list of lists of images for each prompt. the len of media_list must be the same as the len of prompts"
  with torch.no_grad():
      for rng in range(0, len(prompts), batch_size):
          d = prompts[rng:min(len(prompts), rng+batch_size)]
          if media_list and hasattr(model, 'batch_chat') and any(s for s in d if "<image>" in s):
              # this is a internVLM situation
              generation_config = copy.copy(args)
              generation_config['use_cache'] = use_cache
              generation_config['repetition_penalty'] = repetition_penalty
              generation_config['max_new_tokens'] = max_new_tokens
              if bad_words_ids:
                  generation_config['bad_words_ids']=bad_words_ids
              imgs = media_list[rng:min(len(prompts), rng+batch_size)]
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
              output.extend(responses)
          elif d:
              model_inputs = tokenizer(d, truncation=True, padding=True, return_tensors="pt", add_special_tokens=False, ).to(device)
              prompt_len = model_inputs["input_ids"].shape[-1]
              if bad_words_ids:
                  model_output = model.generate(**model_inputs, bad_words_ids=bad_words_ids,
                                                use_cache=use_cache, repetition_penalty=repetition_penalty,  max_new_tokens=max_new_tokens,  **args)
              else:
                  model_output = model.generate(**model_inputs,
                                                            use_cache=use_cache, repetition_penalty=repetition_penalty,  max_new_tokens=max_new_tokens,  **args )
              if return_continuations_only:
                  model_output = model_output[:, prompt_len:]
              output.extend(tokenizer.batch_decode(model_output, skip_special_tokens=skip_special_tokens,))
                      
  #torch.cuda.empty_cache()
  # we don't want to empty the cache every generation. this could really slow things down. this should be done per batch i think. to prevent fagmentation
  return output

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
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        torchvision.transforms.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
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
    image = image.convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

### FINDING TEXT IN CAPTIONS

def remove_quotes(text):
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
  accum = []
  text = text.replace("'s ", " @s@ ").replace("'ve ", " @ve@ ").replace("'m ", " @m@ ").replace("'t ", " @t@ ")
  for idx, segment in enumerate(text.split("'")):
    if idx % 2 != 0:
      accum.append(segment)
  accum = [a.replace(" @s@ ", "'s ").replace(" @ve@ ", "'ve ").replace( " @m@ ", "'m ").replace(" @t@ ", "'t ").replace("  ", " ").replace("  ", " ").strip() for a in accum]
  accum.sort(key=lambda a: len(a), reverse=True)
  return accum

def strip_left_stopwords(e_text):
  e_text2 = []
  add_rest = False
  for et in e_text.split():
      if add_rest or ((et.lower() not in stopwords_set and et.lower not in common_title_words_set) or et.lower().strip(".") in {"a", "an", "united", "the", "new", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",  "asian", "american", "african", "european", }):
        add_rest = True
        e_text2.append(et)
  return " ".join(e_text2)


def strip_right_stopwords(e_text):
  e_text2 = []
  add_rest = False
  e_text_arr = e_text.split()
  e_text_arr.reverse()
  for et in e_text_arr:
      if add_rest or (et.lower() not in stopwords_set or et.lower().strip(".") in {"act", "code", "statute", "regulation", "regulations", "percent", "feet", "foot", "square", "barrells", "hour", "hours", "people", "asian", "american", "african", "european", "act", "law", "facilities", "facility", "center", "square", "rd", "street", "way", "blvd", "ave", "avenue", "states", "kingdom", "court", "corp", "corporation", "co", "company", "ltd", "llc", "llp", "incorp.", "incorporated"}):
        add_rest = True
        e_text2.append(et)
  return " ".join(reversed(e_text2))

# remove text like info from captions
def augment_for_quotes(caption_array, text_cutoff=20):
    # Modify the original caption by appending adversarial suffix
    caption_array2 = []

    # if this is text we previously created, then let's parse it out
    # we use a specific format of At the {side}, there is the following text "{text}".
    for caption in caption_array:
      start_to_template_text = 0
      for side in all_sides:
          if f"At the {side} there is the following text \"" in caption:
              idx = caption.index(f"At the {side}, there is the following text \"")
              if start_to_template_text == 0:
                  start_to_template_text = idx
              else:
                  start_to_template_text = min(start_to_template_text, idx)
      if start_to_template_text:
            ret = {}
            caption2 = caption          
            first_part = caption2[:start_to_template_text]
            for side in all_sides:
              if f"At the {side} there is the following text \"" in caption:
                  for _ in range(5):
                    color = random.choice(base_colors)
                    if color in first_part and color not in ret:
                      continue
                    break
                  text = caption.split(f"At the {side}, there is the following text \"", 1)[-1].split("\"",1)[0]
                  caption2 = caption2.replace(f"At the {side}, there is the following text \"{text}\".", f"At the {side}, there is a large {color} solid rectangle.")
                  text = text.replace("\\n", "\n")
                  ret[color] = ret.get(color, []) + [(side, text)]
            caption_array2.append((caption, caption2, list(ret.items())))
            continue

      # otherwise, this is a raw caption, so we need to parse for all text element. we will try to insert color rectangle at different locations associated with the text.
      caption = caption.replace("\"", "'")
      caption = caption.replace("infographic", "image").replace("slides", "image").replace("document", "image")
      for _ in range(5):
        color = random.choice(base_colors)
        if color in caption:
          continue
        break
      ret = []
      accum = find_quotes(caption)
      accum2 = []
      caption2 = []
      caption3 =[]
      for sentence in caption.split(". "):
        add = False
        for s in accum:
          if s not in sentence: continue
          if len(s) > text_cutoff:
            sentence = sentence.replace(s, '')
            side = random.choice(default_sides)
            for side2 in all_sides:
                if side2 in sentence.lower() or (side2 == "top" and ("above" in sentence.lower() or "upper" in sentence.lower())) \
                 or (side2 == "bottom" and ("lower" in sentence.lower() or "below" in sentence.lower())) or \
                  (side2 == "center" and "middle" in sentence.lower()):
                  side = side2
                  break
            sentence = sentence.replace("the words ''", f"a large {color} solid rectangle "+random.choice(["at", "in", "on", "by", "to"])+f" the {side}")
            sentence = sentence.replace("title ''",  f"a large {color} solid rectangle "+random.choice(["at", "in", "on", "by", "to"])+f" the {side}")
            sentence = sentence.replace("titled ''",  f"a large {color} solid rectangle "+random.choice(["at", "in", "on", "by", "to"])+f" the {side}")
            sentence = sentence.replace("named ''",  f"a large {color} solid rectangle "+random.choice(["at", "in", "on", "by", "to"])+f" the {side}")
            sentence = sentence.replace("states ''",  f"a large {color} solid rectangle "+random.choice(["at", "in", "on", "by", "to"])+f" the {side}")
            sentence = sentence.replace("reads ''",  f"a large {color} solid rectangle "+random.choice(["at", "in", "on", "by", "to"])+f" the {side}")
            sentence = sentence.replace("stating ''",  f"a large {color} solid rectangle "+random.choice(["at", "in", "on", "by", "to"])+f" the {side}")
            sentence = sentence.replace("reading ''",  f"a large {color} solid rectangle "+random.choice(["at", "in", "on", "by", "to"])+f" the {side}")
            sentence = sentence.replace("which translates to ''", "")
            for word in text_mentioning_phrases:
              sentence = sentence.replace(word, " ")
            sentence = sentence.replace("  ", " ")
            if 'solid rectangle'  in sentence:
              ret.append((color, side, s))
              for _ in range(5):
                color = random.choice(base_colors)
                if color in caption:
                  continue
                break
            elif s not in accum2:
              accum2.append(s.strip(",.")+".")
          break
        # when there is no quote, there may be phrases that denotes discussions or explanations
        if any(b for b in discuss_phrases if b in sentence):
          s = ""
          for word in discuss_phrases:
            if word not in sentence: continue
            _, info = sentence.split(word,1)
            for word2 in discuss_phrases:
              info = info.replace(word2, " ")
            info = strip_left_stopwords(info)
            if len(info) > 10:
              s = info[0].upper()+info[1:]
              s = s.strip(",.") + "."
          if "'" not in sentence:
            for word in text_mentioning_phrases:
              sentence = sentence.replace(word, " ")
          for word2 in discuss_phrases:
            sentence = sentence.split(word2, 1)[0]
          if s:
            if random.randint(0,5) != 0:
              accum2.append(s)
            else:
              side = random.choice(default_sides) # ["top", "bottom", "lower left", "upper left", "lower right", "upper right", "left", "right", "center", ]
              for side2 in all_sides:
                  if side2 in sentence.lower() or (side2 == "top" and ("above" in sentence.lower() or "upper" in sentence.lower())) \
                  or (side2 == "bottom" and ("lower" in sentence.lower() or "below" in sentence.lower())) or \
                    (side2 == "center" and "middle" in sentence.lower()):
                    side = side2
                    break
              sentence = sentence +f" with a large {color} solid rectangle in the {side}"
              if len(sentence) < 20: continue
              ret.append((color, side, s))
              for _ in range(5):
                  color = random.choice(base_colors)
                  if color in caption:
                    continue
                  break
        for word in text_mentioning_phrases:
            sentence = sentence.replace(word, " ")
        caption2.append(sentence)
      caption2 = ". ".join(caption2)
      caption2 = caption2.replace("''", " ")
      caption2 = caption2.strip(".")+"."
      caption2 = ".".join(s for s in caption2.split(".") if s.count("rectangle") + s.count("solid")  < 4)
      caption2 = caption2.replace("  ", " ").replace("  ", " ")
      if accum2:
        if len(accum2) > 10:
          sides = ["left", "right", "top", "bottom", "upper left", "upper right", "lower left", "lower right", "center"]
        else:
          sides = ["top", "bottom", "left", "right", "upper left", "upper right", "lower left", "lower right", "center"]
        for side2 in sides:
          if side2 not in caption2:
            if (side2 == "top" and ("above" in sentence.lower() or "upper" in sentence.lower())) \
                 or (side2 == "bottom" and ("lower" in sentence.lower() or "below" in sentence.lower())) or \
                  (side2 == "center" and "middle" in sentence.lower()): continue
            side = side2
            break
        ret.append((color, side, "\n".join(accum2)))
        for _ in range(5):
          color = random.choice(base_colors)
          if color in caption:
            continue
          break
      accumHash = {}
      colorHash = {}
      for color, side, text in ret:
        colorHash[side] = color
        accumHash[side] = accumHash.get(side, '')
        for t in text.split("\n"):
            if t in accumHash[side]: continue
            accumHash[side] = accumHash[side] + "\n" + t
        accumHash[side] = accumHash[side].strip()
      ret = {}
      for side, text in list(accumHash.items()):
        color = colorHash[side]
        ret[color] = ret.get(color, [])
        ret[color].append((side, text))
        if not text.strip(): continue
        if side not in caption2:
          caption2 = random.choice([f"There is a large {color} solid rectangle "+random.choice(["at", "in", "on", "by", "to"])+f" the {side}.",
                            f"The image is mostly on one side, and there is a large {color} solid rectangle "+random.choice(["at", "in", "on", "by", "to"])+f" the {side}.",]) + " " + caption2

        f"There is a large {color} solid rectangle in the {side}. " + caption2
      caption2 = " "+caption2+" "
      caption2 = caption2.replace("that a", "a").replace(",.", ".").replace("It also.", "").replace(" the.", ".").replace("The image.", "").replace("image is an image", "image").replace("The.", "").replace("The the ", "The ").replace("The a ", "A ").replace("The an ", "An ").replace(" the an ", " an ").replace(" the a ", " a ").replace("The of ", "The ").replace(" the of ", " the ").replace(" the , ", ", ").replace(" a , ", ",").replace(" .", ".").strip()
      caption2 = caption2.replace("that a", "a").replace(",.", ".").replace("It also.", "").replace(" the.", ".").replace("The image.", "").replace("image is an image", "image").replace("The.", "").replace("The the ", "The ").replace("The a ", "A ").replace("The an ", "An ").replace(" the an ", " an ").replace(" the a ", " a ").replace("The of ", "The ").replace(" the of ", " the ").replace(" the , ", ", ").replace(" a , ", ",").replace(" .", ".").strip()
      caption_no_textbox = caption2
      for color, sides in ret.items():
        for side in sides:
          for prep in ["at", "in", "on", "by", "to"]:
            caption_no_textbox = caption_no_textbox.replace(f"There is a large {color} solid rectangle {prep} the {side[0]}.", "").\
              replace(f"The image is mostly on one side, and there is a large {color} solid rectangle {prep} the {side[0]}.", "").\
              replace(f"a large {color} solid rectangle {prep} the {side[0]}", "")
      caption_no_textbox = caption_no_textbox.replace(" with .", ".").replace(" with.", ".").replace("  ", " ").replace(" .", ".").strip()
      caption_no_textbox = caption_no_textbox.replace("that a", "a").replace(",.", ".").replace("It also.", "").replace(" the.", ".").replace("The image.", "").replace("image is an image", "image").replace("The.", "").replace("The the ", "The ").replace("The a ", "A ").replace("The an ", "An ").replace(" the an ", " an ").replace(" the a ", " a ").replace("The of ", "The ").replace(" the of ", " the ").replace(" the , ", ", ").replace(" a , ", ",").replace(" .", ".").strip()
      caption_no_textbox = caption_no_textbox.replace("that a", "a").replace(",.", ".").replace("It also.", "").replace(" the.", ".").replace("The image.", "").replace("image is an image", "image").replace("The.", "").replace("The the ", "The ").replace("The a ", "A ").replace("The an ", "An ").replace(" the an ", " an ").replace(" the a ", " a ").replace("The of ", "The ").replace(" the of ", " the ").replace(" the , ", ", ").replace(" a , ", ",").replace(" .", ".").strip()

      for text_list in ret.values():
          for side, text in text_list:
              caption_no_textbox = caption_no_textbox.strip(".") + f". At the {side} there is the following text \""+text.replace("\n", "\\n")+"\". "

      caption_array2.append ((caption_no_textbox, caption2, list(ret.items())))
      
    return caption_array2

### FUNCTIONS TO INSERT TEXT INTO IMAGES
# Function to convert a color name to an HSV tuple directly for color detection
def get_hsv_from_name(color_name):
    upper, lower =  hsv_color_ranges.get(color_name)
    return [int(upper[0]+lower[0]/2), int(upper[1]+lower[1]/2), int(upper[2]+lower[2]/2) ]

# Function to map rectangles to relative positions
def get_position(x, y, w, h, img_width, img_height):
    cx, cy = x + w // 2, y + h // 2  # Center of the rectangle

    # Determine if the rectangle is primarily aligned on the left, right, top, or bottom
    if cx < img_width // 3:
        if cy < img_height // 3:
            return 'upper left'
        elif cy > 2 * img_height // 3:
            return 'lower left'
        else:
            return 'left'
    elif cx > 2 * img_width // 3:
        if cy < img_height // 3:
            return 'upper right'
        elif cy > 2 * img_height // 3:
            return 'lower right'
        else:
            return 'right'
    elif cy < img_height // 3:
        return 'top'
    elif cy > 2 * img_height // 3:
        return 'bottom'
    else:
        return "center"

def create_default_rectangles(img_width, img_height):
    # Create the OrderedDict with the specified order of rectangles
    default_rectangles = OrderedDict([
        ("top", (int(0.01 * img_width), int(0.01 * img_height), int(0.98 * img_width), int(0.2 * img_height))),
        ("bottom", (int(0.01 * img_width), int(0.75 * img_height), int(0.98 * img_width), int(0.2 * img_height))),
        ("left", (int(0.01 * img_width), int(0.01 * img_height), int(0.48 * img_width), int(0.98 * img_height))),
        ("right", (int(0.51 * img_width), int(0.01 * img_height), int(0.48 * img_width), int(0.98 * img_height))),
        ("upper left", (int(0.01 * img_width), int(0.01 * img_height), int(0.48 * img_width), int(0.2 * img_height))),
        ("upper right", (int(0.51 * img_width), int(0.01 * img_height), int(0.48 * img_width), int(0.2 * img_height))),
        ("lower left", (int(0.01 * img_width), int(0.75 * img_height), int(0.48 * img_width), int(0.2 * img_height))),
        ("lower right", (int(0.51 * img_width), int(0.75 * img_height), int(0.48 * img_width), int(0.2 * img_height))),
        ("center", (int(0.25 * img_width), int(0.25 * img_height), int(0.5 * img_width), int(0.5 * img_height))),

    ])
    return default_rectangles


# Main function to detect rectangles and assign text. Given a list of
# [(color, ((position, text) ....], find color rectangles and clear
# the recatangel to the background color, and write in the text
def replace_color_rectangles_with_text(image, replace_list, clear_background=True):
    # Convert to PIL RGB to cv2, BGR
    original_image = np.array(image)
    original_image = original_image[:, :, ::-1]
    image = original_image.copy()
    for detection_color, text_list in replace_list:
        replace_color_rectangles_with_text_bgr(original_image, image, text_list,  clear_background=clear_background)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)


def replace_color_rectangles_with_text_bgr(original_image, image, text_list, detection_color="pink", clear_background=True):

    # Get the HSV range for the specified detection color
    lower_bound, upper_bound = hsv_color_ranges.get(detection_color, hsv_color_ranges["white"])
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_mean = np.mean(hsv[:, :, 0])  # Average Hue
    s_mean = np.mean(hsv[:, :, 1])  # Average Saturation
    v_mean = np.mean(hsv[:, :, 2])  # Average Value (Brightness)

    # Return the average HSV value
    replace_color =  (h_mean, s_mean, v_mean)

    # Threshold the image to get only colors in the range
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = list(contours)
    if detection_color == "pink":
      lower_bound, upper_bound = hsv_color_ranges.get("magenta", [(0, 0, 200), (180, 30, 255)])
      mask = cv2.inRange(hsv, lower_bound, upper_bound)
      contours2, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
      contours.extend(list(contours2))
    elif detection_color == "blue":
      lower_bound, upper_bound = hsv_color_ranges.get("cyan", [(0, 0, 200), (180, 30, 255)])
      mask = cv2.inRange(hsv, lower_bound, upper_bound)
      contours2, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
      contours.extend(list(contours2))
    elif detection_color == "red":
      lower_bound, upper_bound = hsv_color_ranges.get("red_alt", [(0, 0, 200), (180, 30, 255)])
      mask = cv2.inRange(hsv, lower_bound, upper_bound)
      contours2, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
      contours.extend(list(contours2))
    elif detection_color == "purple":
      lower_bound, upper_bound = hsv_color_ranges.get("indigo", [(0, 0, 200), (180, 30, 255)])
      mask = cv2.inRange(hsv, lower_bound, upper_bound)
      contours2, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
      contours.extend(list(contours2))
    elif detection_color == "green":
      lower_bound, upper_bound = hsv_color_ranges.get("lime greeen", [(0, 0, 200), (180, 30, 255)])
      mask = cv2.inRange(hsv, lower_bound, upper_bound)
      contours2, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
      contours.extend(list(contours2))

    # Assign text to positions
    position_map = OrderedDict()  # Maps positions like 'upper right' to contours
    remaining_text = []  # Stores text that doesn't have a specific position

    img_height, img_width, _ = image.shape

    # Iterate through contours and check for rectangles
    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.05 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check if the contour has 4 points (suggests a rectangle) and is convex
        if len(approx) == 4 and cv2.isContourConvex(approx):
            # Check if the approximated polygon is close to a rectangle
            rect = cv2.boundingRect(approx)
            x, y, w, h = rect
            if w > img_width - 10 and h > img_height - 10:
              continue
            # Filter out very small rectangles
            if w > 150 and h > 60:

                position = get_position(x, y, w, h, img_width, img_height)
                if not position: continue
                position_map[position] = rect  # Store the bounding box for this position
    if not position_map:
      clear_background = False
      position_map = create_default_rectangles(img_height, img_width)
    default_position = create_default_rectangles(img_height, img_width)
    # Process the text list
    for item in text_list:
        if isinstance(item, tuple):  # Tuple containing (position, text)
            position, text = item
            if position in position_map:
                rect = position_map.pop(position)
                del default_position[position]
                x, y, w, h = rect
                replace_color = original_image[max(0,x-10), max(0,y-10)].tolist()
                draw_text_in_rectangle_bgr(image, rect, text, replace_color,  clear_background=clear_background)
            else:
                remaining_text.append(item)
        else:
            remaining_text.append(item)

    for item in remaining_text:
        if isinstance(item, tuple):  # Tuple containing (position, text)
            position, text = item
            remaining_text.pop(0)
            if position in default_position:
                rect = default_position.pop(position)
                x, y, w, h = rect
                replace_color = original_image[max(0,x-10), max(0,y-10)].tolist()
                draw_text_in_rectangle_bgr(image, rect, text, replace_color, clear_background=False)

    for position in list(position_map.keys()):
        rect = position_map[position]
        if remaining_text:
            item = remaining_text.pop(0)
            if isinstance(item, tuple):  # Tuple containing (position, text)
              position, text = item
            else:
              text = item
            del default_position[position]
            x, y, w, h = rect
            replace_color = original_image[max(0,x-10), max(0,y-10)].tolist()
            draw_text_in_rectangle_bgr(image, rect, text, replace_color, clear_background=clear_background)
        else:
          break

    for position in list(default_position.keys()):
        rect = default_position[position]
        if remaining_text:
            item = remaining_text.pop(0)
            if isinstance(item, tuple):  # Tuple containing (position, text)
              position, text = item
            else:
              text = item
            del default_position[position]
            x, y, w, h = rect
            replace_color = image[max(0,x-10), max(0,y-10)].tolist()
            draw_text_in_rectangle_bgr(image, rect, text, replace_color,  clear_background=False)
        else:
          break


def get_color_name(rgb_value, tolerance=100):
    # Define the color table

    def is_within_range(color1, color2, tolerance):
        """Check if two RGB colors are within the specified tolerance."""
        return all(abs(c1 - c2) <= tolerance for c1, c2 in zip(color1, color2))

    # Loop through the color table to find the closest matching RGB value within the tolerance
    for color_name, color_value in color_table_bgr:
        if is_within_range(color_value, rgb_value, tolerance):
            return color_name
    return "Unknown color"

# Function to draw text inside a rectangle, using random fonts, random justification, random line types, and larger text size
# NOTE: image is in BGR not RGB
def draw_text_in_rectangle_bgr(image, rect, text, replace_color, clear_background=True):
    x, y, w, h = rect
    # Function to determine if the background color is dark
    def is_color_dark(b, g, r):
        luminance = (0.299 * r + 0.587 * g + 0.114 * b)
        return luminance < 128

    if clear_background:
      replace_color_name = get_color_name(replace_color)
    else:
      replace_color_name = None
    # Choose the font color based on the brightness of the replace_color
    if is_color_dark(replace_color[0], replace_color[1], replace_color[2]):
        for _ in range(5):
          if random.randint(0,1):
            font_color = random.choice(light_colors_bgr_keys)  # Use a light color for dark backgrounds
          else:
            font_color = "white"
          if font_color == replace_color_name: continue
          break
        font_color = light_colors_bgr[font_color]
    else:
        for _ in range(5):
          if random.randint(0,1):
            font_color = random.choice(dark_colors_bgr_keys)  # Use a dark color for light backgrounds
          else:
            font_color = "black"
          if font_color == replace_color_name: continue
          break
        font_color = dark_colors_bgr[font_color]
    # Draw the background rectangle
    if clear_background:
        cv2.rectangle(image, (x, y), (x + w, y + h),replace_color , -1)

    # Split the text into lines based on newline characters
    lines = text.split("\n")

    # Randomly select a font from OpenCV fonts (adding more options)
    fonts = [
        cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_COMPLEX_SMALL, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
        cv2.FONT_HERSHEY_SCRIPT_COMPLEX, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_ITALIC
    ]
    font = random.choice(fonts)

    # Start with a larger font scale
    font_scale = (random.random() + 2)*2
    font_thickness = max(2, int(font_scale * 2))  # Use a larger font thickness

    # Randomly select a line type
    line_types = [cv2.LINE_AA, cv2.LINE_8, cv2.LINE_4]
    line_type = random.choice(line_types)

    # Find the longest line to fit within the width of the rectangle
    longest_line = max(lines, key=len)
    text_size = cv2.getTextSize(longest_line, font, font_scale, font_thickness)[0]

    # Adjust font_scale to fit the longest line within the rectangle's width
    max_text_width = w * 0.9  # Allow text to take up 90% of the rectangle's width
    max_text_height = h * 0.9  # Allow text to take up 90% of the rectangle's height
    step_size = int(text_size[1] + font_thickness * 1.2)  # Step size for each line

    while text_size[0] > max_text_width and font_scale > 0.5:
        font_scale -= 0.1
        font_thickness = max(1, int(font_scale * 2))
        text_size = cv2.getTextSize(longest_line, font, font_scale, font_thickness)[0]

    step_size = int(text_size[1] + font_thickness * 1.2)  # Step size for each line

    # Calculate total text height to ensure it fits within the rectangle's height
    total_text_height = len(lines) * step_size  # Height for all lines

    # Now that the font scale fits, draw each line of text
    y_offset = y + (h - total_text_height) // 2  # Center the text vertically

    # Randomly choose the text alignment (left, center, right)
    justifications = ["left", "center", "right"]
    justification = random.choice(justifications)

    for i, line in enumerate(lines):
        line_size = cv2.getTextSize(line, font, font_scale, font_thickness)[0]

        # Determine the x-coordinate based on the justification
        if justification == "left":
            text_x = x + int(0.05 * w)  # Left-aligned (5% padding)
        elif justification == "right":
            text_x = x + w - line_size[0] - int(0.05 * w)  # Right-aligned (5% padding)
        else:
            text_x = x + (w - line_size[0]) // 2  # Center-aligned

        text_y = y_offset + step_size  # Move down for each line
        y_offset = text_y + (line_size[1] + font_thickness)

        # Put each line of text on the image with random line type
        cv2.putText(image, line, (text_x, text_y), font, font_scale, font_color, font_thickness, line_type)

#### CREATE IMAGE + TEXT DATA
def tokenize_with_chat_template(tokenizer, messages):
  return tokenizer.apply_chat_template(messages, tokenize=False)

def tokenize_with_assistant_continuation(tokenizer, messages):
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

def strip_left_stopwords(e_text):
  e_text2 = []
  add_rest = False
  for et in e_text.split():
      if add_rest or (et.lower() not in stopwords_set):
        add_rest = True
        e_text2.append(et)
  return " ".join(e_text2)

def assign_uuid(input_string: str = None) -> uuid.UUID:
    """Assign a UUID to a string using MD5 hash.
    
    input_string: str
        The string to be hashed.
    """
    if input_string is None:
        return str(uuid.uuid4())
    return uuid.UUID(hashlib.md5(input_string.encode('UTF-8')).hexdigest())

def pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")  # You can change the format if needed (JPEG, etc.)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

#### IMAGE + TEXT FUSION ROUTINES

def cosim_eval(images, texts):
    # evaluate the generated text by comparing its similarity with flux generated image 
    inputs = clip_processor(images=images, return_tensors="pt")
    clip_vision_output = clip_model.vision_model(**inputs)
    image_features = clip_model.visual_projection(clip_vision_output["pooler_output"])

    inputs = clip_processor(texts, padding=True, truncation=True, max_length=76, return_tensors="pt").to(accelerator.device)
    text_features = clip_model.get_text_features(**inputs)
    cos_scores = cosine_similarity(image_features, text_features, dim=1)

    return cos_scores


def clip_image_to_multitext_score(clip_model, clip_processor, image, text_array, clip_vision_output=None, text_features=None, \
                                  cls_weight=.9, box_add_factor=.65, decompose_image=True, normalized_boxes=None, \
                                  ignore_from_box=None, num_boxes=5, box_segmentation_model=None, image_preprocessor=None,\
                                  score_cutoff = 0.2):
  assert len(text_array) > 0, "No text_array"
  if ignore_from_box is None: ignore_from_box = {}
  p = next(clip_model.parameters())
  frcnn_output = None
  attr_ids = None
  # we use a box segmenter so we can get bounding boxes and hints about where things generally are
  if box_segmentation_model is not None:
   frcnn_output = frcnn_decode_image(asarray(image), box_segmentation_model,  image_preprocessor, max_detections=num_boxes)
   normalized_boxes = frcnn_output["normalized_boxes"][0].cpu()
   #print (frcnn_output["attr_ids"])
   #attr_ids = [frcnn_ids.attrids[int(i)] if prob > .4 and frcnn_ids.attrids[int(i)] != "blue" else "" for i, prob in zip(frcnn_output["attr_ids"][0], frcnn_output["attr_probs"][0]) ]
  if attr_ids is None and normalized_boxes is not None:
    attr_ids = [""]* len(normalized_boxes)

  decomposed_image_features = None
  if type(image) is np.array:
      pil_image = PIL.Image.fromarray(image)
  else:
      pil_image = image
      image = np.array(image)
  box_imgs = []
  if normalized_boxes is not None:
    imgs = [image]
    coords = []
    shape = pil_image.size

    for x1,y1,x2,y2 in normalized_boxes:
      l = ((x2-x1) + (y2-y1))/2.0
      if l < 0.20: continue
      box_img_coord = [int(x1*shape[0]), int(y1*shape[1]), int((x2)*shape[0]), int((y2)*shape[1])]
      #print (box_img_coord)
      coords.append(box_img_coord)
      box_PIL_img = pil_image.crop(box_img_coord)
      box_imgs.append(box_PIL_img)
      imgs.append(np.array(box_PIL_img))
      #display(box_PIL_img)
  else:
    imgs = [image]
    coords = [[0,0,1,1]]
  if clip_vision_output is None:
    inputs = clip_processor(images=imgs, return_tensors="pt")
    if True: # with torch.no_grad():
      inputs['pixel_values'] = inputs['pixel_values'].to(dtype=p.dtype, device=p.device)
      inputs['return_dict'] = True
      clip_vision_output = clip_model.vision_model(**inputs)
      if decompose_image:
        o = (clip_vision_output["last_hidden_state"][0,1:,:] + cls_weight*10*clip_vision_output["last_hidden_state"][0,0,:])/(cls_weight*10+1)
        clip_vision_output.decomposed_image_features = clip_model.visual_projection(clip_model.vision_model.post_layernorm(o))
      # image_features[0] is the main picture, the rest are the box parts of the picture
      clip_vision_output.image_features = clip_model.visual_projection(clip_vision_output["pooler_output"])
  image_features = clip_vision_output.image_features
  if hasattr(clip_vision_output, 'decomposed_image_features'):
    decomposed_image_features = clip_vision_output.decomposed_image_features
  if type(text_array) is str: text_array = [text_array]
  if text_features is None:
   inputs = clip_processor(text_array, padding=True, return_tensors="pt").to(p.device)
   try: # with torch.no_grad():
     text_features = clip_model.get_text_features(**inputs)
   except:
     return None
  scores =  cosine_similarity(image_features[0].unsqueeze(0), text_features, dim=1)
  if len(imgs) > 1:
    box_scores_topk = []
    box_scores = []
    box_image_features = image_features[1:]
    #print (scores)
    #print ('box_image_features.shape', box_image_features.shape)
    text_array2 = []
    for cidx, tfeat in enumerate(text_features):
      if text_array[cidx] in ignore_from_box: continue
      text_array2.append(text_array[cidx])
      scores2 =  min (1.0, (scores[cidx] + box_add_factor)) * cosine_similarity(box_image_features, tfeat.unsqueeze(0), dim=1)
      box_scores_topk.append(scores2.topk(k=min(len(text_array), box_image_features.shape[0])))
      box_scores.append(box_scores_topk[-1].values[0])
    if box_scores:
     box2element = {}
     element2box_cnt = {}
     element2attr = {}
     box_scores = torch.stack(box_scores)
     cindices = box_scores.sort().indices.tolist()
     cindices.reverse()
     for cidx in cindices:
       text, topk = text_array2[cidx], box_scores_topk[cidx]
       for idx, score in zip(topk.indices.tolist(), topk.values.tolist()):
         if idx not in box2element:
           box2element[idx] = (text, score, coords[idx], attr_ids[idx])
           break
       #for idx, score in zip(topk.indices.tolist(), topk.values.tolist()):
       #  if score > score_cutoff:
       #    element2box_cnt[text] = element2box_cnt.get(text,0) + 1
    else:
     box2element = None
     #element2box_cnt = None
     box_scores_topk = None
     box_scores = None
     box_image_features  = None
  else:
    box2element = None
    #element2box_cnt = None
    box_scores_topk = None
    box_scores = None
    box_image_features  = None

  return {'image': image, 'box_images': box_imgs, 'image_features': image_features[0].unsqueeze(0),  \
           'normalized_boxes': normalized_boxes, 'coords': coords, 'box_image_features': box_image_features, 'box2element': box2element, \
           'scores': scores, 'clip_vision_output': clip_vision_output, 'text_features': text_features} # , 'element2box_cnt': element2box_cnt

#given a sentence, break the sentence up into elements (ner, verbs, etc.) and match against the img, in the aggregate as well as against boxes
#return a dict of element -> (score, PIL Image or None)
# TODO: count the items. e.g., two apples
def get_element_to_img(matched_sentence, img, box_segmentation_model,\
  image_preprocessor, clip_processor, clip_model, ignore_from_box=[], other_element_arr=[],\
  get_box_images=True, num_boxes=5, box_add_factor=0.65, box_detect_verbs=True, use_longest_subsuming_text=True,\
                         score_cutoff=0.2, ignore_digits=True, ignore_quotes=True):
  global spacy_nlp

  if ignore_digits:
    matched_sentence = " " + matched_sentence + " "
    for word in digits_to_words: 
      matched_sentence = matched_sentence.replace(" " + word + " ", " ")
  matched_sentence = remove_quotes(matched_sentence)
  width, height = img.size
  doc = spacy_nlp(matched_sentence)
  noun_chunks = [strip_left_stopwords(e.text)  for e in doc.noun_chunks if len(e.text) > 4 and e.text.lower() not in stopwords_set]
  verbs = [strip_left_stopwords(e.text) for e in doc if len(e.text) > 4 and e.tag_.startswith('VB') and e.text.lower() not in stopwords_set] + \
          [a for a in noun_chunks if a.endswith("ed") or a.endswith("ing")]
  ents = [strip_left_stopwords(e.text) for e in doc.ents if len(e.text) > 4 and e.text.lower() not in stopwords_set]
  noun_chunks = [a for a in noun_chunks if not a.endswith("ed") and not a.endswith("ing")]
  ner_and_verbs = dict([((e.lower() if len(e) < 5 else e.lower()[:5]), e)  for e in (ents + verbs + noun_chunks)])
  text4 = list(set([a.strip("()[]0123456789-:,.+? ") for a in (list(ner_and_verbs.values()) + other_element_arr) if a.strip()]))
  text4 = [a for a in text4 if a.strip()]
  if use_longest_subsuming_text: #to get ony longest subsuming text
    text5 = []
    text4.sort(key=lambda a: len(a), reverse=True)
    for atext in text4:
      if any(a for a in text5 if atext in a): continue
      text5.append(atext)
    text4 = text5
  text4 = [" "+a+" " for a in text4]
  text4 = [a.split("''")[0].strip() for a in text4 if  " corner" not in a and "foregr" not in a and "backgr" not in a and " word " not in a and "picture" not in a and "illustration" not in a and\
           " words" not in a and "photo" not in a and  "drawing" not in a and "portrait" not in a and " left " not in a and " right " not in a and \
           a.strip().lower() not in {"some", "more", "others", "other", "the type", "a type", "a color", "the color", "the middle", "the center", "the left", "the center", "the right", "the top", "the bottom", "an image", "the image", "image", "the images", "place", "location"}]
  if text4:
    with torch.no_grad():
      if get_box_images:
        clip_output = clip_image_to_multitext_score(clip_model, clip_processor, img, text4, decompose_image=True, ignore_from_box=([] if box_detect_verbs else verbs) + ignore_from_box, box_add_factor=box_add_factor, num_boxes=num_boxes, box_segmentation_model=box_segmentation_model, image_preprocessor=image_preprocessor)
      else:
        clip_output = clip_image_to_multitext_score(clip_model, clip_processor, img, text4, decompose_image=True, ignore_from_box=([] if box_detect_verbs else verbs) + ignore_from_box, box_add_factor=box_add_factor)
      box_images = clip_output['box_images']
      if clip_output is not None:
        # now get relationship between things as a sentence.
        if clip_output['box2element']:
          box2element = [(a[0], a[1], a[2], box_images[idx], a[3]) for idx, a in clip_output['box2element'].items()]
        else:
          box2element = None
        ent2score =  dict([(a, [b.item(), []]) for a, b in zip(text4, clip_output['scores']) ])
        if box2element:
          for element, score, coord, img, attr in box2element:
            if " corner" not in element and "foregr" not in element and "backgr" not in element:
              rec = ent2score.get(element, [0, []])
              rec[0] = max(rec[0], score)
              rec[1].append((score, img, attr, coord))
              ent2score[element] = rec
              
        sents = []
        if box2element:
          background_element = None
          prev_small_element = None
          for element, score, coord, img, attr in box2element:
            if  (element.endswith("ed") or element.endswith("ing")) and box_detect_verbs: continue
            if score >= score_cutoff and " corner" not in element and "foregr" not in element and "backgr" not in element:
              if attr:
                attr = attr.split(",")[0]
                if attr == "black":
                  attr = "dark colored"
                elif attr == "white":
                  attr = "light colored"
                # print (f"the {element} is also {attr}.", score)
                sents.append(f"the {element} is also {attr}.")
              if coord[0]/width <= 0.03 and coord[1]/height <= 0.03 and  coord[2]/width >= 0.20 and coord[3]/height >= 0.10 and coord[3]/height < 0.30:
                sents.append(f"the {element} is in the background.")
                background_element = element
                continue
              x_center = (coord[0] + (coord[2] - coord[0])/2.0)
              y_center  = (coord[1] + (coord[3] - coord[1])/2.0)
              if ((coord[2] - coord[0])/width <= 0.3 or (coord[3] - coord[1])/height <= 0.3) and prev_small_element:
                prev_element, prev_score, prev_coord = prev_small_element
                if (x_center   - (prev_coord[0] + (prev_coord[2] - prev_coord[0])/2.0))/width > 0.2:
                  if random.randint(0,1) == 0:
                    sents.append(f"the {prev_element} is to the left of the {element}.")
                  else:
                    sents.append(f"the {element} is to the right of the {prev_element}.")
                  prev_small_element = None
                  continue
                elif (x_center   - (prev_coord[0] + (prev_coord[2] - prev_coord[0])/2.0))/width > 0.05:
                  sents.append(f"the {prev_element} is beside the {element}.")
                  prev_small_element = None
                  continue
                elif (y_center   - (prev_coord[1] + (prev_coord[3] - prev_coord[1])/2.0))/height > 0.05:
                  if random.randint(0,1) == 0:
                    sents.append(f"the {prev_element} is above the {element}")
                  else:
                    sents.append(f"the {element} is in front of the {prev_element}")
                  prev_small_element = None
                  continue
                elif (x_center   - (prev_coord[0] + (prev_coord[2] - prev_coord[0])/2.0))/width <= 0.05 and \
                  (y_center   - (prev_coord[1] + (prev_coord[3] - prev_coord[1])/2.0))/height <= 0.05:
                  if (prev_coord[2] - prev_coord[0]) < (coord[2] - coord[0]):
                    sents.append(f"the {prev_element} is on the {element}.")
                  elif (prev_coord[3] - prev_coord[1]) < (coord[3] - coord[1]):
                    sents.append(f"the {prev_element} is on the {element}.")
                  else:
                    sents.append(f"the {element} is on the {prev_element}.")
                  prev_small_element = None
                  continue

              #print (x_center, element)
              if x_center/height < .2:
                sents.append(f"the {element} is on the left.")
              if x_center/width > .8:
                sents.append(f"the {element} is on the right.")
              if (coord[2] - coord[0])/width <= .3 or (coord[3] - coord[1])/height <= .4:
                prev_small_element = (element, score, coord)
        sents = [" "+s+" " for s in sents]
        sents = [s.replace(" the the ", " the ").replace(" the a ", " the ").replace(" the an ", " the ").strip() for s in sents]
        return ent2score, sents
    return {}, []


def add_img_context_to_instruction(instruction):
    added_text = ""
    prefix = suffix = False
    if random.randint(0,1) and "scene" not in instruction and "image" not in instruction and "picture" not in instruction:
        prefix = True
        added_text = random.choice(["Given the image, ", "Look at the image and ", "Ok, given the image, ", "Please look at the image and ", "Can you tell me from the image: ", "Next, can you tell me from the image: ", "Now let's examine this image. ", "I want you to act as an expert image analyzer, here. "])
    elif "scene" not in instruction and "image" not in instruction and "picture" not in instruction:
        suffix = True
        added_text = random.choice([ "given the image", "Please look at the image and answer.", "Can you tell me the answer from the image?", "Now let's examine this image. ", "I want you to act as an expert image analyzer, here. "])
    if random.randint(0,1): added_text = added_text.replace("image", "picture")
    if random.randint(0,1): added_text = added_text.replace("image", "scene")
    if random.randint(0,1): added_text = added_text.replace("I want you to act", "Act")
    if random.randint(0,1): added_text = (added_text + " please ").replace(". please", ", please")
    if random.randint(0,1): added_text = added_text.replace("Given", "Here is")
    if random.randint(0,1): added_text = added_text.replace("Given", "You have")
    if random.randint(0,1): added_text = added_text.replace("Given", "Inspect")
    if random.randint(0,1): added_text = added_text.replace("Given", "Analyze")
    if random.randint(0,1): added_text = added_text.replace("Look at", "Here is")
    if random.randint(0,1): added_text = added_text.replace("Look at", "You have")
    if random.randint(0,1): added_text = added_text.replace("Look at", "Inspect")
    if random.randint(0,1): added_text = added_text.replace("Look at", "Analyze")
    if random.randint(0,1): added_text = added_text.replace("Can you", "Will you")
    if random.randint(0,1): added_text = added_text.replace("Can you", "Please")
    if random.randint(0,1): added_text = added_text.replace("given", "here is")
    if random.randint(0,1): added_text = added_text.replace("given", "inspect")
    if random.randint(0,1): added_text = added_text.replace("given", "analyze")
    if random.randint(0,1): added_text = added_text.replace("look at", "here is")
    if random.randint(0,1): added_text = added_text.replace("look at", "you have")
    if random.randint(0,1): added_text = added_text.replace("look at", "inspect")
    if random.randint(0,1): added_text = added_text.replace("look at", "analyze")
    if random.randint(0,1): added_text = added_text.replace("can you", "will you")
    if random.randint(0,1): added_text = added_text.replace("can you", "please")
    if random.randint(0,1): added_text = added_text.replace("examine", "analyze")
    if random.randint(0,1): added_text = added_text.replace("examine", "deleve into")
    if random.randint(0,1): added_text = added_text.replace("analyzer", "AI image understander")
    if random.randint(0,1): added_text = added_text.replace("analyzer", "multimodal AI")
    if prefix:
      if added_text.endswith(".") or added_text.endswith(". "):
        instruction = added_text + " "+ instruction
      else:
        instruction = added_text + " "+ instruction[0].upper()+instruction[1:]
    elif suffix:
      if added_text[0] == added_text[0].upper():
        instruction = instruction + " " + added_text
      else:
        instruction = instruction[:-1] + " " + added_text + instruction[0]

    instruction = instruction.replace("  ", " ")
    return instruction

#TODO: put all these prompts in a prompt library/config file
# should probably move this to create_multimodal data
def generate_image_aware_instruction(captions, instructions, model, tokenizer):
  device = model.device
  if random.randint(0, 1):
    instr_revision_prompts = [tokenize_with_assistant_continuation(tokenizer, [{"role": "user", "content": f"You are given the below image:\n{caption}\n===\nRevise the below instruction such that events, physical conditions, attributes, color, actions, feelings, objects, people or other information from the image are removed from the instruction, and the instruction refers to those things in the image instead. Do not refer to proper names in the instruction if those names are already in the image. Do not refer to any context document. Do not refer to the 'description' of the image. Retain the theme of the instruction. Do not repeat this instruction or the information from the image in your revised instruction. The instruction is:\n{instruction}"}, 
                                                                              {"role": "assistant", "content": "Revised Instruction:"}]) for caption, instruction in zip(captions, instructions)]
    revised_instructions = generate_with_batching(model, tokenizer, instr_revision_prompts, device, batch_size=len(instr_revision_prompts))
    revised_instructions = [
        revised_instruction.split("Answer:", 1)[0]
        .split("answer:", 1)[0]
        .split("instruction:", 1)[-1]
        .split("Instruction:", 1)[-1]
        .split("Revised Instruction:", 1)[-1]
        .split("Revised instruction:", 1)[-1]
        .split("Assistant\n", 1)[-1]
        .split("assistant\n", 1)[-1]
        .replace("caption", "image").strip()
        for revised_instruction in revised_instructions
    ]
  else:
    captions_concepts_list = [set([a.strip(",.\'\"?").lower() for a in caption.split() if (a.endswith("ly") or a.endswith("ion") or a.endswith("ing") or a.endswith("ity")) and len(a) > 5]) for caption in captions]
    
    captions_concepts = [set([a.strip(",.\'\"?").lower()[:-3] for a in caption.split() if (a.endswith("ly") or a.endswith("ion") or a.endswith("ing") or a.endswith("ity")) and len(a) > 5]) for caption in captions]
    instructions_concepts = [list(set([a.strip(",.\'\"?").lower() for a in instruction.split() if (a.endswith("ly") or a.endswith("ion") or a.endswith("ing") or a.endswith("ity")) and len(a) > 5 if a.strip(",.\'\"?").lower()[:-3] not in caption_concepts])) for caption_concepts, instruction in zip(captions_concepts, instructions)]
    instructions_concepts = [", ".join(instruction_concepts) for instruction_concepts in instructions_concepts]
    captions_concepts_list = [", ".join(caption_concepts_list) for caption_concepts_list in captions_concepts_list]
    if instructions_concepts:
      if captions_concepts_list:
        prompts = [tokenize_with_assistant_continuation(tokenizer, [{"role": "user", "content": f"You are given the below image:\n{caption}\n===\nRevise the below instruction such that events, physical conditions, attributes, color, actions, feelings, objects, people or other concepts from the image are removed from the instruction, and the instruction refers to those things in the image instead. DO NOT re-use words from the image description into the revised instruction. Do not refer to the 'description' of the image. Retain the theme of the instruction.\nDo not simply repeat or paraphrase the instruction or the information from the image such as {caption_concepts_list}.\nThe instruction is:\n{instruction}\n===\nBe sure to keep these concepts in the revised instruction: {instruction_concepts}. When referring the image, use phrases like 'the <genric term like boy, girl, man, woman ... > in the image'."},
                                                                  {"role": "assistant", "content": "Revised question: Referring to this image"}]) for caption, caption_concepts_list, instruction, instruction_concepts in zip(captions, captions_concepts_list, instructions, instructions_concepts)]
      else:  
        prompts = [tokenize_with_assistant_continuation(tokenizer, [{"role": "user", "content": f"You are given the below image:\n{caption}\n===\nRevise the below instruction such that events, physical conditions, attributes, color, actions, feelings, objects, people or other concepts from the image are removed from the instruction, and the instruction refers to those things in the image instead. DO NOT re-use words from the image description into the revised instruction. Do not refer to the 'description' of the image. Retain the theme of the instruction\nDo not simply repeat or paraphrase the instruction or the information from the image in your revised instruction.\nThe instruction is:\n{instruction}\n===\nBe sure to keep these concepts in the revised instruction: {instruction_concepts}. When referring the image, use phrases like 'the <genric term like boy, girl, man, woman ... > in the image'."},
                                                                  {"role": "assistant", "content": "Revised question: Referring to this image"}]) for caption, instruction, instruction_concepts in zip(captions, instructions, instructions_concepts) ]
    else:
      prompts = [tokenize_with_assistant_continuation(tokenizer, [{"role": "user", "content": f"You are given the below image:\n{caption}\n===\nRevise the below instruction such that events, physical conditions, attributes, color, actions, feelings, objects, people or other concepts from the image are removed from the instruction, and the instruction refers to those things in the image instead. DO NOT re-use words from the image description into the revised instruction. Do not refer to any context document. Do not refer to the 'description' of the image. Retain the theme of the instruction.\nDo not simply repeat or paraphrase the instruction or the information from the image in your revised instruction.\nThe instruction is:\n{instruction}\n===\nWhen referring the image, use phrases like 'the <genric term like boy, girl, man, woman ... > in the image'."},
                                                                {"role": "assistant", "content": "Revised instruction: Referring to this image"}]) for caption, instruction in zip(captions, instructions)]
      
    revised_instructions = generate_with_batching(model, tokenizer, prompts, device, batch_size=len(prompts))
    revised_instructions = [revised_instruction.split("Revised instruction:",1)[-1].strip() for revised_instruction in revised_instructions]

  if random.randint(0, 1):
    return [add_img_context_to_instruction(revised_instruction) for revised_instruction in revised_instructions] 
  else:
    return revised_instructions


### STORY GENEREATION

STORY_PROMPTS = ["""Revise this story to make it compelling and more logical and detailed. Keep as much of the feelings and actions as possible, but remove anything that doesn't make sense. Make the story at least 10 paragraphs. Start with a title. %(warning)s The story should unfold through the characters interactions, decisions, and the consequences of their actions. Aim to weave in common sense lessons and social cues. The narrative should cater to a diverse age group, including at least one dialogue and presenting both positive and negative outcomes. Do not start with classic sentences like "Once upon a time", be creative:""",
          """Revise this story to make it compelling and more logical and detailed. Keep as much of the feelings and actions as possible, but remove anything that doesn't make sense. Make the story at least 10 paragraphs. Start with a title. %(warning)s Write as a real-life story shared by someone in a social media forum. The story should include:
- Niche interests or humor: dive into specific hobbies, interests, or humorous situations
- An unexpected plot twist or engaging conflict: introduce a relatable yet challenging situation or dilemma that the author faced.
- Reflection and insight: end with a resolution that offers a new understanding, a sense of community, or a personal revelation, much like the conclusions drawn in forum discussions.
Start the story right away. Do not start with sentences like  "Once upon a time" as this is a reddit post and not a novel, you should also avoid starting with classic sentences like "A few years ago" or "A few years back", be creative:""",
          """Revise this story to make it compelling and more logical and detailed. Keep as much of the feelings and actions as possible, but remove anything that doesn't make sense. Make the story at least 10 paragraphs. Start with a title. %(warning)s Write the story in the style of real-life situations that people share in forums. The story needs to include a compelling and unexpected plot twist. Your narrative should resonate with the authenticity and personal touch found in forum discussions. Include relatable events and emotional depth. Do not start with classic sentences like "Once upon a time", "A few years back" or "A few months ago", be creative:""",
          """Revise this story to make it compelling and more logical and detailed. Keep as much of the feelings and actions as possible, but remove anything that doesn't make sense. Make the story at least 10 paragraphs. Start with a title. %(warning)s The story should incorporate the following elements:
- Dialogue: the story must feature at least one meaningful dialogue that reveals character depth, advances the plot, or unravels a crucial piece of the mystery
- Interesting themes: explore themes resonant with a mature audience, such as moral ambiguity, existential queries, personal transformation, or the consequences of past actions.
Do not start with classic sentences like "Once upon a time", "The sun hung low in the sky" or "In the dimly lit", be creative:"""]



#if __name__ == '__main__':
#    split = get_splits('./scripts/atomic_stories.jsonl', 0, 2, 10000)
#    print(split)
#    print(len(split))



# Example usage:
#elements = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#world_size = 3

#for rank in range(world_size):
#    result = get_sublist(elements, rank, world_size)
#    print(f"Sublist for rank {rank}: {result}")


#### CODE UTILS
import subprocess
import json
import tempfile
import os
import sqlite3

def create_random_input(mapping):
    """ Generate a random test input based on variable mappings. """
    return {var: random.randint(1, 500) for var in mapping}

def check_python_syntax(code):
    try:
        compile(code, "<string>", "exec")
        return True, ''
    except Exception as e:
        code = code.split("\n")
        line = str(e)
        line = line.split("line ")[-1].split()[0].strip(",)")
        try:
            line = int(line)
            e = str(e)+"\n>"+code[line-1]
        except:
            print ("couldn't get line", line, code)
            e = str(e)
        e = e.replace("<string>,", "at")
        return False, e
    
def check_python_with_guessing(python, min_len=50):
    if len(python.strip()) < min_len:
      return '', False, ''
    if "```python" not in python:
      python = "```python"+python
    if "`" not in python[-100:]:
      python = python + "```"
    code = python.split("```python")[1]
    if code.count("```") > 1:
      code = code.split("```")[1]
    code = code.split("`")[0].strip("\n ")
    if len(code.strip()) < min_len:
      return '', False, ''
    # TODO: do heuristic check for gabage code
    is_valid, err_str = check_python_syntax(code)
    if not is_valid and "'" in code:
      code2 = code.replace("'", "\"")
      is_valid, err_str = check_python_syntax(code2)
      if is_valid:
          code = code2
    if not is_valid:
      code = "\n".join(code.split("\n")[:-1])
    if len(code.strip()) < min_len:
      return '', False, ''
    is_valid, err_str = check_python_syntax(code)
    if not is_valid:
      code = "\ndef ".join(code.split("\ndef ")[:-1])
    if len(code.strip()) < min_len:
      return '', False, ''
    is_valid, err_str = check_python_syntax(code)
    if not is_valid:
      code = "\nclass ".join(code.split("\nclass ")[:-1])
    if len(code.strip()) < min_len:
      return '', False, ''
    is_valid, err_str = check_python_syntax(code)
    if is_valid:
        code = code.strip()
        before, after = python.split("```python",1)
        before = before+"```python"
        after = "```"+after.split("```",1)[-1]
        python =  before+"\n"+code+"\n"+after
    return python, is_valid, err_str

#DANGER OF SQL INJECTION!!
def check_sqlite_syntax(stmnt, temp_db):
    for st in stmnt.split(";"):
        try:
            temp_db.execute(st)
        except Exception as e:
            return False, str(e)
    return True, ''

def check_js_syntax(js_code):
    """
    Check JavaScript code for syntax errors using Node.js
    
    Args:
        js_code (str): JavaScript code to check
        
    Returns:
        tuple: (bool, str) - (is_valid, error_message)
    """
    # Create a temporary file to store the JavaScript code
    with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as temp_file:
        temp_file.write(js_code)
        temp_filename = temp_file.name

    try:
        # Use Node.js to parse the JavaScript code
        process = subprocess.run(
            ['node', '--check', temp_filename],
            capture_output=True,
            text=True
        )
        
        # Check if there were any syntax errors
        if process.returncode == 0:
            return True, "JavaScript code is syntactically valid"
        else:
            return False, process.stderr.strip()
            
    except subprocess.CalledProcessError as e:
        return False, f"Error running Node.js: {str(e)}"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"
    finally:
        # Clean up the temporary file
        os.unlink(temp_filename)

def execute_python_code(code, mapping):
    """
    Executes Python code with a specified input mapping.
    This code assumes the function 'solution' is defined within the passed code.
    """
    local_locals = mapping.copy()
    wrapped_code = code + f"\nresult = solution({', '.join(f'{k}={v}' for k, v in mapping.items())})"
    try:
        exec(wrapped_code, {"__builtins__": None}, local_locals)
        if 'result' in local_locals:
            return local_locals['result'], ''
        else:
            #print("No 'result' key was found in the local variables after executing the code.")
            return None, ''
    except Exception as e:
        #print(f"Error when executing Python code: {e}")
        return None, str(e)
