import json
import time
import argparse
import spacy
import glob
import itertools
import random
import numpy as np
from collections import deque
import copy
import torch
import transformers
from torch.nn.functional import cosine_similarity
from PIL import Image
import PIL
from diffusers import FluxPipeline
from transformers import pipeline
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel, AutoModel, AutoTokenizer, AutoModelWithLMHead
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from torch import multiprocessing
from torch import threading
from transformers.utils import logging as transformers_logging
from src.utils import chunkify, chatml_format_instructions, generate_with_batching, assign_uuid, tokenize_with_assistant_continuation, cleanup_data_batch, standardize_data_fields, cleanup_and_serialize_params, augment_for_quotes, \
                      STORY_PROMPTS, generate_image_aware_instruction

from src.frcnn.visualizing_image import SingleImageViz
from src.frcnn.processing_image import Preprocess as FRCNNPreprocess
from src.frcnn.modeling_frcnn import GeneralizedRCNN
from src.frcnn.utils import Config as FRCNNConfig
from src.frcnn.utils import decode_image as frcnn_decode_image
from src.purpleteam.autoredteam import auto_redteam
from src.purpleteam.templates.seed import verb_templates, obj_templates
from urllib.parse import unquote


import pyarrow
from pyarrow import parquet
from io import BytesIO
from PIL import Image
import os, torch
import cv2
import numpy as np
from matplotlib import colors
import random
from collections import OrderedDict


import logging
logger = logging.getLogger(__name__)  

logging.basicConfig(
    format='%(asctime)s : %(processName)s : %(threadName)s : %(levelname)s : %(message)s',
    level=logging.WARNING)

transformers_logging.set_verbosity(transformers.logging.ERROR)

spacy_nlp = None
max_detections = 5
num_devices = torch.cuda.device_count()

args = None  
node_name = None
task = None
image_generator = None
caption_generator_model = None
caption_generator_processor = None
LLM_small_model = None
LLM_small_tokenizer = None
LLM_medium_model = None
LLM_medium_tokenizer = None
LLM_large_model = None
LLM_large_tokenizer = None
image_text_score_model = None
image_text_processor = None
device  = None
box_detect_image_preprocessor = None
box_segmentation_model = None
evaluator_model = None
evaluator_tokenizer = None


def initialize(args_new):
  global spacy_nlp, \
      max_detections, \
      num_devices, \
      args, \
      node_name, \
      task, \
      image_generator, \
      caption_generator_model, \
      caption_generator_processor, \
      LLM_small_model, \
      LLM_small_tokenizer, \
      LLM_medium_model, \
      LLM_medium_tokenizer, \
      LLM_large_model, \
      LLM_large_tokenizer, \
      image_text_processor, \
      device, \
      image_text_score_model, \
      box_detect_image_preprocessor, \
      box_segmentation_model, \
      evaluator_model, \
      evaluator_tokenizer, \
      tasks_configs
  
  args = args_new
  device_number = args.device_number
  node_name = args.node_name
  task = args.task
  config = tasks_configs[task]
  max_detections = args.max_detections
  if spacy_nlp is None: # this could be an config
    spacy_nlp = spacy.load('en_core_web_sm')  # this could be a config
  if device is None:
    device = os.environ["CUDA_VISIBLE_DEVICES"] = "cuda:"+str(device_number)
    logger.warning ('SETTING '+ device)
    
  #TODO: test to see if fa2 really does make things faster or not. from our prelimnary tests, there were no differences
  if 'image_generator' in config['models_needed'] and image_generator is None:
    # is there a way to pass params to the initializer??
    logger.warning (f'CREATING {args.image_generator_model} MODEL '+ device)
    if 'black-forest-labs/FLUX.1-schnell' in args.image_generator_model:
      image_generator = FluxPipeline.from_pretrained(args.image_generator_model, torch_dtype=torch.bfloat16, cache_dir=args.cache_dir).to(device, attn_implementation="flash_attention_2") # , 
    else:
      assert False, f"{args.image_generator_model} not yet supported"

  if 'caption_generator' in config['models_needed'] and caption_generator_processor is None:
    logger.warning(f'CREATING {args.caption_generator_model} MODEL')    
    caption_generator_model = AutoModelForCausalLM.from_pretrained(args.caption_generator_model, trust_remote_code=True, torch_dtype=torch.bfloat16, cache_dir=args.cache_dir, attn_implementation="flash_attention_2").train().to(device)
    caption_generator_processor = AutoProcessor.from_pretrained(args.caption_generator_model, trust_remote_code=True, cache_dir=args.cache_dir)

  if (("LLM_model" in config['models_needed']) or 'LLM_small' in config['models_needed']) and 'small' in args.use_LLM_size  and LLM_small_tokenizer is None:
    logger.warning(f'CREATING {args.LLM_small_model} MODEL')    
    LLM_small_model = AutoModelForCausalLM.from_pretrained(args.LLM_small_model, trust_remote_code=True, torch_dtype=torch.bfloat16, cache_dir=args.cache_dir, attn_implementation="flash_attention_2").train().to(device)
    LLM_small_tokenizer = AutoTokenizer.from_pretrained(args.LLM_small_model, trust_remote_code=True, cache_dir=args.cache_dir)
    if not LLM_small_tokenizer.pad_token:
      LLM_small_tokenizer.pad_token = LLM_small_tokenizer.eos_token

  if (("LLM_model" in config['models_needed']) or 'LLM_medium' in config['models_needed']) and 'medium' in args.use_LLM_size  and LLM_medium_tokenizer is None:
    logger.warning(f'CREATING {args.LLM_medium_model} MODEL')    
    LLM_medium_model = AutoModelForCausalLM.from_pretrained(args.LLM_medium_model, trust_remote_code=True, torch_dtype=torch.bfloat16, cache_dir=args.cache_dir, attn_implementation="flash_attention_2").train().to(device)
    LLM_medium_tokenizer = AutoTokenizer.from_pretrained(args.LLM_medium_model, trust_remote_code=True, cache_dir=args.cache_dir)
    if not LLM_medium_tokenizer.pad_token:
      LLM_medium_tokenizer.pad_token = LLM_medium_tokenizer.eos_token

  if (("LLM_model" in config['models_needed']) or 'LLM_large' in config['models_needed']) and 'large' in args.use_LLM_size and LLM_large_tokenizer is None:
    logger.warning(f'CREATING {args.LLM_large_model} MODEL')    
    LLM_large_model = AutoModelForCausalLM.from_pretrained(args.LLM_large_model, trust_remote_code=True, torch_dtype=torch.bfloat16, cache_dir=args.cache_dir, attn_implementation="flash_attention_2").train().to(device)
    LLM_large_tokenizer = AutoTokenizer.from_pretrained(args.LLM_large_model, trust_remote_code=True, cache_dir=args.cache_dir)
    if not LLM_large_tokenizer.pad_token:
      LLM_large_tokenizer.pad_token = LLM_large_tokenizer.eos_token

  if 'image_text_scorer' in config['models_needed'] and image_text_processor is None:
    logger.warning(f'CREATING {args.image_text_score_model} MODEL')
    if 'openai/clip' in args.image_text_score_model:
      image_text_score_model = CLIPModel.from_pretrained(args.image_text_score_model, trust_remote_code=True, torch_dtype=torch.bfloat16, cache_dir=args.cache_dir).eval().to(device)
      image_text_processor = CLIPProcessor.from_pretrained(args.image_text_score_model, cache_dir=args.cache_dir)
    else:
      assert False, f"{args.image_text_score_model} not yet supported"

  if 'box_segementer' in config['models_needed'] and box_segmentation_model is None:
    if "unc-nlp/frcnn-vg-finetuned" in args.box_segmenter_model:
      frcnn_config = json.load(open(args.box_segmenter_config_path)) # "src/frcnn/config.jsonl"
      frcnn_config = FRCNNConfig(frcnn_config)
      box_detect_image_preprocessor= FRCNNPreprocess(frcnn_config).half().cuda()
      box_segmentation_model= GeneralizedRCNN.from_pretrained(args.box_segementer_model, frcnn_config, trust_remote_code=True, cache_dir=args.cache_dir).half().eval().to(device)
    else:
      assert False, f"{args.box_segmenter_model} not yet supported"
  
  if 'evaluator' in config['models_needed'] and evaluator_model is None:
    if "llamas-community/LlamaGuard-7b" in args.evaluator_model:
      evaluator_model = AutoModelForCausalLM.from_pretrained(args.evaluator_model, trust_remote_code=True, torch_dtype=torch.bfloat16, cache_dir=args.cache_dir, attn_implementation="flash_attention_2").train().to(device)
      evaluator_tokenizer = AutoTokenizer.from_pretrained(args.evaluator_model, trust_remote_code=True, cache_dir=args.cache_dir)
      if not evaluator_tokenizer.pad_token:
        evaluator_tokenizer.pad_token = evaluator_tokenizer.eos_token
    else:
      assert False, f"{args.evaluator_model} not yet supported"

      
  return device


def initialize_gpus_and_models(pool, args):
  t0 = time.time()
  # initialize the GPU per process
  initializer_args = []
  for device_number in range(num_devices):
    arg2 = copy.deepcopy(args)
    arg2.device_number = device_number
    initializer_args.append(arg2)
  for arg2, gpu_device in zip(initializer_args, pool.map(initialize, initializer_args, chunksize=1)):
    assert f"cuda:{arg2.device_number}" == gpu_device, "Something went wrong with GPU assignment"
  init_time = time.time() - t0  
  logger.warning(f"Initializing gpus and models took {init_time} secs")

### TASK SPECIFIC FUNCTIONS

# Create a template so we can use python's format functionality to
# fill in captions in the 'text' field.
def create_caption_template_hash(data_list):
  caption_template_hash = {}
  for data in data_list:
    if "<caption>" not in data['text']:
      caption_list0 = [data['text']]
    else:
      caption_list0 = [t.split("</caption>")[0].strip() for t in data['text'].split("<caption>") if "</caption>" in t]
    template = data['text']
    if not template:
      template = "<caption>{CAPTION_1}<caption>"
    else:
      if "<caption>" not in template:
        template = "<caption>"+template+"</caption>"
      caption_list0 = list(enumerate(caption_list0))
      caption_list0.sort(key=lambda a: len(a[1]), reverse=True)
      # we want to create templates by replacing things that are longest first in case there are overlaps
      for i, caption in caption_list0:
        caption = caption.replace("<caption>", "").replace("<image>", "").replace("</caption>", "")
        template = template.replace(">"+caption+"<", ">{CAPTION_"+str(i)+"}<")
        if ">{CAPTION_"+str(i)+"}<" not in template:
          logger.warning("Something went wrong and the caption template could not be created")
    caption_template_hash[data['idx']] = template
  return caption_template_hash

# use a box-element detection model and a text-image scorer to check
# if words in caption really do exist to minimize
# hallucination. remove elements that are found but low scoring. if we
# don't detect an actual box for the element but the text-image
# comparison model thinks there is the element SOMEWHERE in the
# picture, then we want a higher cutoff
def detect_elements_and_remove_missing_elements(caption, image, score_cutoff, \
                                                working_caption_with_elements_removed, \
                                                existing_elements, \
                                                existing_elements_with_spatial_relationships):
  aHash, rel_sents = get_element_to_img(caption, image, box_segmentation_model,\
                                        box_detect_image_preprocessor, image_text_processor, image_text_processor, score_cutoff=score_cutoff)
  for element, val in list(aHash.items()):
      if element not in caption or ((val[1] and val[0] < score_cutoff) or (not val[1] and val[0] < score_cutoff + 0.05)):
          del aHash[element]
          caption = caption.replace(element+" ", " ")
          caption = caption.replace(element+"es ", " ")
          caption = caption.replace(element+"s ", " ")                        
          caption = caption.replace(" "+ element, " ")
          caption = caption.replace(element, " ")
  for element, val in list(aHash.items()):
      if not val[1]: continue
      all_detected_imgs = val[1]
      count = len([a for a in all_detected_imgs if a[0] >= score_cutoff])
      if count > 1 and not element.endswith("ing"):
        if element.split()[0].lower() in {"the", "an", "a",}:
          element = " ".join(element.split()[1:])
        if caption.count(" "+element) == 1:          
          caption = caption.replace(" " + element, " " + digits_to_words[count] + " " + element)
        else:
          caption = caption.strip(".") + ". There are " + digits_to_words[count] + " " + element+"."
  caption = caption.replace(" es ", " ").replace(" ed ", " ").replace(" ly ", " ").replace(" ing ", " ").replace("  ", " ").strip()            
  existing_elements.append(", ".join(a for a in aHash.keys() if not a.endswith("ing"))+".")
  existing_elements_with_spatial_relationships.append(existing_elements[-1].strip(". ") + ". " + " ".join(rel_sents).strip())
  working_caption_with_elements_removed.append(caption)
  return aHash

possible_image_dims = ([256]*8) + ([512]*8)+ ([1024]*5) + ([2048]*3) + ([4096]*2)

# Given a list of data items with a text field with captions inside
# it, each keyed to an idx, generate list of list of images keyed to
# the idx.  NOTE: we don't filter out problem data at the idx/element
# level. instead, if there is a single element that is a problem, we
# remove the whole data at idx.
def generate_images(data_list):
  global device, image_generator, caption_generator_model, caption_generator_processor, args
  logger.warning(f"Starting Image Generation "+ device + " " + str(image_generator.device))  
  idx2DataHash = dict([(data['idx'], data) for data in data_list])
  problem_idxs = []
  model_time = 0
  items_processed = 0
  new_batch_size = args.batch_size
  width=args.image_width
  height=args.image_height
  if width <= 0:
    if height > 0 and random.randint(0,1):
      width = height
    else:
      width = random.choice(possible_image_dims)
    if width == 512:
      new_batch_size = new_batch_size/2
    elif width == 1024:
      new_batch_size = new_batch_size/3
    elif width == 2048:
      new_batch_size = new_batch_size/4
    elif width == 4096:
      new_batch_size = new_batch_size/10

  if height <= 0:
    if width > 0 and random.randint(0,1):
      height = width
    else:
      height = random.choice(possible_image_dims)
    if height == 512:
      new_batch_size = new_batch_size/2
    elif height == 1024:
      new_batch_size = new_batch_size/3
    elif height == 2048:
      new_batch_size = new_batch_size/4
    elif height == 4096:
      new_batch_size = new_batch_size/10
  new_batch_size = max(1, int(new_batch_size))

  #We use idx_pair: (idx, i) where i is the poisition in the media_list
  with torch.no_grad():
    caption_list = []
    idx_pairs = []
    for data in data_list:
      idx = data['idx']
      text = data['text']
      if not text:
        problem_idxs.append(idx)
        continue
      captions = []
      if "<caption>" in text:
        for j, t in enumerate(text.split("<caption>")):
          if (j+1) % 2 == 0:
            t = t.replace("<image>", "").split("</caption>")[0].strip()
            captions.append(t)
      else:
        captions = [text]
      for i in range(len(captions)):
        idx_pairs.append((idx, i))
      caption_list.extend(captions)
    if len(caption_list) == 0:
      logger.warning("There were no data item with valid captions")
      return [], model_time, items_processed, list(problem_idxs)
            
    time0 = time.time()
    #detected_and_cleaned_texts = augment_for_quotes(caption_list)      
    #caption_list_no_text = [a[0] for a in detected_and_cleaned_texts]  
    #draw_caption_list = [a[1] for a in detected_and_cleaned_texts]  
    # Generate image with diffuser pipeline
    images = []
    logger.warning("drawing at " + str(width) +"x"+str(height))
    for rng in range(0, len(caption_list), new_batch_size):
      images.extend(image_generator(
        caption_list[rng: min(len(caption_list), rng+new_batch_size)],
        guidance_scale=0.0,
        num_inference_steps=4,
        max_sequence_length=args.image_gen_caption_max_sequence,
        width=width, height=height,
        generator=torch.Generator(image_generator.device).manual_seed(0)
      ).images)
    model_time = time.time() - time0
    items_processed = len(images)
    logger.warning(f"Image Gen time: {model_time} ")
  #torch.cuda.empty_cache()
  for data  in data_list:
    data['media_list'] = []
    data['media_coordinates_list'] = []
    data['media_caption_scores_list'] = []    
    data['media_types_list'] = []                            
  for idx_pair, image in zip(idx_pairs, images):
    idx, i = idx_pair
    data = idx2DataHash[idx]
    if len(data['media_list']) < i+1:
      data['media_list'].extend([None]*(i+1-len(data['media_list'])))
    if len(data['media_coordinates_list']) < i+1:
      data['media_coordinates_list'].extend([None]*(i+1-len(data['media_coordinates_list'])))
    if len(data['media_types_list']) < i+1:
      data['media_types_list'].extend([None]*(i+1-len(data['media_types_list'])))
    data['media_list'][i] = image
    data["media_coordinates_list"][i] = [0, 0]+ list(image.size) # we can consider shifting the coordinates over for multiple images. this can be done at run time.
    data["media_types_list"][i] = "image"

  for data in data_list:
    # add the caption tag and the image tag if it's not already there
    if "<caption>" not in data["text"]:
      if random.randint(0,1):
        data["text"] = f"<caption><image>{data['text']}</caption>"
      else:
        data["text"] = f"<caption>{data['text']}<image></caption>"
    else:
      segments = []
      for i, t in enumerate(data['text'].split("<caption>")):
        if (i+1) % 2 == 0:
          t = "<caption>"+t
          if "</caption>" not in t:
            t = t+"</caption>"

          # randomly put the image either at the beginning or end of the caption. e.g., the caption appears above the image or below the image in a web-page.
          if "<image>" not in t:
            if random.randint(0,1):
              t = t.replace("<caption>", "<caption><image>")
            else:
              t = t.replace("</caption>", "<image><caption>>")            
          segments.append(t)
      data['text'] = " ".join(segments)
  problem_idxs = set(problem_idxs)    
  data_list = [data for data in data_list if data['idx'] not in problem_idxs]                      
  return data_list, model_time, items_processed, list(problem_idxs)

# Given a lists of data items, with a 'media_list' field that has
# Images and associated with an idx, caption it and return a text field
# of the form <caption> ... </caption><caption>..., etc.  NOTE: we
# don't filter out problem data at the idx/element level. instead, if
# there is a single element that is a problem, we remove the whole data
# at idx.
def generate_captions(data_list, clear_images=False):
  # consider making the thumbnail BEFORE we send to the processes to lower interprocess communcation.
  
  # NOTE: we run the captioning on a thumbnail version of no more than
  # 256x256. TODO: make the thumbnail dimension a arg.
  global device, image_generator, caption_generator_model, caption_generator_processor, args
  logger.warning(f"Starting Captions Generation "+ device + " " + str(caption_generator_model.device)) 
  idx2DataHash = dict([(data['idx'], data) for data in data_list]) 
  time0 = time.time()
  caption_generator_model_prompt = '<MORE_DETAILED_CAPTION>'
  problem_idxs = []
  model_time = 0
  items_processed = 0
  with torch.no_grad():
    #optimize this to remove whole records of idx if there is at least
    #one image that is a problem. right now we do captions for all
    #valid images, and the caption for invalid images are empty.
    images = []
    idx_pairs = [] # in the form of [(idx, i)...]
    #logger.warning ('Checking for corrupted images for batch size: ' +str(len(idx_and_images)))
    for data in data_list:
      idx = data['idx']
      image_set = data['media_list']
      if not image_set:
        problem_idxs.append(idx)
        logger.warning (f"problem with {idx} "+ str(e))        
        continue
      for i, image in enumerate(image_set):
        if type(image) is str:
          image_path = image
          if image_path[0] != "/":
            image_path + args.output_dir+"/"+input_path
          try:
            image = Image.open(image_path)
          except Exception as e:
            problem_idxs.append(idx)
            logger.warning (f"problem with {idx} "+ str(e))
            continue
        if image.mode == 'L':
          image = image.convert('RGB')
        try:
            if not clear_images: image = image.copy()
            image.thumbnail((256,256), Image.Resampling.LANCZOS)
        except:
          logger.warning(f"Couldn't create thumbnail for {idx}")
        try:
          caption_generator_processor(text=[caption_generator_model_prompt], images=[image], return_tensors="pt")
          images.append(image)
          idx_pairs.append((idx, i))
        except Exception as e:
          problem_idxs.append(idx)
          logger.warning (f"problem with {idx} "+ str(e))
          continue
    if len(images) == 0:
      logger.warning("There were no data item with valid images to caption")
      return [], model_time, items_processed, list(problem_idxs)
      
    inputs = caption_generator_processor(text=[caption_generator_model_prompt]*len(images), images=images, return_tensors="pt").to(caption_generator_model.device)
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    generated_ids = caption_generator_model.generate(
          **inputs,
          max_new_tokens=args.caption_max_sequence,
          early_stopping=True,
      )
    recaption_lists = caption_generator_processor.batch_decode(generated_ids, skip_special_tokens=True)
    items_processed = len(recaption_lists)
    assert items_processed != 0, f"Something went wrong and we have no captions {recaption_list}!"      
    time1 = time.time()
    model_time = time1-time0
    logger.warning(f"Caption Generation: {model_time} seconds")

  
  if clear_images:
    for data  in data_list:
      data['media_list'] = []
      data['media_coordinates_list'] = []
      data['media_caption_scores_list'] = []          
      data['media_types_list'] = []                            

  # let's create templates of the original text with potentially
  # interleaved captions to replace with the new captions
  template_hash = create_caption_template_hash(data_list)
  # a temporary hash table to hold idx->[caption, caption ...]
  template_params_hash = {}
  for idx_pair, text in zip(idx_pairs, recaption_lists):
    idx, i = idx_pair
    template_params = template_params_hash[idx] = template_params_hash.get(idx, [])
    if len(template_params) < i+1:
      template_params.extend(['']*(i+1-len(template_params)))
    template_params[i] = text
    
  for idx, caption_list in template_params_hash.items():
    data = idx2DataHash[idx]
    caption_hash = dict([("CAPTION_"+str(i), text) for i, text in enumerate(caption_list)])
    data['text'] = template_hash[idx].format(**caption_hash)

  #torch.cuda.empty_cache()
  problem_idxs = set(problem_idxs)
  data_list = [data for data in data_list if data['idx'] not in problem_idxs]
  return data_list, model_time, items_processed, list(problem_idxs)


def generate_captions_and_clear_images(data_list):
  return generate_captions(data_list, clear_images=True)

def generate_captions_and_dont_clear_images(data_list):
  return generate_captions(data_list, clear_images=False)

# Given an image and optionally an original caption, we will
# (re)caption the image.  we will then correct for any hallucinationed
# elements and incorrect counting.  we will generate at one to two
# upsampled captions with these corrections. then the correction with
# the highest score will be considered the caption for the particular
# image. Do some accounting to keep track of elements of the form (idx,
# image_position). Another complexity is the data might be in the form
# of interleaved captions.  assumes there are images in media_list.
def fix_and_upsample_caption(data_list, do_recaption=True):
  idx2DataHash = dict([(data['idx'], data) for data in data_list])
  model = None
  tokenizer = None
  new_batch_size = args.batch_size
  if args.use_LLM_size == 'small':
    model = LLM_small_model
    tokenizer = LLM_small_tokenizer
  elif args.use_LLM_size == 'large':
    model = LLM_large_model
    tokenizer = LLM_large_tokenizer
    # to account for mismatches between the smaller image geeneration
    # and caption_generator_model models and the large LLMs, create sub_batches to
    # prevent OOM.
    new_batch_size = max(1, int(args.batch_size/2))
  else:
    model = LLM_medium_model
    tokenizer = LLM_medium_tokenizer
  if not data_list[0]['media_list']:
    assert False, "We need images to do (re)captioning"

  score_cutoff = args.score_cutoff
  # NOTE: since there are several images/captions per item, we need to index everything by (idx, i) where i is the position of the element in the images/captions list.
  caption_hash = {}
  if data_list[0]['text']:
    for data in data_list:
      if "<caption>" not in data['text']:
        data['text'] = "<caption>"+data['text']+"</caption>"
        caption_list0 = [data['text']]
      else:
        caption_list0 = [t.split("</caption>")[0].strip() for t in data['text'].split("<caption>") if "</caption>" in t]
      for i, caption in enumerate(caption_list0):
        caption = caption.replace("<caption>", "").replace("<image>", "").replace("</caption>", "").strip()
        caption_hash[(data['idx'], i)] = caption

  if not data_list[0]['text'] or do_recaption:
    # in this case, the user only passed us data items with images but
    # no captions, or we need to do a recaption.
    data_list, model_time, items_processed, problem_idxs = generate_captions_and_dont_clear_images(data_list)
  else:
    model_time = 0
    items_processed = 0
    problem_idxs = []

  recaption_hash = {}
  assert data_list[0]['text'], "Something went wrong and there are no captions"
  for data in data_list:
    if "<caption>" not in data['text']:
      data['text'] = "<caption>"+data['text']+"</caption>"        
      recaption_list0 = [data['text']]
    else:
      recaption_list0 = [t.split("</caption>")[0].strip() for t in data['text'].split("<caption>") if "</caption>" in t]
    for i, recaption in enumerate(recaption_list0):
      recaption = recaption.replace("<caption>", "").replace("<image>", "").replace("</caption>", "").strip()        
      recaption_hash[(data['idx'], i)] = recaption
  
  if not caption_hash or not do_recaption:
    caption_hash = recaption_hash
    recaption_hash = {}

  problem_idxs= list(set(problem_idxs))

  for idx, data in list(idx2DataHash.items()):
    if len(data['media_list']) !=  data['text'].count("<caption>"):
      logger.warning("Something went wrong. There are missing captions for this image for item {idx}. Deleting the whole item")
      problem_itdxs.append(idx)
      del idxDataHash[idx]
      
  problem_idxs= list(set(problem_idxs))    
  for idx in problem_idxs:
    for idx_pair in list(caption_hash.keys()):
      if idx_pair[0] == idx:
        del caption_hash[idx_pair]
    for idx_pair in list(recaption_hash.keys()):
      if idx_pair[0] == idx:
        del recaption_hash[idx_pair]
    if idx in idx2DataHash: del idx2DataHash[idx]

  # create temprory working captions. Remove digits as words because
  # we will count object through object detection.  we will use a box
  # element model and a image-text similarity model to see if there
  # are any spurious elements in the prompts and remove the spurious
  # elements.

  working_caption = []
  working_recaption = []
  images = [] # list of images that we use for box element detection
  idx_pairs = [] # this is used for upsampling. to match the working_caption/recaptions to the images

  # we also need to account for text-image pairs we want to do cosim later.
  text_sim_check_idx_pairs = [] # this is used for keep track of matching a list of text against an image
  text_for_sim_check = [] # these are the candidate text to do sim check
  
  for idx, data in list(idx2DataHash.items()):
    image_set = data['media_list']
    # do some sanity checks
    for i, image in enumerate(image_set):
      caption = caption_hash.get((idx, id))
      recaption = recaption_hash.get((idx, id))      
      if not caption:
        logger.warning("Something went wrong. There are missing captions for this image for item {idx}. Deleting the whole item")
        del idx2DataHash[idx]
        break
    if not idx2DataHash.get(idx):
      continue
    for i, image in enumerate(image_set):
      images.append(image)
      caption = caption_hash.get((idx, i))
      text_sim_check_idx_pairs.append((idx,i))
      text_for_sim_check.append(caption)
      for word in digits_to_words: 
        caption = caption.replace(" " + word + " ", " ")
        working_caption.append(caption)
      idx_pairs.append((idx, i))
      # let's do the same for recaption
      recaption = recaption_hash.get((idx, i))      
      if recaption:
        text_sim_check_idx_pairs.append((idx,i))
        text_for_sim_check.append(caption)
        recaption = " "+ recaption +" "
        for word in digits_to_words: 
          recaption = recaption.replace(" " + word + " ", " ")
          working_recaption.append(recaption)
          
  #  remove missing elements and correct the count of elements
  #  accumlate the exsting elements
  working_caption_with_elements_removed = []
  existing_elements = []
  existing_elements_with_spatial_relationships = []
  for caption, image in zip(working_caption, images):
    aHash = detect_elements_and_remove_missing_elements(caption, image, score_cutoff, \
                                                working_caption_with_elements_removed, \
                                                existing_elements, \
                                                existing_elements_with_spatial_relationships)

  # if we also have a recaptioned text, we want this to also be used to fix the captioning, and a candidate for scoring
  if not working_recaption:
    working_recaption_with_elements_removed = [''] * len(working_caption_with_elements_removed)
  else:
    working_recaption_with_elements_removed = []
    #NOTE: we are going to ignore these tmp elements. We may want to revisit this.
    tmp_existing_elements = []
    tmp_existing_elements_with_spatial_relationships = []
    for recaption, image in zip(working_recaption, images):
      aHash = detect_elements_and_remove_missing_elements(recaption, image, score_cutoff, \
                                                working_recaption_with_elements_removed, \
                                                tmp_existing_elements, \
                                                tmp_existing_elements_with_spatial_relationships)


  # todo: save away the boxes  
  # now upsample the caption with the fixes, to make it grammatical
  upsampled_caption = []
  all_prefixes = []
  for caption, recaption, existing_element, existing_element_with_spatial_relationships , idx_pair in zip(working_caption_with_elements_removed, working_recaption_with_elements_removed, existing_elements, existing_elements_with_spatial_relationships, idx_pairs):
    prefix = random.choice(["an image of", "a photo of", "a photograph of", "a picture of", "a screenshot of", "a screen shot of"])
    all_prefixes.append(prefix)
    existing_element = existing_element.strip().replace("  ", " ")
    existing_element_with_spatial_relationships = existing_element_with_spatial_relationships.strip().replace("  ", " ")
    if recaption:
      upsampled_caption.append(tokenize_with_assistant_continuation(tokenizer, [{"role": "user", "content": f"Modify this image caption to make it grammatical and depicting a matter-of-fact scenary. Do not add new color, objects or people. Do not make up details about the image and stick strictly to the caption given. DO NOT add any comments, just give the modified caption. Caption:\n {caption}. In more detail; {recaption}.\n\n=====\n\nRemember to include these elements:\n{existing_element}"},
                                                                                            {"role": "assistant", "content": f"Modified Caption: {prefix}"}]))
    else:
      upsampled_caption.append(tokenize_with_assistant_continuation(tokenizer, [{"role": "user", "content": f"Modify this image caption to make it grammatical and depicting a matter-of-fact scenary. Do not add new color, objects or people. Do not make up details about the image and stick strictly to the caption given. DO NOT add any comments, just give the modified caption. Caption:\n {caption}.\n\n=====\n\nRemember to include these elements:\n{existing_element}"},
                                                                                            {"role": "assistant", "content": f"Modified Caption: {prefix}"}]))
      
    text_sim_check_idx_pairs.append(idx_pair)
    if existing_element != existing_element_with_spatial_relationships:
      if recaption:
        upsampled_caption.append(tokenize_with_assistant_continuation(tokenizer, [{"role": "user", "content": f"Modify this image caption to make it grammatical and depicting a matter-of-fact scenary. Do not add new color, objects or people. Do not make up details about the image and stick strictly to the caption given. DO NOT add any comments, just give the modified caption. Caption:\n {caption}. In more detail; {recaption}.\n\n=====\n\nRemember to include these elements:\n{existing_element_with_spatial_relationships}"}, 
                                                                                              {"role": "assistant", "content": f"Modified Caption: {prefix}"}]))
      else:
        upsampled_caption.append(tokenize_with_assistant_continuation(tokenizer, [{"role": "user", "content": f"Modify this image caption to make it grammatical and depicting a matter-of-fact scenary. Do not add new color, objects or people. Do not make up details about the image and stick strictly to the caption given. DO NOT add any comments, just give the modified caption. Caption:\n {caption}..\n\n=====\n\nRemember to include these elements:\n{existing_element_with_spatial_relationships}"}, 
                                                                                              {"role": "assistant", "content": f"Modified Caption: {prefix}"}]))
      all_prefixes.append(prefix)        
      text_sim_check_idx_pairs.append(idx_pair)

  with torch.no_grad():
    # we potenially doubled the size of the batch, so do this upsampling with the original batch_size
    upsampled_caption = generate_with_batching(model, tokenizer, upsampled_caption, batch_size=new_batch_size)
    upsampled_caption = [prefix+ " " + o.strip() for prefix, o in zip(all_prefixes, upsampled_caption)]

  items_processed += len(upsampled_caption)
  model_time += time.time()-time0

  text_for_sim_check.extend(upsampled_caption)
  assert len(text_for_sim_check) == len(text_sim_check_idx_pairs), "Something went wrong and the cosine similarity candidates indexing don't match"
  
  # evaluate the generated text by comparing its similarity with the whole image
  text_for_sim_check_hash = dict([(idx_pair, text) for idx_pair, text in zip(text_sim_check_idx_pairs, text_for_sim_check)])

  image_text_cosine_batch = {}
  for idx, data in idx2DataHash.items():
    # create field in params for text/score pairs for each caption in each item. this is a list of lists.
    data['metadata']['params']['related_caption_to_media_scores_list'] = [[]]* len(data['media_list'])
    for i, image in enumerate(data['media_list']):
      if (idx, i) in text_for_sim_check_hash:
        image_text_cosine_batch[(idx,i)] =  image_and_texts = image_text_cosine_batch.get((idx,i), [None, []])
        image_and_texts[0] = image
        image_and_texts[1].append(text_for_sim_check_hash[(idx,i)])
        
  for idx_pair, image_and_texts in image_text_cosine_batch.items():
    idx, i = idx_pair
    image, text = image_and_texts
    cos_scores = cosim_eval([image], texts)
    text_and_scores = list(zip(texts, [ss.item() for ss in cos_scores]))
    text_and_scores.sort(key=lambda a: a[1], reverse=True)
    data = idx2DataHash[idx]
    data['metadata']['params']['related_caption_to_media_scores_list'][i] = text_and_scores

  # now set the text field as the best matching caption and save away the scores
  # use the template we created to keep the same interleaved structure if there are any
  caption_template_hash = create_caption_template_hash(data_list)  
  for idx, data in idx2DataHash.items():
    template = caption_template_hash[data['idx']]
    template_params = dict([('CAPTION_'+str(i), related[0][0]) for i, related in enumerate(data['metadata']['params']['related_caption_to_media_scores_list'])])
    data['text'] = template.format(**template_params)
    data['media_caption_scores_list'] = [related[0][1] for i, related in enumerate(data['metadata']['params']['related_caption_to_media_scores_list'])]
  problem_idxs = set(problem_idxs)
  data_list = [data for data in data_list if data['idx'] not in problem_idxs]
  return data_list, model_time, items_processed, list(problem_idxs)

  
def generate_captions_then_generate_people_images(data_list):
  data_list, model_time, items_processed, problem_idxs = generate_captions_and_dont_clear_images(data_list)
  data_list1 = []
  data_list2 = []
  for data in data_list:
    caption = " "+data['text']+" "
    if 'people' in caption or 'person' in caption or " man" in caption or "woman" in caption or "boy" in caption or "girl" in caption:
      data['media_list'] = []
      data['media_coordinates_list'] = []
      data['media_caption_scores_list'] = []          
      data['media_types_list'] = []
      data_list2.append(data)
    else:
      data_list1.append(data)
  if data_list2:
    data_list2, new_model_time, new_items_processed, new_problem_idxs = generate_images(data_list2)
    return data_list1+data_list2, model_time+new_model_time, items_processed+new_items_processed, problem_idxs+new_problem_idxs
  else:
    return data_list, model_time, items_processed, problem_idxs

def generate_captions_then_images(data_list):
  data_list, model_time, items_processed, problem_idxs = generate_captions_and_clear_images(data_list)
  data_list, new_model_time, new_items_processed, new_problem_idxs = generate_images(data_list)
  return data_list, model_time+new_model_time, items_processed+new_items_processed, problem_idxs+new_problem_idxs

def generate_images_then_recaptions(data_list):
  data_list, model_time, items_processed, problem_idxs = generate_images(data_list)
  data_list, new_model_time, new_items_processed, new_problem_idxs = generate_captions_and_dont_clear_images(data_list)
  return data_list, model_time+new_model_time, items_processed+new_items_processed, problem_idxs+new_problem_idxs

# split up text into segments and create images and captions
# interleaved. of the form caption text caption text, etc.
# upsampling. just use the image generator to generate based on raw
# text, divided by sentences.  this is fixed size image generation for
# every 5 sentences. maximum of 3 captions.
# TODO:
# - add consistency between captions.
# - LLM upsample. 
def generate_interleaved_images_and_captions_from_text(data_list, max_captions=1):
  if max_captions > 3:
    logger.warning("Max captions is 3 for now. Setting to 3")
    max_captions = 3

  for data in data_list:
    original_text = []
    rest = ""
    if len(data['text']) < 100:
      original_text = [data['text']]
    else:
      text_arr = data['text'].split(".")
      if len(text_array) <= 5:
        original_text = [data['text']]
      elif len(text_array) <= 10:
        original_text = [".".join(text_array[:5])+".", ".".join(text_array[5:])]
      elif len(text_array) <= 15:
        original_text = [".".join(text_array[:5])+".", ".".join(text_array[5:10])+".",  ".".join(text_array[10:])]
      else:
        original_text = [".".join(text_array[:5])+".", ".".join(text_array[5:10])+".",  ".".join(text_array[10:15])+"."]
        rest = ".".join(text_array[15:])
    if max_captions == 1:
      rest = " ".join(original_text[1:])
      original_text = original_text[:1]
    elif max_captions == 2:
      rest = " ".join(original_text[2:])
      original_text = original_text[:2]
    data['text'] = ""
    if random.randint(0,1):
      for ot in original_text:
        ot = ot.rstrip(".")+"."
        data['text'] += "<caption>"+ot+"</caption>"+ot+" "
    else:
      for ot in original_text:
        ot = ot.rstrip(".")+"."
        data['text'] += ot+"<caption>"+ot+"</caption> "
    data['text'] += rest
    data['text'] = data['text'].rstrip()
  data_list, model_time, items_processed, problem_idxs = generate_images(data_list)
  data_list, new_model_time, new_items_processed, new_problem_idxs = generate_captions_and_dont_clear_images(data_list)
  return data_list, model_time+new_model_time, items_processed+new_items_processed, problem_idxs+new_problem_idxs

def generate_stories(data_list, clear_images=True):
  idx2DataHash = dict([(data['idx'], data) for data in data_list])  
  model = None
  tokenizer = None
  new_batch_size = args.batch_size
  if args.use_LLM_size == 'small':
    model = LLM_small_model
    tokenizer = LLM_small_tokenizer
  elif args.use_LLM_size == 'large':
    model = LLM_large_model
    tokenizer = LLM_large_tokenizer
    # to account for mismatches between the smaller image geeneration
    # and caption_generator_model models and the large LLMs, create sub_batches to
    # prevent OOM.
    new_batch_size = max(1, int(args.batch_size/2))
  else:
    model = LLM_medium_model
    tokenizer = LLM_medium_tokenizer
  problem_idxs = []
  prompts = []
  prompt = random.choice(STORY_PROMPT)
  for data in data_list:
    text = data['text']
    if 'assault' in text or 'robbery' in text or 'arson' in text or 'fellatio' in text or 'hand job' in text or 'prostitu' in text or 'handjob' in text or 'fucks' in text or 'blow job' in text or 'blowjob' in text or ' incest' in text or ' porn' in text or ' rape' in text or ' killer' in text or ' murder' in text or ' kidnap' in text or ' abduct' in text or ' sex ' in text  or ' torture' in text or ' kills ' in text:
      warning = "If this story contains themes of sex or violence, give a warning at the beginning of the story with an explanation."
    else:
      warning = ""
    prompt = prompt %{'warning': warning} + "\n\n" + text
    prompts.append(tokenie_with_chat_template(tokenizer, prompt))
    idxs.append(data['idx'])
  t0 = time.time()  
  with torch.no_grad():
    stories = generate_with_batching(model, tokenizer, prompts, batch_size=new_batch_size)
  model_time = time.time()-t0
  items_processed = len(stories)

  for idx, story, prompt in zip(idxs, stories, prompts):
    # do error checks and throw-away stories that are garbage
    data = idx2DataHash[idx]
    data['text'] = story
    data['metadata']['params']['story_prompt'] = prompt
  if clear_images:
    for data  in data_list:
      data['media_list'] = []
      data['media_coordinates_list'] = []
      data['media_caption_scores_list'] = []          
      data['media_types_list'] = []                            
  problem_idxs = set(problem_idxs)
  data_list = [data for data in data_list if data['idx'] not in problem_idxs]
  return data_list, model_time, items_processed, list(problem_idxs)

def generate_caption_from_instr(data_list):
  idx2DataHash = dict([(data['idx'], data) for data in data_list])  

  model = None
  tokenizer = None
  new_batch_size = args.batch_size
  if args.use_LLM_size == 'small':
    model = LLM_small_model
    tokenizer = LLM_small_tokenizer
  elif args.use_LLM_size == 'large':
    model = LLM_large_model
    tokenizer = LLM_large_tokenizer
    # to account for mismatches between the smaller image geeneration
    # and caption_generator_model models and the large LLMs, create sub_batches to
    # prevent OOM.
    new_batch_size = max(1, int(args.batch_size/2))
  else:
    model = LLM_medium_model
    tokenizer = LLM_medium_tokenizer
  
  idxs = []
  problem_idxs = []
  prompts = []
  for data in data_list:
    instr = data['text'].split("### Response:",1)[0].split("### Instruction:")[-1]  
    prefix = random.choice(["an image of", "a photo of", "a photograph of", "a picture of", "a screenshot of", "a screen shot of"])
    prompts.append(tokenize_with_assistant_continuation(tokenizer, [{"role": "user", "content": f"Create an image caption that would be useful for answering this instruction, including topics, people, places, things and details as necessary.\n\n{instr}"},
                                                    {"role": "assistant", "content": f"Caption: {prefix}"}]))
    idxs.append(data['idx'])
    if len(instr.split()) <= 2:
      problem_idxs.append(data['idx'])

  t0 = time.time()  
  with torch.no_grad():
    captions = generate_with_batching(model, tokenizer, prompts, batch_size=new_batch_size)
  model_time = time.time()-t0
  items_processed = len(captions)

  for idx, caption in zip(idxs, captions):
    # do error checks and throw-away captions that are garbage
    data = idx2DataHash[idx]
    #clean
    caption = caption.replace("Caption:", "").replace("caption:", "").replace("caption", "").replace("Caption", "").strip()
    caption = caption.replace("1.","").replace("2.","").replace("3.","").replace("1)","").replace("2)","").replace("3)","").strip('-\' "').strip()
    data['media_list'] = [caption]
    # data['metadata']['params']['instr_to_caption_prompt'] = prompt
                       
  problem_idxs = set(problem_idxs)
  data_list = [data for data in data_list if data['idx'] not in problem_idxs]
  return data_list, model_time, items_processed, list(problem_idxs)

def generate_revised_instr_response(data_list):
  idx2DataHash = dict([(data['idx'], data) for data in data_list])  

  model = None
  tokenizer = None
  new_batch_size = args.batch_size
  if args.use_LLM_size == 'small':
    model = LLM_small_model
    tokenizer = LLM_small_tokenizer
  elif args.use_LLM_size == 'large':
    model = LLM_large_model
    tokenizer = LLM_large_tokenizer
    # to account for mismatches between the smaller image geeneration
    # and caption_generator_model models and the large LLMs, create sub_batches to
    # prevent OOM.
    new_batch_size = max(1, int(args.batch_size/2))
  else:
    model = LLM_medium_model
    tokenizer = LLM_medium_tokenizer
  
  problem_idxs = []
  prompts = []
  idxs = [data['idx'] for data in data_list]
  captions = [data['media_list'][0] for data in data_list]
  instrs = [data['text'].split("### Response:",1)[0].split("### Instruction:")[-1] for data in data_list]
  revised_instrs = generate_image_aware_instruction(captions, instrs, model, tokenizer)
  responses = [data['chosen'] for data in data_list]
  prompts = [tokenizer.apply_chat_template([{"role": "user", "content": instruction}, {"role": "assistant", "content": response},
                                              {"role": "user", "content": f"Given the below image:\n{caption}\n===\n{revised_instruction}\nIf instruction cannot be answered based on the image alone, refer to the prior conversation, and explain why. Do not refer to any context document in your answer. If the question cannot be answered by the image state so politely and state why."}], tokenize=False)
              for caption, instruction, revised_instruction, response in zip(captions, instrs, revised_instrs, responses)]

  t0 = time.time()  
  with torch.no_grad():
    revised_responses = generate_with_batching(model, tokenizer, prompts, batch_size=new_batch_size)
  model_time = time.time()-t0
  items_processed = len(revised_responses)

  for idx, revised_instr, revised_response, prompt in zip(idxs, revised_instrs, revised_responses, prompts):
    # do error checks and throw-away garbage
    data = idx2DataHash[idx]

    if len(revised_instr.split()) <= 2 or len(revised_response.split()) <= 2:
      problem_idxs.append(idx)

    #clean
    revised_response = revised_response.replace("\n\n", "\n").strip()
    data['metadata']['params'] = json.loads(data['metadata']['params'])
    data['metadata']['params']['revised_instruction'] = revised_instr
    data['metadata']['params']['revised_response'] = revised_response
    data['metadata']['params']['revised_instr_response_prompt'] = prompt
                       
  problem_idxs = set(problem_idxs)
  data_list = [data for data in data_list if data['idx'] not in problem_idxs]
  return data_list, model_time, items_processed, list(problem_idxs)

def generate_autoredteam(data_list):
  _data_list, data_list = copy.deepcopy(data_list), []
  model = None
  tokenizer = None
  new_batch_size = args.batch_size
  if args.use_LLM_size == 'small':
    model = LLM_small_model
    tokenizer = LLM_small_tokenizer
  elif args.use_LLM_size == 'large':
    model = LLM_large_model
    tokenizer = LLM_large_tokenizer
    # to account for mismatches between the smaller image geeneration
    # and caption_generator_model models and the large LLMs, create sub_batches to
    # prevent OOM.
    new_batch_size = max(1, int(args.batch_size/2))
  else:
    model = LLM_medium_model
    tokenizer = LLM_medium_tokenizer
  
  t0 = time.time()  
  for row in _data_list:
    data = auto_redteam(
                    target_model=model, target_tokenizer=tokenizer, 
                    purpleteam_generative_model=model, purpleteam_generative_tokenizer=tokenizer, 
                    blueteam_llamaguard_model=evaluator_model, blueteam_llamaguard_tokenizer=evaluator_tokenizer,
                    verb_type=row["verb_type"], obj_type=row["obj_type"], batch_size=new_batch_size, blueteam_batch_size=new_batch_size,
                  )
    data_list.extend(data)
  model_time = time.time()-t0
  items_processed = len(data_list)
  return data_list, model_time, items_processed, []

### MAIN FUNCTIONS THAT CALLS THE GENERATE FUNCTIONS
def write_image(image_then_path_list):
  for image, image_path in image_then_path_list:
    image.save(image_path)

# for other media, we will need to create other write functions

def do_one_batch(base_path, outfile, rng, pool, generate_function, curr_data, all_writers, save_time=0, all_model_time=0, items_processed=0):
  # this is a map_reduce function.
  # this function maps one batch into each GPU/process and then reduces back to a single output file.
  # optionally separate images can be written to disk as well in a thread so that the writing is faster.
  global num_devices, args, node_name
  problem_idxs = []
  # curr_data = cleanup_data_batch(curr_data) 
  for idx, data in enumerate(curr_data):
    data['idx'] = idx+rng # these are temporary indexes
  chunks = chunkify(curr_data, num_devices)
  seen_subdir = {}
  for batch, model_time, new_items_processed, new_problem_idxs in pool.imap_unordered(generate_function, chunks, chunksize=1): # imap_unordered doesn't seem to speed things up??
    problem_idxs.extend(new_problem_idxs)
    image_and_path = []
    all_model_time += model_time
    items_processed += new_items_processed
    t0 = time.time()
    for data in batch:

      # Image handling
      # for other media, we will need to write other write functions
      if data["media_list"] and type(data["media_list"][0]) is PIL.Image.Image:
        #TODO - do for other media like sound, video and images
        idx = data['idx']
        images = data['media_list']
        data['media_list'] = []
        #TODO, see if we have already saved this image by using SH-1 fingerprint, so we don't save to disk drive
        for image_idx, image in enumerate(images):
          if not image: continue
          subdir = int(idx//1000) # for 1M records, there will be 1000 directories, each with 1000 files.
          if subdir not in seen_subdir:
            width, height = image.size
            os.makedirs(f"{args.output_dir}/{base_path}/{subdir}/", exist_ok=True)
            seen_subdir[subdir] = 1
          image_and_path.append((image, f"{args.output_dir}/{base_path}/{subdir}/{idx}-{image_idx}-{width}x{height}.png"))
          #image.save(f"{args.output_dir}/{base_path}-{idx}-{image_idx}.png")
          data["media_list"].append(f"{base_path}/{subdir}/{idx}-{image_idx}-{width}x{height}.png")

      data["metadata"]["source"] += f"|{args.input_path}.{node_name}.{args.task}.{idx}"
      if 'idx' in data: del data['idx'] # these are termporary indexes
      data = cleanup_and_serialize_params(data)
      outfile.write(json.dumps(data)+"\n")
      data = None
    if image_and_path:
      all_writers.append(threading.Thread(target=write_image, args=(image_and_path,)))
      all_writers[-1].start()
    save_time += time.time()-t0
  return all_writers, save_time, all_model_time, items_processed, problem_idxs

#TODO: set tansformers logging in args
def main_standard_format():
  # this function reads jsonls in the format we expect. e.g., text, chosen, rejected_list, media_list, ... etc. see definition in utils.py
  global args, node_name
  task = args.task
  generate_function = tasks_configs[task]['generate_function']
  all_writers = []
  start = time.time()
  items_processed = 0
  all_model_time = 0
  save_time = 0
  problem_idxs = []
  data_load_time = 0
  os.makedirs(args.output_dir, exist_ok=True)
  multiprocessing.set_start_method('spawn', force=True)
  with  multiprocessing.Pool(processes=num_devices) as pool:
    initialize_gpus_and_models(pool, args)
    # consider creating an outputfile per inputfile, and passing in inputfile patterns
    if args.input_path[-1] != '*':
      input_files = [args.input_path]
      output_files = [args.output_path]
    else:
      input_files = list(glob.glob(args.intput_path))
      if args.output_path:
        logger.warning("Input path is a pattern, so we will ignore output path. To processs multiple input files, we will use the outpur_dir so we can mirror the input path")
      if not args.output_suffix:
        assert False, "To processs multiple input files, we will use the output_dir/input_file+output_suffix. No output_suffix specified."
      output_files = [args.output_dir+"/"+file.split(args.input_dir,1)[-1].lstrip("/").replace(".jsonl", "")+args.output_suffix+".jsonl" for file in input_files]
    for input_file, output_file in zip(input_files, output_files):
      base_path = output_file.split(args.output_dir,1)[-1].lstrip("/").replace(".jsonl", "")
      os.makedirs(args.output_dir+"/"+base_path, exist_ok=True)
      with open(output_file, "w") as outfile: 
        # read data of various forms
        # potentially have a reader thread to read while we wait for GPUs to finish
        t0 = time.time()
        all_data = [json.loads(l) for l in open(input_file).read().split("\n") if l.strip()]
        data_load_time += time.time()-t0
        # to get the true batch per caption, we need to get the average num of captions and divde by the batch size
        if 'caption' in args.task or 'image' in args.task:
          avg_num_caption = max(1, sum(data['text'].count("<caption>") for data in all_data)/len(all_data)) 
          avg_num_image = max(1, sum(len(data['media_list']) for data in all_data)/len(all_data))
          avg_num = max(avg_num_caption, avg_num_image)
        else:
          avg_num = 1
        # create super batches for each of the N GPUs
        step_size = max(1, int(args.batch_size/avg_num))*num_devices
        for rng in range(0, len(all_data), step_size):
          curr_data = all_data[rng: min(len(all_data), rng+step_size)]
          all_writers, save_time, all_model_time, items_processed, new_problem_idxs = do_one_batch(base_path, outfile, rng, pool, generate_function, curr_data, all_writers, save_time, all_model_time, items_processed)
          problem_idxs.extend(new_problem_idxs)
  for writer in all_writers:
    writer.join()
  stop = time.time()
  total_time = stop-start
  # consider writing a log file per outputfile so that we can push the data for that shard to HF                                                                                                                                                                               
  logger.warning(f"{node_name}.{args.task} : {items_processed} in {total_time} seconds total time on {num_devices} GPUs and processes. All model time took {all_model_time}. Disk load time took {data_load_time}. Disk save time took {save_time} " + str(args))
  logger.warning(f"{node_name}.{args.task} : Problem items " + str(problem_idxs))

def main_autoredteam():
  # reads verb and obj pairs from 'templates/seed.py' , and processes into our standard format, with the generate_function applied to each data item
  global args, node_name
  task = args.task
  generate_function = tasks_configs[task]['generate_function']
  all_writers = []
  start = time.time()
  items_processed = 0
  all_model_time = 0
  save_time = 0
  problem_idxs = []
  data_load_time = 0
  # #TODO - do multiple input files
  if not args.output_path:
    args.output_path = args.output_dir+"/"+args.output_suffix+".jsonl"
  base_path = args.output_path.split(args.output_dir,1)[-1].lstrip("/").replace(".jsonl", "")
  os.makedirs(args.output_dir, exist_ok=True)
  os.makedirs(args.output_dir+"/"+base_path, exist_ok=True)
  multiprocessing.set_start_method('spawn', force=True)
  with  multiprocessing.Pool(processes=num_devices) as pool:
    initialize_gpus_and_models(pool, args)
    # consider whether we want to use imap_unordered instead of startmap so we get data right away and start processing.
    # might not matter as all GPUs will probably finish at the same time.
    # consider creating an outputfile per inputfile, and passing in inputfile patterns
    with open(args.output_path, "w") as outfile: 
        # read data of various forms
        # potentially have a reader thread to read while we wait for GPUs to finish, if we do multiple files at the same time
        t0 = time.time()
        all_data = [{"verb_type": verb_type, "obj_type": obj_type} for verb_type in verb_templates.keys() for obj_type in obj_templates.keys()]
        data_load_time += time.time() - t0
        avg_num = 1 # there is only one image.
        # create super batches for each of the N GPUs
        step_size = max(1, int(args.batch_size/avg_num))*num_devices
        for rng in range(0, len(all_data), step_size):
          curr_data = all_data[rng: min(len(all_data), rng+step_size)]
          all_writers, save_time, all_model_time, items_processed, new_problem_idxs = do_one_batch(base_path, outfile, rng, pool, generate_function, curr_data, all_writers, save_time, all_model_time, items_processed)
          problem_idxs.extend(new_problem_idxs)
  for writer in all_writers:
    writer.join()
  stop = time.time()
  total_time = stop-start
  # consider writing a log file per outputfile so that we can push the data for that shard to HF
  logger.warning(f"{node_name}.{args.task} : {items_processed} in {total_time} seconds total time on {num_devices} GPUs and processes. All model time took {all_model_time}. Disk load time took {data_load_time}. Disk save time took {save_time} " + str(args))
  logger.warning(f"{node_name}.{args.task} : Problem items " + str(problem_idxs))

#TODO: set tansformers logging in args
def main_commoncatalog():
  # reads from commoncatalog, and processes into our standard format, with the generate_function applied to each data item
  global args, node_name
  task = args.task
  generate_function = tasks_configs[task]['generate_function']
  all_writers = []
  start = time.time()
  items_processed = 0
  all_model_time = 0
  save_time = 0
  problem_idxs = []
  data_load_time = 0
  #TODO - do multiple input files
  if args.input_path[-1] == '*':
    assert False, "Multiple input for commoncatalog not yet impelemented"
  if not args.output_path:
    args.output_path = args.output_dir+"/"+args.input_path.split("/")[-1].replace(".parquet", "")+args.output_suffix+".jsonl"
  base_path = args.output_path.split(args.output_dir,1)[-1].lstrip("/").replace(".jsonl", "")
  os.makedirs(args.output_dir, exist_ok=True)
  os.makedirs(args.output_dir+"/"+base_path, exist_ok=True)
  multiprocessing.set_start_method('spawn', force=True)
  with  multiprocessing.Pool(processes=num_devices) as pool:
    initialize_gpus_and_models(pool, args)
    # consider whether we want to use imap_unordered instead of startmap so we get data right away and start processing.
    # might not matter as all GPUs will probably finish at the same time.
    # consider creating an outputfile per inputfile, and passing in inputfile patterns
    with open(args.output_path, "w") as outfile: 
        # read data of various forms
        # potentially have a reader thread to read while we wait for GPUs to finish, if we do multiple files at the same time
        t0 = time.time()
        df = parquet.read_table(args.input_path)
        all_data = []
        for image, caption, blip_text, title, usertags, url in zip(df['jpg'], df['caption'], df['blip2_caption'], df['title'], df['usertags'], df['downloadurl']):
          all_data.append({'media_list': [image], 'text': unquote((title.as_py()+": "+ blip_text.as_py()+". In more detail: "+caption.as_py()).replace("+", " ")),\
                           'metadata': {'source': '', 'params': { 'title': title.as_py(), 'blip_text': blip_text.as_py(), 'orig_caption': caption.as_py(), 'usertags': usertags.as_py(), 'url': url.as_py()}}})
        df = None
        data_load_time += time.time() - t0
        avg_num = 1 # there is only one image.
        step_size = max(1, int(args.batch_size/avg_num))
        for idx, data in enumerate(all_data):
            data['media_list'] = [Image.open(BytesIO(image.as_py()))  for image in data['media_list']]
        all_writers, save_time, all_model_time, items_processed, new_problem_idxs = do_all_batch(base_path, outfile, pool, generate_function, all_data, step_size, all_writers, save_time, all_model_time, items_processed)
        problem_idxs.extend(new_problem_idxs)
  for writer in all_writers:
    writer.join()
  stop = time.time()
  total_time = stop-start
  # consider writing a log file per outputfile so that we can push the data for that shard to HF
  logger.warning(f"{node_name}.{args.task} : {items_processed} in {total_time} seconds total time on {num_devices} GPUs and processes. All model time took {all_model_time}. Disk load time took {data_load_time}. Disk save time took {save_time} " + str(args))
  logger.warning(f"{node_name}.{args.task} : Problem items " + str(problem_idxs))

### TASK CONFIGS

tasks_configs = {"generate_images": {"models_needed": ["image_generator", ], "main_function": main_standard_format, 'generate_function': generate_images},
                 "generate_captions": {"models_needed": ["caption_generator", ], "main_function": main_standard_format, 'generate_function': generate_captions_and_dont_clear_images},
                 "generate_stories": {"models_needed": ["LLM_model"], "main_function": main_standard_format, 'generate_function': generate_stories},                                                   
                 "generate_images_then_recaption": {"models_needed": ["caption_generator", 'image_generator'],  "main_function": main_standard_format, 'generate_function': generate_images_then_recaptions},                 
                 "generate_captions_then_images": {"models_needed": ["caption_generator", 'image_generator'],  "main_function": main_standard_format, 'generate_function': generate_captions_then_images},
                 "generate_interleaved_images_and_captions_from_text": {"models_needed": ["caption_generator", 'image_generator'],  "main_function": main_standard_format, 'generate_function': generate_interleaved_images_and_captions_from_text},                 
                 # common catalog voersion
                 "generate_captions_from_commoncatalog": {"models_needed": ["caption_generator", ], "main_function": main_commoncatalog, 'generate_function': generate_captions_and_clear_images},
                 "generate_stories_from_commoncatalog": {"models_needed": ["LLM_model"], "main_function": main_commoncatalog, 'generate_function': generate_stories},                                  
                 "generate_images_from_commoncatalog": {"models_needed": ["image_generator", ], "main_function": main_commoncatalog, 'generate_function': generate_images},
                 "generate_captions_then_generate_people_images_from_commoncatalog": {"models_needed": ["caption_generator", "image_generator"], "main_function": main_commoncatalog, 'generate_function': generate_captions_then_generate_people_images},
                 "generate_images_then_recaption_from_commoncatalog": {"models_needed": ["box_segmenter", "image_text_scorer", "caption_generator", 'image_generator'],  "main_function": main_commoncatalog, 'generate_function': generate_images_then_recaptions},
                 "generate_captions_then_images_from_commoncatalog": {"models_needed": ["caption_generator", 'image_generator'],  "main_function": main_commoncatalog, 'generate_function': generate_captions_then_images},
                 # auto redteam voersion
                 "generate_autoredteam": {"models_needed": ["LLM_model", "evaluator"], "main_function": main_autoredteam, 'generate_function': generate_autoredteam},
                 "generate_captions_from_autoredteam": {"models_needed": ["LLM_model"], "main_function": main_standard_format, 'generate_function': generate_caption_from_instr},
                 "generate_images_then_recaption_from_autoredteam": {"models_needed": ["box_segmenter", "image_text_scorer", "caption_generator", 'image_generator'], "main_function": main_standard_format, 'generate_function': generate_images_then_recaptions},
                 "generate_revised_instruction_then_response_from_autoredteam": {"models_needed": ["LLM_model"], "main_function": main_standard_format, 'generate_function': generate_revised_instr_response},
                }

def parse_args():
  global args, node_name
  parser = argparse.ArgumentParser(description="Set up data generation and models with specific configurations.")
  parser.add_argument("--use_LLM_size", type=str, default="medium", help="Use a particular LLM for generation")
  parser.add_argument("--task", type=str, default="generate_images", help="Task: one of "+", ".join(tasks_configs.keys()))
  parser.add_argument("--batch_size", type=int, default=50, help="Batch size")
  parser.add_argument("--caption_max_sequence", type=int, default=512, help="Max caption sequence")
  parser.add_argument("--image_gen_caption_max_sequence", type=int, default=256, help="Image text encoder max sequence")                                            
  parser.add_argument("--image_width", type=int, default=256, help="Image width for generation. If <= 0, then a random number will be chosen between 256, 512, 1024, 2048, 4096")
  parser.add_argument("--image_height", type=int, default=256, help="Image height for generation. If <= 0, then a random number will be chosen between 256, 512, 1024, 2048, 4096")
  parser.add_argument("--max_detections", type=int, default=5, help="Maximum number of boxes to detect")  
  parser.add_argument("--score_cutoff", type=float, default=0.14, help="score cutoff")
  parser.add_argument("--cache_dir", type=str, default="/leonardo_work/EUHPC_E03_068/.cache", help="Path to cache directory.")
  parser.add_argument("--LLM_small_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Small LLM generative model hf path.")
  parser.add_argument("--LLM_medium_model", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="Medium LLM generative model hf path.")
  parser.add_argument("--LLM_large_model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Large LLM generative model hf path.")  
  parser.add_argument("--image_text_score_model", type=str, default="openai/clip-vit-base-patch32", help="Model used to get the image-text cosine similarity.")
  parser.add_argument("--caption_generator_model", type=str, default='microsoft/Florence-2-large', help="Model used for generating caption of an image.")
  parser.add_argument("--image_generator_model", type=str, default="black-forest-labs/FLUX.1-schnell", help="Image generator model compatible with diffuser")
  parser.add_argument("--box_segementer_config_path", type=str, default="/leonardo_work/EUHPC_E03_068/safellm/src/frcnn/config.jsonl", help="local config file for frcnn")
  parser.add_argument("--box_segementer_model", type=str, default="unc-nlp/frcnn-vg-finetuned", help="Model used to do mox segmentation")
  parser.add_argument("--evaluator_model", type=str, default="llamas-community/LlamaGuard-7b", help="Evaluator model for generating policy based data") 
  parser.add_argument("--input_dir", type=str, default="", help="Path to the input file.")    
  parser.add_argument("--output_dir", type=str, default="", help="Path to the input file.")
  parser.add_argument("--output_path", type=str, default="", help="Path to save output for this step. If the output_path is empty, we will assume it is output_dir/input_name+output_suffix.jsonl")
  parser.add_argument("--output_suffix", type=str, default="", help="Suffix to append to input_file base path. Output file is then output_dir/input_name+output_suffix.jsonl")
  parser.add_argument("--input_path", type=str, default="", help="Path to load the input for this step. Can be a single file or file pattern that ends with an '*' for multiple input files. The output will be output_dir/input_name+output_suffix.jsonl")

  # TODO, create the output_dir if it doesn't exist with mkdir -p
  node_name = parser.prog.replace(".py", "")
  args = parser.parse_args()
  args.node_name = node_name
  if args.output_path and args.output_dir:
    assert args.output_dir in args.output_path or args.output_dir[0] != "/", "The output_path must be in the output_dir or must be a relative path (without '/' at the beginning)"
  if args.output_path and args.output_path[0] != '/':
    args.output_path = (args.output_dir+"/"+args.output_path).replace("//", "/")
  if args.input_path and args.input_dir:
    assert args.input_dir in args.input_path or args.input_dir[0] != "/", "The input_path must be in the input_dir or must be a relative path (without '/' at the beginning)"
  if args.input_path and args.input_path[0] != '/':
    args.input_path = (args.input_dir+"/"+args.input_path).replace("//", "/")
  logger.warning(f"RUNNING: {node_name} on {num_devices} GPUs")
  logger.warning(args)


## NOTE that model load time can be quite long (in minutes), so we
## should process as many items in one run as possible to amortize
## that time cost. DO NOT launch this script multiple times for small
## shards of a dataset. Instead, loop through the shards.
if __name__ == "__main__":
  parse_args()
  main = tasks_configs[args.task]['main_function']
  main()
  logger.warning("Completed!!")
