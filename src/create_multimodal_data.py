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
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, LlavaForConditionalGeneration
from torch import multiprocessing
from torch import threading
#import multiprocessing, threading
from transformers.utils import logging as transformers_logging
from src.utils import chunkify, chatml_format_instructions, generate_with_batching, assign_uuid, tokenize_with_assistant_continuation, cleanup_data_batch, standardize_data_fields, cleanup_and_serialize_params, augment_for_quotes, \
                      STORY_PROMPTS, generate_image_aware_instruction
from src.purpleteam.autoredteam import auto_redteam
from src.purpleteam.templates.seed import verb_templates, obj_templates
from src.frcnn.visualizing_image import SingleImageViz
from src.frcnn.processing_image import Preprocess as FRCNNPreprocess
from src.frcnn.modeling_frcnn import GeneralizedRCNN
from src.frcnn.utils import Config as FRCNNConfig
from src.frcnn.utils import decode_image as frcnn_decode_image
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

#transformers_logging.set_verbosity(transformers.logging.ERROR)

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
image_text_model = None
image_text_processor = None
device  = None
box_detect_image_preprocessor = None
box_segmentation_model = None
evaluator_model = None
evaluator_tokenizer = None
llavaguard_model = None
llavaguard_processor = None

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
      image_text_model, \
      image_text_processor, \
      device, \
      box_detect_image_preprocessor, \
      box_segmentation_model, \
      evaluator_model, \
      evaluator_tokenizer, \
      llavaguard_model, \
      llavaguard_processor, \
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
    LLM_small_tokenizer = AutoTokenizer.from_pretrained(args.LLM_small_model, trust_remote_code=True, cache_dir=args.cache_dir, padding_side='left')
    if not LLM_small_tokenizer.pad_token:
      LLM_small_tokenizer.pad_token = LLM_small_tokenizer.eos_token

  if (("LLM_model" in config['models_needed']) or 'LLM_medium' in config['models_needed']) and 'medium' in args.use_LLM_size  and LLM_medium_tokenizer is None:
    logger.warning(f'CREATING {args.LLM_medium_model} MODEL')    
    LLM_medium_model = AutoModelForCausalLM.from_pretrained(args.LLM_medium_model, trust_remote_code=True, torch_dtype=torch.bfloat16, cache_dir=args.cache_dir, attn_implementation="flash_attention_2").train().to(device)
    LLM_medium_tokenizer = AutoTokenizer.from_pretrained(args.LLM_medium_model, trust_remote_code=True, cache_dir=args.cache_dir, padding_side='left')
    if not LLM_medium_tokenizer.pad_token:
      LLM_medium_tokenizer.pad_token = LLM_medium_tokenizer.eos_token

  if (("LLM_model" in config['models_needed']) or 'LLM_large' in config['models_needed']) and 'large' in args.use_LLM_size and LLM_large_tokenizer is None:
    logger.warning(f'CREATING {args.LLM_large_model} MODEL')    
    LLM_large_model = AutoModelForCausalLM.from_pretrained(args.LLM_large_model, trust_remote_code=True, torch_dtype=torch.bfloat16, cache_dir=args.cache_dir, attn_implementation="flash_attention_2").train().to(device)
    LLM_large_tokenizer = AutoTokenizer.from_pretrained(args.LLM_large_model, trust_remote_code=True, cache_dir=args.cache_dir, padding_side='left')
    if not LLM_large_tokenizer.pad_token:
      LLM_large_tokenizer.pad_token = LLM_large_tokenizer.eos_token

  if 'image_text_scorer' in config['models_needed'] and image_text_processor is None:
    logger.warning(f'CREATING {args.image_text_score_model} MODEL')
    if 'openai/clip' in args.image_text_model:
      image_text_model = CLIPModel.from_pretrained(args.image_text_score_model, trust_remote_code=True, torch_dtype=torch.bfloat16, cache_dir=args.cache_dir).eval().to(device)
      image_text_processor = CLIPProcessor.from_pretrained(args.image_text_score_model, cache_dir=args.cache_dir)
    else:
      assert False, f"{args.image_text_score_model} not yet supported"

  if 'box_segementer' in config['models_needed'] and box_segmentation_model is None:
    logger.warning(f'CREATING {args.box_segementer_model} MODEL')    
    if "unc-nlp/frcnn-vg-finetuned" in args.box_segmenter_model:
      frcnn_config = json.load(open(args.box_segmenter_config_path)) 
      frcnn_config = FRCNNConfig(frcnn_config)
      box_detect_image_preprocessor= FRCNNPreprocess(frcnn_config).half().cuda()
      box_segmentation_model= GeneralizedRCNN.from_pretrained(args.box_segementer_model, frcnn_config, trust_remote_code=True, cache_dir=args.cache_dir).half().eval().to(device)
    else:
      assert False, f"{args.box_segmenter_model} not yet supported"
      
  if 'evaluator' in config['models_needed'] and evaluator_model is None:
    logger.warning(f'CREATING {args.evaluator_model} MODEL')    
    if "LlamaGuard-7b" in args.evaluator_model:
      evaluator_model = AutoModelForCausalLM.from_pretrained(args.evaluator_model, trust_remote_code=True, torch_dtype=torch.bfloat16, cache_dir=args.cache_dir, attn_implementation="flash_attention_2").train().to(device)
      evaluator_tokenizer = AutoTokenizer.from_pretrained(args.evaluator_model, trust_remote_code=True, cache_dir=args.cache_dir)
      if not evaluator_tokenizer.pad_token:
        evaluator_tokenizer.pad_token = evaluator_tokenizer.eos_token
    else:
      assert False, f"{args.evaluator_model} not yet supported"

  if 'llavaguard' in config['models_needed'] and llavaguard_model is None:
    logger.warning(f'CREATING {args.llavaguard_model} MODEL')    
    if "AIML-TUDA/LlavaGuard-v1.1-7B-hf" in args.llavaguard_model:
      evaluator_model = LlavaForConditionalGeneration.from_pretrained(args.llavaguard_model, trust_remote_code=True, torch_dtype=torch.bfloat16, cache_dir=args.cache_dir, attn_implementation="flash_attention_2").train().to(device)
      evaluator_processor = AutoProcessor.from_pretrained(args.llavaguard_model, trust_remote_code=True, cache_dir=args.cache_dir)
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

# Create a template so we can use python's format functionality to fill
# in captions in the 'text' field. Cleanup the text field to add
# caption tags if none are there.
def create_caption_template_hash(data_list):
  caption_template_hash = {}
  params_template_hash = {}
  for data in data_list:
    if "<image>" in data['text'] and '<caption>' not in data['text']:
     data['text'] = data['text'].replace('<image>', '<caption><image></caption>')
    text_lower = data['text'].lower()
    if "<caption>" not in data['text']:
      if ("<table" in text_lower or  "<div>" in text_lower or "<pre>" in text_lower or \
          "<!" in text_lower or "<a href" in text_lower or "<br>" in text_lower or \
          "<p>" in text_lower or "<h1" in text_lower or "<h2" in text_lower or "<h3" in text_lower or "<h4" in text_lower or "<h5" in text_lower or "<h6" in text_lower):
        data['text'] = "<caption></caption>"+data['text']
      else:
        data['text'] = "<caption>"+data['text']+"</caption>"
    caption_count = data['text'].count("<caption>")
    if data['media_list'] and data['media_list'][0] and len(data['media_list']) > caption_count:
      data['text'] = data['text']+"".join(["<caption><image></caption>" for _ in range(len(data['media_list']) - caption_count)])
    #logger.warning("1 "+data['text'])
    template = ""
    new_text = ""
    params_hash = {}
    for i, t in enumerate(data['text'].split("<caption>")):
        if (i+1) % 2 == 0:
          j = i-1 # we start indexing by 0
          caption =  t.split("</caption>",1)[0]
          t = "<caption>"+t
          if "</caption>" not in t:
            t = t+"</caption>"
          # manage other tags
          if caption.startswith("<"):
            template = template+"<caption><image>{CAPTION_"+str(j)+"}</caption>"+t.split("</caption>",1)[-1]
            caption = caption.split(">",1)[-1]
          elif caption.endswith(">"):
            template = template+"<caption>{CAPTION_"+str(j)+"}<image></caption>"+t.split("</caption>",1)[-1]
            caption = caption.split("<")
            caption = "<".join(caption[:-1])
          elif "<" in caption and ">" in caption: 
            if " " not in caption.split(">",1)[0].split("<",1)[-1]: # we might have captions that have <> and not tags? probably do a regex or test??
              assert False, "<image> or other media tags must occur before or after the caption, and not within the caption"
          else:
            template = template+"<caption>{CAPTION_"+str(j)+"}</caption>"+t.split("</caption>",1)[-1]            
          params_hash[f"CAPTION_{j}"] = caption
          new_text = new_text + t
        else:
          template = template + t
          new_text = new_text + t
    data['text'] = new_text
    #logger.warning ("2 "+ data['text'])
    caption_template_hash[data['_tmp_idx']] = template
    params_template_hash[data['_tmp_idx']] = params_hash
    
  return caption_template_hash, params_template_hash
def set_caption(params_template, j, text):
  params_template[f"CAPTION_{j}"] = text

def reset_all_captions(params_template, text_array):
  for j, text in enumerate(text_array):
    params_template[f"CAPTION_{j}"] = text

def apply_params_to_caption_text(data_list, caption_template_hash, params_template_hash):
  for data in data_list:
    caption_template = caption_template_hash[data['_tmp_idx']]
    params_template = params_template_hash[data['_tmp_idx']]
    #logger.warning("applying " + str(params_template))
    data['text'] = caption_template.format(**params_template)
    

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

# we can make these probabilities parameters
possible_image_dims = ([256]*20) + ([512]*10)+ ([1024]*5) + ([2048]*3) + ([4096]*2)

# Given a list of data items with a text field with captions inside
# it, each keyed to an idx, generate list of list of images keyed to
# the idx.  NOTE: we don't filter out problem data at the idx/element
# level. instead, if there is a single element that is a problem, we
# remove the whole data at idx.
def generate_images(data_list):
  global device, image_generator, caption_generator_model, caption_generator_processor, args
  logger.warning(f"Starting Image Generation "+ device + " " + str(image_generator.device))  
  idx2DataHash = dict([(data['_tmp_idx'], data) for data in data_list])
  problem_idxs = []
  model_time = 0
  items_processed = 0
  new_batch_size = args.batch_size
  width=args.image_width
  height=args.image_height
  if width <= 0:
    if height > 0 and random.randint(0,5) == 0: # increase the probability of square ratio
      width = height
    else:
      width = random.choice(possible_image_dims)
    if width == 512:
      new_batch_size = new_batch_size/1.5
    elif width == 1024:
      new_batch_size = new_batch_size/2
    elif width == 2048:
      new_batch_size = new_batch_size/4.3
    elif width == 4096:
      new_batch_size = new_batch_size/12

  if height <= 0:
    if width > 0 and random.randint(0,5) == 0:
      height = width
    else:
      height = random.choice(possible_image_dims)
    if height == 512:
      new_batch_size = new_batch_size/1.5
    elif height == 1024:
      new_batch_size = new_batch_size/2
    elif height == 2048:
      new_batch_size = new_batch_size/4.3
    elif height == 4096:
      new_batch_size = new_batch_size/12

  #HACK - On leonardo, we get OOM for 4096x4096
  if height == 2048 and width == 4096:
    height = 1024
    logger.warning("4096x2048 will OOM on Leonardo even at batchsize==1. Resetting to 4096x1024")
  if height == 4096 and width == 2048:
    width = 1024
    logger.warning("2048x4096 will OOM on Leonardo even at batchsize==1. Resetting to 1024x4096")
  if height == 4096 and width == 4096:
    height = 1024
    logger.warning("4096x4096 will OOM on Leonardo even at batchsize==1. Resetting to 4096x1024")
  new_batch_size = max(1, int(new_batch_size))
  logger.warning(f"Setting images  in datalist " + str(len(data_list)))
  caption_template_hash, params_template_hash = create_caption_template_hash(data_list)  
  for data in data_list:
    caption_template = caption_template_hash[data['_tmp_idx']]
    params_template = params_template_hash[data['_tmp_idx']]
    for key, val in list(params_template.items()):
      if '<image>' not in val:
        if random.randint(0,1):
          val = val + "<image>"
        else:
          val = "<image>"+val
        params_template[key] = val
  apply_params_to_caption_text(data_list, caption_template_hash, params_template_hash)
  logger.warning ("finished setting image in datalist " + str(len(data_list)))
  #We use idx_pair: (idx, i) where i is the poisition in the media_list
  with torch.no_grad():
    caption_list = []
    idx_pairs = []
    for data in data_list:
      idx = data['_tmp_idx']
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
    logger.warning("drawing at " + str(width) +"x"+str(height) + " at batch size " + str(new_batch_size))
    for rng in range(0, len(caption_list), new_batch_size):
      caption_batch = caption_list[rng: min(len(caption_list), rng+new_batch_size)]
      logger.warning("Doing one batch of captioning " + str(len(caption_batch)))      
      images.extend(image_generator(
        caption_batch,
        guidance_scale=0.0,
        num_inference_steps=4,
        max_sequence_length=args.image_gen_caption_max_sequence,
        width=width, height=height,
        generator=torch.Generator(image_generator.device).manual_seed(0)
      ).images)
    model_time = time.time() - time0
    items_processed = len(images)
    logger.warning(f"Image Gen time: {model_time} "+str(len(data_list)) + " " + str(len(images)))
  #torch.cuda.empty_cache()
  for data  in data_list:
    data['media_list'] = []
    data['media_coordinates_list'] = []
    data['media_caption_scores_list'] = []    
    data['media_types_list'] = []
  logger.warning(f"Cleared images ")    
  for idx_pair, image in zip(idx_pairs, images):
    #logger.warning("setting " + str(idx_pair))
    idx, i = idx_pair
    data = idx2DataHash[idx]
    if len(data['media_list']) < i+1:
      data['media_list'].extend([None]*(i+1-len(data['media_list'])))
    if len(data['media_coordinates_list']) < i+1:
      data['media_coordinates_list'].extend([None]*(i+1-len(data['media_coordinates_list'])))
    if len(data['media_types_list']) < i+1:
      data['media_types_list'].extend([None]*(i+1-len(data['media_types_list'])))
    data['media_list'][i] = image
    data["media_coordinates_list"][i] = [0, 0]+ list(image.size) # is this widthxheight? we can consider shifting the coordinates over for multiple images. this can be done at run time.
    data["media_types_list"][i] = "image"
  problem_idxs = set(problem_idxs)
  logger.warning("finished setting")
  data_list = [data for data in data_list if data['_tmp_idx'] not in problem_idxs]
  
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
  idx2DataHash = dict([(data['_tmp_idx'], data) for data in data_list]) 
  time0 = time.time()
  caption_generator_model_prompt = '<MORE_DETAILED_CAPTION>'
  problem_idxs = []
  model_time = 0
  items_processed = 0
  # we don't know if the data_list was expanded somehow during
  # processing so that the size > batch_size.
  new_batch_size = args.batch_size
  with torch.no_grad():
    #optimize this to remove whole records of idx if there is at least
    #one image that is a problem. right now we do captions for all
    #valid images, and the caption for invalid images are empty.
    images = []
    idx_pairs = [] # in the form of [(idx, i)...]
    #logger.warning ('Checking for corrupted images for batch size: ' +str(len(idx_and_images)))
    for data in data_list:
      idx = data['_tmp_idx']
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
    recaption_list = []
    for rng in range(0, len(images), new_batch_size):
      image_batch = images[rng: min(len(images), rng+new_batch_size)]
      logger.warning("Doing one batch of captioning " + str(len(image_batch)))
      inputs = caption_generator_processor(text=[caption_generator_model_prompt]*len(image_batch), images=image_batch, return_tensors="pt").to(caption_generator_model.device)
      inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
      generated_ids = caption_generator_model.generate(
          **inputs,
          max_new_tokens=args.caption_max_sequence,
          early_stopping=True,
      )
      captions = caption_generator_processor.batch_decode(generated_ids, skip_special_tokens=True)
      recaption_list.extend(captions)
      
    items_processed = len(recaption_list)
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

  #logger.warning("Creating template hash")
  # let's create templates of the original text with potentially
  # interleaved captions to replace with the new captions
  caption_template_hash, params_template_hash = create_caption_template_hash(data_list)
  for idx_pair, text in zip(idx_pairs, recaption_list):
    idx, i = idx_pair
    params_template = params_template_hash[idx] = params_template_hash.get(idx, [])
    if len(params_template) < i+1:
      params_template.extend(['']*(i+1-len(params_template)))
    set_caption(params_template, i, text)
  logger.warning("Created params_template " + str(len(params_template_hash)))
  apply_params_to_caption_text(data_list, caption_template_hash, params_template_hash)
  logger.warning("Setting the text " + str(len(data_list)))
  #torch.cuda.empty_cache()
  problem_idxs = set(problem_idxs)
  data_list = [data for data in data_list if data['_tmp_idx'] not in problem_idxs]
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
  idx2DataHash = dict([(data['_tmp_idx'], data) for data in data_list])
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
        caption_hash[(data['_tmp_idx'], i)] = caption

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
      recaption_hash[(data['_tmp_idx'], i)] = recaption
  
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
  # use the template to keep the same interleaved structure if there are any
  caption_template_hash, params_template_hash = create_caption_template_hash(data_list)  
  for idx, data in idx2DataHash.items():
    params_template = params_template_hash[data['_tmp_idx']]
    reset_all_captions(params_template, [related[0][0] for related in enumerate(data['metadata']['params']['related_caption_to_media_scores_list'])])
    data['media_caption_scores_list'] = [related[0][1] for i, related in enumerate(data['metadata']['params']['related_caption_to_media_scores_list'])]
  apply_params_to_caption_text(data_list, caption_template_hash, params_template_hash)
  
  problem_idxs = set(problem_idxs)
  data_list = [data for data in data_list if data['_tmp_idx'] not in problem_idxs]
  return data_list, model_time, items_processed, list(problem_idxs)

  
def generate_captions_then_filter_people_images(data_list):
  data_list, model_time, items_processed, problem_idxs = generate_captions_and_dont_clear_images(data_list)
  people_images = 0
  for data in data_list:
    caption = " "+data['text']+" "
    if 'people' in caption or 'person' in caption or " man" in caption or "woman" in caption or "boy" in caption or "girl" in caption:
      data['media_list'] = []
      data['media_coordinates_list'] = []
      data['media_caption_scores_list'] = []          
      data['media_types_list'] = []
      people_images += 1
  logger.warning("Not saving people images " + str(people_images))
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
# interleaved. of the form: ...  caption text caption text ..., etc.
# upsampling. just use the image generator to generate based on raw
# text, divided by sentences.  this is fixed size image generation for
# every 5 sentences. maximum of 3 captions.  TODO: - add consistency
# between captions.  - LLM upsample.
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
  idx2DataHash = dict([(data['_tmp_idx'], data) for data in data_list])  
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
  idxs = []
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
    idxs.append(data['_tmp_idx'])
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
  data_list = [data for data in data_list if data['_tmp_idx'] not in problem_idxs]
  return data_list, model_time, items_processed, list(problem_idxs)

example_array = [  
"""### Python code:
```python
def solution(A, B):
    return A - B
```
### Math word problem: 
Lin had {A} bowls of noodles. She was trying to sell noodles to feed her family If she sells {B} bowls to cusotmers. How many bowls does Lin have left?""",
"""### Python code:
```python
def solution(E, F, G, H):
    years = H * 10 # number of years
    Y = F * G * years
    people = E * Y
    max_F = 5 # 5 days a week
    max_G = 52 # 52 weeks in a year
    max_Y = max_F * max_G * years
    max_people = E * max_Y
    return people, max_people
```
### Math word problem: 
A country has one border crossing that has many migrants seeking refuge. The country lets in {E} migrants per day. If the crossing operates for {F} weekdays a week and {G} weeks per year, how many people does the country let in in {H} decades?
What is the maximum number of migrants per year assuming {E} per day for {H} decades?""",
"""### Python code:
```python
def solution(I, J, K, L):
    return (I * J * K) / (L / 100)
```
### Math word problem: 
A hospital has {I} docotors. Each doctor puts on their timesheet that she worked {J} hours per week. If the average hourly wage is ${K} and the hosptial spends {L} percent of its revenue on salaries, please compute the hospital's weekly revenue.""",
"""### Python code:
```python
def solution(D, E, F):
    X = D // E
    F = F / 100
    Y = int(D * F)
    Z = Y // E
    return (X, Y)
```
### Math word problem: 
A city has {D} police officers on duty. If {E} police officers retire once every year, how many years will before there are no more police officers?
Now assume the city will need to re-hire police officers when the number of officers is {F}% of {D}. In what year will the city need to hire a new police officer?
Tell me the answer for both questions.""",
"""### Python code:
```python
def solution(B, C, D):
    steps = "Let's solve the problem step-by-step:\n
    X = (C + D)
    steps +=f"\nFirst, let's compute how many students and teachers, let's call them `X`.\n`X = ( C + D )`.\nHere, `{X}= ({C} + {D})`"
    steps +=f"\nNext, let's compute how many of these people are in each classroom. We know from the problem that all the people in the school, and thus in each class are either student or teachers. So each classroom has X people, which is {X}."
    answer =  X * B
    steps +=f"\nNow let's compute the number of total people in the whole school, assuming there are only classrooms in the school. Here, it would be `answer = X * B`, or `{answer} = {X} * {B}`."
    steps +=f"\nThus, the total number of peoples in the whole school is {answer}."
    return answer, steps
```
### Math word problem: 
A school has B classrooms. If each classroom has C students and D teachers, and there are only students and teachers in the school, how many people are in the school? Give me both the answer, and a description showing your work step-by-step, and note any assumptions.""",
  ]

def generate_python_code(data_list):
  idx2DataHash = dict([(data['_tmp_idx'], data) for data in data_list])    
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
  for data in data_list:
    context = data['text']
    if len(context) > 200: context = context[:200]
    random.shuffle(example_array )
    examples ="\n".join(example_array)
    
    prompt = f"""You are a highly qualified expert in writing math prompts, and finding an expert in Python code for a given math problem in natural language.
You are given some example Python code and math problem pairs below as an exmple, and you will provide some creative Phython code and a creative math word problem that is solved by the Python code. 
The Python code must solve the problem exactly, must be valid and can be multi-steps. 
Be creative, and don't just include examples about apples, and oranges. Include surpising situations where you might use math.
Do not provide commentary, just provide the Python code and math word problems.

Here are some examples:
{examples}
====

Now provide me another creative Python code and corresponding math problem. Include details from the below context.
Context document:
{context}

====

In addition ot the context, the code must be multi-steps, and must include the operations: """+", ".join(random.sample(["*", "**", "//", "+", "-", "%"], 2))+""".
The word problems must include the words or concepts for "+"""+", ".join(random.sample(["compound interest", "laser", "bone", "community standards", "standard elevation"], 2))+"""."""
    prompts.append(tokenize_with_assistant_continuation([{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': "### Python code:\n```python\n"}]))
  with torch.no_grad():
    # we potenially doubled the size of the batch, so do this upsampling with the original batch_size
      python_and_word_problems = generate_with_batching(model, tokenizer, prompts, batch_size=new_batch_size)
  answers = []
  batch2 = []
  idxs2 = []
  codes = []
  word_problem_templates = []
  for data, code_and_word_problem in enumerate(data_list, python_and_word_problems):
    if "### Math" in code_and_word_problem:
      code, word_problem = code_and_word_problem.split("### Math",1)
      word_problem = word_problem.split(":",1)[-1].split("\n")[-1]
    elif "```" in code_and_word_problem.strip("```"):
      code, word_problem = code_and_word_problem.strip("```").split("```",1)
    elif "problem:" in code_and_word_problem:
      code, word_problem = code_and_word_problem.split("word_problem:",1)
      code = "\n".join(code.split("\n")[:-1])
    else:
      problem_idxs.append(data['_tmp_idx'])
      continue
    code = code.strip("`\n ")
    if code.startswith("python"):
      code = code[len("python"):].strip()
    word_problem = word_problem.strip()
    if not code.startswith("def ") or "solution(" not in code:
      problem_idxs.append(data['_tmp_idx'])
      continue
    mapping = [a.strip() for a in code.split("(",1)[1].split(")",1)[0].strip().split(",")]
    word_problem = " " + word_problem + " "
    missing_var = False
    for var in mapping:
      if "{"+var+"}" in word_problem:
        continue
      for after in ["%", "$", "ft", "cm", "lbs"]:
        if " "+var +after in word_problem:
          word_problem = word_problem.replace(" " + var + after, " "+ var+ " " + after)
          break
      for before in ["%", "$", ]:
        if before+var +" " in word_problem:
          word_problem = word_problem.replace(before+" " + var + " ", before+" "+var+" ")
          break
        if before+var +"." in word_problem:
          word_problem = word_problem.replace(before+" " + var + ".", before+" "+var+".")
          break
        if before+var +"," in word_problem:
          word_problem = word_problem.replace(before+" " + var + ",", before+" "+var+",")
          break
        if before+var +":" in word_problem:
          word_problem = word_problem.replace(before+" " + var + ":", before+" "+var+":")
          break
      if " " + var + " " in word_problem:
        word_problem = word_problem.replace(" " + var + " ", " {"+var+"} ")
        continue
      missing_var = True
      break
    if missing_var:
      problem_idxs.append(data['_tmp_idx'])
      continue
    answer_set = []
    input_set = []
    for _ in range(3):
      inputs = create_random_input(mapping)
      answer = execute_python_code(code, inputs)
      if answer is None:
        problem_idxs.append(data['_tmp_idx'])
        break
      answer_set.append(answer)
      input_set.append(inputs)
    if problem_idxs[-1] == data['_tmp_idx']:
      continue
    for answer, inputs in zip(answer_set, input_set):
      word_problem_with_inputs = word_problem.format(**inputs)
      word_problem_templates.append(word_problem)
      codes.append(code)
      answers.append(answer)
      idxs2.append(data['_tmp_idx'])
      batch2.append(tokenize_with_assistant_continuation([{'role': 'user', 'content':word_problem_with_inputs+"\nSolve this problem step-by-step."}, {'role': 'assistant', 'content': "Let's think step-by-step:\n"}]))
  with torch.no_grad():
    cot_answers = generate_with_batching(model, tokenizer, batch2, batch_size=new_batch_size)
  for code, word_problem_template, cot_answer, answer, idx, prompt in zip(cot_answers, answers, idxs, batch2):
    answer = str(answer)
    end_of_cot = cot_answer[-min(len(cot_answer), 50)]
    if " "+answer in end_of_cot or "\n"+answer in end_of_cot or "#" +answer in cot_answer:
      data = idx2Data[idx]
      data['text'] = "Q:\n"+prompt+"\nA:\n"+cot_answer
      data['metadata']['params']['num_verified'] =   data['metadata']['params'].get('num_verified', 0) + 1
      data['metadata']['params']['answer'] = answer
      data['metadata']['params']['code'] = code
      data['metadata']['params']['word_problem_template'] = word_problem_template
    
  for data in data_list:
    if data['_tmp_idx'] in problem_idxs: continue
    if data['metadata']['params']['num_verified'] < 2:
      problem_idxs.append(data['_tmp_idx'])
      
  model_time = time.time()-t0
  problem_idxs = set(problem_idxs)
  data_list = [data for data in data_list if data['_tmp_idx'] not in problem_idxs]
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
  captions = [data['media_list'][0] for data in data_list]
  instrs = [data['text'].split("### Response:",1)[0].split("### Instruction:")[-1] for data in data_list]
  revised_instrs = generate_image_aware_instruction(captions, instrs, model, tokenizer)
  responses = [data['chosen_response'] for data in data_list]
  prompts = [purpleteam_generative_tokenizer.apply_chat_template([{"role": "user", "content": instruction}, {"role": "assistant", "content": response},
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
    data['metadata']['params']['revised_instruction'] = revised_instr
    data['metadata']['params']['revised_response'] = revised_response
    data['metadata']['params']['revised_instr_response_prompt'] = prompt
                       
  problem_idxs = set(problem_idxs)
  data_list = [data for data in data_list if data['idx'] not in problem_idxs]
  return data_list, model_time, items_processed, list(problem_idxs)


def generate_autoredteam(data_list):
  model = None
  tokenizer = None
  new_batch_size = args.batch_size
  if args.use_LLM_size == 'small':
    model = LLM_small_model
    tokenizer = LLM_small_tokenizer
  elif args.use_LLM_size == 'large':
    model = LLM_large_model
    tokenizer = LLM_large_tokenizer
    new_batch_size = max(1, int(args.batch_size/2))
  else:
    model = LLM_medium_model
    tokenizer = LLM_medium_tokenizer
  blueteam_batch_size = max(5, new_batch_size)
  if "7b" in args.evaluator_model:
    blueteam_batch_size = max(5,args.batch_size//10)
  problem_idxs = []
  t0 = time.time()
  new_data_list = []
  for data in  auto_redteam(
      target_model=model, target_tokenizer=tokenizer,
      purpleteam_generative_model=model, purpleteam_generative_tokenizer=tokenizer,
      blueteam_llamaguard_model=evaluator_model, blueteam_llamaguard_tokenizer=evaluator_tokenizer,
      data_list = data_list, batch_size=new_batch_size, blueteam_batch_size=blueteam_batch_size,
  ):
    logger.warning(data)
    new_data_list.append(data)
  model_time = time.time()-t0
  # this does not account for multiple llm calls or problem items. should be really passed into auto_redteam.
  items_processed = len(new_data_list) 
  return new_data_list, model_time, items_processed, problem_idxs

### MAIN FUNCTIONS THAT CALLS THE GENERATE FUNCTIONS
# for other media, we will need to create other write functions
def write_image(image_and_path_list):
  #logger.warning("Starting to write images "+ str(len(image_and_path_list)))
  for image, image_path in image_and_path_list:
    image.save(image_path)
  #logger.warning("Finished write images "+ str(len(image_and_path_list))    )


# to prevent blocking between sub-batches, we create a yield, which
# returns a chunk as soon as available.  there is still a block
# between finishing one input file and starting another file. if this
# is significant, we can consider doing yield_batches over many files.
def yield_batches(step_size, all_data):
  global num_devices, args, node_name
  for rng in range(0, len(all_data), step_size):
    curr_data = all_data[rng: min(len(all_data), rng+step_size)]
    yield curr_data
  
def do_all_batch(base_path, output_file, pool, generate_function, all_data, step_size, all_writers, save_time=0, all_model_time=0, items_processed=0):
  # this is a map_reduce function.  this function maps one batch into
  # each GPU/process and then reduces back to a single output file.
  # optionally separate images can be written to disk as well in a
  # thread so that the writing is faster.  ASSUMES all_data is not a
  # generator, or at least a generator that doesn't produce too much
  # data since we load all_data into memory for faster processing.
  global num_devices, args, node_name
  all_data = cleanup_data_batch(all_data) # be careful to do this BEFORE setting the tmp_idx, as the cleanup routine will delete this field.
  for idx, data in enumerate(all_data):
    data['_tmp_idx'] = idx
  problem_idxs = []
  seen_subdir = {}
  with open(output_file, "w") as outfile:   
    for batch, model_time, new_items_processed, new_problem_idxs in pool.imap_unordered(generate_function, yield_batches(step_size, all_data)):
      problem_idxs.extend(new_problem_idxs)
      image_and_path = []
      all_model_time += model_time
      items_processed += new_items_processed
      t0 = time.time()
      for data in batch:
        # we need to save separately things that are not json
        # serializable. E.g., Image handling and for other media. We
        # will need to write other write functions
        if data["media_list"] and data["media_list"][0] and type(data["media_list"][0]) is not str:
          # check if PIL.Image.Image or other types, like Jpeg
          #TODO - do for other media like sound, video and images,
          idx = data['_tmp_idx']
          images = data['media_list']
          data['media_list'] = []
          #TODO, see if we have already saved this image by using SH-1 fingerprint, so we don't save to disk drive
          for image_idx, image in enumerate(images):
            if not image: continue
            width, height = image.size
            subdir = int(idx//1000) # for 1M records, there will be 1000 directories, each with 1000 files.
            if subdir not in seen_subdir:
              logger.warning("creating directory "+ f"{args.output_dir}/{base_path}/{subdir}/")
              os.makedirs(f"{args.output_dir}/{base_path}/{subdir}/", exist_ok=True)
              logger.warning("finished creating dir")
              seen_subdir[subdir] = 1
            image_and_path.append((image, f"{args.output_dir}/{base_path}/{subdir}/{idx}-{image_idx}-{width}x{height}.png"))
            #image.save(f"{args.output_dir}/{base_path}/{subdir}/{idx}-{image_idx}-{width}x{height}.png")
            data["media_list"].append(f"{base_path}/{subdir}/{idx}-{image_idx}-{width}x{height}.png")
        data["metadata"]["source"] += f"|{base_path}|{node_name}.{args.task}.{idx}"
        del data['_tmp_idx'] # these are termporary indexes
        data = cleanup_and_serialize_params(data)
        outfile.write(json.dumps(data)+"\n")
        data = None
        
      if image_and_path:
        all_writers.append(threading.Thread(target=write_image, args=(image_and_path,)))
        all_writers[-1].start()
      #cleanup the writers that are no longer alive
      for j, writer in enumerate(all_writers):
        if writer:
          if not writer.is_alive():
            writer.join()
            all_writers[j] = None
      # TODO: let's cleaup threads that are dead
      save_time += time.time()-t0
      logger.warning("saved one batch " + str(len(batch)))
      
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
      # create batches taking into account that there may be multiple items per data records
      step_size = max(1, int(args.batch_size/avg_num))
      all_writers, save_time, all_model_time, items_processed, new_problem_idxs = do_all_batch(base_path, output_file, pool, generate_function, all_data, step_size, all_writers, save_time, all_model_time, items_processed)
      problem_idxs.extend(new_problem_idxs)
            
  for writer in all_writers:
    if writer:
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
    # read data of various forms
    # potentially have a reader thread to read while we wait for GPUs to finish, if we do multiple files at the same time
    t0 = time.time()
    df = parquet.read_table(args.input_path)
    all_data = []
    for image, caption, blip_text, title, usertags, url in zip(df['jpg'], df['caption'], df['blip2_caption'], df['title'], df['usertags'], df['downloadurl']):
      all_data.append({'media_list': [image], 'text': '',\
                       'metadata': {'source': '', 'params': { 'title': title.as_py(), 'blip_text': blip_text.as_py(), 'orig_caption': caption.as_py(), 'usertags': usertags.as_py(), 'url': url.as_py()}}})
    df = None
    data_load_time += time.time() - t0
    avg_num = 1 # there is only one image.
    step_size = max(1, int(args.batch_size/avg_num))
    for idx, data in enumerate(all_data):
        data['media_list'] = [Image.open(BytesIO(image.as_py()))  for image in data['media_list']]
    all_writers, save_time, all_model_time, items_processed, new_problem_idxs = do_all_batch(base_path, args.output_path, pool, generate_function, all_data, step_size, all_writers, save_time, all_model_time, items_processed)
    problem_idxs.extend(new_problem_idxs)
  for writer in all_writers:
    if writer: writer.join()
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
  if not args.output_path:
    args.output_path = args.output_dir+"/"+args.output_suffix+".jsonl"
  base_path = args.output_path.split(args.output_dir,1)[-1].lstrip("/").replace(".jsonl", "")
  os.makedirs(args.output_dir, exist_ok=True)
  os.makedirs(args.output_dir+"/"+base_path, exist_ok=True)
  multiprocessing.set_start_method('spawn', force=True)
  with  multiprocessing.Pool(processes=num_devices) as pool:
    initialize_gpus_and_models(pool, args)
    # read data of various forms                                                                                                                                                                                                                                           
    t0 = time.time()
    all_data = [{'text': '','metadata': {'params': { "verb_type": verb_type, "obj_type": obj_type}}} for verb_type in verb_templates.keys() for obj_type in obj_templates.keys()]
    data_load_time += time.time() - t0
    step_size = args.batch_size
    all_writers, save_time, all_model_time, items_processed, new_problem_idxs = do_all_batch(base_path, args.output_path, pool, generate_function, all_data, step_size, all_writers, save_time, all_model_time, items_processed)
    problem_idxs.extend(new_problem_idxs)
  for writer in all_writers:
    if writer:
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
                 # common catalog version
                 "generate_captions_from_commoncatalog": {"models_needed": ["caption_generator", ], "main_function": main_commoncatalog, 'generate_function': generate_captions_and_clear_images},
                 "generate_stories_from_commoncatalog": {"models_needed": ["LLM_model"], "main_function": main_commoncatalog, 'generate_function': generate_stories},                                  
                 "generate_images_from_commoncatalog": {"models_needed": ["image_generator", ], "main_function": main_commoncatalog, 'generate_function': generate_images},
                 "generate_captions_then_filter_people_images": {"models_needed": ["caption_generator", "image_generator"], "main_function": main_commoncatalog, 'generate_function': generate_captions_then_filter_people_images},
                 "generate_images_then_recaption_from_commoncatalog": {"models_needed": ["box_segmenter", "image_text_scorer", "caption_generator", 'image_generator'],  "main_function": main_commoncatalog, 'generate_function': generate_images_then_recaptions},
                 "generate_captions_then_images_from_commoncatalog": {"models_needed": ["caption_generator", 'image_generator'],  "main_function": main_commoncatalog, 'generate_function': generate_captions_then_images},
                 # auto redteam version
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
  parser.add_argument("--lavaguard", type=str, default="AIML-TUDA/LlavaGuard-v1.1-7B-hf", help="Llavaguard model hf path.")  
  parser.add_argument("--image_text_score_model", type=str, default="openai/clip-vit-base-patch32", help="Model used to get the image-text cosine similarity.")
  parser.add_argument("--caption_generator_model", type=str, default='microsoft/Florence-2-large', help="Model used for generating caption of an image.") #microsoft/Florence-2-large
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
