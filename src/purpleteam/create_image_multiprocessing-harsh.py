import json
import time
import argparse
import spacy
import glob
import itertools
import random
import numpy as np
from src.purpleteam.utils import get_element_to_img, pil_image_to_base64
from collections import deque
import torch
from torch.nn.functional import cosine_similarity
from PIL import Image
from diffusers import FluxPipeline
from transformers import pipeline
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel, AutoModel, AutoTokenizer, AutoModelWithLMHead
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from torch import multiprocessing
from src.purpleteam.utils import chatml_format_instructions, generate_with_batching, assign_uuid, tokenize_with_assistant_continuation, chunkify

from src.frcnn.visualizing_image import SingleImageViz
from src.frcnn.processing_image import Preprocess
from src.frcnn.modeling_frcnn import GeneralizedRCNN
from src.frcnn.utils import Config
from src.frcnn.utils import decode_image

import pyarrow
from pyarrow import parquet
from io import BytesIO
from PIL import Image
import os, torch


# Load necessary data
digits_to_words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
                  'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen',
                  'nineteen', 'twenty']

spacy_nlp = spacy.load('en_core_web_sm')
max_detections = 36
num_devices = torch.cuda.device_count()

flux_pipe = None
fluo_model = None
fluo_processor = None
device  = None

def setup_images_and_recaptions(args, d):
  global device, flux_pipe, fluo_model, fluo_processor
  if device is None:
    device = os.environ["CUDA_VISIBLE_DEVICES"] = "cuda:"+str(d)    
  if flux_pipe is None:
    # is there a way to pass params to the initializer??
    flux_pipe = FluxPipeline.from_pretrained(args.image_generator_model_path, torch_dtype=torch.bfloat16, cache_dir=args.cache_dir, attn_implementation="flash_attention_2").to(device) # , attn_implementation="flash_attention_2"
    #flux_pipe.model.train()
  if fluo_model is None:
    fluo_model = AutoModelForCausalLM.from_pretrained(args.caption_generator_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, cache_dir=args.cache_dir, attn_implementation="flash_attention_2").train().to(device)
    fluo_processor = AutoProcessor.from_pretrained(args.caption_generator_model_path, trust_remote_code=True, cache_dir=args.cache_dir)

  return device, flux_pipe, fluo_model, fluo_processor


def setup_captions(args, d):
  global device, flux_pipe, fluo_model, fluo_processor
  if device is None:
    device = os.environ["CUDA_VISIBLE_DEVICES"] = "cuda:"+str(d)    
  if fluo_pipe is None:
    fluo_model = AutoModelForCausalLM.from_pretrained(args.caption_generator_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, cache_dir=args.cache_dir, attn_implementation="flash_attention_2").train().to(device)
    fluo_processor = AutoProcessor.from_pretrained(args.caption_generator_model_path, trust_remote_code=True, cache_dir=args.cache_dir)

  return device, flux_pipe, fluo_model, fluo_processor



def setup_images(args, d):
  global device, flux_pipe, fluo_model, fluo_processor
  if device is None:
    device = os.environ["CUDA_VISIBLE_DEVICES"] = "cuda:"+str(d)    
  if flux_pipe is None:
    flux_pipe = FluxPipeline.from_pretrained(args.image_generator_model_path, torch_dtype=torch.bfloat16, cache_dir=args.cache_dir, attn_implementation="flash_attention_2").to(device) # , attn_implementation="flash_attention_2"
    #flux_pipe.model.train()

  return device, flux_pipe, fluo_model, fluo_processor


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

def generate_images_and_recaptions(args, d, idx_and_prompts):
    score_cutoff = args.score_cutoff
    device, flux_pipe, fluo_model, fluo_processor = setup_images_and_recaptions(args, d)
    
    # Modify the original prompt by appending adversarial suffix
    # prompt_array = [f"{prompt} {suffix}".strip() for prompt in prompt_array]

    # remove quotation text from the prompt
    # detected_and_cleaned_texts = augment_for_quotes(prompt_array)
    with torch.no_grad():
      # prompt_array = [obj[1] if len(obj[1]) > 10 else prompt for obj, prompt in zip(detected_and_cleaned_texts, prompt_array)]
      prompt_array = [prompt for idx, prompt in idx_and_prompts]
      time0 = time.time()
      # Generate image with Flux pipeline
      images1 = flux_pipe(
          prompt_array,
          guidance_scale=0.0,
          num_inference_steps=4,
          max_sequence_length=args.image_gen_caption_max_sequence,
          width=args.image_width, height=args.image_heigth,
          generator=torch.Generator(device).manual_seed(0)
      ).images
      model_time = time.time() - time0
      print(os.getpid(), flux_pipe.device, f"@TIME | Image Gen time: {model_time}")
  
      time0 = time.time()
      fluo_prompt = '<MORE_DETAILED_CAPTION>'
      # Process the image with fluorence and generate caption
      images = []
      idxs = []
      for idx_and_prompt, image in zip(idx_and_prompts, images1):
        try:
          fluo_processor(text=[fluo_prompt], images=[image], return_tensors="pt")
          images.append(image)
          idxs.append(idx_and_prompt[0])
        except:
          print (f"problem with {idx}")
          
      inputs = fluo_processor(text=[fluo_prompt]*len(images), images=images, return_tensors="pt").to(device)
      inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
      generated_ids = fluo_model.generate(
            **inputs,
            max_new_tokens=args.caption_max_sequence,
            early_stopping=True,
        )
      generated_texts = fluo_processor.batch_decode(generated_ids, skip_special_tokens=True)
      model_time2 = time.time() - time0
    print(os.getpid(), fluo_model.device, f"@TIME | Caption Generation: {model_time2} seconds")
    return list(zip(idxs, generated_texts, images)), model_time + model_time2


def generate_captions(args, d, idx_and_images):
    score_cutoff = args.score_cutoff
    device, flux_pipe, fluo_model, fluo_processor = setup_captions(args, d)
    time0 = time.time()
    fluo_prompt = '<MORE_DETAILED_CAPTION>'
    # Process the image with fluorence and generate caption
    with torch.no_grad():
      images = []
      idxs = []
      for idx, image in idx_and_images:
        try:
          fluo_processor(text=[fluo_prompt], images=[image], return_tensors="pt")
          images.append(image)
          idxs.append(idx)
        except:
          print (f"problem with {idx}")
          
      inputs = fluo_processor(text=[fluo_prompt]*len(images), images=images, return_tensors="pt").to(device)
      inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
      generated_ids = fluo_model.generate(
            **inputs,
            max_new_tokens=args.caption_max_sequence,
            early_stopping=True,
        )
      generated_texts = fluo_processor.batch_decode(generated_ids, skip_special_tokens=True)
    time1 = time.time()
    model_time = time1-time0
    print(os.getpid(), fluo_model.device, f"@TIME | Caption Generation: {model_time} seconds")
    return list(zip(idxs, generated_texts, [None]*len(idxs))), model_time


def generate_images(args, d, idx_and_prompts):
    score_cutoff = args.score_cutoff
    device, flux_pipe, fluo_model, fluo_processor = setup_images(args, d)
    
    # Modify the original prompt by appending adversarial suffix
    # prompt_array = [f"{prompt} {suffix}".strip() for prompt in prompt_array]

    # remove quotation text from the prompt
    # detected_and_cleaned_texts = augment_for_quotes(prompt_array)
    with torch.no_grad():
      # prompt_array = [obj[1] if len(obj[1]) > 10 else prompt for obj, prompt in zip(detected_and_cleaned_texts, prompt_array)]
      prompt_array = [prompt for idx, prompt in idx_and_prompts]
      time0 = time.time()
      # Generate image with Flux pipeline
      images = flux_pipe(
          prompt_array,
          guidance_scale=0.0,
          num_inference_steps=4,
          max_sequence_length=args.image_gen_caption_max_sequence,
          width=args.image_width, height=args.image_heigth,
          generator=torch.Generator(device).manual_seed(0)
      ).images
      model_time = time.time() - time0
      print(os.getpid(), flux_pipe.device, f"@TIME | Image Gen time: {model_time}")
    return [(idx_and_prompt[0], idx_and_prompt[1], image) for idx_and_prompt, image in zip(idx_and_prompts, images)] , model_time



def main():
    parser = argparse.ArgumentParser(description="Set up models with quantization and specific configurations.")
    parser.add_argument("--task", type=str, default="image", help="Task: one of generate_images, generate_images_and_recpations, generate_captions.")
    parser.add_argument("--input_dir", type=str, default="", help="Path to the input file.")    
    parser.add_argument("--output_dir", type=str, default="", help="Path to the input file.")    
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size")
    parser.add_argument("--caption_max_sequence", type=int, default=512, help="Max caption sequence")
    parser.add_argument("--image_gen_caption_max_sequence", type=int, default=512, help="Image text encoder max sequence")                                            
    parser.add_argument("--image_width", type=int, default=512, help="Image width")
    parser.add_argument("--image_heigth", type=int, default=512, help="Image heigth")        
    parser.add_argument("--score_cutoff", type=float, default=0.14, help="score cutoff")
    parser.add_argument("--cache_dir", type=str, default="", help="Path to cache directory.")
    parser.add_argument("--purpleteam_generative_model_path", type=str, default="teknium/OpenHermes-2.5-Mistral-7B", help="Purpleteam generative model hf path.")
    parser.add_argument("--cos_score_model_path", type=str, default="openai/clip-vit-base-patch32", help="Model used to get the image-text cosine similarity.")
    parser.add_argument("--caption_generator_model_path", type=str, default='microsoft/Florence-2-large', help="Model used for generating caption of an image.")
    parser.add_argument("--image_generator_model_path", type=str, default="black-forest-labs/FLUX.1-schnell")
    parser.add_argument("--output_path", type=str, default="", help="Path to save output for this step.")
    parser.add_argument("--input_path", type=str, default="", help="Path to save input for this step.")
    node_name = parser.prog.replace(".py", "")
    start = time.time()
    items_processed = 0
    args = parser.parse_args()
    global clip_processor, clip_model, fluo_model, fluo_processor
    global purpleteam_generative_tokenizer, purpleteam_generative_model
    global flux_pipe, image_preprocessor, box_segmentation_model
    base_path = args.output_path.split("/")[-1].split(".jsonl")[0]    
    multiprocessing.set_start_method('spawn', force=True)
    with  multiprocessing.Pool(processes=num_devices) as pool:    
      with open(args.output_path, "w") as outfile: 
        with open(args.input_path, "r") as infile:
          rng = 0
          all_model_time = 0
          seen = {}
          while True:
            # Read a batch of lines from the input file
            lines = list(itertools.islice(infile, args.batch_size*num_devices))
            if not lines:
                break  # Exit the loop if no lines are left

            # Apply the algo over the batched data
            
            all_data = [json.loads(l) for l in lines]
            captions = [data['caption'] for data in all_data]
            captions = list(enumerate(captions))
            captions.sort(key=lambda a: len(a[1]))
            chunks = chunkify(captions, num_devices)
            save_time = 0
            for batch, model_time in pool.starmap(generate_images, zip([args]*num_devices, range(num_devices), chunks), chunksize=1):
              all_model_time += model_time
              t0 = time.time()
              for (idx, caption, image) in batch:
                data = all_data[idx]
                if "metadata" not in data:
                  data["metadata"] = {}
                if idx+rng not in seen: 
                  image.save(f"{args.output_dir}/{base_path}-{rng+idx}.png")
                seen[idx+rng] = 1
                data["caption"] = caption
                data["images"] = [f"{args.output_dir}/{base_path}-{rng+idx}.png"]
                if 'source' not in data['metadata']:
                  data['metdata'] = ""
                data["metadata"]["source"] += f"|{node_name}.{task}.{rng+idx}.{args.input_path}"
                data["metadata"]["create_img-params"] = json.dumps(vars(args))
                outfile.write(json.dumps(data)+"\n")
                items_processed += 1
              save_time += time.time()-t0
            rng += args.batch_size*num_devices
    stop = time.time()
    print(os.getpid(), f"@TIME | {node_name}.{task} : {items_processed} in {stop - start:.2f} seconds total time on {num_devices} GPUs and processes. All model time took {all_model_time}. Disk save time took {save_time}", args)



if __name__ == "__main__":
      main()
      print("Completed!!")
