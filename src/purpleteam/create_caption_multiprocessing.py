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

fluo_model = None
fluo_processor = None
device  = None

def setup(args, d):
  global device, fluo_model, fluo_processor
  if fluo_model is None:
    # is there a way to pass params to the initializer??
    device = os.environ["CUDA_VISIBLE_DEVICES"] = "cuda:"+str(d)
    fluo_model = AutoModelForCausalLM.from_pretrained(args.caption_generator_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, cache_dir=args.cache_dir).eval().to(device)
    fluo_processor = AutoProcessor.from_pretrained(args.caption_generator_model_path, trust_remote_code=True, cache_dir=args.cache_dir)

  return device, fluo_model, fluo_processor


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
  
def generate_captions(args, d, idx_and_images):
    score_cutoff = args.score_cutoff
    device, fluo_model, fluo_processor = setup(args, d)
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
            max_new_tokens=1024,
            early_stopping=True,
        )
      generated_texts = fluo_processor.batch_decode(generated_ids, skip_special_tokens=True)
    time1 = time.time()
    print(os.getpid(), f"@TIME | 'create_caption' | Caption Generation: {time1 - time0:.2f} seconds")
    return list(zip(idxs, generated_texts))



def main():
    parser = argparse.ArgumentParser(description="Set up models with quantization and specific configurations.")
    parser.add_argument("--input_dir", type=str, default="", help="Path to the input file.")
    parser.add_argument("--batch_size", type=int, default=70, help="Batch size")
    parser.add_argument("--score_cutoff", type=float, default=0.14, help="score cutoff")
    parser.add_argument("--cache_dir", type=str, default="", help="Path to cache directory.")
    parser.add_argument("--purpleteam_generative_model_path", type=str, default="teknium/OpenHermes-2.5-Mistral-7B", help="Purpleteam generative model hf path.")
    parser.add_argument("--cos_score_model_path", type=str, default="openai/clip-vit-base-patch32", help="Model used to get the image-text cosine similarity.")
    parser.add_argument("--caption_generator_model_path", type=str, default='multimodalart/Florence-2-large-no-flash-attn', help="Model used for generating caption of an image.")
    parser.add_argument("--output_path", type=str, default="", help="Path to save output for this step.")
    start = time.time()
    items_processed = 0
    args = parser.parse_args()
    global clip_processor, clip_model, fluo_model, fluo_processor
    global purpleteam_generative_tokenizer, purpleteam_generative_model
    global flux_pipe, image_preprocessor, box_segmentation_model
    multiprocessing.set_start_method('spawn', force=True)
    with  multiprocessing.Pool(processes=num_devices) as pool:    
      with open(args.output_path, "w") as outfile: 
        for file in glob.glob(args.input_dir.rstrip("/")+"/*/*/*/*"):
          seen = {}
          df = parquet.read_table(file)
          all_data = []
          for image, caption, blip_text, title, usertags, url in zip(df['jpg'], df['caption'], df['blip2_caption'], df['title'], df['usertags'], df['downloadurl']):
            all_data.append({'image': image, 'orig_caption': caption.as_py(), 'blip2_text': blip_text.as_py(), 'title': title.as_py(), 'usertags': usertags.as_py(), 'url': url.as_py(), 'source': file})
          image_array = [Image.open(BytesIO(data['image'].as_py())) for data in all_data]

          for rng in range(0, len(image_array), args.batch_size*num_devices):
            images = image_array[rng:min(len(image_array), rng+(args.batch_size*num_devices))]
                  
            chunks = chunkify(list(enumerate(images)), num_devices)
            for batch in pool.starmap(generate_captions, zip([args]*num_devices, range(num_devices), chunks), chunksize=1):
              for idx, text in batch:
                idx += rng
                metadata = all_data[idx]
                data = {'caption': text, 'metadata': metadata}
                if "image" in data["metadata"]:
                  del data["metadata"]["image"]
                data["metadata"]["create_caption-params"] = json.dumps(vars(args))
                outfile.write(json.dumps(data)+"\n")
                items_processed += 1
    stop = time.time()
    print(os.getpid(), f"@TIME | 'create_caption' | Caption Generation: {items_processed} in {stop - start:.2f} seconds")



if __name__ == "__main__":
      main()
      print("Completed!!")
