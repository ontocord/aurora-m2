import os
import json
import time
import argparse
import spacy
import glob
import itertools
import random
import numpy as np
from src.purpleteam.utils import get_element_to_img, pil_image_to_base64

import torch
from torch.nn.functional import cosine_similarity
from PIL import Image
from diffusers import FluxPipeline
from transformers import pipeline
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel, AutoModel, AutoTokenizer, AutoModelWithLMHead
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
# from src.accelerator import accelerator
# from accelerate.utils import gather_object
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

# Load necessary data
digits_to_words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
                  'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen',
                  'nineteen', 'twenty']

max_detections = 36
num_devices = torch.cuda.device_count()

device, spacy_nlp = "cuda", None
clip_model, clip_processor = None, None
fluo_model, fluo_processor = None, None
purpleteam_generative_model, purpleteam_generative_tokenizer = None, None
box_segmentation_model, image_preprocessor = None, None

def setup(args):
    global spacy_nlp, image_preprocessor, box_segmentation_model, clip_processor, clip_model, fluo_model, fluo_processor, purpleteam_generative_tokenizer, purpleteam_generative_model

    if spacy_nlp is None:
      spacy_nlp = spacy.load('en_core_web_sm')   
    
    # if device is None:
    #   device = os.environ["CUDA_VISIBLE_DEVICES"] = "cuda:"+str(d)
    #   print (os.getpid(), 'SETTING ', device)    
            
    if not clip_model:
      clip_model = CLIPModel.from_pretrained(args.cos_score_model_path, cache_dir=args.cache_dir).eval().to(device)
      clip_processor = CLIPProcessor.from_pretrained(args.cos_score_model_path, cache_dir=args.cache_dir)

    if not fluo_model:
      fluo_model = AutoModelForCausalLM.from_pretrained(args.caption_generator_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", cache_dir=args.cache_dir).eval().to(device)
      fluo_processor = AutoProcessor.from_pretrained(args.caption_generator_model_path, trust_remote_code=True, cache_dir=args.cache_dir)
      # fluo_model = fluo_model.to(accelerator.device)

    if not purpleteam_generative_model:
      purpleteam_generative_tokenizer = AutoTokenizer.from_pretrained(args.purpleteam_generative_model_path, cache_dir=args.cache_dir)
      purpleteam_generative_model = AutoModelForCausalLM.from_pretrained(args.purpleteam_generative_model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", cache_dir=args.cache_dir).eval().to(device)
      purpleteam_generative_tokenizer.pad_token = purpleteam_generative_tokenizer.eos_token
      # purpleteam_generative_model = purpleteam_generative_model.to(accelerator.device)

    if not box_segmentation_model:
      frcnn_config = json.load(open("src/frcnn/config.jsonl"))
      frcnn_config = Config(frcnn_config)
      image_preprocessor= Preprocess(frcnn_config).half().cuda()
      box_segmentation_model= GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", frcnn_config, cache_dir="/leonardo_scratch/fast/EUHPC_E03_068/.cache").half().eval().to(device)
      # box_segmentation_model = box_segmentation_model.to(accelerator.device)

    return image_preprocessor, box_segmentation_model, clip_processor, clip_model, fluo_model, fluo_processor, purpleteam_generative_tokenizer, purpleteam_generative_model


def cosim_eval(images, texts, device):
    # evaluate the generated text by comparing its similarity with flux generated image 
    inputs = clip_processor(images=images, return_tensors="pt").to(device)
    clip_vision_output = clip_model.vision_model(**inputs)
    image_features = clip_model.visual_projection(clip_vision_output["pooler_output"])

    inputs = clip_processor(texts, padding=True, truncation=True, max_length=76, return_tensors="pt").to(device)
    text_features = clip_model.get_text_features(**inputs)
    cos_scores = cosine_similarity(image_features, text_features, dim=1)

    return cos_scores

def generate_captions(args, idx_and_images):
    batch_size, score_cutoff = args.batch_size, args.score_cutoff
    
    image_preprocessor, box_segmentation_model, clip_processor, clip_model, fluo_model, fluo_processor, purpleteam_generative_tokenizer, purpleteam_generative_model = setup(args)

    time0 = time.time()
    images = [image for idx, image in idx_and_images]
    idxs = [idx for idx, image in idx_and_images]
    
    fluo_prompt = '<MORE_DETAILED_CAPTION>'
    # Process the image with fluorence and generate caption
    with torch.no_grad():
        inputs = fluo_processor(text=[fluo_prompt]*len(images), images=images, return_tensors="pt").to(device)
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        generated_ids = fluo_model.generate(
            **inputs,
            max_new_tokens=1024,
            early_stopping=False,
            temperature=0.85,
        )
        generated_texts = fluo_processor.batch_decode(generated_ids, skip_special_tokens=True)
    time1 = time.time()
    print(os.getpid(), f"@TIME | 'create_caption_from_img' | Caption Generation: {time1 - time0:.2f} seconds")

    #create working batches
    return_text = []
    images_idxs = []

    # save away a reference of the image->various text  
    for generated_text, image_idx in zip(generated_texts, range(len(images))):
      return_text.append(generated_text)
      images_idxs.append(image_idx)

    # Remove digits as words
    _working_prompt = []
    for prompt in generated_texts:
      prompt = " "+ prompt +" "
      for word in digits_to_words: 
          prompt = prompt.replace(" " + word + " ", " ")
      _working_prompt.append(prompt)
    

    working_prompt = []
    elements = []
    elements_ = []
    for prompt, image in zip(_working_prompt, images):
      # for working_prompt
      aHash, rel_sents = get_element_to_img(prompt, image, box_segmentation_model,\
                                            image_preprocessor, clip_processor, clip_model, score_cutoff=score_cutoff)
      for element, val in list(aHash.items()):
          # if we don't detect an actual image but clip thinks there is the element SOMEWHERE in the picture, then we want a higher cutoff
          if element not in prompt or ((val[1] and val[0] < score_cutoff) or (not val[1] and val[0] < score_cutoff + 0.05)):
              del aHash[element]
              prompt = prompt.replace(element+" ", " ")
              prompt = prompt.replace(" "+ element, " ")
              prompt = prompt.replace(element, "")
      for element, val in list(aHash.items()):
          if not val[1]: continue
          all_detected_imgs = val[1]
          count = len([a for a in all_detected_imgs if a[0] >= score_cutoff])
          if count > 1 and not element.endswith("ing"):
              prompt = prompt.replace(" " + element, " " + digits_to_words[count] + " " + element)
      prompt = prompt.strip()
      working_prompt.append(prompt)
      elements.append(", ".join(a for a in aHash.keys() if not a.endswith("ing")))
      elements_.append(elements[-1] + " " + " ".join(rel_sents))

    # upsample the caption and correct the count of elements
    up_prompt = []
    prefix = random.choice(["an image of", "a photo of", "a photograph of", "a picture of", "a screenshot of", "a screen shot of"])
    for prompt, e1, e2 , image_idx in zip(working_prompt, elements, elements_, range(len(images))):
      e1 = e1.strip().replace("  ", " ")
      e2 = e2.strip().replace("  ", " ")
      up_prompt.append(tokenize_with_assistant_continuation(purpleteam_generative_tokenizer, [{"role": "user", "content": f"Modify this image caption to make it grammatical and depicting a matter-of-fact scenary. Do not add new color, objects or people. Do not make up details about the image and stick strictly to the caption given. DO NOT add any comments, just give the modified caption. Caption:\n {prompt}.\n\n=====\n\nRemember to include these elements:\n{e1}"},
                                                                                              {"role": "assistant", "content": f"Modified Caption: {prefix}"}]))
      images_idxs.append(image_idx)
      if e1 != e2:
        up_prompt.append(tokenize_with_assistant_continuation(purpleteam_generative_tokenizer, [{"role": "user", "content": f"Modify this image caption to make it grammatical and depicting a matter-of-fact scenary. Do not add new color, objects or people. Do not make up details about the image and stick strictly to the caption given. DO NOT add any comments, just give the modified caption. Caption:\n {prompt}.\n\n=====\n\nRemember to include these elements:\n{e2}"}, 
                                                                                                {"role": "assistant", "content": f"Modified Caption: {prefix}"}]))
        images_idxs.append(image_idx)
    
    time2 = time.time()
    print(os.getpid(), f"@TIME | 'create_caption_from_img' | Caption Upsampling Prep: {time2 - time1:.2f} seconds")
    outputs = generate_with_batching(purpleteam_generative_model, purpleteam_generative_tokenizer, up_prompt, device,  use_cache=True, repetition_penalty=1.2, no_repeat_ngram_size=4, max_new_tokens=400 ,batch_size=batch_size)
    outputs = [o.split("Modified Caption:",1)[-1] for o in outputs]
    outputs = [o.replace("Caption:", "").replace("caption:", "").replace("Modified Caption:", "").replace("Modified caption:", "").replace("modified caption:", "").strip() for o in outputs]
    return_text.extend(outputs)
    
    time3 = time.time()
    print(os.getpid(), f"@TIME | 'create_caption_from_img' | Upsampled Caption Generation: {time3 - time2:.2f} seconds")
    
    # evaluate the generated text by comparing its similarity with flux generated image 
    ret = []
    cosine_batch = {}
    for image_idx, text in zip(images_idxs, return_text):
      cosine_batch[image_idx] = cosine_batch.get(image_idx, [])+ [text]
      
    for image_idx, texts in cosine_batch.items():
      cos_scores = cosim_eval([images[image_idx]], texts, device)
      ret.extend([(idxs[image_idx], text, score.item(), list(zip(texts, [ss.item() for ss in cos_scores]))) for text, score in zip(texts, cos_scores)])
    time4 = time.time()
    print(os.getpid(), f"@TIME | 'create_caption_from_img' | Cosine Scoring: {time4 - time3:.2f} seconds")
        
    return ret


def main():
    parser = argparse.ArgumentParser(description="Set up models with quantization and specific configurations.")
    parser.add_argument("--input_dir", type=str, default="", help="Path to the input file.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--score_cutoff", type=float, default=0.14, help="score cutoff")
    parser.add_argument("--cache_dir", type=str, default="", help="Path to cache directory.")
    parser.add_argument("--purpleteam_generative_model_path", type=str, default="teknium/OpenHermes-2.5-Mistral-7B", help="Purpleteam generative model hf path.")
    parser.add_argument("--cos_score_model_path", type=str, default="openai/clip-vit-base-patch32", help="Model used to get the image-text cosine similarity.")
    parser.add_argument("--caption_generator_model_path", type=str, default='multimodalart/Florence-2-large-no-flash-attn', help="Model used for generating caption of an image.")
    parser.add_argument("--output_path", type=str, default="", help="Path to save output for this step.")

    args = parser.parse_args()

    start = time.time()
    items_processed = 0
        
    # multiprocessing.set_start_method('spawn', force=True)
    # with  multiprocessing.Pool(processes=num_devices) as pool:  
    with open(args.output_path, "w") as outfile: 
      for file in glob.glob(args.input_dir.rstrip("/")+"/*/*/*/*"):
        df = parquet.read_table(file)
        idx = 0
        all_data = []
        for image, caption, blip_text, title, usertags, url in zip(df['jpg'], df['caption'], df['blip2_caption'], df['title'], df['usertags'], df['downloadurl']):
          all_data.append({'image': image, 'orig_caption': caption.as_py(), 'blip2_text': blip_text.as_py(), 'title': title.as_py(), 'usertags': usertags.as_py(), 'url': url.as_py(), 'source': file})
        image_array = [Image.open(BytesIO(data['image'].as_py())) for data in all_data]

        for rng in range(0, len(image_array), args.batch_size*num_devices):
          images = image_array[rng:min(len(image_array), rng+args.batch_size)]
          
          # img_chunks = chunkify(list(enumerate(images)), num_devices)
          
          # collator = []
          # for out_batch in pool.starmap(generate_captions, zip([args]*num_devices, img_chunks, [id for id in range(num_devices)]), chunksize=1):
          out_batch = generate_captions(args, list(enumerate(images)))
          for (idx, text, score, related) in out_batch:
            idx += rng
            metadata = all_data[idx]
            data = {'caption': text, 'metadata': metadata}
            if "image" in data["metadata"]:
              del data["metadata"]["image"]
            data["metadata"]["create_caption_from_img-params"] = json.dumps(vars(args))
            outfile.write(json.dumps(data)+"\n")
            items_processed += 1
          if rng >= 20*args.batch_size: break # for testing ONLY
    # stop = time.time()
    # print(os.getpid(), f"@TIME | 'create_caption_from_img' {items_processed} in {stop - start:.2f} seconds")

            # with accelerator.split_between_processes(list(enumerate(images))) as images_split:
            #   _tmp = generate_captions(images_split, batch_size=args.batch_size, score_cutoff=args.score_cutoff)
            #   collator.extend(_tmp)
            
            # # collect results from all the GPUs
            # tmp=gather_object(collator)

            # if accelerator.is_main_process:
            #     timediff=time.time()-start
            #     print(f"@TIME elapsed: {timediff}")

            # # add batch_id to idx
            # for idx, tmpp in enumerate(tmp):
            #   tmp[idx] = (rng + tmpp[0],) + tmpp[1:]
            # idx_text_score_related = tmp
            # for (idx, text, score, related) in idx_text_score_related:
            #   metadata = all_data[idx]
            #   data = {'caption': text, 'metadata': metadata}
            #   data["metadata"]["caption_media_score"] = score
            #   data["metadata"]["related"] = related
            #   if "image" in data["metadata"]:
            #     del data["metadata"]["image"]
            #   else:
            #     print("'image' key not in data['metadata']")
            #   data["metadata"]["create_caption_from_img-params"] = json.dumps(vars(args))
              
            #   outfile.write(json.dumps(data)+"\n")
            # if rng >= 20*args.batch_size: break


if __name__ == "__main__":
    main()
    print("Completed!!")
