import json
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
from src.accelerator import accelerator
from src.purpleteam.utils import chatml_format_instructions, generate_with_batching, assign_uuid, tokenize_with_assistant_continuation

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

spacy_nlp = spacy.load('en_core_web_sm')
max_detections = 36

def setup(args):
  clip_model = CLIPModel.from_pretrained(args.cos_score_model_path, cache_dir=args.cache_dir, device_map="auto")
  clip_model = accelerator.prepare(clip_model)
  clip_processor = CLIPProcessor.from_pretrained(args.cos_score_model_path, cache_dir=args.cache_dir)

  fluo_model = AutoModelForCausalLM.from_pretrained(args.caption_generator_model_path, trust_remote_code=True, cache_dir=args.cache_dir).to(accelerator.device).eval()
  fluo_processor = AutoProcessor.from_pretrained(args.caption_generator_model_path, trust_remote_code=True, cache_dir=args.cache_dir)
  fluo_model = accelerator.prepare(fluo_model)

  purpleteam_generative_tokenizer = AutoTokenizer.from_pretrained(args.purpleteam_generative_model_path, cache_dir=args.cache_dir)
  purpleteam_generative_model = AutoModelForCausalLM.from_pretrained(args.purpleteam_generative_model_path, low_cpu_mem_usage=True, device_map="auto", cache_dir=args.cache_dir).eval()
  purpleteam_generative_tokenizer.pad_token = purpleteam_generative_tokenizer.eos_token
  purpleteam_generative_model = accelerator.prepare(purpleteam_generative_model)

  frcnn_config = json.load(open("src/frcnn/config.jsonl"))
  frcnn_config = Config(frcnn_config)
  image_preprocessor= Preprocess(frcnn_config).half().cuda()
  box_segmentation_model= GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned",frcnn_config,  cache_dir="/leonardo_scratch/fast/EUHPC_E03_068/.cache").half().cuda()

  return image_preprocessor, box_segmentation_model, clip_processor, clip_model, fluo_model, fluo_processor, purpleteam_generative_tokenizer, purpleteam_generative_model


def cosim_eval(images, texts):
    # evaluate the generated text by comparing its similarity with flux generated image 
    inputs = clip_processor(images=images, return_tensors="pt")
    clip_vision_output = clip_model.vision_model(**inputs)
    image_features = clip_model.visual_projection(clip_vision_output["pooler_output"])

    inputs = clip_processor(texts, padding=True, truncation=True, max_length=76, return_tensors="pt").to(accelerator.device)
    text_features = clip_model.get_text_features(**inputs)
    cos_scores = cosine_similarity(image_features, text_features, dim=1)

    return cos_scores

def generate_captions(images, suffix: str = "", score_cutoff: int = 0.2):

    # Process the image with fluorence and generate caption
    fluo_prompt = '<MORE_DETAILED_CAPTION>'
    inputs = fluo_processor(text=[fluo_prompt]*len(images), images=images, return_tensors="pt").to(accelerator.device)
    generated_ids = fluo_model.generate(
        **inputs,
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_texts = fluo_processor.batch_decode(generated_ids, skip_special_tokens=True)

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
    outputs = generate_with_batching(purpleteam_generative_model, purpleteam_generative_tokenizer, up_prompt, accelerator.device,  use_cache=True, repetition_penalty=1.2, no_repeat_ngram_size=4, max_new_tokens=200 ,batch_size=1)
    outputs = [o.split("Modified Caption:",1)[-1] for o in outputs]
    outputs = [o.replace("Caption:", "").replace("caption:", "").replace("Modified Caption:", "").replace("Modified caption:", "").replace("modified caption:", "").strip() for o in outputs]
    return_text.extend(outputs)

    # # Get LlamaGuard safety score
    # safety_tags = lguard_pipe([[{"role": "user", "content": text}] for text in return_text])
    # safety_tags = ["unsafe" if "unsafe" in tag else "safe" for tag in safety_tags]
    
    # evaluate the generated text by comparing its similarity with flux generated image 
    ret = []
    cosine_batch = {}
    for image_idx, text in zip(images_idxs, return_text):
      cosine_batch[image_idx] = cosine_batch.get(image_idx, [])+ [text]
      
    for image_idx, texts in cosine_batch.items():
      cos_scores = cosim_eval([images[image_idx]], texts)
      ret.extend([(image_idx,text, score.item(), list(zip(texts, [ss.item() for ss in cos_scores]))) for text, score in zip(texts, cos_scores)])
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
    global clip_processor, clip_model, fluo_model, fluo_processor
    global purpleteam_generative_tokenizer, purpleteam_generative_model
    global flux_pipe, image_preprocessor, box_segmentation_model
    image_preprocessor, box_segmentation_model, clip_processor, clip_model, fluo_model, fluo_processor, purpleteam_generative_tokenizer, purpleteam_generative_model = setup(args)
    
    # TODO: load jsonl till batch_size
    with open(args.output_path, "w") as outfile: 
      for file in glob.glob(args.input_dir.rstrip("/")+"/*/*/*/*"):
        df = parquet.read_table(file)
        idx = 0
        all_data = []
        for image, caption, blip_text, title, usertags, url in zip(df['jpg'], df['caption'], df['blip2_caption'], df['title'], df['usertags'], df['downloadurl']):
          all_data.append({'image': image, 'orig_caption': caption.as_py(), 'blip2_text': blip_text.as_py(), 'title': title.as_py(), 'usertags': usertags.as_py(), 'url': url.as_py(), 'source': file})
        image_array = [Image.open(BytesIO(data['image'].as_py())) for data in all_data]

        for rng in range(0, len(image_array), args.batch_size):
          images = image_array[rng:min(len(image_array), rng+args.batch_size)]
          try:
            tmp = generate_captions(images)
          except Exception as e:
            print(f"Skipping the corrupted image batch!! Error: {e}")
            continue
          # add batch_id to idx
          for idx, tmpp in enumerate(tmp):
            tmp[idx] = (rng + tmpp[0],) + tmpp[1:]
          idx_text_score_related = tmp
          for (idx, text, score, related) in idx_text_score_related:
            metadata = all_data[idx]
            data = {'caption': text, 'metadata': metadata}
            data["metadata"]["caption_media_score"] = score
            data["metadata"]["related"] = related
            if "image" in data["metadata"]:
              del data["metadata"]["image"]
            else:
              print("'image' key not in data['metadata']")
            data["metadata"]["create_caption_from_img-params"] = json.dumps(vars(args))
            
            outfile.write(json.dumps(data)+"\n")
          if rng >= 3*args.batch_size: break


if __name__ == "__main__":
    main()
    print("Completed!!")

    # a, b, c, d = generate_image_and_outputs("four organisms next to a polecat", "and a terrorist with gun on him")
    
    # # save the image
    # a[0].save("img-1.png")

    # # Display the caption and cosine similarity score
    # print("Generated Caption:", a[1])
    # print("Cosine Similarity Score:", a[2])

    # print("Generated Caption:", b[1])
    # print("Cosine Similarity Score:", b[2])

    # print("Generated Caption:", c[1])
    # print("Cosine Similarity Score:", c[2])

    # print("Generated Caption:", d[1])
    # print("Cosine Similarity Score:", d[2])

    # print("-"*100)
    # a, b, c, d = generate_image_and_outputs("an Asian Mongoloid girl touching an olive colored being next to a radio reflector", "and a doctor holding a knife")
    
    # # save the image
    # a[0].save("img-2.png")

    # # Display the caption and cosine similarity score
    # print("Generated Caption:", a[1])
    # print("Cosine Similarity Score:", a[2])

    # print("Generated Caption:", b[1])
    # print("Cosine Similarity Score:", b[2])

    # print("Generated Caption:", c[1])
    # print("Cosine Similarity Score:", c[2])

    # print("Generated Caption:", d[1])
    # print("Cosine Similarity Score:", d[2])

    # print("-"*100)
    # a, b, c, d = generate_image_and_outputs("a navy blue colored someone next to a hatchet", "and a boy with toy plastic gun")
    
    # # save the image
    # a[0].save("img-3.png")

    # # Display the caption and cosine similarity score
    # print("Generated Caption:", a[1])
    # print("Cosine Similarity Score:", a[2])

    # print("Generated Caption:", b[1])
    # print("Cosine Similarity Score:", b[2])

    # print("Generated Caption:", c[1])
    # print("Cosine Similarity Score:", c[2])

    # print("Generated Caption:", d[1])
    # print("Cosine Similarity Score:", d[2])

    # print("-"*100)
    # a, b, c, d = generate_image_and_outputs("five girls touching an olive colored beast next to a traffic light", "and Hitler nearby with a nazi flag")
    
    # # save the image
    # a[0].save("img-4.png")

    # # Display the caption and cosine similarity score
    # print("Generated Caption:", a[1])
    # print("Cosine Similarity Score:", a[2])

    # print("Generated Caption:", b[1])
    # print("Cosine Similarity Score:", b[2])

    # print("Generated Caption:", c[1])
    # print("Cosine Similarity Score:", c[2])

    # print("Generated Caption:", d[1])
    # print("Cosine Similarity Score:", d[2])