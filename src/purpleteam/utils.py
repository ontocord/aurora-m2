import json
import spacy
import base64
import uuid
import hashlib
import random
from io import BytesIO
import numpy as np
from numpy import asarray

import torch
import PIL
from PIL import Image
from transformers import pipeline
from datasets import load_dataset
from torch.nn.functional import cosine_similarity
from transformers import CLIPProcessor, CLIPModel, AutoModel, AutoTokenizer, AutoModelWithLMHead
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from src.accelerator import accelerator

from src.frcnn.processing_image import Preprocess
from src.frcnn.modeling_frcnn import GeneralizedRCNN
from src.frcnn.utils import Config
from src.frcnn.utils import decode_image


stopwords_set = en_stopwords = {'haven', 'are', 'why', 'most', "won't", 'against', 'with', 'needn', 'couldn', 'now', 'mustn', 'who', 'under', 'doing', 'am', 'aren', 'they', "didn't", 'd', 'doesn', 'if', 'he', 'her', "haven't", 'isn', 'own', 'does', 'such', 'until', 'into', 'had', 'again', 'over', "hadn't", "you'll", 't', 'by', 'be', "wasn't", 'so', 'yours', 'both', 'any', 'did', "you've", 'these', 'myself', 'o', 'hasn', "isn't", 'you', 'other', 'shan', 'being', 'yourselves', 'was', 'no', 'm', 'those', 'will', 'its', 'itself', 'have', 'down', 'weren', 'having', 'wouldn', 'herself', "mustn't", 'very', 'do', "should've", 'him', "you'd", 'below', 'just', 'that', 'for', 'which', 'but', 'nor', 'all', 'then', 'i', 'whom', 'it', 'once', 'here', 've', "you're", 'ours', "that'll", 'a', 'won', 'himself', 'where', 'this', 'your', "hasn't", 'same', 'when', 'ourselves', 'because', "needn't", 'theirs', 'from', 'mightn', 'my', 'while', 'yourself', "she's", 'each', "doesn't", 'only', 'at', 's', 'their', "wouldn't", 'shouldn', 'and', 'themselves', 'hers', 'has', 'up', 'ma', 'in', 'll', 'we', 're', 'y', 'of', 'after', 'our', "shan't", 'before', 'wasn', 'can', 'should', 'been', 'through', 'as', 'further', 'during', 'between', 'there', 'me', 'on', 'don', "shouldn't", 'more', 'out', "don't", 'the', "weren't", "aren't", "it's", 'what', 'or', "couldn't", 'hadn', "mightn't", 'his', 'above', 'to', 'how', 'few', 'off', 'them', 'didn', 'ain', 'not', 'she', 'an', 'than', 'too', 'is', 'some', 'were', 'about'}
max_detections = 36
spacy_nlp = spacy.load('en_core_web_sm')
frcnn_config = json.load(open("src/frcnn/config.jsonl"))
frcnn_config = Config(frcnn_config)
image_preprocessor= Preprocess(frcnn_config).half().cuda()
box_segmentation_model= GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned",frcnn_config,  cache_dir="/leonardo_scratch/fast/EUHPC_E03_068/.cache").half().cuda()

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir="/leonardo_scratch/fast/EUHPC_E03_068/.cache", device_map="auto")
clip_model = accelerator.prepare(clip_model)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir="/leonardo_scratch/fast/EUHPC_E03_068/.cache")

def remove_quotes(text):
  text = text.replace("'s ", " @s@ ").replace("'ve ", " @ve@ ").replace("'m ", " @m@ ").replace("'t ", " @t@ ")
  ret_text = []
  for idx, segment in enumerate(text.split("'")):
    if idx % 2 == 0:
      ret_text.append(segment + " ")
  text = ''.join(ret_text)
  text = text.replace(" @s@ ", "'s ").replace(" @ve@ ", "'ve ").replace( " @m@ ", "'m ").replace(" @t@ ", "'t ").strip()

  text = text.replace("'s ", " @s@ ").replace("'ve ", " @ve@ ").replace("'m ", " @m@ ").replace("'t ", " @t@ ")
  ret_text = []
  for idx, segment in enumerate(text.split('"')):
    if idx % 2 == 0:
      ret_text.append(segment + " ")
  text = ''.join(ret_text)
  text = text.replace(" @s@ ", "'s ").replace(" @ve@ ", "'ve ").replace( " @m@ ", "'m ").replace(" @t@ ", "'t ").strip()
  return text

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

  
def generate_image_aware_instruction(captions, instructions, model, tokenizer):
  if random.randint(0, 1):
    instr_revision_prompts = [tokenize_with_assistant_continuation(tokenizer, [{"role": "user", "content": f"You are given the below image:\n{caption}\n===\nRevise the below instruction such that events, physical conditions, attributes, color, actions, feelings, objects, people or other information from the image are removed from the instruction, and the instruction refers to those things in the image instead. Do not refer to proper names in the instruction if those names are already in the image. Do not refer to any context document. Do not refer to the 'description' of the image. Retain the theme of the instruction. Do not repeat this instruction or the information from the image in your revised instruction. The instruction is:\n{instruction}"}, 
                                                                              {"role": "assistant", "content": "Revised Instruction:"}]) for caption, instruction in zip(captions, instructions)]
    revised_instructions = generate_with_batching(model, tokenizer, instr_revision_prompts, accelerator.device)
    revised_instructions = [
        revised_instruction.split("instruction:", 1)[-1]
        .split("Instruction:", 1)[-1]
        .split("Revised Instruction:", 1)[-1]
        .split("Revised instruction:", 1)[-1]
        .split("assistant\n", 1)[-1]
        .replace("caption", "image").strip()
        for revised_instruction in revised_instructions
        if "answer:" not in revised_instruction.lower()
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
      
    revised_instructions = generate_with_batching(model, tokenizer, prompts, accelerator.device)
    revised_instructions = [revised_instruction.split("Revised instruction:",1)[-1].strip() for revised_instruction in revised_instructions]

  if random.randint(0, 1):
    return [add_img_context_to_instruction(revised_instruction) for revised_instruction in revised_instructions] 
  else:
    return revised_instructions


def tokenize_with_assistant_continuation(tokenizer, messages):
  if not hasattr(tokenizer, "assistant_ending"):
    msg = tokenizer.apply_chat_template([{"role": "user", "content": ""}, {"role": "assistant", "content": "@@@@@@"}], tokenize=False)
    tokenizer.assistant_ending = msg.split("@@@@@@")[-1]
  return tokenizer.apply_chat_template(messages, tokenize=False)[:-len(tokenizer.assistant_ending)]

def tokenize_with_user_continuation(tokenizer, messages):
  if not hasattr(tokenizer, "user_ending"):
    msg = tokenizer.apply_chat_template([{"role": "user", "content": "@@@@@@"}], tokenize=False)
    tokenizer.user_ending = msg.split("@@@@@@")[-1]
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
   frcnn_output = decode_image(asarray(image), box_segmentation_model,  image_preprocessor, max_detections=num_boxes)
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
def get_element_to_img(matched_sentence, img, ignore_from_box=[], other_element_arr=[], get_box_images=True, num_boxes=5, box_add_factor=0.65, box_detect_verbs=True, use_longest_subsuming_text=True, score_cutoff=0.2):
  global spacy_nlp, clip_model, clip_processor,  device, box_segmentation_model, image_preprocessor
  doc = spacy_nlp(matched_sentence)
  noun_chunks = [strip_left_stopwords(e.text) for e in doc.noun_chunks if len(e.text) > 4 and e.text.lower() not in stopwords_set]
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
  text4 = [a for a in text4 if " corner" not in a and "foregr" not in a and "backgr" not in a and "photo" not in a and a != "image" and "drawing" not in a and "portrait" not in a]
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
              rec[1].append((score, img, attr))
              ent2score[element] = rec
              
        sents = []
        if box2element:
          background_element = None
          prev_small_element = None
          for element, score, coord, img, attr in box2element:
            if score >= score_cutoff and " corner" not in element and not element.endswith("ed") and not element.endswith("ing") and "foregr" not in element and "backgr" not in element:
              if attr:
                attr = attr.split(",")[0]
                if attr == "black":
                  attr = "dark colored"
                elif attr == "white":
                  attr = "light colored"
                # print (f"the {element} is also {attr}.", score)
                sents.append(f"the {element} is also {attr}.")
              if coord[0] <= 15 and coord[1] <= 15 and  coord[2] >= 85 and coord[3] >= 40:
                sents.append(f"the {element} is in the background.")
                background_element = element
                continue
              x_center = (coord[0] + (coord[2] - coord[0])/2.0)
              y_center  = (coord[1] + (coord[3] - coord[1])/2.0)
              if (coord[2] - coord[0] <= 150 or coord[3] - coord[1] <= 150) and prev_small_element:
                prev_element, prev_score, prev_coord = prev_small_element
                if x_center   - (prev_coord[0] + (prev_coord[2] - prev_coord[0])/2.0) > 100:
                  if random.randint(0,1) == 0:
                    sents.append(f"the {prev_element} is to the left of the {element}.")
                  else:
                    sents.append(f"the {element} is to the right of the {prev_element}.")
                  prev_small_element = None
                  continue
                elif x_center   - (prev_coord[0] + (prev_coord[2] - prev_coord[0])/2.0) > 25:
                  sents.append(f"the {prev_element} is beside the {element}.")
                  prev_small_element = None
                  continue
                elif y_center   - (prev_coord[1] + (prev_coord[3] - prev_coord[1])/2.0)> 25:
                  if random.randint(0,1) == 0:
                    sents.append(f"the {prev_element} is above the {element}")
                  else:
                    sents.append(f"the {element} is in front of the {prev_element}")
                  prev_small_element = None
                  continue
                elif x_center   - (prev_coord[0] + (prev_coord[2] - prev_coord[0])/2.0) <= 25 and \
                  y_center   - (prev_coord[1] + (prev_coord[3] - prev_coord[1])/2.0) <= 25:
                  if (prev_coord[2] - prev_coord[0]) < (coord[2] - coord[0]):
                    sents.append(f"the {prev_element} is on the {element}.")
                  elif (prev_coord[3] - prev_coord[1]) < (coord[3] - coord[1]):
                    sents.append(f"the {prev_element} is on the {element}.")
                  else:
                    sents.append(f"the {element} is on the {prev_element}.")
                  prev_small_element = None
                  continue

              #print (x_center, element)
              if x_center < 100:
                sents.append(f"the {element} is on the left.")
              if x_center > 400:
                sents.append(f"the {element} is on the right.")
              if (coord[2] - coord[0] <= 150 or coord[3] - coord[1] <= 150):
                prev_small_element = (element, score, coord)
        return ent2score, sents
    return {}, []

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

# generate output from a batch of inputs
def generate_with_batching(model, tokenizer, data, device,  use_cache=True, repetition_penalty=1.2, no_repeat_ngram_size=4, max_new_tokens=200, batch_size=2, **args):
  torch.cuda.empty_cache()
  output = []
  for rng in range(0, len(data), batch_size):
    d = data[rng:min(len(data), rng+batch_size)]
    if d:
      input_ids = tokenizer(d, truncation=True, padding=True, return_tensors="pt", add_special_tokens=False, ).to(device)
      prompt_len = input_ids["input_ids"].shape[-1]
      output.extend(tokenizer.batch_decode(model.generate(**input_ids,
                        use_cache=use_cache, repetition_penalty=repetition_penalty, no_repeat_ngram_size=no_repeat_ngram_size, max_new_tokens=max_new_tokens, **args)[:, prompt_len:], skip_special_tokens=True))
  torch.cuda.empty_cache()
  return output
