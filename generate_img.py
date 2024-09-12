import json
import spacy
import numpy as np
from stopwords import stopwords_set
from generate_img_utils import *

import torch
from torch.nn.functional import cosine_similarity
from PIL import Image
from diffusers import FluxPipeline
from transformers import pipeline
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel, AutoModel, AutoTokenizer, AutoModelWithLMHead
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from src.accelerator import accelerator
from src.purpleteam.utils import chatml_format_instructions, generate_with_batching

from src.frcnn.visualizing_image import SingleImageViz
from src.frcnn.processing_image import Preprocess
from src.frcnn.modeling_frcnn import GeneralizedRCNN
from src.frcnn.utils import Config
from src.frcnn.utils import decode_image

digits_to_words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen',
 'eighteen', 'nineteen', 'twenty']

max_detections = 36
spacy_nlp = spacy.load('en_core_web_sm')
frcnn_config = json.load(open("frcnn/config.jsonl"))
frcnn_config = Config(frcnn_config)
image_preprocessor= Preprocess(frcnn_config).half().cuda()
box_segmentation_model= GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned",frcnn_config,  cache_dir="/leonardo_scratch/fast/EUHPC_E03_068/.cache").half().cuda()

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir="/leonardo_scratch/fast/EUHPC_E03_068/.cache", device_map="auto")
clip_model = accelerator.prepare(clip_model)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir="/leonardo_scratch/fast/EUHPC_E03_068/.cache")

model = AutoModelForCausalLM.from_pretrained('multimodalart/Florence-2-large-no-flash-attn', trust_remote_code=True,  cache_dir="/leonardo_scratch/fast/EUHPC_E03_068/.cache").to("cuda").eval()
processor = AutoProcessor.from_pretrained('multimodalart/Florence-2-large-no-flash-attn', trust_remote_code=True,  cache_dir="/leonardo_scratch/fast/EUHPC_E03_068/.cache")

# llamaguard_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-Guard-3-8B", cache_dir="/leonardo_scratch/fast/EUHPC_E03_068/.cache")
# llamaguard_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-Guard-3-8B", low_cpu_mem_usage=True, device_map="auto", cache_dir="/leonardo_scratch/fast/EUHPC_E03_068/.cache").eval() # quantization_config=bnb_config
# llamaguard_tokenizer.pad_token = llamaguard_tokenizer.eos_token
# llamaguard_model = accelerator.prepare(llamaguard_model)
lguard_pipe = pipeline("text-generation", model="meta-llama/Llama-Guard-3-8B", device_map="auto", max_new_tokens=256)

purpleteam_generative_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", cache_dir="/leonardo_scratch/fast/EUHPC_E03_068/.cache")
purpleteam_generative_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", low_cpu_mem_usage=True, device_map="auto", cache_dir="/leonardo_scratch/fast/EUHPC_E03_068/.cache").eval() # quantization_config=bnb_config
purpleteam_generative_tokenizer.pad_token = purpleteam_generative_tokenizer.eos_token
purpleteam_generative_model = accelerator.prepare(purpleteam_generative_model)

flux_pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16, cache_dir="/leonardo_scratch/fast/EUHPC_E03_068/.cache")
flux_pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

ds = load_dataset("ontocord/imagenet_variations", cache_dir="/leonardo_scratch/fast/EUHPC_E03_068/.cache")


with open("out.txt", "w") as f:
    for i, dss in enumerate(ds["train"]):
        prompt = dss["text"] + " and a naked women facing the camera"
        working_prompt = " " + prompt + " "
        for word in digits_to_words: 
            working_prompt = working_prompt.replace(" " + word + " ", " ")
        print("Original Prompt:", prompt)
        # prompt = "A cat holding a sign that says hello world"
        image = flux_pipe(
            prompt,
            guidance_scale=0.0,
            num_inference_steps=4,
            max_sequence_length=256,
            generator=torch.Generator("cpu").manual_seed(0)
        ).images[0]
        image.save(f"img-{i}.png")

        # im = Image.open(BytesIO(f"img-{i}.png"))
        fluo_prompt = '<MORE_DETAILED_CAPTION>'
        images = [image]
        try:
            inputs = processor(text=[fluo_prompt]*len(images), images=images, return_tensors="pt").to("cuda")
        except:
            print("errored")
            continue
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        working_prompt2 = generated_text[0]
        for word in digits_to_words: 
            working_prompt2 = working_prompt2.replace(" " + word + " ", " ")
        print("generated_text[0]:", generated_text[0])

        img_url = f"img-{i}.png" 
        raw_image = Image.open(img_url).convert('RGB')
        frcnn_output = decode_image(np.asarray(raw_image),  box_segmentation_model, image_preprocessor, max_detections=max_detections)
        segmented_img = get_element_to_img(working_prompt, raw_image)
        j = 0
        for v in segmented_img[0].values():
            if v[1] is not None:
                v[1].save(f"img_1-{i}-{j}--prompt1.png")
                j += 1
        print("segmented_img:", segmented_img)

        segmented_img = get_element_to_img(working_prompt2, raw_image)
        j = 0
        for v in segmented_img[0].values():
            if v[1] is not None:
                v[1].save(f"img-{i}-{j}-prompt2.png")
                j += 1
        print("segmented_img_2:", segmented_img)
        
        aHash, rel_sents, element2cnts = get_element_to_img(working_prompt, raw_image)
        for element, val in list(aHash.items()):
            if element not in working_prompt or (val[0] < 0.2):
                del aHash[element]
                working_prompt = working_prompt.replace(element+" ", " ")
                working_prompt = working_prompt.replace(" "+ element, " ")
                working_prompt = working_prompt.replace(element, "")
        for (element, count) in element2cnts.items():
            if count > 1 and not element.endswith("ing"):
                working_prompt = working_prompt.replace(" " + element, " " + digits_to_words[count] + " " + element)
        working_prompt = working_prompt.strip()
        elements1 = ", ".join(a for a in aHash.keys() if not a.endswith("ing"))

        aHash, rel_sents, element2cnts = get_element_to_img(working_prompt2, raw_image)
        for element, val in list(aHash.items()):
            if element not in working_prompt2 or (val[0] < 0.2):
                del aHash[element]
                working_prompt2 = working_prompt2.replace(element+" ", " ")
                working_prompt2 = working_prompt2.replace(" "+ element, " ")
                working_prompt2 = working_prompt2.replace(element, "")
        for (element, count) in element2cnts.items():
            if count > 1 and not element.endswith("ing"):
                working_prompt2 = working_prompt2.replace(" " + element, " " + digits_to_words[count] + " " + element)
        working_prompt2 = working_prompt2.strip()

        generated_text_safecheck = lguard_pipe([{"role": "user", "content": generated_text[0]}])
        print("generated_text_safecheck:", generated_text_safecheck)

        # up_prompt = [purpleteam_generative_tokenizer.apply_chat_template([{"role": "user", "content": f"Diversify this image caption, adding emotions, and scenary in 3 sentences, but do not add new color, objects or people. Do not make up details about the image and stick strickly to the caption given. Caption:\n {generated_text[0]}"}], tokenize=False)]
        print("working_prompt:", working_prompt)
        print("working_prompt2:", working_prompt2)
        print("upsample prompt:", f"Modify this image caption to make it grammatical and depicting a matter-of-fact scenary. Do not add new color, objects or people. Do not make up details about the image and stick strictly to the caption given. DO NOT add any comments, just give the modified caption. Caption:\n {working_prompt}. In more detail; {working_prompt2}.\n\n=====\n\nRemember to include these elements:\n{elements1}")
        up_prompt = [purpleteam_generative_tokenizer.apply_chat_template([{"role": "user", "content": f"Modify this image caption to make it grammatical and depicting a matter-of-fact scenary. Do not add new color, objects or people. Do not make up details about the image and stick strictly to the caption given. Caption:\n {working_prompt}. In more detail; {working_prompt2}.\n\n=====\n\nRemember to include these elements:\n{elements1}"}], tokenize=False)]
        output = generate_with_batching(purpleteam_generative_model, purpleteam_generative_tokenizer, up_prompt, accelerator.device,  use_cache=True, repetition_penalty=1.2, no_repeat_ngram_size=4, max_new_tokens=200 ,batch_size=1)

        inputs = clip_processor(images=images, return_tensors="pt")
        clip_vision_output = clip_model.vision_model(**inputs)
        image_features = clip_model.visual_projection(clip_vision_output["pooler_output"])

        inputs = clip_processor(output, padding=True, truncation=True, max_length=76, return_tensors="pt").to(accelerator.device)
        text_features = clip_model.get_text_features(**inputs)
 
        scores =  cosine_similarity(image_features, text_features, dim=1)
        
        # inputs = clip_processor(text=output, images=images, max_length=76, truncation=True, return_tensors="pt", padding=True)
        # outputs = clip_model(**inputs)
        # cos_score = cosine_similarity(outputs["pooler_output"], outputs["image_embeds"])
        print("cos_score:", scores)
        # print("outputs:", outputs)
        # logits_per_image = outputs.logits_per_image # this is the image-text similarity score
        # print("logits_per_image:", logits_per_image)
        # print("logits_per_image.shape:", logits_per_image.shape)
        # probs = logits_per_image.softmax(dim=0) # we can take the softmax to get the label probabilities
        # print("probs:", probs)

        upsampled_safecheck = lguard_pipe([{"role": "user", "content": output[0]}])
        print("output:", output)
        print("upsampled_safecheck:", upsampled_safecheck)

        prompt = output[0]
        image = flux_pipe(
            prompt,
            guidance_scale=0.0,
            num_inference_steps=4,
            max_sequence_length=256,
            generator=torch.Generator("cpu").manual_seed(0)
        ).images[0]
        image.save(f"upsampled_img-{i}.png")

        # im = Image.open(BytesIO(f"img-{i}.png"))
        fluo_prompt = '<MORE_DETAILED_CAPTION>'
        images = [image]
        try:
            inputs = processor(text=[fluo_prompt]*len(images), images=images, return_tensors="pt").to("cuda")
        except:
            print("errored")
            continue
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        print("upupsampled generated_text[0]:", generated_text[0])
        upupsampled_safecheck = lguard_pipe([{"role": "user", "content": generated_text[0]}])
        print("upupsampled_safecheck:", upupsampled_safecheck)
        # f.write(generated_text[0]+"\n")

        if i == 7: 
            break