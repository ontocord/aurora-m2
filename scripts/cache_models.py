
import time
#from liger_kernel.transformers import AutoLigerKernelForCausalLM as AutoModelForCausalLM

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import requests

cache_dir = "/leonardo_work/EUHPC_E03_068/.cache/"
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from transformers import LlavaNextForConditionalGeneration, AutoProcessor
import torch
import json, os

# translation based stuff
import transformers

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
if False:
    tokenizer = AutoTokenizer.from_pretrained('Ray2333/GRM-Gemma2-2B-rewardmodel-ft', cache_dir=cache_dir)
    reward_model = AutoModelForSequenceClassification.from_pretrained('Ray2333/GRM-Gemma2-2B-rewardmodel-ft', cache_dir=cache_dir)

from transformers import (
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
    BertModel,
    BertTokenizerFast,
)
if False:
    m2m100_model =         M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M", cache_dir=cache_dir)
    
    m2m100_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M", cache_dir=cache_dir)
    labse_model =         BertModel.from_pretrained("sentence-transformers/LaBSE", cache_dir=cache_dir)
    labse_tokenizer = BertTokenizerFast.from_pretrained("sentence-transformers/LaBSE", cache_dir=cache_dir)

if False:
    from multilingual_clip import pt_multilingual_clip
    import transformers
if False:
    model_name = 'M-CLIP/LABSE-Vit-L-14'
    model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    model_name = "failspy/Phi-3-medium-4k-instruct-abliterated-v3"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True, cache_dir=cache_dir)


if False:
  for path in  ['ibm-granite/granite-3.0-8b-instruct', 'ibm-granite/granite-3.0-2b-instruct']: #  'microsoft/Phi-3-vision-128k-instruct', 'Qwen/Qwen2.5-Coder-1.5B-Instruct', 'Qwen/Qwen2.5-Math-1.5B-Instruct', 'Qwen/Qwen2.5-0.5B-Instruct', 'Qwen/Qwen2.5-1.5B-Instruct']:# 'Qwen/Qwen2-VL-2B-Instruct', 
    #model = AutoModelForCausalLM.from_pretrained(
    #    path,
    #    trust_remote_code=True, cache_dir=cache_dir)
    try:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, cache_dir=cache_dir)
    except:
        tokenizer = AutoProcessor.from_pretrained(path, trust_remote_code=True, cache_dir=cache_dir)

#model = LlavaForConditionalGeneration.from_pretrained('AIML-TUDA/LlavaGuard-v1.1-7B-hf', cache_dir=cache_dir)
#processor = AutoProcessor.from_pretrained('AIML-TUDA/LlavaGuard-v1.1-7B-hf', cache_dir=cache_dir)
#print (model)
#model_name = "cognitivecomputations/dolphin-2.9.2-Phi-3-Medium"
import ctranslate2
langs = ['mul']

['bg',
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
 'sv', 'vi', 'zh', 'ar', 'ru', 'hi', 'ar', 'sw', 'jap', 'ko', 'id']

for lang in langs:
    if not os.path.exists(f"{cache_dir}opus-mt-en-{lang}"):
        try:
            os.system(f"ct2-transformers-converter --model HPLT/translate-en-{lang}-v1.0-hplt_opus --output_dir {cache_dir}opus-mt-en-{lang}")
            transformers.AutoTokenizer.from_pretrained(f"HPLT/translate-en-{lang}-v1.0-hplt_opus", cache_dir=cache_dir)
        except:
            try:
                os.system(f"ct2-transformers-converter --model Helsinki-NLP/opus-mt-en-{lang} --output_dir {cache_dir}opus-mt-en-{lang}")
                transformers.AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-"+lang, cache_dir=cache_dir)
            except:
                try:
                    os.system(f"ct2-transformers-converter --model Helsinki-NLP/opus-mt-tc-big-en-{lang} --output_dir {cache_dir}opus-mt-en-{lang}")
                    transformers.AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-tc-big-en-"+lang, cache_dir=cache_dir)
                except:
                    print (f"NO MODEL FOR en-{lang}")
                    pass
    if not os.path.exists(f"{cache_dir}opus-mt-{lang}-en"):
        try:
            os.system(f"ct2-transformers-converter --model HPLT/translate-{lang}-en-v1.0-hplt_opus --output_dir {cache_dir}opus-mt-{lang}-en")
            transformers.AutoTokenizer.from_pretrained(f"HPLT/translate-{lang}-en-v1.0-hplt_opus", cache_dir=cache_dir)
        except:
            try:
                os.system(f"ct2-transformers-converter --model Helsinki-NLP/opus-mt-{lang}-en --output_dir {cache_dir}opus-mt-{lang}-en")
                transformers.AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-"+lang+"-en", cache_dir=cache_dir)            
            except:
                try:
                    os.system(f"ct2-transformers-converter --model Helsinki-NLP/opus-mt-tc-big-{lang}-en --output_dir {cache_dir}opus-mt-{lang}-en")
                    transformers.AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-tc-big-"+lang+"-en", cache_dir=cache_dir)
                except:
                    print (f"NO MODEL FOR {lang}-en")                    
                    pass


if False:
    
    model_name="nhyha/N3N_gemma-2-9b-it_20241029_1532"
    model_name = "jsgreenawalt/gemma-2-9B-it-advanced-v2.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True, cache_dir=cache_dir)

if False:
# This AutoModel wrapper class automatically monkey-patches the
# model with the optimized Liger kernels if the model is supported.
    for model_name in  [ "multimodalart/Florence-2-large-no-flash-attn", "microsoft/Florence-2-large",]:
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True) 
#transformers==4.43.4
if False: 
  for model_name in  ["Qwen/Qwen2.5-Coder-7B-Instruct", "Qwen/Qwen2.5-Math-7B-Instruct", "NousResearch/Hermes-3-Llama-3.1-8B", "Lyte/Llama-3.2-3B-Overthinker", "artificialguybr/Qwen2.5-0.5B-OpenHermes2.5", "Qwen/Qwen2.5-3B-Instruct", "multimodalart/Florence-2-large-no-flash-attn", "microsoft/Florence-2-large", "argilla/CapybaraHermes-2.5-Mistral-7B", "llamas-community/LlamaGuard-7b", "HuggingFaceH4/zephyr-7b-beta", "alpindale/Llama-Guard-3-1B", "NousResearch/Hermes-3-Llama-3.1-8B", "Lyte/Llama-3.2-3B-Overthinker",  "teknium/OpenHermes-2.5-Mistral-7B"]:
    print (model_name)
    #model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
    except:
        processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
