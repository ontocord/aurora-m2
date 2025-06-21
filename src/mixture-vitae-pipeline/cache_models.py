import time
#from liger_kernel.transformers import AutoLigerKernelForCausalLM as AutoModelForCausalLM

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
#from PIL import Image
import requests

cache_dir = "./cache/"
import numpy as np
import torch
#import torchvision.transforms as T
#from decord import VideoReader, cpu
#from PIL import Image
#from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import ctranslate2

from transformers import LlavaNextForConditionalGeneration, AutoProcessor
import torch
import json, os


import transformers

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSpeechSeq2Seq, WhisperFeatureExtractor


if False:
    #model_name = "ontocord/starcoder2-8b-ls-autoredteam"
    model_name = "ontocord/wide_3b-stage1_shuf_sample1_jsonl-pretrained"
    model_name = "microsoft/Phi-4-mini-instruct"
    model_name = "nvidia/AceMath-1.5B-Instruct"
    model_name = "MaziyarPanahi/calme-2.1-phi3.5-4b"
    model_name = "Qwen/Qwen3-1.7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
    model_name = "Qwen/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
    
    model_name = "Qwen/Qwen3-14B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
    
    model_name = "Qwen/Qwen3-32B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
    
if False:
    model_name = "huihui-ai/Phi-4-mini-instruct-abliterated"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
    
if False: # True:
    model_name = "sometimesanotion/Lamarck-14B-v0.6" # huihui-ai/Qwen2.5-14B-Instruct-abliterated-v2" # "huihui-ai/Qwen2.5-1.5B-Instruct-abliterated" # "huihui-ai/Qwen2.5-7B-Instruct-abliterated-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
if False:
#cahya/whisper-tiny-audio-captioning-v1.0    
    AutoProcessor.from_pretrained("cahya/whisper-base-audio-captioning-v1.0", cache_dir=cache_dir)
    AutoModelForSpeechSeq2Seq.from_pretrained("cahya/whisper-base-audio-captioning-v1.0", cache_dir=cache_dir)
    WhisperFeatureExtractor.from_pretrained("cahya/whisper-base-audio-captioning-v1.0", cache_dir=cache_dir)
    
if False:
    AutoProcessor.from_pretrained("MU-NLPC/whisper-tiny-audio-captioning", cache_dir=cache_dir)
    AutoModelForSpeechSeq2Seq.from_pretrained("MU-NLPC/whisper-tiny-audio-captioning", cache_dir=cache_dir)
    WhisperFeatureExtractor.from_pretrained("MU-NLPC/whisper-tiny-audio-captioning", cache_dir=cache_dir)    

if False:
    AutoProcessor.from_pretrained("openai/whisper-large-v3-turbo", cache_dir=cache_dir)
    AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v3-turbo", cache_dir=cache_dir)

if False:
    tokenizer = AutoTokenizer.from_pretrained('Ray2333/GRM-Gemma2-2B-rewardmodel-ft', cache_dir=cache_dir)
    reward_model = AutoModelForSequenceClassification.from_pretrained('Ray2333/GRM-Gemma2-2B-rewardmodel-ft', cache_dir=cache_dir)

from transformers import (
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
    BertModel,
    BertTokenizerFast,
)
import glob
files = list(glob.glob("../../.cache"))
if False:
    import torch
    from diffusers import FluxPipeline

    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", cache_dir=cache_dir)

if False: # True:
    ## Clap
    from transformers import ClapModel, ClapProcessor
    clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused", cache_dir=cache_dir)
    clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused", cache_dir=cache_dir)
  
if False:
    for model_name in  ["zelk12/MT-Merge4-gemma-2-9B", "sometimesanotion/Qwen2.5-14B-Vimarckoso-v3", ]: # "teknium/OpenHermes-2.5-Mistral-7B", 
        print (model_name.replace("/", "--") in files)
        #if  (model_name.replace("/", "--") in files): continue
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
        except:
            processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
            
if False:
    for model_name in  ["microsoft/Florence-2-large"]: #multimodalart/Florence-2-large-no-flash-attn
        print (model_name.replace("/", "--") in files)
        #if  (model_name.replace("/", "--") in files): continue
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
        except:
            processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)

if False:
    for model_name in  [  "Qwen/Qwen2.5-3B-Instruct", ]: #  "llamas-community/LlamaGuard-7b", "alpindale/Llama-Guard-3-1B",  "teknium/OpenHermes-2.5-Mistral-7B",'Qwen/Qwen2.5-Coder-1.5B-Instruct', 'Qwen/Qwen2.5-Math-1.5B-Instruct', 'ministral/Ministral-3b-instruct', 'Qwen/Qwen2.5-1.5B-Instruct', 'Qwen/Qwen2.5-0.5B-Instruct', "Qwen/Qwen2.5-Coder-7B-Instruct", "Qwen/Qwen2.5-Math-7B-Instruct",
        print (model_name.replace("/", "--") in files)
        #if  (model_name.replace("/", "--") in files): continue
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
        except:
            processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)

    m2m100_model =         M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M", cache_dir=cache_dir)
    
    m2m100_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M", cache_dir=cache_dir)
    labse_model =         BertModel.from_pretrained("sentence-transformers/LaBSE", cache_dir=cache_dir)
    labse_tokenizer = BertTokenizerFast.from_pretrained("sentence-transformers/LaBSE", cache_dir=cache_dir)
    model_name = 'M-CLIP/LABSE-Vit-L-14'
    model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

if False:
    from transformers import AutoTokenizer, AutoModelForMaskedLM
        
    model = LlavaForConditionalGeneration.from_pretrained('AIML-TUDA/LlavaGuard-v1.1-7B-hf', cache_dir=cache_dir)
    processor = AutoProcessor.from_pretrained('AIML-TUDA/LlavaGuard-v1.1-7B-hf', cache_dir=cache_dir)

    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-large", cache_dir=cache_dir)
    model = AutoModelForMaskedLM.from_pretrained("FacebookAI/xlm-roberta-large", cache_dir=cache_dir)

    from multilingual_clip import pt_multilingual_clip
    import transformers
    model_name = 'M-CLIP/XLM-Roberta-Large-Vit-B-32'
    model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
#../        ./         afr_Latn/  arb_Arab/  azj_Latn/  ben_Beng/  bul_Cyrl/  ces_Latn/  cmn_Hani/  dan_Latn/  deu_Latn/  ekk_Latn/  ell_Grek/  fas_Arab/  fin_Latn/  fra_Latn/  gle_Latn/  glg_Latn/  guj_Gujr/
#heb_Hebr/  hin_Deva/  hrv_Latn/  hun_Latn/  ind_Latn/  ita_Latn/  jpn_Jpan/  kaz_Cyrl/  khk_Cyrl/  khm_Khmr/  kor_Hang/  lit_Latn/  lvs_Latn/  mal_Mlym/  mar_Deva/  mkd_Cyrl/  mlt_Latn/  mya_Mymr/  nld_Latn/
#npi_Deva/  pbt_Arab/  pol_Latn/  por_Latn/  ron_Latn/  rus_Cyrl/  sin_Sinh/  slk_Latn/  slv_Latn/  spa_Latn/  swe_Latn/  swh_Latn/  tam_Taml/  tel_Telu/  tha_Thai/  tur_Latn/  ukr_Cyrl/  urd_Arab/  vie_Latn/
#xho_Latn/  zsm_Latn/


if False:
    #print (model)
    #model_name = "cognitivecomputations/dolphin-2.9.2-Phi-3-Medium"
    os.system(f"ct2-transformers-converter --model Mitsua/elan-mt-bt-en-ja  --output_dir {cache_dir}opus-mt-en-ja")
    transformers.AutoTokenizer.from_pretrained("Mitsua/elan-mt-bt-en-ja", cache_dir=cache_dir)
    os.system(f"ct2-transformers-converter --model Helsinki-NLP/opus-mt-de-pl --output_dir {cache_dir}opus-mt-de-pl")
    transformers.AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-pl", cache_dir=cache_dir)
    os.system(f"ct2-transformers-converter --model Helsinki-NLP/opus-mt-fr-sl --output_dir {cache_dir}opus-mt-fr-sl")
    transformers.AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-sl", cache_dir=cache_dir)

if True:
        langs = ['ja', 'fa', 'tr', 'th', "he", "ta", "az", "bn", "uk", "ms", "az", "mk", "af", "tl", "kk", "gl",
                 "gu",
                 "km",
                 "ml",
                 "mn",
                 "mr",
                 "my",
                 "ne",
                 "ps",
                 "si",
                 "xh",
        'bg',
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
         'sv', 'vi', 'zh', 'ar', 'ru', 'hi', 'sw', 'jap', 'ko', 'id']
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
                            try:
                                os.system(f"ct2-transformers-converter --model gsarti/opus-mt-tc-en-{lang} --output_dir {cache_dir}opus-mt-en-{lang}")
                                transformers.AutoTokenizer.from_pretrained("gsarti/opus-mt-tc-en-"+lang, cache_dir=cache_dir)
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
                            try:
                                os.system(f"ct2-transformers-converter --model gsarti/opus-mt-tc-{lang}-en --output_dir {cache_dir}opus-mt-{lang}-en")
                                transformers.AutoTokenizer.from_pretrained("gsarti/opus-mt-tc-"+lang+"-en", cache_dir=cache_dir)            
                            except:
                                print (f"NO MODEL FOR {lang}-en")                    
                                pass


if False:
    
    model_name="nhyha/N3N_gemma-2-9b-it_20241029_1532"
    model_name = "jsgreenawalt/gemma-2-9B-it-advanced-v2.1"
    model_name = "zake7749/gemma-2-2b-it-chinese-kyara-dpo"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True, cache_dir=cache_dir)
