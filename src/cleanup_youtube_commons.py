#@title cleanup youtube commons
import os
from google.colab import drive
drive.mount('/content/drive')

try:
  import langid
except:
  !pip install langid
import langid
langs = {
        "af": "Afrikaans",
        "als": "Tosk Albanian",
        "am": "Amharic",
        "an": "Aragonese",
        "ar": "Arabic",
        "arz": "Egyptian Arabic",
        "ast": "Asturian",
        "as": "Assamese",
        "av": "Avaric",
        "azb": "South Azerbaijani",
        "az": "Azerbaijani",
        "bar": "Bavarian",
        "ba": "Bashkir",
        "bcl": "Central Bikol",
        "be": "Belarusian",
        "bg": "Bulgarian",
        "bh": "Bihari",
        "bn": "Bengali",
        "bo": "Tibetan",
        "bpy": "Bishnupriya",
        "br": "Breton",
        "bs": "Bosnian",
        "bxr": "Russia Buriat",
        "ca": "Catalan",
        "cbk": "Chavacano",
        "ceb": "Cebuano",
        "ce": "Chechen",
        "ckb": "Central Kurdish",
        "cs": "Czech",
        "cv": "Chuvash",
        "cy": "Welsh",
        "da": "Danish",
        "de": "German",
        "diq": "Dimli",
        "dsb": "Lower Sorbian",
        "dv": "Dhivehi",
        "el": "Modern Greek",
        "eml": "Emilian-Romagnol",
        "en": "English",
        "eo": "Esperanto",
        "es": "Spanish",
        "et": "Estonian",
        "eu": "Basque",
        "fa": "Persian",
        "fi": "Finnish",
        "frr": "Northern Frisian",
        "fr": "French",
        "fy": "Western Frisian",
        "ga": "Irish",
        "gd": "Scottish Gaelic",
        "gl": "Galician",
        "gn": "Guarani",
        "gom": "Goan Konkani",
        "gu": "Gujarati",
        "he": "Hebrew",
        "hi": "Hindi",
        "hr": "Croatian",
        "hsb": "Upper Sorbian",
        "ht": "Haitian",
        "hu": "Hungarian",
        "hy": "Armenian",
        "ia": "Interlingua",
        "id": "Indonesian",
        "ie": "Interlingue",
        "ilo": "Iloko",
        "io": "Ido",
        "is": "Icelandic",
        "it": "Italian",
        "ja": "Japanese",
        "jbo": "Lojban",
        "jv": "Javanese",
        "ka": "Georgian",
        "kk": "Kazakh",
        "km": "Central Khmer",
        "kn": "Kannada",
        "ko": "Korean",
        "krc": "Karachay-Balkar",
        "ku": "Kurdish",
        "kv": "Komi",
        "kw": "Cornish",
        "ky": "Kirghiz",
        "la": "Latin",
        "lb": "Luxembourgish",
        "lez": "Lezghian",
        "li": "Limburgan",
        "lmo": "Lombard",
        "lo": "Lao",
        "lrc": "Northern Luri",
        "lt": "Lithuanian",
        "lv": "Latvian",
        "mai": "Maithili",
        "mg": "Malagasy",
        "mhr": "Eastern Mari",
        "min": "Minangkabau",
        "mk": "Macedonian",
        "ml": "Malayalam",
        "mn": "Mongolian",
        "mrj": "Western Mari",
        "mr": "Marathi",
        "ms": "Malay",
        "mt": "Maltese",
        "mwl": "Mirandese",
        "my": "Burmese",
        "myv": "Erzya",
        "mzn": "Mazanderani",
        "nah": "Nahuatl languages",
        "nap": "Neapolitan",
        "nds": "Low German",
        "ne": "Nepali",
        "new": "Newari",
        "nl": "Dutch",
        "nn": "Norwegian Nynorsk",
        "no": "Norwegian",
        "oc": "Occitan",
        "or": "Oriya",
        "os": "Ossetian",
        "pam": "Pampanga",
        "pa": "Panjabi",
        "pl": "Polish",
        "pms": "Piemontese",
        "pnb": "Western Panjabi",
        "ps": "Pushto",
        "pt": "Portuguese",
        "qu": "Quechua",
        "rm": "Romansh",
        "ro": "Romanian",
        "ru": "Russian",
        "sah": "Yakut",
        "sa": "Sanskrit",
        "scn": "Sicilian",
        "sd": "Sindhi",
        "sh": "Serbo-Croatian",
        "si": "Sinhala",
        "sk": "Slovak",
        "sl": "Slovenian",
        "so": "Somali",
        "sq": "Albanian",
        "sr": "Serbian",
        "su": "Sundanese",
        "sv": "Swedish",
        "sw": "Swahili",
        "ta": "Tamil",
        "te": "Telugu",
        "tg": "Tajik",
        "th": "Thai",
        "tk": "Turkmen",
        "tl": "Tagalog",
        "tr": "Turkish",
        "tt": "Tatar",
        "tyv": "Tuvinian",
        "ug": "Uighur",
        "uk": "Ukrainian",
        "ur": "Urdu",
        "uz": "Uzbek",
        "vec": "Venetian",
        "vi": "Vietnamese",
        "vo": "Volapük",
        "war": "Waray",
        "wa": "Walloon",
        "wuu": "Wu Chinese",
        "xal": "Kalmyk",
        "xmf": "Mingrelian",
        "yi": "Yiddish",
        "yo": "Yoruba",
        "yue": "Yue Chinese",
        "zh": "Chinese",
    }

import random
from typing import List
import re
from huggingface_hub import hf_hub_download
try:
  import fasttext
except:
  !pip install fasttext
import fasttext
if True:
  try:
    from punctuators.models import PunctCapSegModelONNX
  except:
    !pip install -q punctuators
  from punctuators.models import PunctCapSegModelONNX

  try:
    if model is None: assert False
  except:
    model = PunctCapSegModelONNX.from_pretrained(
        "1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase"
    )
if True:
  try:
    import detoxify
  except:
    !pip install detoxify
  from detoxify import Detoxify

  try:
    if classifier is None: assert False
  except:
    classifier = Detoxify('original')


try:
    if edu_model is None: assert False
except:
    edu_model = fasttext.load_model(hf_hub_download("kenhktsui/llm-data-textbook-quality-fasttext-classifer-v2", "model.bin"))


try:
    if oh_el15 is None: assert False
except:
    oh_el15 = fasttext.load_model(hf_hub_download("mlfoundations/fasttext-oh-eli5", "openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin"))

%cd /content/


def replace_newlines(text: str) -> str:
  return re.sub("\n+", " ", text)


score_dict = {
  '__label__': 0,
  '__label__Low': 0,
  '__label__Mid': 1,
  '__label__High': 2
}


def predict_educational_value(text):
  text = replace_newlines(text)
  pred = edu_model.predict(text, k=-1)
  score_list = []
  l, s = pred
  score = 0
  for _l, _s in zip(l, s):
    score += score_dict[_l] * _s
  return to_finew_edu_score(float(score))

def to_finew_edu_score(score):
  if score <= 0.448785: return 0
  if score <= 0.866190: return 1
  if score <= 1.195964: return 2
  if score <= 1.276247: return 3
  return 4

import json

!mkdir -p /content/drive/Shareddrives/ontocord_llc/safe_llm/youtube_commons_output2

import glob, os
import time
import random, json
try:
  import translators
except:
  !pip install -q translators
import translators as ts
import random
import time
lang_supported = {"bing": {'af', 'am', 'ar', 'as', 'az', 'ba', 'bg', 'bho', 'bn', 'bo', 'brx', 'bs', 'ca', 'cs', 'cy', 'da', 'de', 'doi', 'dsb', 'dv', 'el', 'en', 'es', 'et', 'eu', 'fa', 'fi', 'fil', 'fj', 'fo', 'fr', 'fr-CA', 'ga', 'gl', 'gom', 'gu', 'ha', 'he', 'hi', 'hne', 'hr', 'hsb', 'ht', 'hu', 'hy', 'id', 'ig', 'ikt', 'is', 'it', 'iu', 'iu-Latn', 'ja', 'ka', 'kk', 'km', 'kmr', 'kn', 'ko', 'ks', 'ku', 'ky', 'ln', 'lo', 'lt', 'lug', 'lv', 'lzh', 'mai', 'mg', 'mi', 'mk', 'ml', 'mn-Cyrl', 'mn-Mong', 'mr', 'ms', 'mt', 'mww', 'my', 'nb', 'ne', 'nl', 'nso', 'nya', 'or', 'otq', 'pa', 'pl', 'prs', 'ps', 'pt', 'pt-PT', 'ro', 'ru', 'run', 'rw', 'sd', 'si', 'sk', 'sl', 'sm', 'sn', 'so', 'sq', 'sr-Cyrl', 'sr-Latn', 'st', 'sv', 'sw', 'ta', 'te', 'th', 'ti', 'tk', 'tlh-Latn', 'tn', 'to', 'tr', 'tt', 'ty', 'ug', 'uk', 'ur', 'uz', 'vi', 'xh', 'yo', 'yua', 'yue', 'zh-Hans', 'zh-Hant', 'zu'},
 "yandex": {'az', 'be', 'bg', 'ca', 'cs', 'da', 'de', 'el', 'en', 'es', 'et', 'fi', 'fr', 'hr', 'hu', 'hy', 'it', 'lt', 'lv', 'mk', 'nl', 'no', 'pl', 'pt', 'ro', 'ru', 'sk', 'sl', 'sq', 'sr', 'sv', 'tr', 'uk'},
 "sogou": {'ar', 'cs', 'da', 'de', 'en', 'es', 'fi', 'fr', 'hu', 'it', 'ja', 'ko', 'nl', 'pl', 'pt', 'ru', 'sv', 'th', 'vi', 'zh-CHS'}}

translators = ["google", "sogou", "yandex",]

def grammar_fix(text):
  text = text.lower()
  #prev_dat['grammar_fix'] = True
        #text = dat['text']
  text = "\n".join([b[0].upper()+b[1:] for b in [a.strip(" ,") for a in text.split("\n")] if len(b) > 1]).replace(",,", ",").replace(". ,", ". ").replace("  ", " ").strip(" .,")
  text = text.replace("!.", "!").replace("?.", "?").replace(",,", ",").replace("??", "?").replace(". .. ", ".").replace("? ?", "?")
  transcript =  model.infer(
    texts=[text], apply_sbd=True,
    )[0]
  text = " ".join([b[0].upper()+b[1:] for b in [a.strip(" ,") for a in transcript] if len(b) > 1]).replace(",,", ",").replace(". ,", ". ").replace("  ", " ").strip(" .,")
  if text[-1] not in ".?!":
    text = text+"."

  return text

def output_prev_data(prev_id, prev_dat, outf):
    text_arr = prev_dat['text'].split("<|endoftext|>")
    lang_arr = prev_dat['metadata']['transcription_language'].split("<|endoftext|>")
    #prev_dat['grammar_fix'] = False
    #print ((prev_id, edu_score, embed_text))
    text2 = []
    embed_text = ""
    en_text = ""
    prev_dat['metadata']['edu_score'] = 0
    lang_arr2 = []
    beg = ""
    for text, lang in zip(text_arr, lang_arr):
      beg, text = text.split("Transcript:",1)
      beg = "\n".join(beg.strip().split("\n")[1:])
      if len(text) < 150: continue
      if lang in lang_arr2:
        continue
      if lang == 'en':
        edu_score = predict_educational_value(text)
        prev_dat['metadata']['edu_score'] = edu_score
        en_text = text.replace(". ", " ")
        if edu_score <= 0: return
        #if edu_score <= 1 and random.randint(0,5): return
        continue
      lang_arr2.append(lang)
      text2.append(text)
    text = ""
    prev_dat['metadata']['toxicity'] = []
    if not en_text:
      if not text2: return
      from_text = text2[0]
      lang = lang_arr[0]
      from_text = from_text[:min(len(from_text), 200)]
      tran_text = ""
      for _ in range(5):
        for _ in range(5):
          translator = random.choice(translators)
          if translator == "google" or lang in lang_supported[translator]:
            break
          else:
            translator = ""
        if translator:
          try:
            tran_text = ts.translate_text(from_text, from_language=lang, to_language="en", translator=translator)
            break
          except:
            print ('problem 1', lang, translator)
            if random.randint(0,5)==0: time.sleep(random.randint(0,2))
            continue
      if tran_text:
        edu_score = predict_educational_value(tran_text)
        prev_dat['metadata']['edu_score'] = edu_score
        if edu_score <= 0: return
        predict = classifier.predict(tran_text)
        prev_dat['metadata']['toxicity'] = [[a[0], float(a[1])] for a in predict.items()]
        prev_dat['embed_text'] = tran_text[:min(len(tran_text), 200)]
    if en_text:
      lang_arr2 = ['en'] + lang_arr2
      text2 = [en_text] + text2
      predict = classifier.predict(en_text)
      prev_dat['metadata']['toxicity'] = [[a[0], float(a[1])] for a in predict.items()]
      prev_dat['embed_text'] = en_text[:min(len(en_text), 200)]
    if not lang_arr2: return
    text2[0] = beg+"\nTranscript:\n" + text2[0]
    for lang, text3 in zip(lang_arr2, text2):
      if "Transcript:" in text3:
        beg, text4 = text3.split("Transcript:",1)
        text4 = text4.strip()
        text4 = text4[0].upper() + text4[1:]
        text3 = beg + "Transcript:\n" + text4
      else:
        text4 = text3
      if text4 in text: continue
      if lang in {'zh', 'ja', 'ko'} and text4.count('。') <= 2:
        text3 = grammar_fix(text3)
      elif lang not in {'zh', 'ja', 'ko'} and text4.count('.') <= 2:
        text3 = grammar_fix(text3)
      text3 = text3.replace(" :", ":").replace("  ", " ").replace("Channel name:", "Channel Name:")
      text3 = text3.replace("Channel Name: \n", "Channel Name: ").replace("Channel Name:\n", "Channel Name: ").replace("Title:", "Title: ").strip()
      text3 = text3.replace(" 've ", "'ve ").replace(" 't ", "'t ").replace(" 're ", "'re ").replace(" 'd ", "'d ").replace(" 'm ", "'m ").replace(" 's ", "'s ")
      for c in "QWERTYUIOPASDFGHJKLZXCVBNM":
        text3 = text3.replace("."+c, ". "+c)
      text3 = text3.replace("U. S.", "U.S.").replace("U. K.", "U.K.").replace("\n\n", "\n").replace(" ?", "?").replace("Dr..", "Dr.").replace("Mr..", "Mr.").replace("Mrs..", "Mrs.")
      text3 = text3.replace("!.", "!").replace("?.", "?").replace("?,", "?").replace(".,", ",").replace(",,", ",").replace("??", "?").replace(". .. ", ".").replace("? ?", "?").replace("..", ".")
      if text3[:100] in text: continue
      text3 = text3[0].upper() + text3[1:]
      text3 = text3.replace("Transcript:", "\nTranscript:")
      if lang != 'en':
        lang2 = langs.get(lang, lang)
        if "Transcript:" in text3:
          text3 = text3.replace("Transcript:", lang2 + " Transcript: ")
        else:
          text3 = lang2+" Transcript: " + text3
      text3 = text3.replace("\n\n", "\n").replace(":.", ":").replace(": .", ": ")
      if random.randint(0,1):
        text3 = text3.replace("Transcript:\n", "Transcript: ")
      if random.randint(0,1):
        text3 = text3.replace("Transcript: \n", "Transcript: ")
      if random.randint(0,1):
        text3 = text3.replace("Transcript: ", "Transcript:\n")
      if random.randint(0,1):
        text3 = text3.replace("Transcript:", "Transcript:\n")
      if random.randint(0,1):
        text3 = text3.replace(" Transcript:", ":")
      if random.randint(0,1):
        text3 = text3.replace("Title: ", "")
      if random.randint(0,1):
        text3 = text3.replace("Channel Name: ", "")
      text3 = text3.replace("\n ", "\n").replace("  ", " ").replace("\n\n", "\n")
      if text == "":
        text = text3
      else:
        text = text +"<|endoftext|>"+text3
    prev_dat['text'] = text
    prev_dat['metadata']['transcription_language'] =  lang_arr2
    del prev_dat['metadata']["word_count"]
    del prev_dat['metadata']["character_count"]
    outf.write (json.dumps(prev_dat)+"\n")

import os
import tqdm

def process_one(file):
  outfile = "/content/drive/Shareddrives/ontocord_llc/safe_llm/youtube_commons_output2/"+file.split("/")[-1]
  if os.path.exists(outfile) and os.path.getsize(outfile) > 1000:
    return
  !touch $outfile
  print (outfile)
  prev_id = None
  prev_dat = None
  prev_source = ""
  !sort --parallel=5 $file -o $file
  with open(outfile, "w") as outf:
    idx = -1
    for l in open(file):
      idx+= 1
      id, l = l.split("\t")
      id = id.strip()
      try:
        dat = json.loads(l)
      except:
        continue
      if dat['metadata']['source'] == prev_source:
        print ("found dup", prev_id, id)
        continue
      prev_source = dat['metadata']['source']
      if prev_id != id and prev_id is not None:
        if prev_dat:
          output_prev_data(prev_id, prev_dat, outf)
        prev_id = id
        prev_dat = None
      if dat['text'].count("[") > 1:
        prev_id = None
        prev_dat = None
        continue
      lang = dat['metadata']['transcription_language']
      orig_lang = dat['metadata']['original_language']
      transcript = dat['text']
      transcript = [a.strip() for a in transcript.split(". ")]
      transcript = ". ".join([b[0].upper()+b[1:] for b in transcript if len(b) > 1]).replace(",,", ",").replace(". ,", ". ").replace("  ", " ")
      #print ('not infer', transcript)
      dat['text'] = transcript
      dat['text'] = dat['text'].replace(",,", ",").replace(".,", ".").replace(" .. ", " ").replace("..", ".").replace("??", "?").replace(" . ", " ")
      if prev_dat is None:
        prev_dat = dat
      if id == prev_id:
        #print ('appending')
        text =  prev_dat['text']+"<|endoftext|>" + dat['text']
        lang = prev_dat['metadata']['transcription_language']+"<|endoftext|>"+dat['metadata']['transcription_language']
        prev_dat['metadata']['transcription_language'] = lang
        prev_dat['text'] = text
      prev_id = id
    if prev_dat:
      output_prev_data(prev_id, prev_dat, outf)
    !sort --parallel=5 $outfile -o $outfile

from multiprocessing import Pool
files = list(glob.glob("/content/drive/Shareddrives/ontocord_llc/safe_llm/youtube_commons_input2/*_out.jsonl"))
random.shuffle(files)
with Pool(10) as p:
  p.map(process_one, files)
