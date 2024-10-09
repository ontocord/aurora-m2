#@title megawika govt+cc-by source references
from google.colab import drive
import os
if not os.path.exists("/content/drive"):
  drive.mount('/content/drive')
from typing import List
import re
import json, os, random
try:
  import langid
except:
  !pip install langid
import langid

from multiprocessing import Pool

from collections import Counter

import string

from typing import List
import re
try:
  from huggingface_hub import hf_hub_download
except:
  !pip install huggingface_hub
from huggingface_hub import hf_hub_download


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



strip_chars = " ,،、_–+=-{}[]|()\"'“”《》«»!:;¿?。゜。…．꧈꧉꧋ꧏ"
punc_char = string.punctuation + "¿？,،、º。゜ "".!:;?。…．꧈꧉꧋ꧏ"
special_char = "�꧊ ꧊꧌꧍꧁꧂꧇꧃꧋꧆꧉,{}[]()|\\\"'“”《》«»~!@#$%^&*{}[]()_–+=-0987654321`<>,、،./?':;“”\"\t\n\\πه☆●¦″．۩۱（☛₨➩°・■↑☻、๑º‹€σ٪’Ø·−♥ıॽ،٥《‘©。¨﴿！★×✱´٬→±x：¹？£―▷ф¡Г♫∟™ª₪®▬「—¯；¼❖․ø•」٣，٢◦‑←§١ー٤）˚›٩▼٠«¢¸٨³½˜٭ˈ¿¬ι۞⌐¥►†ƒ∙²»¤…﴾⠀》′ا✓→¶'"
junk = set("�꧊꧌꧍꧁꧂꧇꧃꧋꧆꧉,{}[]()|\\\"'“”《》«»~!@#$%^&*{}[]()_–+=-0987654321`<>,、،./?':;“”\"\t\n\\πه☆●¦″．۩۱（☛₨➩°・■↑☻、๑º‹€σ٪’Ø·−♥ıॽ،٥《‘©。¨﴿！★×✱´٬→±x：¹？£―▷ф¡Г♫∟™ª₪®▬「—¯；¼❖․ø•」٣，٢◦‑←§١ー٤）˚›٩▼٠«¢¸٨³½˜٭ˈ¿¬ι۞⌐¥►†ƒ∙²»¤…﴾⠀》′ا✓→¶'")

def get_special_char_score (text, lang="en", special_characters_default=None):
  global junk
  if not text or len(text) == 0: return 1
  #TODO: do we want to do any lang specific special_chars?
  if special_characters_default is None: special_characters_default = junk
  return len([a for a in text if a in special_characters_default])/len(text)

import re

def cjk_detect(texts):
    # chinese
    if re.search("[\u4e00-\u9FFF]", texts):
        return "zh"
    # korean
    if re.search("[\uac00-\ud7a3]", texts):
        return "ko"
    # japanese
    if re.search("[\u3040-\u30ff]", texts):
        return "ja"
    # thai
    if re.search("[\u0E01-\u0E5B]", texts):
        return "th"
    # traditional javanese
    if re.search("[\uA980-\uA9DF]", texts):
       return "jv_tr"
    return None

def lang_is_cjk(lang):
    return lang in {'zh', 'zh-classical', 'zh-min-nan', 'zh-yue', 'ko', 'ja', 'th', 'jv_tr'}

def get_ngram(text, lang="en", window_size=3, ):
  if lang_is_cjk(lang):
    tokens = text
    ret= ["".join(tokens[i : i + window_size])   for i in range(len(tokens) - window_size)]
  else:
    tokens = text.split()
    ret= [" ".join(tokens[i : i + window_size])   for i in range(len(tokens) - window_size)]
  return Counter(ret)

def get_ngram_score(text, lang="en", window_size=3, ):
  if not text: return 1
  aHash = get_ngram(text, lang=lang, window_size=window_size)
  text_len = text.count(" ")+1
  for key in list(aHash.keys()):
    aHash[key] = aHash[key]/text_len
  if not aHash: return 0.0
  return aHash.most_common(1)[0][1]

def process_one(file):
  if os.path.exists(f"/content/drive/Shareddrives/ontocord_llc/safe_llm/megawiki/{file}"): return
  parent_dir = file.split("/")[0]
  !mkdir -p $parent_dir
  parent_dir2 = f"/content/drive/Shareddrives/ontocord_llc/safe_llm/megawiki/{parent_dir}"
  !mkdir -p $parent_dir2
  file2 = f"/content/drive/Shareddrives/ontocord_llc/safe_llm/megawiki/{file}"
  !touch $file2
  if not os.path.exists(file):
    #print (file, os.path.exists(file))
    url = f"https://huggingface.co/datasets/hltcoe/megawika/resolve/main/data/{file}"
    !wget -O $file $url
  urls = {}
  seen = {}
  source_lang = langs.get(parent_dir, parent_dir, )
  with open(f"/content/drive/Shareddrives/ontocord_llc/safe_llm/megawiki/{file}", "w") as outf:
    with open(file) as info:

      for l in info:
        reference = 1
        dat = json.loads(l)
        translation = ""
        edu_score = 0
        if parent_dir in {'zh', 'ko' 'ja'} and len(dat['article_text']) < 10: continue
        if parent_dir not in {'zh', 'ko' 'ja'} and len(dat['article_text']) < 100: continue
        article_text = "### Wikipedia Article: "+ dat['article_text'].replace("(,", "(").replace(";  )", ")").replace(", )", ")").replace("\n\n\n", "\n\n").replace("( ,", "(").replace(", ,", ",").replace(" ; ;", ";").replace("(;", "(").\
                  replace("thumb|right", "IMAGE: ").replace("thumb|left", "IMAGE: ").replace("thumb|", "IMAGE: ").replace("()", " ").replace("( )", "")
        for c in "QWERTYUIOPASDFGHJKLZXCVBNM":
          article_text = article_text.replace("."+c, ". "+c)
        article_text = article_text.replace("U. S.", "U.S.").replace("U. K.", "U.K.").replace(":", ": ").replace("IMAGE: |", "IMAGE: ").replace("|IMAGE:", "IMAGE:").\
              replace(".[", ". [").replace(":|", ":")
        embed_text = ""
        seen_articles = {}
        for b in dat['entries']:
          if 'repetitious_translation' not in b or not b['repetitious_translation']:
            if 'translation' in b:
               translate_key = b['translation'].strip().lower().replace(" ", "")
               translate_key = translate_key[:min(len(translate_key), 30)]
               if translate_key not in translation.strip().lower().replace(" ", "") and get_ngram_score(b['translation']) < 0.1 and get_special_char_score(b['translation']) < 0.1:
                  translation = (translation+ " " + b['translation']).strip()

          if b['source_url']  not in urls and b['source_text'] and len(b['source_text']) > 100 and get_special_char_score(b['source_text']) < 0.1:
              urls[b['source_url']] = 1
              lang2 = cjk_detect(b['source_text'][:100])
              if not lang2: lang2 = "en"
              if get_ngram_score(b['source_text'], lang2) < 0.1:
                #print (('found',b['source_url'] ,b['source_text'] ))
                head = b['source_text'][:100].lower()
                tail = b['source_text'][-100:].lower()
                if "cc-by " in head or "cc-by " in tail or "cc-0 " in head or "cc-0 " in tail or "cc-by-" in head or "cc-by-" in tail or \
                  "creative common" in head or "creative common" in tail:
                  if "cc-by " in head or "cc-0 " in head or "cc-by-" in head or \
                    "creative common" in head:
                      if 'Flickr' in head or 'Flicker' in head or 'flickr' in head or 'flicker' in head or "NonCommercial" in head or "-NC" in head: continue
                  if "cc-by " in tail or "cc-0 " in tail or "cc-by-" in tail or \
                      "creative common" in tail:
                      if 'Flickr' in tail or 'Flicker' in tail or 'flickr' in tail or 'flicker' in tail or "NonCommercial" in head or "-NC" in head: continue
                  is_cc = True
                if is_cc or  "europa.eu/" in b['source_url'] or ".un/" in b['source_url'] or ".int/" in b['source_url'] or ".gov/" in b['source_url'] or '.gov.' in b['source_url'] or '.gouv.' in b['source_url']:
                  if "ymca.int" in b['source_url'] or "census" : continue
                    t = b['source_text']
                    t = t.strip()
                    t = "\n".join(t0.lstrip(".,:;").strip("|") for t0 in t.split("\n") if t0.lstrip(".,:;").strip("|"))
                    if get_special_char_score(t, l) > 0.1:
                      continue
                    t_arr = []
                    t_split = t.split("\n")
                    mean_line_len = sum(len(t0) for t0 in t_split)/len(t_split)
                    if mean_line_len < 50:
                      continue
                    for t0 in t_split:
                      if get_special_char_score(t0, l) > 0.1:
                        continue
                      t0 = " ".join(t1 for t1 in t0.split(" ") if not ("." in t1 and "@" in t1) and "http" not in t1 and "www" not in t1)
                      if not t0:
                        continue
                      if len(t0) > 10:
                        t2 = hash(t0.replace(" ", "").lower())
                        if t2 in seen:
                          continue
                        seen[t2] = 1
                      t_arr.append(t0)
                    t = "\n".join(t_arr)
                    lang, _ = langid.classify(t.replace("\n", " ")[:min(len(t), 200)])
                    if lang == 'en':
                      embed_text = t[:min(len(embed_text),400)]
                    if lang in {'ja', 'ko', 'zh'} and len(t) < 20: continue
                    if lang not in {'ja', 'ko', 'zh'} and len(t) < 100: continue
                    if lang != 'en' and lang in langs:
                      seen_articles[b['source_url'] ] = "### LANGUAGE: "+ langs[lang]+"\n### URL: "+b['source_url']+"\n"+t
          reference = str(abs(hash(b['source_url'] )))[:3]
          if len(b['passage']['text']) > 1:
            remove = b['passage']['text'][1]
          else:
            remove = b['passage']['text'][0]
          if remove in article_text:
            article_text = article_text.replace(remove, f" [REFERENCE {reference}] ")
            if random.randint(0,1):
              seen_articles[b['source_url'] ] = seen_articles[b['source_url'] ]+f"\n[REFERENCE {reference}]"
            else:
              seen_articles[b['source_url'] ] = f"[REFERENCE {reference}]\n"+seen_articles[b['source_url'] ]
            article_text = article_text.replace(remove, " ")
        if seen_articles or len(translation) > 200:
            translation = translation.replace("()", "").replace("( )", "").replace("(,", "(").replace(";  )", ")").replace(", )", ")").replace("\n\n\n", "\n\n").replace("( ,", "(").replace(", ,", ",").replace(" ; ;", ";").replace("(;", "(").\
                  replace("thumb|right", "IMAGE: ").replace("thumb|left", "IMAGE: ").replace("thumb|", "IMAGE: ").replace("()", " ").replace("( )", "")
            if not embed_text and translation:
              embed_text = translation[:min(len(translation),400)]
            if not embed_text:
              embed_text = dat['article_text']
              embed_text = embed_text[:min(len(embed_text),400)]
            seen_articles['wikipedia'] = article_text
            if len(translation) > 200:
              seen_articles['translation'] = f"### This is a machine translation of snippets of a Wikipedia article about '{dat['article_title']}' from {source_lang} to English:\n"+translation.strip()
              if random.randint(0,1):
                seen_articles['translation'] = seen_articles['translation'].replace("This is a machine translation", "Translation")
                seen_articles['translation'] = seen_articles['translation'].replace("of snippets of a Wikipedia article", "of Wikipedia text")
            seen_articles = list(seen_articles.values())
            random.shuffle(seen_articles)
            article = "\n<|endoftext|>".join(seen_articles)
            article = article.replace("\n\n", "\n")
            print ({'embed_text': embed_text, 'text': article, 'metadata': {'source':'hltcoe/megawika', 'edu_score': edu_score, 'title': dat['article_title'], 'lang': parent_dir }})
            outf.write(json.dumps({'embed_text': embed_text, 'text': article, 'metadata': {'source':'hltcoe/megawika', 'edu_score': edu_score, 'title': dat['article_title'], 'lang': parent_dir }})+"\n")
    !rm $file

def process():
  !mkdir -p /content/drive/Shareddrives/ontocord_llc/safe_llm/megawiki
  random.shuffle(files)
  if True:
    with Pool(50) as p:
      p.map(process_one, files)

