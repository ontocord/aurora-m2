"""
Copyright 2023, LAION contributors, inclduing Ontocord, LLC
and the other authors of OIG
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at
  http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
"""

import json, os

# translation based stuff
import transformers


from transformers import (
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
    BertModel,
    BertTokenizerFast,
)
import langid


import torch
from torch.nn.functional import cosine_similarity
import string

punc = string.punctuation + "¿？,،、º。゜ "

# We assume we are running on CPU only
# use labse to do a comparison
try:
    if m2m100_model is None:
        pass
except:
    # m2m100_model = M2M100ForConditionalGeneration.from_pretrained("alirezamsh/small100").half().eval().to('cuda')
    # m2m100_tokenizer = SMALL100Tokenizer.from_pretrained("alirezamsh/small100")
    m2m100_model = (
        M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
        .half()
        .eval()
        .to("cuda")
    )
    m2m100_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
    labse_model = (
        BertModel.from_pretrained("sentence-transformers/LaBSE")
        .half()
        .eval()
        .to("cuda")
    )
    labse_tokenizer = BertTokenizerFast.from_pretrained("sentence-transformers/LaBSE")

from collections import Counter


def get_ngram(sent, window_size=3, lang="en"):
    if lang in {"zh", "ja", "ko", "th"}:
        tokens = sent
        ret = [
            "".join(tokens[i : i + window_size])
            for i in range(len(tokens) - window_size)
        ]
    else:
        tokens = sent.split()
    ret = [
        " ".join(tokens[i : i + window_size]) for i in range(len(tokens) - window_size)
    ]
    return Counter(ret)


def high_ngram(sent, cutoff=0.15, window_size=3, lang="en"):
    aHash = get_ngram(sent, window_size, lang)
    sent_len = sent.count(" ") + 1
    for key in list(aHash.keys()):
        aHash[key] = aHash[key] / sent_len
    return any(a > cutoff for a in aHash.values())


xglm_langs = {
    "en",
    "ru",
    "zh",
    "de",
    "es",
    "fr",
    "ja",
    "it",
    "pt",
    "el",
    "ko",
    "fi",
    "id",
    "tr",
    "ar",
    "vi",
    "th",
    "bg",
    "ca",
    "hi",
    "et",
    "bn",
    "ta",
    "ur",
    "sw",
    "te",
    "eu",
    "my",
    "ht",
    "qu",
}

langs2 = []
# TODO - add some multitrans, backtrans to get more diversity
# TODO - add option to return as paragraph by lang.
def get_translation_set(
    text,
    threshold=0.75,
    langs=[
        "af",
        "am",
        "ar",
        "ast",
        "az",
        "ba",
        "be",
        "bg",
        "bn",
        "br",
        "bs",
        "ca",
        "ceb",
        "cs",
        "cy",
        "da",
        "de",
        "el",
        "en",
        "es",
        "et",
        "fa",
        "ff",
        "fi",
        "fr",
        "fy",
        "ga",
        "gd",
        "gl",
        "gu",
        "ha",
        "he",
        "hi",
        "hr",
        "ht",
        "hu",
        "hy",
        "id",
        "ig",
        "ilo",
        "is",
        "it",
        "ja",
        "jv",
        "ka",
        "kk",
        "km",
        "kn",
        "ko",
        "lb",
        "lg",
        "ln",
        "lo",
        "lt",
        "lv",
        "mg",
        "mk",
        "ml",
        "mn",
        "mr",
        "ms",
        "my",
        "ne",
        "nl",
        "no",
        "ns",
        "oc",
        "or",
        "pa",
        "pl",
        "ps",
        "pt",
        "ro",
        "ru",
        "sd",
        "si",
        "sk",
        "sl",
        "so",
        "sq",
        "sr",
        "ss",
        "su",
        "sv",
        "sw",
        "ta",
        "th",
        "tl",
        "tn",
        "tr",
        "uk",
        "ur",
        "uz",
        "vi",
        "wo",
        "xh",
        "yi",
        "yo",
        "zh",
        "zu",
    ],
    return_original=False,
):
    ret = []
    if type(text) is str:
        text = [text]
    else:
        text = list(text)
    with torch.no_grad():
        labse_text = labse_tokenizer(text, max_length=512, truncation=True, padding=True, return_tensors="pt").to("cuda")
        en_embed = labse_model(**labse_text).pooler_output

        for target_lang in langs:

            input = m2m100_tokenizer(text, max_length=512, truncation=True, padding=True, return_tensors="pt").to("cuda")
            generated_tokens = m2m100_model.generate(
                **input, forced_bos_token_id=m2m100_tokenizer.get_lang_id(target_lang)
            )
            trans_text = m2m100_tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )
            trans_text = [tt for tt in trans_text if not high_ngram(tt, lang=target_lang)]
            if not trans_text: continue
            labse_text = labse_tokenizer(
                trans_text, padding=True, return_tensors="pt"
            ).to("cuda")

            all_trans_embed = labse_model(**labse_text).pooler_output
            try:
              similarity = cosine_similarity(en_embed, all_trans_embed, dim=1)
            except:
              print ('prob with sim')
              continue
            # trs = []
            for sim, tr in zip(similarity, trans_text):
                # print (sim, tr)
                if sim >= threshold and not high_ngram(tr):
                    ret.append({'text':tr, 'lang': target_lang, 'score': float(sim)})
                    # trs.append(tr)
            # if add_backtrans:
        return ret


common_langs_plus = [
    "ar",
    "bg",
    "bn",
    "ca",
    "de",
    "el",
    "en",
    "es",
    "et",
    "fi",
    "fr",
    "hi",
    "ht",
    "id",
    "it",
    "ko",
    "my",
    "pt",
    "ru",
    "sw",
    "ta",
    "th",
    "tr",
    "ur",
    "zh",
    'vi', 'da', 'sv', "is", "no", "ja",
    'vi', 'da', 'sv', "is", "no", "ja",
    'vi', 'da', 'sv', "is", "no", "ja",
    'vi', 'da', 'sv', "is", "no", "ja",
]
try:
  import datasets
except:
  !pip install datasets
from datasets import load_dataset
try:
  if minipile is None: assert False
except:
  minipile = load_dataset("JeanKaddour/minipile")
import random
batch = []
multilingual_data = []

try:
  if minipile_data is None: assert False
except:
  minipile_data = minipile['train']['text']

random.shuffle(minipile_data)
with open("/content/drive/Shareddrives/MDEL/dataset/minipile_multilingual.jsonl", "w") as output:
  for idx, text in enumerate(minipile_data):
      batch = []
      src_lang = 'en'
      multilingual_data.append({"text": text, "lang": src_lang})
      if "Python" in text or ("{" in text and '}' in text):
          continue  # we need to make sure not to translate the functions and code themselves. only the comments
      lang_choices = random.sample(common_langs_plus, 5)
      lang_choices = [l for l in lang_choices if l != src_lang]
      text = text.replace("\n", "<n>").replace("  ", "<w>")
      if len(text) > 500:
          sent = text.split(". ")
          all_text = []
          for lang in lang_choices:
            batch2 = []
            for s in sent:
              if not s.strip(): continue
              if batch2 and len(batch2[-1]) < 450:
                if batch2[-1]:
                  batch2[-1] += ". " + s
                else:
                  batch2[-1] = s
              else:
                if batch2: batch2[-1] += "."
                batch2.append(s)
              if len(batch2) >= 10:
                if batch2 and batch2[-1][-1] != '.': batch2[-1] += "."
                trans = get_translation_set(batch2, langs=[lang])
                for text2_data in trans:
                    text2 = text2_data['text']
                    lang = text2_data['lang']
                    if langid.classify(text2)[0] != lang:
                        print("bad trans", text2)
                        continue
                    all_text.append(text2)
                batch2 = []
            if batch2:
                if batch2 and batch2[-1][-1] != '.': batch2[-1] += "."
                trans = get_translation_set(batch2, langs=[lang])
                for text2_data in trans:
                    text2 = text2_data['text']
                    lang = text2_data['lang']
                    if langid.classify(text2)[0] != lang:
                        print("bad trans", text2)
                        continue
                    all_text.append(text2)
                batch2 = []
            if all_text:
              all_text = " ".join(all_text)
              multilingual_data.append({"text": all_text.replace("<n>", "\n").replace("<w>", "  "), "lang": lang})
            all_text = []
      else:
          batch.append(data)
          if len(batch) >= 40:
              trans = get_translation_set(batch, langs=lang_choices)
              for text2_data in trans:
                  text2 = text2_data['text']
                  lang = text2_data['lang']
                  if langid.classify(text2)[0] != lang:
                      print("bad trans", text2)
                      continue
                  multilingual_data.append({"text": text2.replace("<n>", "\n").replace("<w>", "  "), "lang": lang})
          batch = []
      for data in multilingual_data:
        output.write(json.dumps(data)+"\n")
      multilingual_data = []
  if batch:
      trans = get_translation_set(batch, langs=lang_choices)
      for text2_data in trans:
          text2 = text2_data['text']
          lang = text2_data['lang']
          if langid.classify(text2)[0] != lang:
              print("bad trans", text2)
              continue
          multilingual_data.append({"text": text2.replace("<n>", "\n").replace("<w>", "  "), "lang": lang})
      batch = []
  for data in multilingual_data:
      output.write(json.dumps(data)+"\n")
  multilingual_data = []
