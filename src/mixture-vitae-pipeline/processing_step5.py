import os
import multiprocessing
import glob, random
import subprocess
from pathlib import Path
from tqdm import tqdm
import json
import json, glob, os
import math
import multiprocessing, functools, json, glob
from flagged_words import *
from collections import Counter
import string
import argparse
from collections import defaultdict
import sys
import pyarrow.parquet as pq
import time, random
import json, os, glob, random
import multiprocessing
from multiprocessing import set_start_method
import os
import json, os, glob
from tqdm import tqdm
from cdifflib import CSequenceMatcher
from typing import List
import glob, json
import re
from huggingface_hub import hf_hub_download
import fasttext
from multiprocessing import Pool
from shared import *
from transformers import AutoTokenizer
import re
from itertools import islice
import wn, itertools


information_extraction_templates = ['are defined as', 'refer to', 'can be described as', 'are characterized by', 'belong to', 'are part of', 'fall under', 'are included in', 'are also called', 'are alternatively referred to as', 'are synonymous with', 'go by the name', 'are classified as', 'are grouped under', 'are categorized as', 'fit into the category of', 'are used for', 'serve as', 'function as', 'operate as', 'result in', 'lead to', 'are caused by', 'are the result of', 'are similar to', 'resemble', 'are comparable to', 'are analogous to', 'are associated with', 'are related to', 'are owned by', 'occur before', 'happen after', 'coincide with', 'take place during', 'such as', 'for example', 'is defined as', 'refers to', 'is characterized by', 'belongs to', 'is part of', 'falls under', 'is included in', 'also called', 'alternatively referred to as', 'is synonymous with', 'goes by the name', 'is classified as', 'is grouped under', 'is categorized as', 'fits into the category of', 'is used for', 'serves as', 'functions as', 'operates as', 'results in', 'leads to', 'is caused by', 'is the result of', 'is similar to', 'resembles', 'is comparable to', 'is analogous to', 'is associated with', 'is related to', 'is owned by', 'occurs before', 'happens after', 'coincides with', 'takes place during', 'is a']


def is_num(n):
  try:
    float(n)
    return True
  except:
    return False



def guess_ner_label(ent, label=""):
  hype_lemma = []
  exact_match = False
  ent = ent.replace("Symptom", "Disease").replace("symptom", "disease").replace("syndrome", "disease").replace("Syndrome", "Disease").replace("Death", "Disease").replace("death", "disease").replace("Cancer", "Disease").replace("cancer", "disease").replace("Disorder", "Disease").replace("disorder", "disease")

  #now do wordnet checking
  ss_list = list(en_wn.synsets(ent, pos='n'))
  if ss_list:
    exact_match = True
  if not ss_list and "'s " in ent:
    ss_list = list(en_wn.synsets(ent.split("'s ")[-1].strip(), pos='n'))
  if not ss_list and ent[-1] not in "1234567890":
    ss_list = list(en_wn.synsets(ent.split()[-1], pos='n'))
  if not ss_list and ent[-1] in "1234567890":
    ss_list = list(en_wn.synsets(ent.split()[0], pos='n'))
  if not ss_list:
      ent2 = ent.replace("'s", " 's").replace("(", " ( ").replace(")", " ) ").replace("-", " - ").replace(":", " : ")
      for s in ["(", ")", ":", "-", "of", "for", "in", "on", "from", "at"]:
        ent2 = ent2.replace(" "+s+" ", "@#@")
        ent2 = ent2.replace(" "+s.upper()+" ", "@#@")
      if "@#@" in ent2:
        first_word = ent2.strip().split("@#@")[0]
        ss_list = list(en_wn.synsets(first_word, pos='n'))
  orig_label = label
  if ss_list:
    ss = ss_list[0]
    hype = ss.hypernyms()
    hype_lemma = []
    if hype:
      hype_lemma = hype[0].lemmas()
      hype_hype = hype[0].hypernyms()
      while hype_hype:
        hype_lemma.extend(hype_hype[0].lemmas())
        hype_hype = hype_hype[0].hypernyms()
    if 'microorganism' in hype_lemma or 'harm' in hype_lemma or  'death' in hype_lemma or 'unhealthiness' in hype_lemma or \
          'disorder' in hype_lemma or 'disease' in hype_lemma or 'illness' in hype_lemma or \
          'pathogen' in hype_lemma or 'symptom' in hype_lemma or 'mental condition' in hype_lemma or 'enlargement' in hype_lemma:
      label = "DISEASE_OR_HARM"
    elif 'chemical' in hype_lemma or 'drug' in hype_lemma or 'substance' in hype_lemma or 'molecule' in hype_lemma:
      label = "CHEMICAL"
    elif 'monetary unit' in hype_lemma:
      label = "MONEY"
    elif 'commodity' in hype_lemma and exact_match:
      label = "PRODUCT"
    elif 'animal' in hype_lemma and exact_match:
      label = "ANIMAL"
    elif 'plant' in hype_lemma and exact_match:
      label = "PLANT"

  return label, hype_lemma, exact_match

def templatize(text2, key, label):
  text2 = " "+text2+" "
  key= key.strip("{} ")
  text2 = text2.replace(" "+key,' {'+label+'} ').strip()
  text2 = text2.replace(key+" ",' {'+label+'} ').strip()
  return text2

# do reverb like extraction NER obj verb subj
def get_verb_relation(text):
  global spacy_nlp
  #TODO: do RB and "-" and ":" before verb
  #TODO: do basic relationships such as "known as", "defined as", "included in", "is a", "known for"
  doc = spacy_nlp(text)
  verb_relationship = ""
  orig_verb = ""
  prev_be = ""
  for token in doc:
    if token.lemma_ in {'do', 'be', 'have'}:
        prev_be = token.lemma_+"_"
        continue
    if token.tag_.startswith("VB") and token.tag_ not in {"VBZ", }:
      orig_verb = token.text
      verb_relationship = prev_be+str(token.lemma_)
      prev_be = ""
      continue
    if verb_relationship:
      if token.tag_ in {"RB","IN"}:
        orig_verb += " "+token.text
        verb_relationship += "_"+str(token.lemma_)
        prev_be = ""
        break
      else:
        break
  obj = None
  if not orig_verb:
    for rel in information_extraction_templates:
        if " "+rel+" " in text:
            orig_verb = rel
            verb_relationship = rel.replace(" ", "_").replace("is_" "be_").replace("are_" "be_")
            break
  if orig_verb:
    obj = text.split(orig_verb,1)[-1]
    obj = strip_right_stopwords(obj)
    obj = strip_left_stopwords(obj)
        
  return verb_relationship, orig_verb, obj

strip_chars = '–- ,~!@#^&*()-_=+" \n<>\\>/|:[]'

def basic_cleanup_word(ent, lang="en"):
  ent = ent.strip(strip_chars)
  ent = strip_right_stopwords(ent, lang=lang)
  ent = strip_left_stopwords(ent, lang=alang)
  ent = ent.strip(strip_chars)
  if ent.startswith("s ") or ent.startswith("'s ") or ent.startswith("’s ") or ent.startswith(". "):
    ent = ent[2:].strip()
  if ent.endswith("’s") or ent.endswith("'s"):
    ent = ent[:-2].strip()
  for word in ["A", "An", "The", "Mr.", "Mrs.", "Dr.",]:
    if ent.startswith(word+" "):
      ent = ent.split(word+" ",1)[-1].strip()
      break
  for word in numbering_list:
    if ent.startswith(word+" "):
      ent = ent.split(word+" ",1)[-1].strip()
      break
  return ent


def extract_capitalized_ngrams(text, min_n=2, max_n=4, topk=.1):
    """
    Extract 2-4 grams of words that start and end with capitalized words.
    
    Args:
    text (str): The input text.
    min_n (int): Minimum size of n-grams. Default is 2.
    max_n (int): Maximum size of n-grams. Default is 4.

    Returns:
    list: A list of n-grams as strings.
    """
    # Tokenize the text into words
    words = re.findall(r'\b[A-Za-z]+\b', text)

    # Function to generate n-grams
    def ngrams(words, n):
        return zip(*(islice(words, i, None) for i in range(n)))

    # Find n-grams of specified sizes that start and end with capitalized words
    capitalized_ngrams = []
    for n in range(min_n, max_n + 1):
        for ngram in ngrams(words, n):
            if ngram[0][0].isupper() and ngram[-1][0].isupper() and ngram[0].lower() not in stopwords_set and ngram[-1].lower() not in stopwords_set:
                capitalized_ngrams.append(' '.join(ngram))
    counter = Counter(capitalized_ngrams)
    return [a[0] for a in counter.most_common(int(len(counter)*topk))]

import time, re
def ner_scispacy(text, min_ner_guess_len=10, length_for_rel=200):
  global spacy_nlp, sci_spacy

  orig_text = text
  text = re.sub(r'\[\d+\]', '', text)  
  text2 = text.replace("{", "-lbracket-").replace("}", "-rbracket-")
  text = "\n"+text+"\n"
  ner_cnt = {}
  ents = {}

  # gather disease and chemical NER
  text_lower = text.lower()
  doc =sci_spacy(text)
  total_ents = len(doc.ents)
  if 'disease' in text_lower or 'bio' in text_lower or 'medic' in text_lower or total_ents > 3:
    chunks0 = dict([(ent.text, ent.label_) for ent in doc.ents])
  else:
    chunks0 = {}

def get_verb_instr():
    while True:
        ss = list(wn.synsets())
        random.shuffle(ss)
        for s in ss:
            if s.pos == "v":
                d = s.definition().split()
                if d[0] == "be":
                    d[0] = "being"
                elif len(d) > 3 and d[0][-1] == "e":
                    d[0]  = d[0][:-1]+"ing"
                else:
                    d[0]  +="ing"
                d = " ".join(d)
                forms = [a.forms() for a in s.words()]
                examples = s.examples()
                if examples:
                    examples[-1] += ", "+ d
                else:
                    examples = [d]
                spelling = ""
                if random.randint(0,5)==0:
                    spelling = " ".join(forms[0][0])
                    if random.randint(0,2)==0:
                        spelling = spelling.upper()
                    spelling = "Also spell the word '"+forms[0][0] +"' as '" + spelling + "' in your response."
                ret = ("In the revision below use one of these verbs or actions: "+ ", ".join(list(itertools.chain(*forms)))+ ". Examples of usage: "+"; ".join(examples)) + ". " + spelling
                #if spelling:
                #    print (ret)
                yield ret


# splitting doesn't always work because some languages aren't space separated
def get_aligned_text(sent1, sent2, target_lang="en"):
  if target_lang not in  {"ja", "zh", "ko"}:
    sep = " "
    sent1 = sent1.replace("\n", " ** ").split()
    sent2 = sent2.replace("\n", " ** ").split()
  else:
    sep = ""
  aMatch = CSequenceMatcher(None,sent1, sent2)
  score0 = aMatch.ratio()
  blocks = aMatch.get_matching_blocks()
  blocks2 = []
  prevEndA = 0
  prevEndB = 0
  matchLen = 0
  nonMatchLen = 0
  for blockI in range(len(blocks)):
      if blockI > 0 or (blockI==0 and (blocks[blockI][0] != 0 or blocks[blockI][1] != 0)):
          blocks2.append([sep.join(sent1[prevEndA:blocks[blockI][0]]), sep.join(sent2[prevEndB:blocks[blockI][1]]), 0])
          nonMatchLen += max(blocks[blockI][0] - prevEndA, blocks[blockI][1] - prevEndB)
      if blocks[blockI][2] != 0:
        blocks2.append([sep.join(sent1[blocks[blockI][0]:blocks[blockI][0]+blocks[blockI][2]]), sep.join(sent2[blocks[blockI][1]:blocks[blockI][1]+blocks[blockI][2]]), 1])
        prevEndA = blocks[blockI][0]+blocks[blockI][2]
        prevEndB = blocks[blockI][1]+blocks[blockI][2]
        matchLen += blocks[blockI][2]
  score = float(matchLen+1)/float(nonMatchLen+1)
  return (blocks2, score, score0)

# Define filter phrases (with surrounding spaces for safety)
filter_phrases = [" click ", " posted ", " website ", " thank"]

def contains_filter_phrase(t):
  t_lower = " "+ t.lower() +" "
  return any(phrase in t_lower for phrase in filter_phrases)

  

def create_summary(text):
    # Build dictionaries for sentence-level and paragraph-level fragments
    starts_with = {}
    ends_with = {}
    for t in text.split(". "):
        t_arr = t.split(" ")
        if len(t_arr) > 3:
            t2 = " ".join(t_arr[:3]).strip(".?!|;,")
            starts_with[t2] = starts_with.get(t2, 0) + 1
        if len(t_arr) > 4:
            t2 = " ".join(t_arr[:4]).strip(".?!|;,")
            starts_with[t2] = starts_with.get(t2, 0) + 1
        if len(t_arr) > 5:
            t2 = " ".join(t_arr[:5]).strip(".?!|;,")
            starts_with[t2] = starts_with.get(t2, 0) + 1

        if len(t_arr) > 5:
            t2 = " ".join(t_arr[-3:]).strip(".?!|;,")
            ends_with[t2] = ends_with.get(t2, 0) + 1
        if len(t_arr) > 6:
            t2 = " ".join(t_arr[-4:]).strip(".?!|;,")
            ends_with[t2] = ends_with.get(t2, 0) + 1
        if len(t_arr) > 7:
            t2 = " ".join(t_arr[-5:]).strip(".?!|;,")
            ends_with[t2] = ends_with.get(t2, 0) + 1

    starts_with_para = {}
    ends_with_para = {}
    for t in text.split("\n"):
        t_arr = t.split(" ")
        if len(t_arr) > 3:
            t2 = " ".join(t_arr[:3]).strip(".?!|;,")
            starts_with_para[t2] = starts_with_para.get(t2, 0) + 1
        if len(t_arr) > 4:
            t2 = " ".join(t_arr[:4]).strip(".?!|;,")
            starts_with_para[t2] = starts_with_para.get(t2, 0) + 1
        if len(t_arr) > 5:
            t2 = " ".join(t_arr[:5]).strip(".?!|;,")
            starts_with_para[t2] = starts_with_para.get(t2, 0) + 1

        if len(t_arr) > 5:
            t2 = " ".join(t_arr[-3:]).strip(".?!|;,")
            ends_with_para[t2] = ends_with_para.get(t2, 0) + 1
        if len(t_arr) > 6:
            t2 = " ".join(t_arr[-4:]).strip(".?!|;,")
            ends_with_para[t2] = ends_with_para.get(t2, 0) + 1
        if len(t_arr) > 7:
            t2 = " ".join(t_arr[-5:]).strip(".?!|;,")
            ends_with_para[t2] = ends_with_para.get(t2, 0) + 1

    # Filtering for paragraph-level starts_with entries with additional phrase checks
    for key in list(starts_with_para.keys()):
        t = " " + key + " "
        if starts_with_para[key] < 2 or contains_filter_phrase(t):
            del starts_with_para[key]

    # Remove redundant keys for paragraph-level starts_with
    for key in list(starts_with_para.keys()):
        for key2 in list(starts_with_para.keys()):
            if key != key2 and key.startswith(key2):
                del starts_with_para[key2]

    # Filtering for paragraph-level ends_with entries with additional phrase checks
    for key in list(ends_with_para.keys()):
        t = " " + key + " "
        if ends_with_para[key] < 2 or contains_filter_phrase(t):
            del ends_with_para[key]
    for key in list(ends_with_para.keys()):
        for key2 in list(ends_with_para.keys()):
            if key != key2 and key.endswith(key2):
                del ends_with_para[key2]

    # Filtering for non-paragraph starts_with entries with additional phrase checks
    for key in list(starts_with.keys()):
        t = " " + key + " "
        if starts_with[key] < 2 or contains_filter_phrase(t):
            del starts_with[key]
    for key in list(starts_with.keys()):
        for key2 in list(starts_with.keys()):
            if key != key2 and key.startswith(key2):
                del starts_with[key2]

    # Filtering for non-paragraph ends_with entries with additional phrase checks
    for key in list(ends_with.keys()):
        t = " " + key + " "
        if ends_with[key] < 2 or contains_filter_phrase(t):
            del ends_with[key]
    for key in list(ends_with.keys()):
        for key2 in list(ends_with.keys()):
            if key != key2 and key.endswith(key2):
                del ends_with[key2]

    sentences = []

    # Sentence-level fragments summary
    for key, count in starts_with.items():
        sentences.append(f"{count} sentence{'s' if count != 1 else ''} that start{'s' if count == 1 else ''} with '{key}',")

    for key, count in ends_with.items():
        sentences.append(f"{count} sentence{'s' if count != 1 else ''} that end{'s' if count == 1 else ''} with '{key}',")

    # Paragraph-level fragments summary
    for key, count in starts_with_para.items():
        sentences.append(f"{count} paragraph{'s' if count != 1 else ''} that start{'s' if count == 1 else ''} with '{key}',")

    for key, count in ends_with_para.items():
        sentences.append(f"{count} paragraph{'s' if count != 1 else ''} that end{'s' if count == 1 else ''} with '{key}',")

    # Concatenate all summary sentences into one string (separated by a space or newline)
    summary_string = random.choice([" ", "\n", "\n* "]).join(sentences)
    if "*" in summary_string:
        summary_string = "* "+summary_string

    # Now summary_string contains the concatenated summary sentences
    if summary_string and len(summary_string) > 50 and len(summary_string) < 500:
      summary_string = "This below document has "+ summary_string
      summary_string = summary_string[:-1]+"."
      text = text.replace("\n\n", "\n").replace("\n\n", "\n").replace("\n\n", "\n").replace("\n", " \n").replace("!", ".").replace("?", ".").strip()+". "
      summary_string += " It has approximately "+ str(len(text.split()))+ " words, "+ str(text.count(". "))+" sentences"
      if 1+text.count("\n") < text.count(". "):
        summary_string += ", and " + str(1+text.count("\n")) + " paragraph(s)."
      else:
        summary_string += "."
      return summary_string, starts_with, ends_with, starts_with_para, ends_with_para
    return None, None,None,None,None



def process(arg):
    global model, tokenizer, device, args, num_devices
    file, device_no, args = arg
    model, tokenizer = init_model(device_no, args)
    verb_instr = get_verb_instr()
    batch = []
    for l in open(file, "rb"):
        try:
            data= json.loads(l)
        except:
            continue
        if (data['text'].count("<|endoftext|>")+1) != len(data['metadata']):
            print ("PROBLEM B", len(text_arr), len(data['metadata']), text_arr, data)
            continue
        fix_idx(data)
          
        if "fandom.com" in data['idx'] or ("gutenberg.org" in data['idx'] and "wiki" not in data['idx']):
            data['text'] = "<|endoftext|>".join(a if not a.strip() else (random.choice(["### Fiction: ", "Synopsis of a story: ", "This is a fan created fiction. ", "Fan/fiction website: ", "Fictional writing: ", "Story: ", "Short fiction\n"]) + a) for a in data['text'].split("<|endoftext|>"))
        elif "wikipedia.org" in data['idx']:
            data['text'] = "<|endoftext|>".join(a if not a.strip() else (random.choice(["### Wikipedia: ", "Encycolpedia\n", "Detailed Report.  ", "Wikipedia page: ", "Informational Report: ", "Wikipedia Article: "]) + a) for a in data['text'].split("<|endoftext|>"))
        if args.create_upsample:
            meta_text = list(zip(data['metadata'], data['text'].split("<|endoftext|>")))
            meta_text = [a for a in meta_text if "hq-" in a[0].get('domain','')]
            if not meta_text:
                meta_text = list(zip(data['metadata'], data['text'].split("<|endoftext|>")))                    
            meta_text.sort(key=lambda a: len(a[1]), reverse=True)
            meta, text = meta_text[0]
            meta_text = [(meta, text) for meta, text in meta_text if (text == text.upper() or text.lower() == text) and len(text) > 200]
            number_instructions_added = 0
            for meta, text in meta_text:
                content_type = random.choice(["Fiction", "Story", "fiction", "fictional story", "story including dialogue, ", "excerpt from a novel", "short story", "alternate fictional history", "narrative and drama", "obituary", "knowledge graph", "highly educational and detailed descriptions", "fictional narrative", "highly poetic text", "song", "rap battle", "humorous dialogue", "rap song", "multiple choice questions with answers, and in depth explanations", "joke book", "pop song", "lyrical and narrative text", "essay", "textbook","textbook","textbook","textbook","textbook","textbook","textbook","textbook", "scientific article", "recipe", "infographic", "textbook with question and answer excercises", "class outline", "dramatic play", "detailed literary criticism", "advertisement", "blog", "social media post", "religious sermon", "confessional story", "personal story found on an online social media platform", "simple math problem and solution", "summary", "logic problem", "mystery problem", "business email", "personal email", "diary entry", "how-to manual", "python psuedo code", "wikipedia article", "satirical piece", "opinion piece", "editorial", "romantic story", "sci-fi story", "western story", "comic book or graphic novel", "commencement speech", "political speech", "debate between two people", "workbook", ])                    
                tmp_instr = random.choice(["Make this suitable for a 5 year old in the form of a","Make the below suitable for a middle school student in the form of a","Make the below suitable for a high school student in the form of a","Make the document suitable for a college student in the form of a","Make the content set forth below suitable for a graduate student in the form of a","### Rewrite this into modern language","Revise this into a story and ","Make this text to be a","Provide a revision of this in the form of a","I need this document in the form of a","Based on the below, let's tell a","Please create for the below a","Generate from the below a","Using the context, make a historical","Please find the content below. Create literature and","Based on your background knowledge and the content below, create media type:","Based on the document I am giving you, create content type:"])+ " " + content_type+", inspired by the subject matter below. Where appropriate, change names of characters, genders, nationalities, dates, time periods, and regions to make them more diverse. "
                tmp_instr += ("" if ("multilingual" in data['domain']) else "You should focus on the "+data['domain'].replace("_", " ").replace("-", " ")+" domain. ")+ "If there are any incorrect facts or archaic bias, remove it. Additionally, "+ next(verb_instr)
                tmp_instr += " Do not provide commentary. Just return the revised " + content_type +("" if ("spell the word" in tmp_instr or data.get('lang', 'en') != 'en') else (" in English" if random.randint(0,1) != 0 else (" in "+ random.choice(regions))))+  ". \n===\n"
                if random.randint(0,3)==0:
                    tmp_instr += " "+random.choice(["Think", "Solve", "Work", "Reason", "Describe", "Write"])+" step-by-step."                                            
                tmp_instr += random.choice([": ", "-\n\n\n", "--\n\n", "\n######\n", "\nContext:\n"])
                do_revise_instr = False
                if random.randint(0,3)==0:
                    do_revise_instr = True
                    tmp_instr = "Below is an instruction, followed by the context for the instruction. Deeply reflect on and then revise the below instruction to fix any issues, add depth and complexity, vary the grammar, word choice and sentence structure, and then respond to the revised instruction. Don't just repeat the original instruction:\n==\n"+ tmp_instr
                tmp_instr = tmp_instr.replace("characters, genders, nationalities, dates, time periods, and regions", random.choice(["characters, genders, nationalities, dates, time periods, and regions", "people, places and things", "objects and things", "characters, nationalities", "characters, time periods", "characters and regions"]))
                tmp_instr = tmp_instr.replace("to make them more diverse", random.choice(["to make them more diverse", "to vary the context", "to create interests", "to add depth"]))
                tmp_instr = tmp_instr.replace("Make", random.choice(["Make", "Generate", "Provide", "Improve", "Draft", "Revise"]))
                tmp_instr = tmp_instr.replace("Change", random.choice(["Change", "Edit", "Modify", ]))                                        
                tmp_instr = tmp_instr.replace("inspired by", random.choice(["inspired by", "based on", "sticking loosly", "starting from", ]))                    
                tmp_instr = tmp_instr.replace("Rewrite", random.choice(["Rewrite", "Revise", "Think deeply and create", "Greatly improve",]))
                tmp_instr = tmp_instr.replace("Revise", random.choice(["Rewrite", "Revise", "Think deeply and create", "Vastly improve",]))                    
                tmp_instr = tmp_instr.replace("document", random.choice(["document", "content", "work", "text",]))
                tmp_instr = tmp_instr.replace("email", random.choice(["email", "mail", "letter", "correspondence",]))                    
                tmp_instr = tmp_instr.replace(" text", random.choice([" text", " epic content", " very well written content",]))
                tmp_instr = tmp_instr.replace("song", random.choice(["song", "lyrics", "ballad", "poem"]))
                tmp_instr = tmp_instr.replace("story", random.choice(["story", "narrative", "fictional account", "journal"]))
                tmp_instr = tmp_instr.replace("remove", random.choice(["remove", "revise", "delete"]))
                tmp_instr = tmp_instr.replace("textbook", random.choice(["textbook", "coursebook", "handbook", "highly accurate and modern teaching materials"]))                    
                if "story" in tmp_instr or "fiction" in tmp_instr or "song" in tmp_instr or "poem" in tmp_instr:
                    tmp_instr = tmp_instr.replace("\n===\n", "Be creative and avoid cliches.\n===\n")
                if len(doc) < 2000:
                    doc = doc[:2000]
                if random.randint(0,1):
                  doc = register_text_for_upsample(doc, meta['lang'], add_trans_prob=0.5, sent_reorder_prob=0.2, sent_upsample_prob=0.2, sent_shuffle_prob=0.2)
                else:
                  doc = register_text_for_upsample(doc, meta['lang'], add_trans_prob=-1, sent_reorder_prob=0.2, sent_upsample_prob=0.2, sent_shuffle_prob=0.2)                  
                tmp_instr = tmp_instr + doc
                if random.randint(0,5) == 0:
                    tmp_instr += "<|im_start|>"
                elif random.randint(0,5) == 0:
                    tmp_instr += "<|im_start|>system\n"+random.choice(["You", "You are", "You are a", "You are a helpful", "I", "Use tools", "Reflect", "Always", "Rule", "Requirements"])+" "
                elif random.randint(0,5) == 0:
                    tmp_instr += "<|im_start|>user\n"+random.choice(["Please", "What", "Where", "When", "Why", "How", "In", "On", "Under", "By", "With"])+" "
                else:
                    tmp_instr = tokenize_with_chat_template(tokenizer, system= "You are a helpful "+random.choice(["virtual", "artificial", "synthetic"])+" "+random.choice(["assistant", "friend", "guide", "teacher", "person", "parent"]), instruction=tmp_instr)
                    if do_revise_instr:
                        tmp_instr += "### Revised Instruction:" 
                if random.randint(0,5)==0 and "### Revised Instruction:" not in tmp_instr:
                    tmp_instr = tmp_instr.replace("\n\n</think>\n\n", "")
                tmp_instr = synonym_textaugment(tmp_instr, 0.50)
                data['text'] += "<|endoftext|>" + tmp_instr
                data['metadata'].append(copy.copy(meta))                    
                number_instructions_added += 1
                if number_instructions_added >= args.create_upsample: break
                
        meta_arr = []
        text_arr = []
        for meta, text in zip(data['metadata'], data['text'].split("<|endoftext|>")):
            extra_summary = ""
            if 'kind2=distill' in meta['source']:
              extra_summary = "Distilled and short document\n"
            elif 'kind2=diverse_qa_pairs' in meta['source'] and "?" in text:
              extra_summary = "Document with questions\n"
            elif 'kind2=extract_knowledge' in meta['source']:
              extra_summary = "Knwowledge specific summary\n"
            elif 'kind2=knowledge_list' in meta['source']:
              if "1. " in text or "* " in text or "1)" in text:
                extra_summary = "List form\n"


            create_noun_hypernyms = False
            if random.randint(0,4) == 0:
              create_noun_hypernyms = True
            if len(text) < 300:
              create_noun_hypernyms = False
            if "<|im_start|>" in text:
              create_noun_hypernyms = False                    
            safety_issue, safety_score, matched_words, ner, template2, noun_hypernyms = basic_safety_processing(text, create_noun_hypernyms=create_noun_hypernyms)
            do_strict_anon = False
            added_summary = False                    
            # science and math has many numbers in the text, which could be mistaken for IDs
            if 'science' in data['domain'] or 'engine' in data['domain'] or 'ology' in data['domain'] or 'physi' in data['domain'] or "math" in data['domain']:
                do_strict_anon = True
            # we always anononimize people and PII IDs
            if "PII" in safety_issue:
              ner, text =  pii_anonymize(ner, template2, lang=meta['lang'], extended_anonymize=False, do_person=True, do_public_people=False, do_gender_swap=False, do_strict_id=do_strict_anon)
            # anonymize names and do gender swap if appropriate
            if (" born " in text[:200] and (") is a " in text[:200] or ") is an " in text[:200]) or meta['lang'] == 'en') and \
               (any('PUBLIC' in key or 'NORP' in key for key in ner) or (text.count(" him ") + text.count(" he ") + text.count(" his ") + text.count(" Him ") + text.count(" he ") + text.count(" his ") > 4)):
              ner, text1 =  pii_anonymize(ner, template2, lang=meta['lang'], extended_anonymize=True, do_person=True, do_public_people=True, do_gender_swap=random.randint(0,1), do_strict_id=do_strict_anon)
            else:
              text1 = text
            text = text.strip()
            text1 = text1.strip()
            swap = False

            # let's decide to swap the anonymized text with the non-anonymized text
            if (not noun_hypernyms and random.randint(0,1)) or  "<|im_start|>" in text:
              tmp = text1
              text1 = text
              text = tmp
              swap = True

            # only augment the non instruction text
            if "<|im_start|>" not in text:
                if text == text1 and random.randint(0,5)==0 and len(ner) > 2:
                  text = template2
                  extra_summary += "This is a template document.\n"
                instruction = ""
                if text != text1 and "hq-" in data['domain']:
                  text1 = synonym_textaugment(text1, 1.0 if not do_strict_anon else 0.5)
                if len(text) > 500:
                  summary_string, starts_with, ends_with, starts_with_para, ends_with_para = create_summary(text)
                  if starts_with and random.randint(0,1)==0:
                    lst = list(starts_with.items())
                    random.shuffle(lst)
                    s,num=lst[0]
                    instruction += f"Add {num} sentences that starts with '{s}'."
                    text= ".".join([t for t in text.split(".") if not t.strip().startswith(s)])
                  elif ends_with and random.randint(0,1)==0:
                    lst = list(ends_with.items())
                    random.shuffle(lst)
                    s,num=lst[0]
                    instruction += f"Add {num} sentences that ends with '{s}'."                        
                    text= ".".join([t for t in text.split(".") if not t.strip().endswith(s)])                        
                  elif "\n" in text and random.randint(0,1):
                    before, after = text.split("\n",1)
                    if len(before) < 100 and "." not in before:
                      instruction += "Add a short header."
                      text = after.strip()
                    elif before[-1] == "." and before.count('.')==1:
                      instruction += "Add an introductory sentence."
                      text = after.strip()
                    elif len(before) >= 100 and before.count('.') > 1:
                      instruction += "Add an introductory paragraph."
                      text = after.strip()
                  elif "\n" in text and random.randint(0,1):
                    arr = text.split("\n")
                    before = "\n".join(arr[:-1])
                    after = arr[-1]
                    if len(after) < 100 and "." not in after:
                      instruction += "Add a short footer."
                      text = before.strip()
                    elif after[-1] == "." and after.count('.')==1:
                      if "Question:" in after:
                        instruction += "Add a concluding question."
                      else:
                        instruction += "Add a concluding sentence."
                      text = before.strip()
                    elif len(after) >= 100 and after.count('.') > 1:
                      if "Question:" in after:
                        instruction += "Add a paragraph at the end with question(s)."
                      else:
                        instruction += "Add a conclusion."
                      text = before.strip()
                  elif text.count('"') >=4  and random.randint(0,1):
                      text = text.replace('."', '".').replace('?"', '?".').replace('!"', '!".')
                      text = ".".join([t for t in text.split(".") if '"' not in t]).strip()
                      instruction += "Add quotations or dialogue."
                  elif text.count('\n') > 3 and text.count("   ") >3  and random.randint(0,1):
                      text = text.replace("\n", " ").replace("  ", " ").replace("  ", " ").replace("  ", " ").strip()
                      text = ".".join([t for t in text.split(".") if '"' not in t]).strip()
                      instruction += "Add formatting."
                  elif (("\n1. " in text and "\n2. " in text) or  ("\n1) " in text and "\n2) " in text)) and random.randint(0,1):
                      text = text.replace("10. ", "").replace("1. ", "").replace("2. ", "").replace("3. ", "").replace("4. ", "").replace("5. ", "").replace("6. ", "").replace("7. ", "").replace("8. ", "").replace("9. ", "").\
                          replace("10) ", "").replace("1) ", "").replace("2) ", "").replace("3) ", "").replace("4) ", "").replace("5) ", "").replace("6) ", "").replace("7) ", "").replace("8) ", "").replace("9) ", "")
                      text = ".".join([t for t in text.split(".") if '"' not in t]).strip()
                      instruction += "Add numbering."
                text = text.strip()
                diff = get_aligned_text(text, text1)
                diff = diff[0]
                if diff[-1][0]=='':
                  diff = diff[:-1]
                diff = [[d[0].replace('**', '\n').replace('*', '\n'), d[1].replace('**', '\n').replace('*', '\n'), d[2]] for d in diff]
                diff = list(set([d[0].strip(".?!|;,")+" => "+d[1].strip(".?!|;,") for d in diff if d[-1] == 0 and d[0] != '' and  d[1] != '' ]))
                if diff:
                    if random.randint(0,1):
                        diff = str(diff)
                        diff = f"```json\n{diff}\n```"
                    else:
                        diff = "\n".join(diff)
                        replace_word =  random.choice([": ", " replace with ", " => ", " change to ", " modify as ", " as "])
                        diff = diff.replace(" => ", replace_word)
                else:
                  diff = ""

                if random.randint(0,1) or diff:
                  if random.randint(0,10) == 0:
                    text = text.upper()
                    if random.randint(0,10) != 0:
                      extra_summary += "Upper case document\n"
                      instruction += "\n"+random.choice(["Convert", "Change", "Edit", "Modify"])+" document to have normal upper and lower-casing."
                    else:
                      text = text.replace(".", "").replace("!", "").replace("?", "").replace(",", "").replace(":", "").replace(";", "")
                      extra_summary += "Upper case document with no punctuation\n"
                      instruction += "\n"+random.choice(["Convert", "Change", "Edit", "Modify"])+" document to have normal upper and lower-casing and add punctuation."                          
                  elif random.randint(0,10) == 0:
                    text = text.lower()
                    if random.randint(0,10) != 0:
                      extra_summary += "Lower case document\n"
                      instruction += "\n"+random.choice(["Convert", "Change", "Edit", "Modify"])+" document to have normal upper and lower-casing."                          
                    else:
                      text = text.replace(".", " ").replace("!", " ").replace("?", " ").replace(",", " ").replace(":", " ").replace(";", " ")
                      extra_summary += "Lower case document with no punctuation\n"
                      instruction += "\n"+random.choice(["Convert", "Change", "Edit", "Modify"])+" document to have normal upper and lower-casing and add punctuation."
                summary_string, starts_with, ends_with, starts_with_para, ends_with_para = create_summary(text)
                text = text.strip()
                text1 = text1.strip()
                if text != text1:
                    if noun_hypernyms:
                      if random.randint(0,1):
                        added_summary = True
                        text = extra_summary + ("" if not summary_string else summary_string+" ")+random.choice(["The article", "The content", "The document", "This text"])+ random.choice([" below has", " here includes", " that follows can be described with", " embodies"]) + \
                          random.choice([" the following entities", "", " this", " this information"]) +":\n```json\n"+json.dumps(noun_hypernyms)+"\n```\n===\n" + text
                      elif random.randint(0,1):
                        text = text + "<|endoftext|>" + (("Above article is " + extra_summary) if extra_summary else "") + random.choice(["The above article", "The above", "The above document", "This previous text"])+ random.choice([" has", " includes", " can be described with", " embodies"]) + \
                          random.choice([" the following entities", "", " this", " this information"]) +":\n```json\n"+json.dumps(noun_hypernyms)+"\n```"
                      elif extra_summary:
                        text = extra_summary + text
                    else:
                      if extra_summary or random.randint(0,5)==0:
                        added_summary = True
                        text = extra_summary + ("" if not summary_string else summary_string+" ") + f"\n===\n" + text                                
                    if diff and instruction:
                      if '<|endoftext|>' in text and text[-1] == "`":
                         text  += f"\n\n" + random.choice(["Modify", "Convert", "Change", "Provide", "Generate", "Make"])+ " " + random.choice(["the article", "the content", "the document", "this text"])+ f" above as follows:\n{diff}\n{instruction}<|endoftext|>"+text1
                      else:
                        if random.randint(0,1):
                         text  += f"<|endoftext|>The above "+ random.choice(["will", "must", "should"])+" be "+random.choice(["modified", "converted", "changed",])+f" as follows:\n{diff}\n{instruction}<|endoftext|>"+text1
                        else:
                         text  += f"<|endoftext|>{instruction} Also please "+random.choice(["modify", "convert", "change",])+f" the above as follows:\n{diff}<|endoftext|>"+text1      
                      #print ((text,))

                    elif diff:
                      if '<|endoftext|>' in text and text[-1] == "`":
                         text  += f"\n\n" + random.choice(["Modifying", "Converting", "Changing", "Providing", "Generating", "Making"])+ " " + random.choice(["the article", "the content", "the document", "this text"])+ f" above as follows:\n{diff}\n<|endoftext|>"+text1
                      else:
                        if random.randint(0,1):
                         text  += f"<|endoftext|>The above "+ random.choice(["will", "must", "should"])+" be " +random.choice(["modified", "converted", "changed",])+f" as follows:\n{diff}\n<|endoftext|>"+text1
                        else:
                          text  += f"<|endoftext|>Please "+random.choice(["modify", "convert", "change",])+f" the above as follows:\n{diff}`\n<|endoftext|>"+text1                                           
                    elif instruction:
                      instruction = instruction.replace("document", "above document")
                      if '<|endoftext|>' in text and text[-1] == "`":
                         text  += f"\n\n{instruction}<|endoftext|>"+text1
                      else:
                         text  += f"<|endoftext|>{instruction}<|endoftext|>"+text1

                else:
                  if extra_summary or (random.randint(0,5) == 0 and summary_string):
                    text = extra_summary + ("" if not summary_string else summary_string+" ") + "<|endoftext|>" + text                        
                if meta['lang'] != 'en' and random.randint(0,5)==0:
                  language= langs2fullname.get(meta['lang'])
                  if language:
                    text = "Language: "+ language + "\n"+text
                if random.randint(0,5)==0 or "fiction" in data['domain']:
                  text = "Domain: "+data['domain'].replace("hq-","").replace("_", " ").replace("-", " ") + "\n"+text

            text_arr.append(text)
            meta['safety_issue'] = [safety_issue, safety_score, matched_words]
            meta['ner'] = ner
            meta_arr.extend([meta]*(1+text.count("<|endoftext|>")))

        if not meta_arr: continue
        data['text'] = "<|endoftext|>".join([t for t in text_arr if t.strip()])
        data['metdata'] = meta_arr
        data['safety_issue'] = [safety_issue, safety_score, matched_words]
        data['ner'] = ner
        # assume each text item is pretty long already - at least 2000 tokens, we can efficiently batch the whole text item instead of batching across
        data = add_translations(data)
        orig_data = data
        data = dedup(data)
        if not data:
          print ("dup", orig_data)
          continue
        batch.append(data)                    

        upsample_batch.append(data)
        if len(upsample_batch)==400:
          generate_upsample(upsample_batch)
          upsample_batch = []
        

    generate_upsample(upsample_batch)
    with open(file.replace(args.input_dir, args.output_dir), "a+") as outf:
        for data in batch:
            #if "<|im_start|>" in data['text']:
            #    print (data)
            outf.write(json.dumps(data)+"\n")
    pid = str(get_rank())+str(os.getpid())            
    os.system("mkdir -p "+ "/".join(file.replace(args.input_dir, args.input_dir+"/done/"+pid+"/").split("/")[:-1]))
    file3 = file.replace(args.input_dir, args.input_dir+"/done/"+pid+"/")
    os.system("mv "+file + " " +file3)            

    #with open(file.replace(args.input_dir, args.output_dir).replace(".jsonl", ".reduced"), "w"): pass
        

if __name__ == "__main__":
    args = parse_args()
    args.input_dir = args.target_dir+"4/"
    args.output_dir = args.target_dir+"5/"
    subset= args.subset
    if subset:
        args.output_dir = args.output_dir.rstrip("/")+"_"+subset+"/"
        args.input_dir = args.input_dir.rstrip("/")+"_"+subset+"/"    
    print (args)
    os.system(f"mkdir -p {args.output_dir}")
    os.system("rm -rf "+args.output_dir+"/"+str(get_rank())+".rank_done")            
    if True: # not os.path.exists(args.output_dir+"/"+str(get_rank())+".rank_done"):
        args.all_files =[]
        args.all_files.extend(list(set(list(glob.glob(args.input_dir + '*.jsonl', recursive=True)) +  list(glob.glob(args.input_dir + '*/*.jsonl', recursive=True)) +  list(glob.glob(args.input_dir + '*/*/*.jsonl', recursive=True)))))
        args.all_files = [file for file in args.all_files if "/done/" not in file]        
        args.all_files.sort()
        ws = get_world_size()
        rank = args.rank = get_rank()
        print ("starting rank", rank, args.all_files)
        rank2files = {}
        j = -1
        for file in args.all_files:
            j += 1
            for k in range(ws):
                if j == k:
                    p = rank2files[k] = rank2files.get(k,[])
                    p.append(file)
                    if j == ws-1:
                        j = -1
                    break
        if rank in rank2files:
          files = rank2files[rank]
          print ("files for rank", rank, files)

          random.shuffle(files)
          files = [(file, i%num_devices, args) for i, file in enumerate(files)]        
          with multiprocessing.Pool(10 if num_devices <= 1 else num_devices) as pool:    
              for _ in pool.imap_unordered(process, files):
                  pass
        wait_for_other_ranks(args.output_dir)        
