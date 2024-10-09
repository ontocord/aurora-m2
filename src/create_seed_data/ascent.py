#@title ascent
import os
import gzip
import json
try:
  from datasets import load_dataset
except:
  !pip install -q datasets
from datasets import load_dataset
try:
  if dataset is None: assert False
except:
  dataset0 = load_dataset("ascent_kb", "open")
  dataset0 = dataset0.sort("subject")
  dataset = load_dataset("ascent_kb", "canonical")
  dataset = dataset.sort("arg1")

try:
  if not aHash1: assert False
except:
  aHash1 = {}
  for subject, obj, predicate in zip(dataset0['train']['subject'],dataset0['train']['object'], dataset0['train']['predicate']):
    aHash1[(subject, obj)] = predicate

prev_subject = ""

aHash = {}
for idx, dat in enumerate(dataset['train']):
  if prev_subject and dat['arg1'] not in prev_subject:
    seen = {}
    if len(prev_subject) <= 2 or len(aHash) < 4:
      aHash = {}
    else:
      aList = list(aHash.items())
      aList.sort(key=lambda a: a[1]['rel']+":"+aHash1.get(a[0], ''))
      title=False
      prev_rel = ""
      for key, dat2 in aList:
        if dat2['source_sentences'] and not any(a for a in  dat2['source_sentences'] if dat2['arg1'][:min(len(dat2['arg1']), 4)] in a['text'].lower()):
          #print ('not found', dat2)
          continue
        if not title:
          print ('##', dat2['arg1'])
          title=True
        if dat2['rel'] != prev_rel:
          print ('###', dat2['rel'])
          prev_rel = dat2['rel']
        if aHash1.get(key) != None:
          pred =  aHash1.get(key)
          if pred == "be":
            pred = "is"
          pred = pred.replace("be ", "is ")
          facets = [a['value'] for a in dat2['facets'] if " i " not in a['value'] and " you " not in a['value'] and " me " not in a['value'] and a['value'] != 'in that']
          facets.sort(key=lambda a: len(a))
          if pred != "is" or facets:
            sent = " ".join([a['text'] for a in dat2['source_sentences'] if 'wikipedia' in a['source'] and  dat2['arg1'][:min(len(dat2['arg1']), 4)] in a['text'].lower()])
            if sent:
              if sent not in seen:
                print (sent)
              seen[sent] = 1
      if False:
        for key, dat2 in aList:
          if dat2['source_sentences'] and not any(a for a in  dat2['source_sentences'] if dat2['arg1'][:min(len(dat2['arg1']), 4)] in a['text'].lower()):
            #print ('not found', dat2)
            continue
          if not title:
            print ('##', dat2['arg1'])
            title=True
          if dat2['rel'] != prev_rel:
            print ('###', dat2['rel'])
            prev_rel = dat2['rel']
          if aHash1.get(key) != None:
            pred =  aHash1.get(key)
            if pred == 'has subgroup':
              sent = dat2['arg2']+' is a type of '+ dat2['arg1']
              if sent not in seen:
                print (sent)
              seen[sent] = 1
            elif pred == 'has aspect' and "'s" in dat2['arg2']:
              sent = dat2['arg2'].replace("'s", " has")
              if sent not in seen:
                print (sent)
              seen[sent] = 1
            else:
              if pred == "be":
                pred = "is"
              pred = pred.replace("be ", "is ")
              facets = [a['value'] for a in dat2['facets'] if " i " not in a['value'] and " you " not in a['value'] and " me " not in a['value'] and a['value'] != 'in that']
              facets.sort(key=lambda a: len(a))
              if pred != "is" or facets:
                sent = " ".join([a['text'] for a in dat2['source_sentences'] if 'wikipedia' in a['source']])
                if sent:
                  if sent not in seen:
                    print (sent)
                  seen[sent] = 1
                elif  len(dat2['source_sentences']) <= 1:
                  continue
                elif len(facets) == 1 and " " not in facets[0]:
                  sent = dat2['arg1']+" "+pred+" "+facets[0]+" "+dat2['arg2']
                  if sent not in seen:
                    print (sent)
                  seen[sent] = 1
                elif facets and " " not in facets[0]:
                  sent = dat2['arg1']+" "+pred+" "+facets[0]+" "+dat2['arg2']+" "+", ".join(facets[1:min(len(facets), 3)])
                  if sent not in seen:
                    print (sent)
                  seen[sent] = 1
                elif facets:
                  sent = dat2['arg1']+" "+pred+" "+ dat2['arg2']+" "+", ".join(facets[:min(len(facets), 3)])
                  if sent not in seen:
                    print (sent)
                  seen[sent] = 1
                else:
                  sent = dat2['arg1']+" "+pred+" "+ dat2['arg2']
                  if sent not in seen:
                    print (sent)
                  seen[sent] = 1
                #if dat2['rel'] == '/r/AtLocation':
                #  sent = dat2['arg1']+ ' is located in '+ dat2['arg2']
                #  if sent not in seen:
                #    print (dat2['rel'], sent)
                #  seen[sent] = 1
        aHash = {}
    if idx > 1000: break

  prev_subject = dat['arg1']
  aHash[(dat['arg1'], dat['arg2'])] = dat
