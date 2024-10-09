#@title genericskb to text dataset
import json
import os
import gzip
import random
if not os.path.exists("generics_kb"):
  !git clone https://huggingface.co/datasets/generics_kb

if not os.path.exists("/content/drive"):
  from google.colab import drive
  drive.mount('/content/drive')

stop_words = {'common', 'another', 'always', 'need', 'about',  'because', 'through', "isn't", 'look', "shouldn't", "i've", 'yourself', 'wouldn', 'else', 'against', "mustn't", 'may', 'named', 'three', 'bring', 'bit', 'my', 'below', 'ma', 'everything', 'o',  'could', 'ready', 'followed', 'mr', 'at', 'to', 'was', 'happy', 'open', 'would', 'say', 'describing', 'weren', 'still', 'more', 'we', 'ever', 'doing',  'please',  'this', 'won', 'all', 'does', 'nor', 'can', 'right', 'hasn', 'illustrating', 'consisting', 'move', 'went', 'someone', 'isn', 'saying', 'find', 'ago', 'thank', 'yea', 'whom', 'they', 'want', 'lost', 'developed', 'of', 'aren', 'whatever', 'some', 'it', 'bad', "you'll", 'why', 'keep', "I'm", 'mine',  'too', 'big', 'true', 'searching', "that'll", 'wasn', 'as', 'including', 'idea', "you'd", 'ain', 'y', 'every', 'heard', 'him', '"', 'm', 'sir', 'one', 'them', 'long', 'by', 'turn', 'next', 'theirs', "wasn't", 'regarding', "he's", "hadn't", 'come', 'had', 'having', "i'm", "can't", 'yet', 'll', 'same', 'reserved', 'also', 'above', 'give', 'got', 'once', 'yourselves', 'sure', 'hers', 'and', 'few', 'those', 'shan', 'feel', 'is', 'let', 'do', 'mrs', 'yes', "mightn't", 'will', 'which', 'own', 'meet', 'just', 'am', 'guys', 'each', 'since', 'between', 'know', 'start', 'something', 'might', "couldn't", "let's", 'hey', 'any', 'illustrated', 'two', 'again', 'up', '.', 'continued', 'me', 'writing', 'ourselves', 'guess', 'search', 'but', 'there', 'not', 'didn', 'maybe', 'online',  'make', 'did', 'stay', 'have', 'be', 're', "didn't", 'nice', 'says', 'shared', 'rights', 'gone', 'last', 'die', 'if', 'haven', 'hope', 'comprising', 'seen', 'when', 'consisted', "you've", 'over', 'naming', 'first', "don't", "needn't", 'nothing', 'off', 'here', 'these', 'from', 'she', 'use', 'gonna', 'an', 'said', 'our', 'even', 'being', 'place', 'very', 'doesn', 'needn', 'their', 'going', 'her', 'on', 'herself', 'than', 'away', 'into', 'phone', 'yeah', '!', 'under', "I've", 'okay', 'tell', 'name', 'play', 'such', "doesn't", 'days', 'further', 'called', 'getting', 'however', "i'll", 'see', '?', 'whole', 'job', 'yours', 'developing', 'his', 'head', 'hard', 'way', 'hello', 'what', 'good', 'where', "you're", 'or', 'ours', "won't", 'been', 'get', 'myself', 'with', "it's", 'until', 'in', 'wanna', 'were', "hasn't", 'should', 'like', 'thanks', 'couldn', "that's", 'gotta', 'knew', 'he', 'how', 't', 'fine', 'mean', 'no', 'described', 'life', 'mustn', 'who', "aren't", 'mightn', 'sorry', 'thing', 'while', 'put', 'during', 'other', 'woman', 'comprised', 'its', 'following', 've', 'found', 'much', 'wait', 'you', "haven't", 'never', 'mom', 'hold',  'most', 'contact', 'i', 'took', "shan't",  'after', 'left', 'user', 'must', 'so', 'the', 'new', 'lot', 'room', 'used', 'that', 'd', 'care', 'boy', 'out', 'guy', 'back', 'huh',  'hear', 'continuing', 'ask', 'done', 'for', 'indeed', 'wants', 'well', "wouldn't", 'told', 's', 'are', 'old', "'", 'nonetheless', 'both', 'night', 'your', 'down', 'themselves', 'miss', 'think', 'before', 'a', 'run', 'alone', 'included', 'don', "there's", 'men', 'now', 'came', 'only', 'himself', 'try', 'then', 'itself', 'according', 'shouldn', 'looks', 'work', 'wrong', 'happened', 'has', "i'd", 'made', "she's", "should've", 'kind', 'saw', 'baby', ':', "weren't", 'hadn'}
if True:
  aHash = {}
  concepts = {}
  categories = []
  titles = []
  headings = []

  prev_ent = ""
  for i, l in enumerate(gzip.open("/content/generics_kb/data/GenericsKB-Best.tsv.gz")):
    if i == 0: continue
    l = l.decode()
    #if i > 500: break
    l_arr = l = l.split("\t")
    source, ent, qual, sent, score =  l_arr
    sent = sent.replace(" .", ".").strip()
    # if we want broader coverage, get everything except conceptnet which has too many problems
    # copyright permissive only
    #if "Concept" not in source and "Wiki" not in source and "Word" not in source: continue
    if "Concept" in source:
      if len(ent) <= 2: continue
      if 'Concept' in source and (len(ent) < 4 or 'men ' in sent or "Men " in sent or "sex" in sent or "Men" in ent or "Women" in ent or "sex" in ent or "Sex" in ent):
        continue
        #print (sent)
    ent_lower = ent.lower()
    if "prostitution" in ent_lower or "sodomy" in ent_lower or "homosex" in ent_lower or "sex" in ent_lower or "lesbian" in ent_lower or " rape" in ent_lower or \
      "molest" in ent_lower or "porn" in ent_lower or  "prostitute" in ent_lower or "fuck" in ent_lower or ent_lower == "rape" or \
      "incest" in ent_lower or "streetwalker" in ent_lower or "whore" in ent_lower or "hooker" in ent_lower or \
      ent_lower in {'prostitute', "fuck", "fucking", "whore", "cunt", "pussy", "cock", "nigger", "terrorist"}:
      if 'Wiki' not in source and "Word" not in source: continue

    #if not ('Word' in source or 'Simple' in source):
    #  continue
    if len(ent) > 2:
      aHash[ent] = aHash.get(ent,'')+"\n* "+ sent



  for i, l in enumerate(gzip.open("/content/generics_kb/data/GenericsKB-SimpleWiki-With-Context.jsonl.gz")):
    l = l.decode()
    #if i > 500: break
    dat = json.loads(l)
    sent = ""
    ent = [a['concept_name'] for a in dat['knowledge']['key_concepts']][0]

    #if 'title' in dat['knowledge']['context'] and dat['knowledge']['context']['title']:
    #    sent += "\n* Title: "+", ".join(dat['knowledge']['context']['title'])
    if 'headings' in dat['knowledge']['context'] and dat['knowledge']['context']['headings']:
        heading = ", ".join(dat['knowledge']['context']['headings'])
        if "Template" not in heading and "Events" not in heading and "Wikipedia" not in heading and heading not in aHash.get(ent,''):
          sent += "\n+ "+ heading
          if 'categories' in dat['knowledge']['context'] and dat['knowledge']['context']['categories']:
            categories = " :: ".join(dat['knowledge']['context']['categories'])
            categories = categories.replace("s which is", "s which are")
            if categories not in sent and categories not in aHash.get(ent,''):
              sent += ": " + categories
    sent2 = (" ".join(dat['knowledge']['context']['sentences_before']) + " " + dat['knowledge']['sentence'] + " " + " ".join(dat['knowledge']['context']['sentences_after'])).strip()
    sent2 = sent2.replace(" .", ".").strip()
    if sent2 not in aHash.get(ent,''):
      sent += "\n  * "+ sent2
    if len(ent) > 2:
      aHash[ent] = aHash.get(ent,'') + sent

  text = ""
  hype = {}
  for key2, val2 in aHash.items():
    key2 = key2.strip()
    val2 = val2.strip()
    if "isa thing." in val2 and "is part of" in val2:
      val2 = val2.replace("isa thing.", "isa location.")
    val = val2.lower()
    key = key2.lower()
    if key+' isa ' in val:
      parent = val.split(key+" isa ",1)[1].split("\n")[0].split(".")[0]
      if len(parent) > 2 and parent.split()[0] not in stop_words and parent.count(" ") < 3 and \
        (parent.count(" ") < 2 or parent.split()[1] not in stop_words) and parent not in stop_words and "oxymoron" not in parent:
        parent = parent.replace("type of ", "").replace("'", "").strip()
        hype[parent] = hype.get(parent, [])+[(key2, val2)]
        continue
    if key+' is a ' in val:
      parent = val.split(key+" is a ",1)[1].split("\n")[0].split(".")[0]
      if len(parent) > 2 and parent.split()[0] not in stop_words and parent.count(" ") < 3 and \
        (parent.count(" ") < 2 or parent.split()[1] not in stop_words) and parent not in stop_words and "oxymoron" not in parent:
        parent = parent.replace("type of ", "").replace("'", "").strip()
        hype[parent] = hype.get(parent, [])+[(key2, val2)]
        continue
    if key+' is an ' in val:
      parent = val.split(key+" is an ",1)[1].split("\n")[0].split(".")[0]
      if len(parent) > 2 and parent.split()[0] not in stop_words and parent.count(" ") < 3 and \
        (parent.count(" ") < 2 or parent.split()[1] not in stop_words) and parent not in stop_words and "oxymoron" not in parent:
        parent = parent.replace("type of ", "").replace("'", "").strip()
        hype[parent] = hype.get(parent, [])+[(key2, val2)]
        continue
    if key+' is ' in val:
      parent = val.split(key+" is ",1)[1].split("\n")[0].split(".")[0]
      if len(parent) > 2 and parent.split()[0] not in stop_words and parent.count(" ") < 3 and \
        (parent.count(" ") < 2 or parent.split()[1] not in stop_words) and parent not in stop_words and "oxymoron" not in parent:
        parent = parent.replace("type of ", "").replace("'", "").strip()
        hype[parent] = hype.get(parent, [])+[(key2, val2)]
        continue
    if key+'s are ' in val:
      parent = val.split(key+"s are ",1)[1].split("\n")[0].split(".")[0]
      if len(parent) > 2 and parent.split()[0] not in stop_words and parent.count(" ") < 3 and \
        (parent.count(" ") < 2 or parent.split()[1] not in stop_words) and parent not in stop_words and "oxymoron" not in parent:
        parent = parent.replace("type of ", "").replace("'", "").strip()
        hype[parent] = hype.get(parent, [])+[(key2, val2)]
        continue
    if key+'es are ' in val:
      parent = val.split(key+"es are ",1)[1].split("\n")[0].split(".")[0]
      if len(parent) > 2 and parent.split()[0] not in stop_words and parent.count(" ") < 3 and \
        (parent.count(" ") < 2 or parent.split()[1] not in stop_words) and parent not in stop_words and "oxymoron" not in parent:
        parent = parent.replace("type of ", "").replace("'", "").strip()
        hype[parent] = hype.get(parent, [])+[(key2, val2)]
        continue
      #print ((key, val))
  hypo = {}

  for parent, val in hype.items():
    parent = parent.replace("type of ", "").replace("'", "").strip()
    for child in val:

      hypo[child[0]] = parent

  for key, val in aHash.items():
    if key not in hypo:
      if " " in key:
        parent = key.split()[-1].lower()
        if parent in hypo:
          parent = parent.replace("type of ", "").replace("'", "").strip()
          hype[parent] = hype.get(parent, [])+[(key, val)]
          continue
        elif parent+"s" in hypo:
          parent = parent.replace("type of ", "").replace("'", "").strip()
          hype[parent+"s"] = hype.get(parent+"s", [])+[(key, val)]
          continue
        elif parent[:-1] in hypo:
          parent = parent.replace("type of ", "").replace("'", "").strip()
          hype[parent[:-1]] = hype.get(parent[:-1], [])+[(key, val)]
          continue
        else:
          parent = key.split()[0].lower()
          parent = parent.replace("type of ", "").replace("'", "").strip()
          if parent in hypo:
            hype[parent] = hype.get(parent, [])+[(key, val)]
            continue
          elif parent+"s" in hypo:
            parent = parent.replace("type of ", "").replace("'", "").strip()
            hype[parent+"s"] = hype.get(parent+"s", [])+[(key, val)]
            continue
          elif parent[:-1] in hypo:
            parent = parent.replace("type of ", "").replace("'", "").strip()
            hype[parent[:-1]] = hype.get(parent[:-1], [])+[(key, val)]
            continue
      if "\n+" in val:
        found = False
        for parent in val.split("\n+",1)[-1].split("\n")[-1].lower().split(":"):
          parent = parent.strip()
          parent = parent.replace("type of ", "").replace("'", "").strip()
          if parent in hypo:
            hype[parent] = hype.get(parent, [])+[(key, val)]
            found = True
            break
          elif parent+"s" in hypo:
            hype[parent+"s"] = hype.get(parent+"s", [])+[(key, val)]
            found = True
            break
          elif parent[:-1] in hypo:
            hype[parent[:-1]] = hype.get(parent[:-1], [])+[(key, val)]
            found = True
            break
        if found: continue
      parent = 'entity'
      hype[parent] = hype.get(parent, [])+[(key2, val2)]
      #if key.lower() not in hypo:
      #  print ((key, val))

  # cluster all sentences. collapse all X is Y1, X is Y2 -> X is Y1, Y2

  hype2 = {}
  for parent, val in hype.items():
        parents = [parent]
        for _ in range(5):
          if parent in hypo:
            parent = hypo[parent]
            if parent == "entity": break
            parents.append(parent)
            continue
          break
        #if len(parents) == 1: continue
        parent = " | ".join(reversed(parents))
        hype2[parent] = val

  hypo = {}
  cnts = {}
  for parent, val in hype2.items():
    for child in val:
      hypo[child[0]] = parent
      for p in parent.split(" | "):
        cnts[p] = cnts.get(p,0) + 1
  trim_cutoff = len(aHash)*.1
  trim_parents = set([a for a, b in cnts.items() if b > trim_cutoff])
  text = ""
  items = list(aHash.items())
  items0 = []
  items1 = []
  for key, val in items:
    add_to_items0 = "Basic English 850" in val
    val = val.replace(" Basic English 850 words ::", "").replace("Basic English 850 words ::", "").replace("Basic English 850 words which are related to", "").replace(" Basic English 850 words", "").replace("Basic English 850 words", "").replace("Basic English 850 words ", "").replace("has (part)", "has a part called")
    val = val.replace(" ::\n", "\n")
    key2 = hypo.get(key, hypo.get(key.lower()))
    if not key2:
      key2 = key
    else:
      add_to_items0= True
      key2 = (key2 + " | " + key.lower()).strip()
    #del aHash[key]
    key2 = key2.replace("activity | job | craft | aircraft", "craft | aircraft").replace("object | ", "").\
          replace("very common | ", "").replace("substance | material | chemical | ", "").\
          replace("individual | ", "person | ").replace("person | adult | professional | ", "person | professional | ").\
          replace("event | activity | crime | attack |", "event | activity | attack |").\
          replace("cheating | fraud", "crime | cheating | fraud").\
          replace("chemical | compound |", "chemical compounds |").replace("body part | appendix", "appendix").\
          replace("body part | arch", "arch").replace("body part | small", "small").replace("accessories | fixing", "accessories | fastener").\
          replace("concept | abstraction |", "abstract concept |").replace("acting |", "act | acting |").\
          replace("actions |", "act |").replace("adjectives |", "concept | attribute | feature").\
          replace("assessment | rating | mark | diacritic | accent mark | acute |", "disease |").\
          replace("art | fortification |", "fortification |").replace("art | genre |", "art genre |").\
          replace("argument between places | travel", "travel").replace("attack | occlusion", "occlusion").\
          replace("ancient thinking | deprivation", "deprivation").replace("arrival | admission | confession", "confession").\
          replace("autococker term | timing", "timing").replace("balancing energy | healing", "healing").\
          replace("relation | possession | expenditure | payment |", "payment |").replace("crime |crime |", "crime |").\
          replace("case in point | ", "").replace("burden on families | ", "").replace("alignment | true", "alignment").\
          replace("arrow | complication", "complication").replace("bodhisattvas purified buddhaland | wisdom", "wisdom").\
          replace("business in mexico | extortion", "extortion").replace("chance | shot", "shot").\
          replace("celtic folk | celtic folk", "celtic folk").replace("children | prostitute", "prostitute").\
          replace("biblical concept | capitalism", "capitalism").replace("badness | disadvantage", "disadvantage").\
          replace("air mass | low", "air mass").replace("arrow | perturbation", "perturbation").\
          replace("anticipation | prophecy", "prophecy").replace("control | ", "").\
          replace("euphemism for totalitarianism | consistency", "consistency").\
          replace("enthusiast | addict", "addict").replace("disgust | disgust", "disgust").\
          replace("digit | pair", "pair").replace("dead-end street | protectionism", "protectionism").\
          replace("crime |crime |", "crime |").replace("employers ideology | conservatism", "conservatism").\
          replace("godly vocation | parenthood", "parenthood").replace("girl | demoiselle | anemone fish", "anemone fish").\
          replace("fact of life | insecurity", "insecurity").replace("typical | typical of anorexia |", "anorexia").\
          replace("question of conscience | intonation", "intonation").replace("invention of idealists | insanity", "insanity").\
          replace("form of wordism | communism", "communism").replace("public transport | local", "public transport").\
          replace("shot | marksman", "marksman").replace("sex organ | fanny", "sex organ | genitalia").replace("servant | domestic | ", "").\
          replace("sense organ | optic", "optic").replace("sculpture | mobile | ", "").replace("rope | hemp", "hemp").replace("reversible | ", "").\
          replace("reaper | combine | recombination","recombination").replace("prediction | projection", "projection").\
          replace("voice | passive | ", "").replace("type | ", "").replace("thief | dip", "dip").replace("subject | ", "").\
          replace("sphere | drop", "drop").replace("young | young mammal", "young mammal").replace("waterproof | slicker", "slicker").\
          replace("waterway | rapid ", "rapids ").replace("warrior | irregular | ", "").\
          replace("warrior | ", "").replace("tense | imperfect | ", "").replace("tense | progressive | ", "").\
          replace("abnormal, unnatural behavior | homosexuality", "homosexuality").replace("acquired habit | sleep", "sleep").\
          replace("abstraction | abstract concept", "abstract concept").replace("abstract concept | abstract concept", "abstract concept").\
          replace("abstract concepts", "abstract concept").replace("abstract concept | concept", "abstract concept").replace("abstract concept | abstract concept", "abstract concept").\
          replace("abstract concept | framework", "framework").replace("abnormal | spinal fluid", "spinal fluid").\
          replace("structure | outlet | store | fat store", "fat store").replace("doctors | human female", "woman").\
          replace("image | oxymoron | soft rock", "soft rock").replace("artifact | art | painting | fresco | queso fresco", "queso fresco").\
          replace("perforation | photo", "artifact | photo").replace("artifact | passageway | orifice", "body part | orifice").\
          replace("pathway | tunnel", "tunnel").replace("artifact | pathway", "pathway").\
          replace("passageway | corridor | fistula", "body part | fistula").replace("membrane", "body part | membrane").\
          replace("instrumentality | device | bomb | blockbuster", "blockbuster").\
          replace("instrumentality | device | circuit | clipper", "device | clipper").\
          replace("accessories | fastener | loop | busy loop", "busy loop").\
          replace("accessories | fastener | loop | coronal loop", "coronal loop").\
          replace("accessories | fastener | loop | extracellular loop", "extracellular loop").\
          replace("accessories | fastener | loop | infinite loop", "infinite loop").\
          replace("sin | inaccuracy", "inaccuracy").replace("advancing deterioration process | individualism", "individualism").\
          replace("adversity | affliction | calvary", "calvary").replace("afflictions | cross", "cross").\
          replace("age-old process | recycling", "recycling").replace("agent | booker", "booker").\
          replace("agent | handler", "person | handler").replace("agent | inducer", "person | inducer").\
          replace("animal | vertebrate | mammal | feline | cat | tom | egyptian tomb", "artefact | tomb | egyptian tomb").\
          replace("animal | vertebrate | mammal | feline | cat | tom | royal tomb", "artefact | tomb | royal tomb").\
          replace("animal | vertebrate | mammal | feline | cat | tom | sarmatian tomb", "artefact | tomb | sarmatian tomb").\
          replace("animal | vertebrate | mammal | feline | cat | tom | wedge tomb", "artefact | tomb | wedge tomb").\
          replace("areas | open | ", "").replace("arrangements | ordering | commanding", "act | commanding").\
          replace("arrangements | ordering | military order", "act | military order").\
          replace("article | newspaper article | personal | ", "").\
          replace("board | job board", "job board").\
          replace("board | medical specialty board", "medical specialty board").\
          replace("board | message board", "message board").\
          replace("board | advisory board", "advisory board").\
          replace("board | board of directors", "board of directors").\
          replace("board | board of education", "board of education").\
          replace("board | board of trustees", "board of trustees").\
          replace("board | bulletin board", "bulletin board").\
          replace("board | currency board", "currency board").\
          replace("board | directorate", "directorate").\
          replace("board | draft board", "draft board").\
          replace("canal | airway", "body part | airway").\
          replace("block | sympathetic block", "sympathetic block").\
          replace("block | programming block", "programming block").\
          replace("block | text block", "text block").\
          replace("block | writers block", "writers block").\
          replace("canal", "body part | body canal").\
          replace("cord | string | cello", "artifact | string instrument | cello").\
          replace("cord | string | viol", "artifact | string instrument | viol").\
          replace("cord | string | dominant string", "artifact | string instrument | dominant string").\
          replace("cord | string | character string", "character string").\
          replace("cord | string | primitive string", "primitive string").\
          replace("cord | string | initialization string", "initialization string").\
          replace("cord | string | filename extension", "filename extension").\
          replace("cord | string | wide string", "wide string").\
          replace("clothing | black", "black").replace("artifact | clothing | blue", "blue").\
          replace("clothing | drag", "drag").\
          replace("facility | station | sampler", "sampler").\
          replace("clothes | garment | neckwear | tie | copulatory tie", "social group | copulatory tie").\
          replace("clothes | garment | neckwear | tie | economic tie", "social group | economic tie").\
          replace("clothes | garment | neckwear | tie | ethnic tie", "social group | ethnic tie").\
          replace("clothes | garment | neckwear | tie | family tie", "social group | family tie").\
          replace("clothes | garment | neckwear | tie | generational tie", "social group | generational tie").\
          replace("clothes | garment | neckwear | tie | hyperlocal tie", "social group | hyperlocal tie").\
          replace("clothes | garment | neckwear | tie | social tie", "social group | social tie").\
          replace("clothes | garment | neckwear | tie | spiritual tie", "social group | spiritual tie").\
          replace("clothes | garment | neckwear | tie | stalemate", "social group | stalemate").\
          replace("clothes | garment | neckwear | tie | strong tie", "social group | strong tie").\
          replace("registration", "registration").replace("artifact | entry", "entry").\
          replace("commodity | stock | fish stock", "fish stock").\
          replace("commodity | stock | beef broth", "beef broth").\
          replace("commodity | stock | chicken broth", "chicken broth").\
          replace("commodity | stock | root stock", "root stock").\
          replace("commodity | stock", "stock").\
          replace("film | abdominal film", "device | abdominal film").\
          replace("film | acetate film", "device | acetate film").\
          replace("film | carbon film", "device | carbon film").\
          replace("film | chromogenic film", "device | chromogenic film").\
          replace("film | dense film", "device | dense film").\
          replace("film | digital film", "device | digital film").\
          replace("film | edible film", "device | edible film").\
          replace("film | epoxy film", "device | epoxy film").\
          replace("film | fast film", "device | fast film").\
          replace("film | faster film", "device | faster film").\
          replace("film | freezing film", "device | freezing film").\
          replace("film | heatpressed film", "device | heatpressed film").\
          replace("film | metal film", "device | metal film").\
          replace("film | intraoral film", "device | intraoral film").\
          replace("film | magnetoelastic film", "device | magnetoelastic film").\
          replace("film | makrofol film", "device | makrofol film").\
          replace("film | microfilm", "device | microfilm").\
          replace("film | mulch film", "device | mulch film").\
          replace("film | multilayered film", "device | multilayered film").\
          replace("film | negative film", "device | negative film").\
          replace("film | organic film", "device | organic film").\
          replace("film | orthochromatic film", "device | orthochromatic film").\
          replace("film | oxide film", "device | oxide film").\
          replace("film | panchromatic film", "device | panchromatic film").\
          replace("film | pet film", "device | pet film").\
          replace("film | plain film", "device | plain film").\
          replace("film | safety film", "device | safety film").\
          replace("film | slide film", "device | slide film").\
          replace("film | panchromatic film", "device | panchromatic film").\
          replace("film | ", "art | film | ").\
          replace("albums | glitter | ", "").\
          replace("albums | jealousy | ", "").replace("albums | live | ", "").\
          replace("albums | more | ", "").replace("artifact | channel | ", "channel | ").\
          replace("container | bicycle", "vehicle | bicycle").\
          replace("container | forklift", "vehicle | forklift").\
          replace("container | bike", "vehicle | bike").\
          replace("container | car", "vehicle | car").\
          replace("container | carriage", "vehicle | carriage").\
          replace("container | vessel | fishing boat", "vessel | fishing boat").\
          replace("container | vessel | iceboat", "vessel | iceboat").\
          replace("container | vessel | merchant vessel", "vessel | merchant vessel").\
          replace("container | vessel | motor vessel", "vessel | motor vessel").\
          replace("container | vessel | passenger vessel", "vessel | passenger vessel").\
          replace("container | vessel | patrol boat", "vessel | patrol boat").\
          replace("container | vessel | sailing ship", "vessel | sailing ship").\
          replace("container | vessel | sailing vessel", "vessel | sailing vessel").\
          replace("container | vessel | shallow vessel", "vessel | shallow vessel").\
          replace("container | vessel | shrimper", "vessel | shrimper").\
          replace("container | vessel | sailing ship", "vessel | sailing ship").\
          replace("container | vessel | thebesian vessel", "body part | thebesian vessel").\
          replace("container | vessel | umbilical vessel", "body part | umbilical vessel").\
          replace("container | case | ", "").\
          replace("container | drawer | ","person | artist | ").\
          replace("container | chariot", "vehicle | chariot").\
          replace("container | go-cart", "vehicle | go-cart").\
          replace("container | go-kart", "vehicle | go-kart").\
          replace("container | gondola", "vehicle | gondola").\
          replace("container | hearse", "vehicle | hearse").\
          replace("container | hearse", "vehicle | hearse").\
          replace("container | locomotive", "vehicle | locomotive").\
          replace("container | motor home", "vehicle | motor home").\
          replace("container | motor scooter", "vehicle | motor scooter").\
          replace("container | motorbike", "vehicle | motorbike").\
          replace("container | motor vehicle", "vehicle | motor vehicle").\
          replace("container | motorcar", "vehicle | motorcar").\
          replace("container | motorcycle", "vehicle | motorcycle").\
          replace("container | pushchair", "vehicle | pushchair").\
          replace("container | railcar", "vehicle | railcar").\
          replace("container | recreational vehicle", "vehicle | recreational vehicle").\
          replace("container | scooter", "vehicle | scooter").\
          replace("container | snowplow", "vehicle | snowplow").\
          replace("container | streetcar", "vehicle | streetcar").\
          replace("container | stroller", "vehicle | stroller").\
          replace("container | railroad car", "vehicle | railroad car").\
          replace("container | tracked vehicle", "vehicle | tracked vehicle").\
          replace("container | tractor", "vehicle | tractor").\
          replace("container | tricycle", "vehicle | tricycle").\
          replace("container | trolley", "vehicle | trolley").\
          replace("container | truck", "vehicle | truck").\
          replace("container | velocipede", "vehicle | velocipede").\
          replace("container | unicycle", "vehicle | unicycle").\
          replace("container | wagon", "vehicle | wagon").\
          replace("device | igniter", "igniter").\
          replace("device | accord", "accord").\
          replace("device | act", "act").\
          replace("device | contraceptive", "contraceptive").\
          replace("device | machine | assembly | legislative body", "social group | legislative body").\
          replace("device | machine | assembly | legislature", "social group | legislature").\
          replace("device | machine | assembly | messianic assembly", "social group | messianic assembly").\
          replace("device | machine | assembly | sabbat", "social group | sabbat").\
          replace("device | machine | assembly | tribunal", "social group | tribunal").\
          replace("written language | transcription | ", "").\
          replace("classes | craft", "vehicle").\
          replace("transport | vehicle |", "vehicle |").\
          replace("assessment | assay", "device | assay").\
          replace("belief | possibility | shot", "shot").\
          replace("trait | propriety | reserve", "reserve").\
          replace("example | sample", "sample").\
          replace("evidence | symptom", "symptom").\
          replace("covering | scale | balance", "balance").\
          replace("currency | specie ", "species ").\
          replace("device | instrument | warrant", "written communication | warrant").\
          replace("device | instrument | permit", "written communication | permit").\
          replace("device | instrument | pact", "written communication | pact").\
          replace("device | instrument | return", "written communication | return").\
          replace("device | instrument | deed", "written communication | deed").\
          replace("device | instrument | order", "written communication | order").\
          replace("device | instrument | court order", "written communication | court order").\
          replace("device | instrument | draft", "written communication | draft").\
          replace("device | instrument | affidavit", "written communication | affidavit").\
          replace("device | instrument | act", "act").\
          replace("device | instrument | ruler", "person | ruler").\
          replace("container | dish", "dish").\
          replace("concept | composite | syndrome", "condition | illness | syndrome").\
          replace("compounds | complex | industrial plant", "industrial plant").\
          replace("auditory communication | music ", "art | music ").\
          replace("action | transportation | delivery", " action | delivery").\
          replace("facility | military installation | post |", "facility | post |").\
          replace("matter | food | diet", "action | diet").\
          replace("indication | translation | shift", "shift").\
          replace("happening | beginning | cause | producer", "producer").\
          replace("generally teleonomic | evolutionary process | medical science", "medical science").\
          replace("food | produce | eater", "eater").\
          replace("art | music | jazz | trad |", "trade |").\
          replace("magnitude | amount | increase | supplement", "supplement").\
          replace("process | medical procedure", "| medical procedure").\
          replace("person | adult | conservative | capitalist", "person | capitalist").\
          replace("party | smoker", "person | smoker").\
          replace("stone | calculus |", "stone |").\
          replace("process | phenomenon | ", "").\
          replace("event | ", "").\
          replace("cognition | ", "").\
          replace("instrumentality | ", "").\
          replace("artifact | ", "").\
          replace("event | ", "").\
          replace("compound | ", "").\
          replace("matter | ", "").\
          replace("material | ", "").\
          replace("indication | ", "").\
          replace("happening | ", "").\
          replace("group | ", "").\
          replace("organism | ", "").\
          replace("state | ", "").\
          replace("quantity | ", "").\
          replace("quality | ", "").\
          replace("relation | part | ", "").\
          replace("relation | ", "").\
          replace("property | ", "")

    if "soldier | marine | " in key2 and "american" not in key2:
      key2 = key2.replace("soldier | marine | ", "")
    if "channel |" in key2 and ("strait" in key2 or "gutter" in key2 or "stream" in key2 or "water" in key2 or "gutter" in key2 or "river" in key2 or "fluvial" in key2):
          key2 = key2.replace("channel |", "water channel |")
    for _ in range(4):
      if key2.count("|") > 1:
        key2_arr = key2.split(" | ")
        if key2_arr[0] in trim_parents:
          key2 = " | ".join(key2_arr[1:])
          continue
      break
    if "|" in key2 and " of " in key2.split(" | ")[0]:
      key2 = key2.split(" | ")[0].split(" of ")[0].strip() + " | " + key2
    if "crime |" in key2 and "event | activity |" not in key2:
      key2 = "event | activity | crime |" +key2.split(" crime |",1)[-1]
    if " web" in key2  and "silken" not in key2 and "spider" not in key2 and "webs" not in key2:
      key2 = key2.replace("web |", "written language | document |")
    if key2.count("|") == 1:
      key2_arr = key2.split(" | ")
      if key2_arr[0] in {"art",}:
        key2 = "art genre | " + key2_arr[1]
    for _ in range(4):
      if key2.count("|") > 1:
        key2_arr = key2.split(" | ")
        if key2_arr[0] in trim_parents:
          key2 = " | ".join(key2_arr[1:])
          continue
      break
    key2 = key2.replace("crime |crime | ", "crime | ").replace("event | ", "").strip(" |")
    aHash[key2] = val
    if add_to_items0:
      items0.append(key2)
    else:
      items1.append(key2)

  items0 = list(set(items0))
  items1 = list(set(items1))
  items0.sort() #key=lambda a: a) # [0].split()[-1].lower())
  items1.sort(key=lambda a: a[0].split()[-1].lower())
  text = ""
  with open("hypo.jsonl", "w") as hypof:
    for key in items0:
      if key.count("|") > 1 and aHash[key].count("*") > 1:
        hypof.write (key+"\n")
