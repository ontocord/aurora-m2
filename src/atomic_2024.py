#@title atomic 2020
import glob, json
personVerbFill = {}
import glob, json
import os
def process_1():
  if not os.path.exists("atomic2020_data-feb2021/"):
    !wget https://ai2-atomic.s3-us-west-2.amazonaws.com/data/atomic2020_data-feb2021.zip
    !unzip atomic2020_data-feb2021.zip
  for file in glob.glob("/content/atomic2020_data-feb2021/*.tsv"):
    with open(file, "rb") as infile:
      for idx, l in enumerate(infile):
        l = l.decode()
        if not l.strip(): continue
        l = l.strip()
        if l == "PersonX begins to lose weight	xEffect	\"medicine takes effect	blows their nose stresses about weight gain prospects\"":
          l = "PersonX begins to lose weight\txEffect\tmedicine takes effect	blows their nose stresses about weight gain prospects"
        try:
          sent1, rel, sent2 = l.split("\t")
        except:
          continue
        sent1 = " "+sent1.replace(" 'd", " had") + " "
        sent2 = " "+sent2+" "
        sent1= sent1.replace(" his ", " PersonX's ")
        sent2= sent2.replace(" her ", " PersonX's ")
        sent1= sent1.replace(" their ", " PersonX's ")
        sent2= sent2.replace(" their ", " PersonX's ")
        sent1 = sent1.strip()
        sent2 = sent2.strip()
        if rel == 'isFilledBy':
          if "___" in sent1:
              sent1, sent3 = sent1.split("___")
              sent1 = sent1.split()
              if sent1[1] not in {"always", "sometimes"} and sent1[1].endswith("s") and (len(sent1) < 3 or sent1[2] not in {'to',}):
                sent3 = " ".join(sent1[2:]) + " "+ "["+ sent2.strip() +"]" + " "+sent3.strip()
                sent1 = " ".join(sent1[:2])
              elif sent1[1] in {"left",} and (len(sent1) < 3 or sent1[2] not in {'to',}):
                sent3 = " ".join(sent1[2:]) + " "+ "["+ sent2.strip() +"]" + " "+sent3.strip()
                sent1 = " ".join(sent1[:2])
              elif sent1[1].endswith("ed") and (len(sent1) < 3 or sent1[2] not in {'to',}):
                sent3 = " ".join(sent1[2:]) + " "+ "["+ sent2.strip() +"]" + " "+sent3.strip()
                sent1 = " ".join(sent1[:2])
              elif sent1[-1] in {'the', 'a', 'an', 'this',  "their"}:
                sent3 = "["+ sent2.strip() +"]" + " "+sent3.strip()
                sent1 = " ".join(sent1[:-1])
              elif sent1[-1] in { 'one', 'two', 'three', 'four', 'five', 'most', 'many', 'all', 'some', 'every', 'all', 'each', 'another', 'more', 'less'}:
                sent3 = "["+ sent1[-1] + " "+ sent2.strip() +"]" + " "+sent3.strip()
                sent1 = " ".join(sent1[:-1])
              elif sent1[-1] in {"PersonZ's", "PersonZ", "PersonX", "PersonY", "others's", "'s", "PersonX's", "PersonY's"}:
                sent3 = sent1[-1] + " "+ "["+ sent2.strip() +"]" + " "+sent3.strip()
                sent1 = " ".join(sent1[:-1])
              else:
                sent3 = "["+ sent2.strip() +"]" + " "+sent3.strip()
                sent1 = " ".join(sent1)
              sent1 = sent1 + " ___"
              sent3 = sent3.replace("  ", " ").strip()
              sent2 = sent3
          sent2 = sent2.strip()
          sent2 = " "+sent2.replace("  ", " ")+" "
          for word in ['the', 'a', 'an', 'this']:
            sent2 = sent2.replace(" "+word + " [", " [")
            sent2 = sent2.replace("["+word + " ", " [")
            sent2 = sent2.replace(" "+word[0].upper()+word[:1] + " [", " [")
            sent2 = sent2.replace("["+word[0].upper()+word[:1] + " ", " [")
          for word in ['PersonX', 'PersonX\'s', 'PersonY', 'PersonY\'s', 'PersonZ', 'PersonZ\'s']:
            sent2 = sent2.replace("["+word+" " , word+" [")
          for word in ['one', 'two', 'three', 'four', 'five', 'most', 'many', 'all', 'some', 'every', 'all', 'each', 'another', 'more', 'less']:
            sent2 = sent2.replace(word + " [", "["+ word + " ")
          sent2 = sent2.replace("  ", " ").strip()
          sent2 = " "+sent2.replace(".]", "]")+" "
          if "need _" in sent1:
            pass
          elif "hated _" in sent1:
            sent1 = sent1.replace("d _", "s _")
          elif "ced _" in sent1 or "hed _" in sent1:
            sent1 = sent1.replace("ed _", "es _")
          else:
            sent1 = sent1.replace("ed _", "s _")
          verbOnly = sent1[len("PersonX "):].strip(" _")
          personVerbFill[verbOnly] = personVerbFill.get(verbOnly,[]) +[sent2.strip()]
          if " son " in sent2:
            sent2 = sent2.replace(" son ", " daugther ").strip()
            personVerbFill[verbOnly] = personVerbFill.get(verbOnly,[]) +[sent2.strip()]
          if " husband " in sent2:
            sent2 = sent2.replace(" husband ", " wife ").strip()
            personVerbFill[verbOnly] = personVerbFill.get(verbOnly,[]) +[sent2.strip()]
          if " daugther " in sent2:
            sent2 = sent2.replace(" daugther ", " son ").strip()
            personVerbFill[verbOnly] = personVerbFill.get(verbOnly,[]) +[sent2.strip()]
          if " wife " in sent2:
            sent2 = sent2.replace(" wife ", " husband ").strip()
            personVerbFill[verbOnly] = personVerbFill.get(verbOnly,[]) +[sent2.strip()]
  
  # do some clean up
  for key, vals in list(personVerbFill.items()):
    orig_key = key
    if not vals: continue
    for val in vals:
      orig_val = val
      if (val.startswith("and ") or val.startswith("or ")) and "comes" not in key and "relaxes" not in key and "sits" not in key and "stops" not in key and "goes" not in key and "tries" not in key and " to " not in val :
        if key.endswith("s"):
          val = val.split()
          val = val[1:]
          if not val[0].endswith("s"):
            val[0] = val[0]+"s"
          verb= val[0]
          if verb== "brokes":
            verb = "breaks"
          if "[" not in verb:
            val = val[1:]
            val = " ".join(val)
            personVerbFill[key] = personVerbFill.get(key,[]) +[val.strip()]
            key = " ".join(key.split()[:-1]) + " "+ verb
            personVerbFill[key] = personVerbFill.get(key,[]) +[val.strip()]
            #print (key, val.strip())
            try:
              personVerbFill[orig_key].remove(orig_val)
            except:
              print ('error')
              pass
              #print (personVerbFill[orig_key])
  
  relStore = {} # 'PersonX': personVerbFill}
  for key, vals in list(personVerbFill.items()):
      personVerbFill[key] = list(set(personVerbFill[key]))
      for val in personVerbFill[key]:
        val = val.split("[")[-1].split("]")[0].strip()
        val = val.split()
        val = " ".join(v for v in val if v not in {'one', 'two', 'three', 'four', 'five', 'most', 'many', 'all', 'some', 'every', 'all', 'each', 'another', 'more', 'less'})
        relStore[val.lower()] = {}
  
  #for key, vals in list(personVerbFill.items()):
  #    if len(personVerbFill[key]) > 20:
  #      print ((key, vals))
  if True:
  
    #@title even more atomic 2020
    import random, glob
  
    if True:
      seen = {}
      for file in glob.glob("/content/atomic2020_data-feb2021/*.tsv"):
        with open(file, "rb") as infile:
          for idx, l in enumerate(infile):
            l = l.decode()
            if not l.strip(): continue
            l = l.strip()
            if l == "PersonX begins to lose weight	xEffect	\"medicine takes effect	blows their nose stresses about weight gain prospects\"":
              l = "PersonX begins to lose weight\txEffect\tmedicine takes effect	blows their nose stresses about weight gain prospects"
            try:
              sent, rel, sent2 = l.split("\t")
            except:
              continue
            sent, rel, sent2 = l.split("\t")
            sent = sent.replace(" 'd", " had")
            sent2= sent2.strip()
            sent = " " + sent + " "
            sent2 = " " + sent2 + " ".lower()
            sent2 = sent2.replace(" x ", " PersonX ").replace(" y ", " PersonY ")
            sent2 = sent2.replace("personx", "PersonX").replace("persony", "PersonY")
            sent3 = ""
            foundSlot = personVerbFill.get(sent)
            if rel == "oEffect" and sent2 != " none ":
              sent3 = sent.strip()+"." + " This causes others "+ sent2.replace(" are ", " to be ")
            elif rel == "oReact" and sent2 != " none ":
              sent3 = sent.strip()+". " + ("This makes others react by feeling " if "PersonY" not in sent2 else "This makes PersonY react by feeling ")+ sent2.replace(" are ", " being ")
            elif rel == "oWant" and sent2 != " none ":
              sent3 = sent.strip()+". "+ ("This makes others want to " if "PersonY" not in sent2 else "This makes PersonY want to ")+ (sent2.strip()[3:].lower() if sent2.startswith(" to ") else sent2)
            elif rel == "xAttr" and sent2 != " none ":
              sent3 =  sent.strip()+". "+ ( "PersonX is feeling ") +sent2
            elif rel == "xEffect" and sent2 != " none ":
              sent3 =  sent.strip()+". "+ "So "+ sent2.strip()
            elif rel == "xIntent" and sent2 != " none ":
              sent3 = sent.strip()+". "+ ( "Therefore, PersonX intended to ") + (sent2.strip()[3:] if sent2.startswith(" to ") else sent2)
            elif rel == "xWant" and sent2 != " none ":
              sent3 = sent.strip()+". "+ ( "This makes PersonX want to ") + (sent2.strip()[3:] if sent2.startswith(" to ") else sent2)
            elif rel == "xReact" and sent2 != " none ":
              sent3 = sent.strip()+". "+ ( "This makes PersonX react by feeling ") + (sent2.replace(" are ", " being "))
            elif rel == "xNeed" and sent2 != " none ":
              sent3 = sent.strip()+". "+ ( "Therefore, PersonX needed to ")+ (sent2.strip()[3:]  if sent2.startswith(" to ") else sent2)
            elif rel == "HinderedBy" and sent2 != " none ":
              if sent.strip().startswith("PersonX"):
                _, verb, sent = sent.strip().split(" ",2)
                sent = "PersonX wants to " + verb.rstrip("s") + " " + sent
              sent3 = sent+". "+ "But "+ (sent2.strip()[3:] if sent2.startswith(" to ") else sent2)
            elif rel == "isBefore" and sent2 != " none ":
              sent3 = sent.strip()+". "+ ( "Then ")+ sent2.strip()
            elif rel == "isAfter" and sent2 != " none ":
              sent3 = sent2.strip()+". "+ ( "Then ")+ sent.strip()
            elif rel == "xReason" and sent2 != " none ":
              sent3 = sent.strip()+". "+ ( "Because ")+ sent2.strip()
            elif rel == "ObjectUse":
              find = sent.strip()
              for key, vals in list(personVerbFill.items()):
                found = [a for a in vals if find + " " in a or find + "]" in a]
                if found:
                  if "s " in sent2:
                    #print (("PersonX "+ sent2.strip() + " ___", "with [" + find + "]"))
                    pass
                  else:
                    sent2_arr = sent2.split()
                    if len(sent2_arr) > 2 and sent2_arr[1] in {"a", "the", "an"}:
                      sent2_arr[2] = "["+sent2_arr[2]
                      sent2_arr[-1] = sent2_arr[-1]+"]"
                      sent2_arr[0] = sent2_arr[0]+"s"
                      #print (("PersonX "+ sent2_arr[0]+"  ___", " ".join(sent2_arr[2:])  + " with " + find))
                    elif not sent2_arr[0].endswith("ing") and sent2_arr[0] not in {"to", "of", "in", "on", "at"}:
                      sent2_arr[0] = sent2_arr[0]+"s"
                      #print (("PersonX "+ " ".join(sent2_arr)  + " ___", "with [" + find + "]"))
                    else:
                      #print (("PersonX uses ___", "["+ find + "] for " + sent2.strip()))
                      pass
                  break
            elif rel in {"HasProperty",}:
              # can be described as ADJ
              pass
              #TODO - only add adjectives
            elif rel in {"CapableOf", "NotDesires","Desires", "Causes"}:
              sent = sent.strip()
              if sent in {"cock", "no person",  "bitch", "catherine havasi"}:
                continue
              elif "white person" in sent and "rap music" in sent2:
                continue
              elif sent == "somw people":
                sent = "PersonX"
              elif sent == "s person":
                sent = "PersonX"
              elif sent == "tv network":
                sent = "a tv network executive"
              elif sent == "music group":
                sent = "a music group member"
              elif sent == "spot":
                sent = "a dog named spot"
              elif sent == "people":
                sent = "PersonX"
              elif sent == "band":
                sent = "band member"
              elif sent in {"adultery", "affair"}:
                sent = "adulterer"
              elif sent == "bigotry":
                sent = "bigot"
              elif sent == "carelessness":
                sent = "careless person"
              elif sent == "too hot":
                sent = "extreme heat"
              elif sent == "please":
                sent = "something pleasing"
              elif sent == "person":
                sent = "PersonX"
              elif sent == "pussy":
                sent = "vagina"
              elif sent == "pisser":
                sent = "penis"
              elif sent == "boob":
                sent = "a woman's breast"
              elif sent in {"streetwalker", "prostitute", "whore", "hooker"}:
                sent = "sex worker"
              elif sent == "murder":
                sent = "murderer"
              elif sent == "rudness":
                sent = "a rude person"
              elif sent in {'society', 'dinosaur', 'dragon', 'skunk', 'iguana', 'mammal', 'insect', 'coyote', 'racoon', 'extreme heat', 'wild game', 'bug', 'terrier','thoroughbred', 'unicorn', 'shark', 'bunny', 'organism','parasite', 'penguin', 'heroin', 'call', 'use', 'creativity', 'hammer', 'corruption', 'leak', 'familiar smell', 'season', 'car', 'bribery', 'cold wind', 'diver', 'violence', 'detergent', 'gravity', 'job', 'smoke', 'funeral', 'noise', 'high altitude', 'inability to breath', 'cold', 'rock salt', 'excercising', 'old age', 'broad knowledge', 'marijuana use', 'poison', 'subjective change in consciousness', 'fatigue', 'religion', 'happiness', 'belief', 'sexual stimulation', 'war', 'plastic surgery', 'homework', 'medication', 'soap', 'anxiety', 'close call with death', 'raise', 'overpopulation', 'virus', 'rain', 'too hot', 'drug use', 'insult', 'car accident', 'age', 'good sex', 'oxygen', 'bribe', 'conceit', 'instrument', 'work', 'robbery', 'germ', 'solar eclipse', 'project', 'democracy', 'excessive sunshine', 'give', 'irc', 'flame retardant', 'arson', 'familiar sound', 'uncooked food', 'cold weather', 'explosion', 'inventive mind', 'grwoing old', 'car crash', 'sun exposure', 'jesse', 'schizophrenia', 'lift', 'wing', 'faith', 'icy road', 'stress', 'thirst', 'meditation', 'exercize', 'sin', 'intimacy', 'puppy', 'damage', 'purity', 'tooth paste', 'snowstorm', 'marble', 'depression', 'religious experience', 'fire', 'lawsuit', 'beer', 'ambulance', 'censorship', 'skin cream', 'kising', 'unemployment', 'coccidiosis', 'resuscitation', 'high blood pressure', 'loneliness', 'birthday', 'eye contact', 'sexual arousal', 'sugar', 'impatience', 'perfection', 'crime', 'crutch', 'unprotected sex', 'drought', 'information', 'lack of sleep', 'infection', 'mine', 'marijuana', 'ignorance', 'coffin', 'jealousy', 'flood', 'bad luck', 'sport', 'cigarette', 'waste', 'alarm', 'nuclear war', 'pressure', 'intense heat', 'irs', 'arousal', 'hate', 'faux pa', 'starvation', 'bag', 'caffiene', 'loud sound', 'mental illness', 'sex', 'pepper', 'produce', 'loud music', 'engine', 'tequila', 'famine', 'fly', 'desire', 'friction', 'keyboard use', 'laughter', 'shave', 'insufficient sleep', 'wathcing movie', 'seat', 'insomnia', 'death of friend', 'collaborative effort', 'kkk', 'depth of knowledge', 'crown', 'hail', 'teach', 'electricity', 'dehydration', 'joke', 'pack', 'injesting poison', 'show', 'vampire', 'puberty', 'goal in football', 'big investment', 'guitar', 'coffee', 'intercorse', 'subject', 'heart attack', 'cocaine use', 'terrorism', 'lot of snow', 'earthquake', 'flling', 'alcohol', 'heat', 'money', 'assault', 'sunburn', 'old injury', 'moisture', 'sex in toilet', 'spark', 'fever', 'struggle', 'collaboration', 'flatulence', 'intoxication', 'hurricane', 'embarrassment', 'injury', 'crisis', 'laziness', 'infidelity',
                             "marble", "project", "diet soda", "viral infection", "snow", "thunderstorm", "sickness", "rudeness", "ring", "hot weather", "presence of cloud", "purchase", "hot weather", "masturbation", "lie", "lack of money", "intense pressure", "red wine", "bad weather", "cabbage", "cancer", "slug", "cancerous tumor", "car chase", "darkness",  "death", "alcoholism", "archaeology", "argument", "hamster", "bull", "sheep", "owl", "duck", "mouth", "gorilla", "sun", "moon", "worm", "study", "browser", "stigma", "pastry", "government", "PersonX","government", "letter", "turtle", "sharp pain", "mule", "letter", "kitten", "pig", "animal", "pet", "fox", "emu", "diner", "hummingbird", "too much sun", "chipmunk", "beagle", "beaver", "root", "rabbit", "hunger", "tortise",  "seal", "vermin", "toad", "goose", "soda", "fear", "bacteria", "avalanche", "aol", "kangaroo", "lighening", "telepathy", "tiger", "rhino", "nasa","childre", "canary", "without good shade", "kangaroo", "elephant", "squirrel", "company", "crane", "butterfly", "tiger", "spider", "chicken", "grass", "bird", "life", "snake", "dolphin", "mug", "bear", "match", "cow", "monkey", "mind", "pain", "parrot", "chicken", "rat", "cat", "wolf", "wind", "cow", "frog", "bat", "flea", "lion","mosquito","dog", "rodent", "camel", "fish", "hand", "deer", "rooster", "pets", "love", "horse", "possum", "chameleon", "foal", "brain", "ibm", "salmon", "cheetah", "bee", "bitch", "lizard", "mouse", "family", "eagle", "ant", "hen", "stallion", "jet fighter", "squirls"} or \
                "plane" in sent or "animal" in sent or "seed" in sent or "snake" in sent or "shower" in sent or "plant" in sent or "smile" in sent or "plant" in sent or "children" in sent or " hand" in sent or "fish" in sent or " dog" in sent or " party" in sent or "bear" in sent or "nasal" in sent or "light" in sent:
                #print (sent+" can "+ sent2)
                if "PersonX" not in sent:
                  verb = ""
                  if rel == "Desires":
                    verb = "wants"
                  elif rel == "NotDesires":
                    verb = "does not want"
                  elif rel == "Causes":
                    verb = "causes"
                  if verb:
                    entity = sent.lower()
                    aHash = relStore[entity] = relStore.get(entity, {})
                    aHash[verb] = aHash.get(verb, []) + [sent2.strip()]
                    #print (aHash[verb])
                pass
              else:
                sent1 = "As a "+sent + ", PersonX "
                verb = ""
                if rel == "Desires":
                    verb = "wants"
                elif rel == "NotDesires":
                    verb = "does not want"
                elif rel == "Causes":
                    verb = "causes"
                if not verb:
                  sent2_arr = sent2.split()
                  if sent2_arr[0].endswith("ing"):
                    sent2_arr[0] = sent2_arr[0][:-3]
                  sent2_arr[0] = sent2_arr[0]+"s"
                  sent2 = " ".join(sent2_arr)
                sent1 = sent1 + " " + verb
                entity = sent.lower()
                aHash = relStore[entity] = relStore.get(entity, {})
                if (" " in sent2):
                  sent2_arr = sent2.split()
                  if sent2_arr[0].endswith("to"):
                    sent1 = sent1 + " "+ sent2_arr[0] + " " + sent2_arr[1]
                    sent2 = " ".join(sent2_arr[2:])
                  elif sent2_arr[0].endswith("s"):
                    sent1 = sent1 + " "+ sent2_arr[0]
                    sent2 = " ".join(sent2_arr[1:])
                verb2 = sent1.split("PersonX", 1)[-1].strip()
                if verb2 == "ss": verb2 = "sings"
                if verb2 == "brs": verb2 = "brings"
                if verb2 == "rs": verb2 = "rings"
                if not verb2:
                  print (sent1)
                aHash[verb2] = aHash.get(verb2, []) + [sent2.strip()]
  
            elif rel in {"MadeUpOf",}:
              #print ("A"+sent +" is made of " + sent2)
              pass
            elif rel in {"AtLocation",}:
              #print ("A"+sent + " "+ "is located at"+" " + sent2)
              pass
            elif rel in {"isFilledBy",   "HasSubEvent", }:
              sent3 = ""
            elif sent2 != " none ":
              #print ((sent, rel, sent2))
              pass
            if sent3:
              if sent3 in seen: continue
              seen[sent3] = 1
              sent3= sent3.replace("  ", " ")
              if "PersonX" not in sent3:
                sent3 = "PersonX " + sent3
              sent3 = sent3.strip().replace(" others Persony", "PersonY").replace("  ", " ").strip()
              #print (sent3)
              if foundSlot:
                pass
              if "___" in sent:
                sent, _ = sent.split("___")
                sent = sent  + " ___"
                #if personVerbFill.get(sent):
                #  print (sent, personVerbFill.get(sent))
            #if idx > 10000000: break
  for key, val in relStore.items():
    if len(val) > 1:
      print (key, val)
    #personVerbFill

def process_2():
  #@title more atomic 2020
  import glob, json
  personVerbFill = {}
  import glob, json
  import os
  if not os.path.exists("atomic2020_data-feb2021/"):
    !wget https://ai2-atomic.s3-us-west-2.amazonaws.com/data/atomic2020_data-feb2021.zip
    !unzip atomic2020_data-feb2021.zip
  
  import random, glob
  ifThen={}
  if True:
    seen = {}
    for file in glob.glob("/content/atomic2020_data-feb2021/*.tsv"):
      with open(file) as infile:
        for idx, l in enumerate(infile):
          if not l.strip(): continue
          l = l.strip()
          if l == "PersonX begins to lose weight	xEffect	\"medicine takes effect	blows their nose stresses about weight gain prospects\"":
            l = "PersonX begins to lose weight\txEffect\tmedicine takes effect	blows their nose stresses about weight gain prospects"
          try:
            sent, rel, sent2 = l.split("\t")
          except:
            continue
          sent, rel, sent2 = l.split("\t")
          sent = sent.replace(" 'd", " had")
          sent2= sent2.strip()
          sent = " " + sent + " "
          sent2 = " " + sent2 + " ".lower()
          sent2 = sent2.replace(" x ", " PersonX ").replace(" y ", " PersonY ")
          sent2 = sent2.replace("personx", "PersonX").replace("persony", "PersonY")
          sent3 = ""
          foundSlot = personVerbFill.get(sent)
          if rel == "oEffect" and sent2 != " none ":
            sent3 = sent.strip()+"."
            if "PersonX" not in sent3:
              sent3 = "PersonX " + sent3
            aHash = ifThen[sent3] =ifThen.get(sent3,{})
            transition = "This causes others "
            aList = aHash[transition] = aHash.get(transition,[])
            sent2 = sent2.replace(" are ", " to be ").strip().replace(" others Persony", "PersonY").replace("  ", " ").strip()
            aList.append(sent2)
          elif rel == "oReact" and sent2 != " none ":
            sent3 = sent.strip()+"."
            if "PersonX" not in sent3:
              sent3 = "PersonX " + sent3
            aHash = ifThen[sent3] =ifThen.get(sent3,{})
            transition = ("This makes others react by feeling " if "PersonY" not in sent2 else "This makes PersonY react by feeling ")
            aList = aHash[transition] = aHash.get(transition,[])
            sent2 = sent2.replace(" are ", " being ").strip().replace(" others Persony", "PersonY").replace("  ", " ").strip()
            aList.append(sent2)
          elif rel == "oWant" and sent2 != " none ":
            sent3 = sent.strip()+". "+ ("This makes others want to " if "PersonY" not in sent2 else "This makes PersonY want to ")+ (sent2.strip()[3:].lower() if sent2.startswith(" to ") else sent2)
          elif rel == "xAttr" and sent2 != " none ":
            sent3 =  sent.strip()+". "+ ( "PersonX is feeling ") +sent2
          elif rel == "xEffect" and sent2 != " none ":
            sent3 =  sent.strip()+". "+ "So "+ sent2.strip()
          elif rel == "xIntent" and sent2 != " none ":
            sent3 = sent.strip()+". "+ ( "Therefore, PersonX intended to ") + (sent2.strip()[3:] if sent2.startswith(" to ") else sent2)
          elif rel == "xWant" and sent2 != " none ":
            sent3 = sent.strip()+". "+ ( "This makes PersonX want to ") + (sent2.strip()[3:] if sent2.startswith(" to ") else sent2)
          elif rel == "xReact" and sent2 != " none ":
            sent3 = sent.strip()+". "+ ( "This makes PersonX react by feeling ") + (sent2.replace(" are ", " being "))
          elif rel == "xNeed" and sent2 != " none ":
            sent3 = sent.strip()+". "+ ( "Therefore, PersonX needed to ")+ (sent2 if sent2.startswith(" to ") else sent2)
          elif rel == "HinderedBy" and sent2 != " none ":
            if sent.startswith("PersonX"):
              sent = "PersonX wants to" + sent
            sent3 = sent+". "+ "But "+ (sent2.strip()[3:] if sent2.startswith(" to ") else sent2)
          elif rel == "isBefore" and sent2 != " none ":
            sent3 = sent.strip()+". "+ ( "Then ")+ sent2.strip()
          elif rel == "isAfter" and sent2 != " none ":
            sent3 = sent2.strip()+". "+ ( "Then ")+ sent.strip()
          elif rel == "xReason" and sent2 != " none ":
            sent3 = sent.strip()+". "+ ( "Because ")+ sent2.strip()
          else:
            continue
          sent3= sent3.replace("  ", " ")
          if "PersonX" not in sent3:
              sent3 = "PersonX " + sent3
          sent3 = sent3.strip().replace(" others Persony", "PersonY").replace("  ", " ").strip()
          ifThen[sent2] = ifThen.get(sent2,[]) + [sent3]

def process_3():
  #@title atoimc 2020, SODA and 10x to KG. KG to stories; probing model for KG.
  import json, os
  import random
  if not os.path.exists("/content/drive"):
    from google.colab import drive
    drive.mount('/content/drive')
  
  #try:
  #  import faker
  #except:
  #  !pip install -q faker
  female_names = ['Raquel', 'Kelly', 'Naireeta', 'Emma', 'Vân', 'Lynn', 'Beverly', 'Kolawole', 'Abril', 'Mariah', 'Sherry', 'Diễm', 'Evelyn', 'Kaitlin', 'Lacey', 'Destiny', 'Isabella', 'Elaine', 'Dominique', 'Berta', 'Kinfeosioluwa', 'Shelby', 'Thư', 'Míriam', 'Tuyền', 'Thùy', 'Patricia', 'Yesenia', 'Hayley', 'Shelia', 'Payal', 'Jayne', 'Crystal', 'Naomi', 'Martina', 'Toni', 'Sabrina', 'Tường', 'Sudarshana', 'Lindsey', 'Mỹ', 'Nayanika', 'Holly', 'Anwesha', 'Robyn', 'Patty', 'Minoo', 'Angel', 'Paloma', 'Sally', 'Ashlee', 'Adaoma', 'Kristine', 'Wanda', 'Yvette', 'Stacey', 'Ngọc', 'Tú', 'Arijita', 'Roser', 'Suzanne', 'Bodunde', 'Ashley', 'Ý', 'Ngân', 'Rituparna', 'Rebecca', 'Kerry', 'Mikayla', 'Tammie', 'Thủy', 'Joan', 'Jill', 'Xènia', 'Sonia', 'Ivet', 'Savannah', 'Alícia', 'Kamalika', 'Brandy', 'Ann', 'Clàudia', 'Maureen', 'Martha', 'Kylie', 'Terri', 'Katy', 'Hiền', 'Michelle', 'Latorunwa', 'Tâm', 'Queralt', 'Jyoti', 'Maliha', 'Jan', 'Sandra', 'Sonya', 'Carrie', 'Briana', 'Kimberley', 'Fiona', 'Monica', 'Sydney', 'Phương', 'Subha', 'Keyshia', 'Joy', 'Adankwo', 'Leah', 'Tuyết', 'Nicole', 'Priscilla', 'Ebunoluwa', 'Kara', 'Wendy', 'Sandy', 'Antònia', 'Neus', 'Bmidele', 'Haley', 'Josephine', 'Kathleen', 'Katie', 'Ánh', 'Montserrat', 'Amàlia', 'Tami', 'Nikita', 'Shalini', 'Sophia', 'Nancy', 'Heather', 'Theresa', 'Jaime', 'Miranda', 'Annette', 'Beverley', 'Carly', 'Dolors', 'Durba', 'Stacy', 'Nghi', 'Marilyn', 'Jeanne', 'Leanne', 'Latasha', 'Sian', 'Elisabet', 'Phúc', 'Phụng', 'Vickie', 'Bimpe', 'Dương', 'Paige', 'Kristi', 'Durga', 'Meredith', 'Cheryl', 'Jamie', 'Romana', 'Anuradha', 'Daniela', 'Breanna', 'Erin', 'Nichole', 'Alicia', 'Chi', 'Ona', 'Sheena', 'Caroline', 'Teresa', 'Melissa', 'Bridget', 'Hương', 'Morgan', 'Sudipta', 'Sataraupa', 'Glenda', 'Georgia', 'Jeanette', 'Rose', 'Nga', 'Ibilola', 'Marina', 'Tracy', 'Melanie', 'Laura', 'Nguyệt', 'Betty', 'Brittany', 'Olga', 'Veronica', 'Whitney', 'Kari', 'Liên', 'Riya', 'Irene', 'Carme', 'Kaitlyn', 'Carla', 'Changezi', 'Trân', 'Mariona', 'Pam', 'Joanna', 'Mai', 'April', 'Joanne', 'Alexis', 'Shreya', 'Rebeca', 'Anushka', 'Donna', 'Marion', 'Ariel', 'Kiều', 'Hilary', 'Valerie', 'Anna', 'Norma', 'Bailey', 'Charlotte', 'Ipshita', 'Lisa', 'Anne', 'Helen', 'Hạnh', 'Mohar', 'Tista', 'Margarita', 'Cassandra', 'Lan', 'Sofía', 'Khuê', 'Caitlin', 'Cassidy', 'Carolyn', 'Trinh', 'Selena', 'Nguyên', 'Christina', 'Núria', 'Casey', 'Cindy', 'Mallory', 'Lori', 'Vicki', 'Eulàlia', 'Hoa', 'Emily', 'Bethany', 'Erica', 'Kate', 'Jackie', 'Alba', 'Ariadna', 'Jacqueline', 'Danielle', 'Judith', 'Tonya', 'Kimberly', 'Minh', 'Kristie', 'Thảo', 'Sue', 'Paula', 'June', 'Duyên', 'Allison', 'Maria', 'Taylor', 'Thi', 'Christy', 'Madhuparna', 'Rita', 'Benazir', 'Meagan', 'Joana', 'Adrija', 'Jaclyn', 'Kathy', 'Thúy', 'Châu', 'Roshni', 'Arlet', 'Mia', 'Faith', 'Jane', 'Kamala', 'Madison', 'Mẫn', 'Thắm', 'Sheryl', 'Madeline', 'Nora', 'Aurora', 'Varsha', 'Nhung', 'Sayani', 'Debapriya', 'Gabriela', 'Giang', 'Thơ', 'Khanh', 'Abigail', 'Jade', 'Stephanie', 'Vanessa', 'Uma', 'Patrícia', 'Lindsay', 'Candace', 'Fasih', 'Elizabeth', 'Claudia', 'Francesca', 'Deanna', 'Gabriella', 'Clara', 'Lídia', 'Nhàn', 'Doyinsola', 'Tracie', 'Renee', 'Diamond', 'Bình', 'Alice', 'Diệu', 'Elisenda', 'Krista', 'Debarati', 'Ellie', 'Trâm', 'Hassim', 'Alexa', 'Irina', 'Kristen', 'Trúc', 'Diekololaoluwalayemi', 'Farahnaz', 'Kelli', 'Bích', 'Brianna', 'Laurie', 'Catherine', 'Shannon', 'Đào', 'Lanre', 'Brittney', 'Chloe', 'Peggy', 'Tricia', 'Karla', 'Natasha', 'Nayan', 'Caitlyn', 'Mackenzie', 'Sheri', 'Raven', 'Jasmin', 'Sharon', 'Amelia', 'Esther', 'Judy', 'Glòria', 'Rachael', 'Kelsey', 'Christine', 'Kristin', 'Jocelyn', 'Tamara', 'Hồng', 'Tuệ', 'Carol', 'Eva', 'Candice', 'Gisela', 'Debra', 'Gabrielle', 'Eileen', 'Amy', 'Darlene', 'Kaylee', 'Prerona', 'Kristy', 'Nivedita', 'Joann', 'Jodi', 'Hailey', 'Felicia', 'Julia', 'Như', 'Jordan', 'Karen', 'Mindy', 'Ibidun', 'Janet', 'Kellie', 'Becky', 'Terry', 'Dawn', 'Regina', 'Jean', 'Sierra', 'Jodie', 'Reema', 'Chelsea', 'Rhonda', 'Shirley', 'Katrina', 'Diane', 'Angie', 'Autumn', 'Hà', 'Moumita', 'Summer', 'Geeta', 'Kerri', 'Traci', 'Nabanita', 'Noemí', 'Linda', 'Diana', 'Dorothy', 'Nhi', 'Jenna', 'Jenny', 'An', 'Nina', 'Natàlia', 'Vi', 'Jo', 'Debanjana', 'Cathy', 'Rachel', 'Chaity', 'Kiara', 'Barbara', 'Doris', 'Lorraine', 'Hollie', 'Pallavi', 'Karina', 'Latoya', 'Mònica', 'Uyên', 'Băng', 'Gemma', 'Marissa', 'Shweta', 'Sílvia', 'Mary', 'Jasmine', 'Phyllis', 'Makayla', 'Carole', 'Chelsey', 'Audrey', 'Anna Maria', 'Rosa Maria', 'Andrea', 'Georgina', 'Brenda', 'Sanghamitra', 'Lynne', 'Julie', 'Kayla', 'Sudeshna', 'Grace', 'Xuân', 'Alexandria', 'Gillian', 'Pamela', 'Rosie', 'Radhika', 'Katherine', 'Inés', 'Shawna', 'Bishakha', 'Lâm', 'Àngela', 'Kathryn', 'Penny', 'Ankita', 'Chandrayee', 'Rupsa', 'Nhã', 'Asmita', 'Ariana', 'Tina', 'Molly', 'Alyssa', 'Lam', 'Jana', 'Upasana', 'Michaela', 'Nhiên', 'Thy', 'Olivia', 'Marian', 'Ruma', 'Frances', 'Krystal', 'Kendra', 'Sònia', 'Amber', 'Misty', 'Mi', 'Robin', 'My', 'Monique', 'Nicola', 'Sarah', 'Lucy', 'Helena', 'Adriana', 'Pampa', 'Stacie', 'Doanh', 'Tammy', 'Pallabi', 'Tiffany', 'Hannah', 'Stefanie', 'Hiếu', 'Megan', 'Cynthia', 'Meghan', 'Lesley', 'Marisa', 'Leslie', 'Isabel', 'Trà', 'Claire', 'Desiree', 'Yvonne', 'Noèlia', 'Hằng', 'Hân', 'Tanurina', 'Sreemoyee', 'Shari', 'Dideoluwakusidede', 'Priya', 'Sophie', 'Di', 'Sayantani', 'Daisy', 'Mar', 'Bonnie', 'Charlene', 'Jody', 'Adrienne', 'Kayleigh', 'Marta', 'Mandy', 'Brooke', 'Brandi', 'Lynda', 'Ibidolapo', 'Christie', 'Arundhuti', 'Rosa', 'Quyên', 'Emilohi', 'Estela', 'Lauren', 'Azhar', 'Tanya', 'Eleanor', 'Ananya', 'Natalie', 'Carolina', 'Harriet', 'Connie', 'Phượng', 'Zoe', 'Alison', 'Anindita', 'Gwendolyn', 'Bianca', 'Lydia', 'Thanh', 'Alisha', 'Susanna', 'Margaret', 'Paromita', 'Hazel', 'Ruth', 'Loan', 'Swagata', 'Ibironke', 'Dung', 'Denise', 'Cèlia', 'Anita', 'Bethan', 'Debbie', 'Clare', 'Thương', 'Geraldine', 'Rebekah', 'Jessica', 'Sylvia', 'Kehinde', 'Samantha', 'Gina', 'Susan', 'Kristina', 'Kim', 'Courtney', 'Linh', 'Ly', 'Colleen', 'Shelly', 'Carmen', 'Ellen', 'Joyce', 'Lia', 'Blanca', 'Emiola', 'Quân', 'Abbie', 'Yolanda', 'Ainhoa', 'Abebi', 'Laia', 'Tiên', 'Ashleigh', 'Virginia', 'Alèxia', 'Louise', 'Dana', 'Judit', 'Rohini', 'Angelica', 'Shoaib', 'Faizan', 'Sara', 'Sherri', 'Eniiyi', 'Katelyn', 'Khánh', 'Abidemi', 'Victòria', 'Lara', 'Verònica', 'Yến', 'Belinda', 'Tasha', 'Jennifer', 'Vy', 'Jemma', 'Melinda', 'Piyali', 'Mercedes', 'Cristina', 'Michele', 'Suparna', 'Heidi', 'Lakshmi', 'Ishita', 'Deborah', 'Pauline', 'Shayoni', 'Roberta', 'Indrani', 'Alejandra', 'Marie', 'Melody', 'Júlia', 'Debasmita', 'Aparna', 'Meritxell', 'Elsa', 'Alexandra', 'Marcia', 'Kirsten', 'Kirsty', 'Rumela', 'Gala', 'Gloria', 'Amanda', 'Jillian', 'Quỳnh', 'Bipasha', 'Erika', 'Carlota', 'Rosemary', 'Đan', 'Aina', 'Mckenzie', 'Iris', 'Sheila', 'Shelley', 'Reshma', 'Cassie', 'Sushmita', 'Tara', 'Loretta', 'Angela', 'Gail', 'Kỳ', 'Ebony', 'Anh', 'Oanh', 'Diệp', 'Priyanka', 'Victoria', 'Mireia', 'Aimee', 'Cheyenne', 'Ana', 'Tabitha', 'Janice', 'Beth', 'Tracey', 'Sania', 'Burhan', 'Mercè', 'Fuad', 'Ân', 'Trang', 'Zoputan', 'Huyền']
  male_names = ['Hayden', 'Kelly', 'Nam', 'Tân', 'Francis', 'Sơn', 'Sandeep', 'Glen', 'Andreu', 'Graham', 'Max', 'Arijit', 'Javier', 'Randy', 'Sumit', 'Anthony', 'Đông', 'Ramon', 'Vernon', 'Alan', 'Thông', 'Tyler', 'Tài', 'Raül', 'Warren', 'Drew', 'Xavier', 'Sebastià', 'Tanner', 'Tường', 'Lluc', 'Indrajit', 'Franklin', 'Hiển', 'Glenn', 'Tathagata', 'Deeptiman', 'Bill', 'Minoo', 'Edgar', 'Angel', 'Marcus', 'Hiệp', 'Abdul', 'Cèsar', 'Ngọc', 'Tú', 'Enric', 'Guillem', 'Allan', 'Paul', 'Elliot', 'Ashley', 'Hào', 'Văn', 'Mark', 'Gary', 'Kerry', 'Garry', 'Swagato', 'Don', 'Darius', 'Joan', 'Sanjay', 'Soham', 'Lộc', 'Martyn', 'Tín', 'Nhân', 'Ankur', 'Siddhartha', 'Genís', 'Guy', 'Teo', 'Brendan', 'Stuart', 'Daniel', 'Dũng', 'Tâm', 'Hải', 'Maliha', 'Jan', 'Miquel Àngel', 'Phương', 'Arghya', 'Ifelewa', 'Frank', 'Jose', 'Vương', 'Pol', 'Shane', 'Nigel', 'Steven', 'Oliver', 'Roberto', 'Hunter', 'Hector', 'Jeremy', 'Ayan', 'Joshua', 'Troy', 'Àlex', 'Banjoko', 'Iles', 'Liam', 'Walter', 'Gabriel', 'Francesc', 'Arthur', 'Colton', 'Tejumola', 'Dillon', 'Òscar', 'Antoni', 'Bách', 'Duy', 'Perry', 'Jackson', 'Gordon', 'Udayan', 'Jaime', 'Chase', 'Clifford', 'Jerome', 'Mohamed', 'Christopher', 'Ferran', 'Phúc', 'Bartolomé', 'Francisco', 'Oscar', 'Raymond', 'Debajyoti', 'Dương', 'Ruben', 'Lance', 'Juan', 'Jamie', 'Romana', 'Corey', 'Nicholas', 'Amit', 'Nathan', 'Cristian', 'Lucas', 'Víctor', 'Todd', 'Scott', 'Marc', 'Aritra', 'Shaun', 'Greg', 'Ritam', 'Howard', 'Ananyo', 'Ben', 'William', 'Willie', 'Gael', 'Pedro', 'Ian', 'Jerry', 'Đăng', 'Tracy', 'Adrià', 'Khoa', 'Clayton', 'Cameron', 'Ralph', 'Jeffery', 'Carles', 'Bankole', 'Changezi', 'Larry', 'Samuel', 'Aitor', 'Biel', 'Reece', 'Gaurav', 'Obasolape', 'Alexis', 'Riley', 'Bruce', 'Chad', 'Bảo', 'Allen', 'Ross', 'Jonathon', 'Mithun', 'Joe', 'Mario', 'Iain', 'Peter', 'Salvador', 'Aniruddha', 'Bryce', 'Eloi', 'Kuntal', 'Charlie', 'Danny', 'Owen', 'Phong', 'Saül', 'Ronnie', 'Nguyên', 'Ropo', 'Casey', 'Thiên', 'Àngel', 'Ganesh', 'Alejandro', 'Oba', 'Reginald', 'Dylan', 'Richard', 'Luân', 'Minh', 'Subrata', 'Kieran', 'Seth', 'Benjamin', 'Alok', 'Terrence', 'Abeo', 'Trí', 'Taylor', 'Benazir', 'Seriki', 'Kha', 'Terence', 'Dominic', 'Vỹ', 'Dakota', 'Brian', 'Cody', 'Amitava', 'Roc', 'Jacob', 'Damien', 'Dhrubo', 'Jon', 'Preston', 'Rahul', 'Aditya', 'Giang', 'Artur', 'Khanh', 'Fernando', 'Dennis', 'Pere', 'Jeff', 'Bhaskar', 'Souvik', 'Thành', 'Khiêm', 'Fasih', 'Roger', 'Edward', 'Esupofo', 'Wayne', 'Francesc Xavier', 'Sam', 'Alec', 'Wesley', 'Patrick', 'Dave', 'Cory', 'Fèlix', 'Sandip', 'Ricky', 'Bình', 'Nghĩa', 'Sayan', 'Hassim', 'Joel', 'Jeremiah', 'Farahnaz', 'Raja', 'Darin', 'Jay', 'Alexandre', 'Noah', 'Duane', 'Nhật', 'Shannon', 'Quý', 'Ranajoy', 'Axel', 'Theodore', 'Dipayan', 'Calvin', 'Ignasi', 'Curtis', 'Việt', 'Tanimola', 'Jimmy', 'Jim', 'Randall', 'Sang', 'Bernard', 'Rajat', 'Dipankar', 'Justin', 'Toby', 'Jonathan', 'Carlos', 'Terrance', 'Marvin', 'Charles', 'Tapan', 'Shayok', 'Nathaniel', 'Somnath', 'Sourojit', 'Elías', 'Johnathan', 'Josep Lluís', 'Jesus', 'Roy', 'Phước', 'Phát', 'Thịnh', 'Cole', 'Lewis', 'Pankaj', 'Timothy', 'Kent', 'Leroy', 'Tomàs', 'Avik', 'Rick', 'Gregg', 'Gregory', 'Sourabh', 'Jason', 'Gbadebo', 'Joan Carles', 'Hưng', 'Levi', 'Mohammed', 'Gene', 'Brett', 'Jordan', 'Bishwadeep', 'Dídac', 'Terry', 'Mohammad', 'Gautam', 'Brad', 'Bernat', 'Dean', 'Steve', 'Emili', 'Gerard', 'Dale', 'Đại', 'Agustí', 'Cesar', 'Inioluwa', 'Ronald', 'Tapas', 'Hèctor', 'Harry', 'Kevin', 'Esteve', 'Saikat', 'Colin', 'Conor', 'Isamotu Olalekan', 'Martin', 'Kenneth', 'Sunny', 'Bobby', 'Keith', 'Llorenç', 'Công', 'An', 'Thắng', 'Abhijit', 'Indranil', 'Elijah', 'Tim', 'Tiến', 'Abhishek', 'Toàn', 'Tristan', 'Chandan', 'Tony', 'Mukul', 'Herbert', 'Michael', 'Kirk', 'Lawrence', 'Agniva', 'Sukumar', 'Phi', 'Sabyasachi', 'Austin', 'Victor', 'Dalton', 'Joan Antoni', 'Bryan', 'Sudipto', 'Ryan', 'Connor', 'Karl', 'Sergi', 'Khang', 'Eugene', 'Jared', 'Jayanta', 'Iranola', 'Billy', 'Arka', 'Abel', 'Devon', 'Kilian', 'Antonio', 'Callum', 'Lâm', 'Leigh', 'Arnau', 'Leonard', 'Johnny', 'Darren', 'Brady', 'Louis', 'Julian', 'Alex', 'Kurt', 'Nhựt', 'Eddie', 'Trọng', 'Damon', 'Wyatt', 'Harold', 'Jack', 'Douglas', 'Trung', 'Eduard', 'Kristopher', 'Phillip', 'Zachary', 'Marcel', 'Vũ', 'Rickey', 'Craig', 'Antony', 'Robin', 'Leon', 'Stanley', 'Neil', 'Nil', 'Andre', 'Ricardo', 'Albert', 'Frederick', 'Thomas', 'Utsab', 'Cường', 'Logan', 'Hiếu', 'Christian', 'Earl', 'Mike', 'Abegunde', 'Dustin', 'Leslie', 'George', 'Miguel', 'Subhashish', 'Geoffrey', 'Malcolm', 'Narcís', 'Dan', 'Josh', 'Lợi', 'Adrian', 'Ernest', 'Kyle', 'Darrell', 'Long', 'Erik', 'Bob', 'Brent', 'Rafel', 'Trevor', 'Danh', 'Derek', 'Sean', 'Stephen', 'Devin', 'Andrew', 'Alexander', 'Ethan', 'Isaac', 'Philip', 'Hậu', 'Samrat', 'Sourav', 'Isaiah', 'Santiago', 'Gavin', 'Ivan', 'Graeme', 'Ratan', 'Manuel', 'Caleb', 'Maurice', 'Parker', 'Quốc', 'Tuấn', 'Triết', 'Carl', 'Hugh', 'Duncan', 'Quang', 'Pau', 'Kiệt', 'Azhar', 'Josep Maria', 'Fred', 'David', 'Mitchell', 'Adam', 'Mason', 'Malik', 'Himadri', 'Josep', 'Thanh', 'Ricard', 'Gaurab', 'Andres', 'Souparna', 'Saptarshi', 'Jeffrey', 'Darryl', 'Russell', 'Clive', 'Eric', 'Marco', 'Lee', 'Alvin', 'Henry', 'Norman', 'Elliott', 'Jesse', 'Hòa', 'Gerald', 'Arko', 'Kiên', 'Vinh', 'Travis', 'Arindam', 'Linh', 'Damian', 'Mathew', 'Lonnie', 'James', 'Praveen', 'Nicolàs', 'Quân', 'Milan', 'Gilbert', 'Miquel', 'Avishek', 'Jishnu', 'Tommy', 'Raghav', 'Jofre', 'Daryl', 'Barry', 'Obatotosinloluwa', 'Edwin', 'Vincent', 'Jermaine', 'Shoaib', 'Declan', 'Faizan', 'Tùng', 'Quim', 'Mạnh', 'Trường', 'Dwayne', 'Oriol', 'Khánh', 'Khải', 'Tantoluwa', 'Chris', 'Sankalpa', 'Soumya', 'Tom', 'Matthew', 'Melvin', 'Luis', 'Tushar', 'Tadenikawo', 'Monoranjan', 'Alfred', 'Rhys', 'Rereloluwa', 'Simon', 'Blake', 'Đức', 'Denis', 'Alfons', 'Martí', 'Joseph', 'Ismael', 'Seye', 'Prasenjit', 'Grant', 'Nicolas', 'Phú', 'Joaquim', 'Khôi', 'Lluís', 'Huy', 'Sergio', 'Tyrone', 'Thuận', 'Shawn', 'Gopal', 'Jake', 'Luke', 'John', 'Vĩ', 'Thái', 'Bikash', 'Omar', 'Robert', 'Brandon', 'Đạt', 'Niladri', 'Manel', 'Preetam', 'Thiện', 'Clarence', 'Garrett', 'Hoàng', 'Jordi', 'Shakale', 'Stewart', 'Abayomrunkoje', 'Jorge', 'Micheal', 'Jaume', 'Hùng', 'Spencer', 'Derrick', 'Donald', 'Gareth', 'Tấn', 'Evan', 'Maxwell', 'Ray', 'Kỳ', 'Marçal', 'Bruno', 'Anh', 'Collin', 'Clinton', 'Sania', 'Bradley', 'Aaron', 'Vicenç', 'Burhan', 'Fuad', 'Obafemi', 'Rodney', 'Ân', 'Aleix', 'Eduardo', 'Jesús']
  
  names = ['Aimee',
   'Molly',
   'Denis',
   'Gerard',
   'Gillian',
   'Karen',
   'Rita',
   'Rachael',
   'Julia',
   'Elliott',
   'Kate',
   'Elaine',
   'Mark',
   'Dylan',
   'Jack',
   'Leanne',
   'Mitchell',
   'Andrea',
   'Leah',
   'Kathryn',
   'Michael',
   'Katy',
   'Joel',
   'Joe',
   'Stanley',
   'Joan',
   'Roger',
   'Mohammed',
   'Dennis',
   'Pauline',
   'Steven',
   'Lawrence',
   'Charles',
   'Marian',
   'Amber',
   'Jodie',
   'Fiona',
   'Dorothy',
   'Damien',
   'Gerald',
   'Jay',
   'Joyce',
   'Eileen',
   'Donald',
   'Keith',
   'Louis',
   'Callum',
   'Ruth',
   'Owen',
   'Carl',
   'Joanne',
   'Ian',
   'Rachel',
   'Edward',
   'Frances',
   'Denise',
   'Rosemary',
   'Dawn',
   'Anne',
   'Bethan',
   'Carly',
   'Christian',
   'Angela',
   'Teresa',
   'Beverley',
   'Ashley',
   'Jacob',
   'Paul',
   'Leigh',
   'Matthew',
   'Jenna',
   'Shannon',
   'Sian',
   'Iain',
   'Ann',
   'Rhys',
   'Victoria',
   'Amanda',
   'Douglas',
   'Stephen',
   'Philip',
   'Lesley',
   'Stephanie',
   'Amelia',
   'Jemma',
   'Melanie',
   'Daniel',
   'Damian',
   'Brian',
   'Connor',
   'Scott',
   'Maria',
   'Ryan',
   'Lindsey',
   'Terry',
   'Josephine',
   'Joseph',
   'George',
   'Howard',
   'Mohammad',
   'Nicola',
   'Grace',
   'Nathan',
   'Leslie',
   'Clive',
   'Valerie',
   'Jonathan',
   'Colin',
   'Luke',
   'Jean',
   'Guy',
   'Gordon',
   'Sarah',
   'Anna',
   'Bethany',
   'Cheryl',
   'Tom',
   'Dale',
   'Jasmine',
   'Amy',
   'Dean',
   'Justin',
   'Christopher',
   'Declan',
   'Yvonne',
   'Lydia',
   'Abdul',
   'Abigail',
   'Kyle',
   'Donna',
   'Toby',
   'Irene',
   'Shaun',
   'Abbie',
   'Benjamin',
   'Catherine',
   'Samantha',
   'Georgia',
   'Claire',
   'Alan',
   'Ben',
   'Kerry',
   'Jessica',
   'Henry',
   'Neil',
   'Emma',
   'Terence',
   'Zoe',
   'Lynn',
   'Jeffrey',
   'Wayne',
   'Kayleigh',
   'Suzanne',
   'David',
   'Joanna',
   'Hilary',
   'Megan',
   'Brett',
   'Robert',
   'Graham',
   'Ronald',
   'Deborah',
   'Lorraine',
   'Sheila',
   'Stewart',
   'Barbara',
   'Marilyn',
   'Eric',
   'Sean',
   'Ellie',
   'Rosie',
   'Jade',
   'Jacqueline',
   'Kenneth',
   'Victor',
   'Peter',
   'Max',
   'Patrick',
   'Francis',
   'Derek',
   'Michelle',
   'Kieran',
   'Nicole',
   'Katie',
   'Marie',
   'Kirsty',
   'Annette',
   'Reece',
   'Allan',
   'Gemma',
   'Wendy',
   'Lynne',
   'Andrew',
   'James',
   'Ashleigh',
   'Karl',
   'Bradley',
   'Maurice',
   'Kelly',
   'Frank',
   'Gregory',
   'Patricia',
   'Raymond',
   'Lisa',
   'Natalie',
   'Margaret',
   'Marion',
   'Jordan',
   'Gavin',
   'Olivia',
   'Garry',
   'Elizabeth',
   'Josh',
   'Richard',
   'Julian',
   'Phillip',
   'Craig',
   'Pamela',
   'Stacey',
   'Dominic',
   'Georgina',
   'Danielle',
   'Alison',
   'Liam',
   'Nigel',
   'Judith',
   'Natasha',
   'Leonard',
   'Adam',
   'Hayley',
   'Frederick',
   'Martin',
   'Gareth',
   'Mary',
   'Oliver',
   'Caroline',
   'Eleanor',
   'Sandra',
   'Bernard',
   'Ross',
   'Conor',
   'Roy',
   'Rebecca',
   'Beth',
   'Christine',
   'Aaron',
   'Norman',
   'Clare',
   'Hugh',
   'Barry',
   'Alexandra',
   'Melissa',
   'Naomi',
   'Timothy',
   'Harriet',
   'Hazel',
   'Bruce',
   'Shane',
   'Alex',
   'Paige',
   'Chelsea',
   'Sharon',
   'Jason',
   'Simon',
   'Trevor',
   'Charlotte',
   'Sam',
   'Janice',
   'Paula',
   'Tracey',
   'Louise',
   'Brenda',
   'Debra',
   'Lee',
   'Hannah',
   'Mandy',
   'Lewis',
   'Ricky',
   'Holly',
   'Adrian',
   'Jill',
   'Susan',
   'Jamie',
   'Marcus',
   'Tracy',
   'Tina',
   'Marc',
   'Malcolm',
   'Stuart',
   'Laura',
   'Elliot',
   'Helen',
   'Katherine',
   'Jayne',
   'Nicholas',
   'Vanessa',
   'Darren',
   'Graeme',
   'Albert',
   'Jake',
   'Janet',
   'Cameron',
   'Kevin',
   'Francesca',
   'Chloe',
   'Alice',
   'Sylvia',
   'Billy',
   'Joshua',
   'Sally',
   'Lynda',
   'Heather',
   'Leon',
   'Geraldine',
   'Kim',
   'Antony',
   'Samuel',
   'Sophie',
   'Lucy',
   'Charlie',
   'Glen',
   'Maureen',
   'Clifford',
   'Martyn',
   'Hollie',
   'Mohamed',
   'Anthony',
   'Jennifer',
   'Geoffrey',
   'Carol',
   'Harry',
   'Kathleen',
   'Brandon',
   'William',
   'Thomas',
   'Charlene',
   'Diane',
   'Lauren',
   'Vincent',
   'Glenn',
   'Emily',
   'Kimberley',
   'Arthur',
   'Alexander',
   'Gail',
   'Bryan',
   'Diana',
   'Jeremy',
   'Linda',
   'Carolyn',
   'Danny',
   'Shirley',
   'Jane',
   'Julie',
   'Gary',
   'Russell',
   'Robin',
   'Mathew',
   'Duncan',
   'Carole',
   'John',
   'Tony',
   'Sara',
   'June']
  
  if not os.path.exists("/content/ATOMIC10X.jsonl"):
    !cp /content/drive/Shareddrives/ontocord_llc/safe_llm/ATOMIC10X.jsonl /content/
  
  import glob, json
  import os
  if not os.path.exists("atomic2020_data-feb2021/"):
    !wget https://ai2-atomic.s3-us-west-2.amazonaws.com/data/atomic2020_data-feb2021.zip
    !unzip atomic2020_data-feb2021.zip
  
  import json
  try:
    from datasets import load_dataset
  except:
    !pip install -q datasets
  from datasets import load_dataset
  
  try:
    if dataset is None: assert False
  except:
    dataset = load_dataset("allenai/soda")
  
  
  
  if not os.path.exists("atomic_train.jsonl"):
    with open("atomic_train.jsonl", "w") as outf:
      for idx, l in enumerate(open("/content/atomic2020_data-feb2021/train.tsv")):
        #l = l.decode()
        if not l.strip(): continue
        l = l.strip()
        if l == "PersonX begins to lose weight	xEffect	\"medicine takes effect	blows their nose stresses about weight gain prospects\"":
          l = "PersonX begins to lose weight\txEffect\tmedicine takes effect	blows their nose stresses about weight gain prospects"
        try:
          head, rel, tail = l.split("\t")
        except:
          continue
        head = " "+head.replace(" 'd", " had") + " "
        tail = " "+tail+" "
        head= head.replace(" his ", " PersonX's ")
        tail= tail.replace(" her ", " PersonX's ")
        head= head.replace(" their ", " PersonX's ")
        tail= tail.replace(" their ", " PersonX's ")
        head = head.strip()
        tail = tail.strip()
        if tail != "none":
          if rel in {"oEffect","oReact","oWant","xAttr" ,"xEffect" ,"xIntent" ,"xWant" ,"xReact" ,
                     "xNeed","HinderedBy","isBefore" ,"isAfter","xReason"}:
            if not head.startswith("PersonX"):
              head = "PersonX "+ head
            head = head.replace("the ___", "something")
            head = head.replace(" a ___", " something")
            head = head.replace(" an ___", " something")
            head = head.replace(" another ___", " something")
            head = head.replace(" some ___", " something")
            if "___" in head:
              head2 = head.replace("___", "PersonY")
              outf.write(json.dumps({'head': head2, "relation": rel, "tail": tail,'p': 1.0 })+"\n")
              head2 = head.replace("___", "something")
              outf.write(json.dumps({'head': head2, "relation": rel, "tail": tail,'p': 1.0 })+"\n")
            else:
              outf.write(json.dumps({'head': head, "relation": rel, "tail": tail,'p': 1.0 })+"\n")
  
      with open("ATOMIC10X.jsonl") as inf:
        for l in inf:
          dat = json.loads(l)
          if dat['split'] != 'train': continue
          head = dat['head']
          if not head.startswith("PersonX"):
              head = "PersonX "+ head
          outf.write(json.dumps({'head': head, "relation": dat['relation'], "tail": dat['tail'],'p': dat['p_valid_model'] })+"\n")
  
      for idx, dat in enumerate(dataset['train']):
        personX = dat['PersonX']
        personY = dat['PersonY']
        personZ = dat['PersonZ']
        head = dat['head']
        if not head.startswith("PersonX"):
          head = "PersonX " + head
        relation = dat['relation']
        tail = dat['tail']
        narrative = dat['narrative'].replace(personX, "PersonX")
        literal = dat['literal'].replace(personX, "PersonX")
        dialogue = [s.replace(personX, "PersonX") for s in dat['dialogue']]
        mapper = {personX: 'PersonX'}
        if personY:
          mapper[personY] =  'PersonY'
        if personZ:
          mapper[personZ] = 'PersonZ'
        speakers = [mapper.get(s, s) for s in dat['speakers']]
        speakers_set = list(set([a for a in speakers if not a.startswith("Person")]))
        for a, b in zip(speakers_set, ["PersonY", "PersonZ"]):
          mapper[a] = b
        dialogue = "\n".join(f"{speaker}: {dialog}" for speaker, dialog in zip(speakers, dialogue)).replace(personX, "PersonX")
        for a, b in mapper.items():
          literal = literal.replace(a, b)
          narrative = narrative.replace(a, b)
          dialogue = dialogue.replace(a, b)
          literal = literal.replace(a.lower(), b)
          narrative = narrative.replace(a.lower(), b)
          dialogue = dialogue.replace(a.lower(), b)
        literal = literal.replace("the PersonY", "PersonY").replace("his PersonY", "PersonY").replace("her PersonY", "PersonY")
        narrative = narrative.replace("the PersonY", "PersonY").replace("his PersonY", "PersonY").replace("her PersonY", "PersonY")
        dialogue = dialogue.replace("the PersonY", "PersonY").replace("his PersonY", "PersonY").replace("her PersonY", "PersonY")
        if "PersonX" not in narrative:
          if " I "  in narrative:
            narrative = narrative.replace(" I ", " PersonX ").replace("I ", "PersonX ").replace(" I.", " PersonX.").replace(" I,", " PersonX,").\
              replace(" my ", " their ").replace(" mine ", " theirs ")
            outf.write (json.dumps({'head': head, "relation": relation, "tail": tail, 'literal': literal, 'narrative': narrative, 'dialogue': dialogue, })+"\n")
        else:
          outf.write (json.dumps({'head': head, "relation": relation, "tail": tail, 'literal': literal, 'narrative': narrative, 'dialogue': dialogue, })+"\n")
  
    !sort atomic_train.jsonl -o atomic_train.jsonl --parallel 32
  
  import json
  
  if not os.path.exists("atomic_train_collapse.jsonl"):
    prev_head = ""
    rel_to_tails = {}
    with open("atomic_train_collapse.jsonl", "w") as outf:
      with open("atomic_train.jsonl") as inf:
        for l in inf:
          dat = json.loads(l)
          head = dat['head'].strip()
          if prev_head and prev_head != head:
            for lst in rel_to_tails.values():
              lst.sort(key=lambda a: a[1], reverse=True)
            while True:
              rels = {}
              for key in list(rel_to_tails.keys()):
                if len(rel_to_tails[key]) > 5:
                  rels[key] = rel_to_tails[key][:5]
                  rel_to_tails[key] = rel_to_tails[key][5:]
                else:
                  rels[key] = rel_to_tails[key]
                  del rel_to_tails[key]
              outf.write(json.dumps({'head': prev_head, "relation": rels})+"\n")
              if any(v for v in rel_to_tails.values() if v):
                continue
              break
  
            prev_head = head
            rel_to_tails = {dat['relation']: [(dat['tail'], round(dat.get('p',1.0)*100)/100, dat.get('literal',''), dat.get('narrative',''), dat.get('dialogue',''))]}
          else:
            prev_head = head
            rel_to_tails[dat['relation']] = rel_to_tails.get(dat['relation'], []) + [(dat['tail'], round(dat.get('p',1.0)*100)/100, dat.get('literal',''), dat.get('narrative',''), dat.get('dialogue',''))]
        for lst in rel_to_tails.values():
            lst.sort(key=lambda a: a[1], reverse=True)
        while True:
          rels = {}
          for key in list(rel_to_tails.keys()):
            if len(rel_to_tails[key]) > 5:
              rels[key] = rel_to_tails[key][:5]
              rel_to_tails[key] = rel_to_tails[key][5:]
            else:
              rels[key] = rel_to_tails[key]
              del rel_to_tails[key]
          outf.write(json.dumps({'head': prev_head, "relation": rels})+"\n")
          if any(v for v in rel_to_tails.values() if v):
            continue
          break
  
  import json

def lst_to_sent(lst, connector=", ", ending=" and "):
  lst = [ret.replace("say, ", "say ").replace("Paul", "PersonY").replace("Jane", "PersonY").replace("John", "PersonY").replace("Susan", "PersonY").\
    replace("Jeff", "PersonY").replace("Mike", "PersonY")
   for ret in list(set(lst))]
  lst = list(set(connector.join(lst).split(connector)))
  if len(lst) > 1:
      last = lst[-1]
      return  connector.join(lst[:-1])+ending+ last
  elif len(lst) == 1:
    return lst[0]
  else:
    return ""

def output_story(txt, outf, dat):
  if " she " in txt or "She " in txt or " her " in txt or "herself" in txt:
    personX = random.choice(female_names)
    txt2 = txt.replace("PersonX", personX).replace("personX", personX).replace("personx", personX).replace("person x", personX).replace(" X's", " "+personX+"'s").replace(" X,", " "+personX+",").replace(" X.", " "+personX+".").replace(" X ", " "+personX + " ")
    name = random.choice(female_names+names)
    if name == personX:
      name = random.choice(female_names+names)
    txt2 = txt2.replace("PersonY", name).replace("personY", name).replace("persony", name).replace("person y", name).replace(" Y's", " "+personY+"'s").replace(" Y,", " "+personY+",").replace(" Y.", " "+personY+".").replace(" Y ", " "+personY + " ")
    if random.randint(0,1):
      txt2 = txt2.replace(" he ", " she ").replace(" He ", " She ").replace(" his ", " her ").replace(" him", " her")
    outf.write(json.dumps({'text': txt2, 'metadata': {'source': 'atomic2020_10x_SODA', 'rels': json.dumps(dat)}})+"\n")

    personX = random.choice(male_names)
    txt2 = txt.replace("PersonX", personX).replace("personX", personX).replace("personx", personX).replace("person x", personX).replace(" X's", " "+personX+"'s").replace(" X,", " "+personX+",").replace(" X.", " "+personX+".").replace(" X ", " "+personX + " ")
    name = random.choice(male_names+names)
    if name == personX:
      name = random.choice(male_names+names)
    txt2 = txt2.replace("PersonY", name).replace("personY", name).replace("persony", name).replace("person y", name).replace(" Y's", " "+personY+"'s").replace(" Y,", " "+personY+",").replace(" Y.", " "+personY+".").replace(" Y ", " "+personY + " ")
    txt2 = txt2.replace(" she ", " he ").replace(" She ", " He ").replace(" her ", " his ").replace(" her", " him")
    outf.write(json.dumps({'text': txt2, 'metadata': {'source': 'atomic2020_10x_SODA', 'rels': json.dumps(dat)}})+"\n")

  elif " he " in txt or "He " in txt or " his " in txt or "himself" in txt or " him " in txt:
    personX = random.choice(male_names)
    txt2 = txt.replace("PersonX", personX).replace("personX", personX).replace("personx", personX).replace("person x", personX).replace(" X's", " "+personX+"'s").replace(" X,", " "+personX+",").replace(" X.", " "+personX+".").replace(" X ", " "+personX + " ")
    name = random.choice(male_names+names)
    if name == personX:
      name = random.choice(male_names)
    if random.randint(0,1): txt2 = txt2.replace(" she ", " he ").replace(" She ", " He ").replace(" her ", " his ").replace(" her", " him")
    txt2 = txt2.replace("PersonY", name).replace("personY", name).replace("persony", name).replace("person y", name).replace(" Y's", " "+personY+"'s").replace(" Y,", " "+personY+",").replace(" Y.", " "+personY+".").replace(" Y ", " "+personY + " ")
    outf.write(json.dumps({'text': txt2, 'metadata': {'source': 'atomic2020_10x_SODA', 'rels': json.dumps(dat)}})+"\n")

    txt2 = txt.replace(" he ", " she ").replace(" He ", " She ").replace(" his ", " her ").replace(" him", " her")
    personX = random.choice(female_names)
    txt2 = txt2.replace("PersonX", personX).replace("personX", personX).replace("personx", personX).replace("person x", personX).replace(" X's", " "+personX+"'s").replace(" X,", " "+personX+",").replace(" X.", " "+personX+".").replace(" X ", " "+personX + " ")
    name = random.choice(female_names+names)
    if name == personX:
      name = random.choice(female_names+names)
    txt2 = txt2.replace("PersonY", name).replace("personY", name).replace("persony", name).replace("person y", name).replace(" Y's", " "+personY+"'s").replace(" Y,", " "+personY+",").replace(" Y.", " "+personY+".").replace(" Y ", " "+personY + " ")
    outf.write(json.dumps({'text': txt2, 'metadata': {'source': 'atomic2020_10x_SODA', 'rels': json.dumps(dat)}})+"\n")
  else:
    personX = random.choice(names)
    txt = txt.replace("PersonX", personX).replace("personX", personX).replace("personx", personX).replace("person x", personX).replace(" X's", " "+personX+"'s").replace(" X,", " "+personX+",").replace(" X.", " "+personX+".").replace(" X ", " "+personX + " ")
    name = random.choice(names)
    if name == personX:
      name = random.choice(names)
    txt = txt.replace("PersonY", name).replace("personY", name).replace("persony", name).replace("person y", name).replace(" Y's", " "+personY+"'s").replace(" Y,", " "+personY+",").replace(" Y.", " "+personY+".").replace(" Y ", " "+personY + " ")
    outf.write(json.dumps({'text': txt, 'metadata': {'source': 'atomic2020_10x_SODA', 'rels': json.dumps(dat)}})+"\n")

def create_stories_from_atomic(likely=0.75, unlikely=0.50):
  prev_head = ""
  rel_to_tails = {}
  with open("atomic_stories.jsonl", "w") as outf:
    with open("atomic_train_collapse.jsonl") as inf:
      for idx, l in enumerate(inf):
        dat = json.loads(l)
        head = dat['head']
        if not head.strip(". \n"): continue
        head = head.replace(". . .", "something")
        rel_to_tails_likely= dict((a, [b1 for b1 in b if b1[1] > likely]) for a, b in dat['relation'].items())
        scenes =  {}
        for key in ['xAttr', 'isAfter', 'xWant', 'xNeed', 'xReason', 'xReact', 'xIntent', 'xEffect', 'oWant', 'oEffect', 'oReact', 'isBefore']:
            scenes[key] =   [a for a in [(("\nSCENE: "+a[2]+"\n") if a[2] else '') + ((a[3]+"\n") if a[3] else '') + ((a[4]+"\n") if a[4] else '')
                    for a in rel_to_tails_likely.get(key,[])] if a.strip()]
        rel_to_tails_likely= dict((a, [b1[:2] for b1 in b if b1[1] > likely]) for a, b in dat['relation'].items())
        rel_to_tails_unlikely= dict((a, [b1[:2] for b1 in b if b1[1] <= likely and b1[1] >= unlikely]) for a, b in dat['relation'].items())
        rels_to_tail_likely_unlikely  = dict((a, [b1[:2] for b1 in b if b1[1] >= unlikely]) for a, b in dat['relation'].items())

        if 'HinderedBy' in rel_to_tails_unlikely: del rel_to_tails_unlikely['HinderedBy']
        if 'HinderedBy' in rels_to_tail_likely_unlikely: del rels_to_tail_likely_unlikely['HinderedBy']
        if 'HinderedBy'in rel_to_tails_likely and rel_to_tails_likely['HinderedBy']:
          dat['relation'] = rel_to_tails_likely
          txt = '#### ' + head.replace("PersonX ", "PersonX wants to ").replace(" is ", " be ") +" but won't or can't.\n\n"
          lst =  [a[0].replace(".", " ")  for a in rel_to_tails_likely.get('xAttr',[])]
          if lst: txt += "PersonX is "+ lst_to_sent(lst)+".\n"

          lst = [a[0].replace(".", " ")  for a in rel_to_tails_likely.get('isAfter',[])]
          if lst:
            lst.sort()
            txt += "First "+ lst_to_sent(lst, ". ", ". And ")+".\n"


          lst =  [a[0].replace(".", " ")  for a in rel_to_tails_likely.get('xWant',[])]
          if lst:
            lst.sort()
            txt += "PersonX wants " +  lst_to_sent(lst)+".\n"

          #print (rel_to_tails_likely)
          lst =  [a[0].replace("to ", "") for a in rel_to_tails_likely.get('xNeed',[])]
          if lst:
            lst.sort()
            txt += "PersonX might " +  (" "+lst_to_sent(lst)).replace(" is ", " becomes ").replace("s ", " ")+".\n"
          ###
          txt += "PersonX might "+(" "+head[len("PersonX "):]).replace(" is ", " becomes ").replace("s ", " ").replace(" a ", " the ").replace(" an ", " the ")+".\n"
          ###
          lst =  [a[0] for a in rel_to_tails_likely.get('xReason',[])]
          if lst:
            lst.sort()
            txt += "Because "+  lst_to_sent(lst)+".\n"

          lst =  [a[0].replace(".", " ") for a in rel_to_tails_likely.get('HinderedBy',[])]
          if lst:
            lst.sort()
            txt += "But PersonX can't or won't because "+  (" "+lst_to_sent(lst)).replace(" is ", " becomes ")+".\n"
          lst =  [a[0] for a in rel_to_tails_likely.get('xIntent',[])]
          if lst:
            lst.sort()
            txt += "So PersonX wont' intended to "+  lst_to_sent(lst)+".\n"
          lst =  [a[0] for a in rel_to_tails_likely.get('xEffect',[])]
          if lst:
            lst.sort()
            txt += "Therefore, PersonX will not "+  (" "+lst_to_sent(lst)).replace(" is ", " be ").replace("s ", " ")+".\n"
          lst =  ["want to "+a[0][len("to "):] if a[0].startswith("to ") else a[0] for a in rel_to_tails_likely.get('oWant',[])]
          if lst:
            lst.sort()
            txt += "Others won't: "+  lst_to_sent(lst)+".\n"

          lst =  [a[0].replace(".", " ")  for a in rel_to_tails_likely.get('oEffect',[])]
          if lst:
            lst.sort()
            txt += "Thus others will also not: " +  lst_to_sent(lst)+".\n"
          lst =  [a[0].replace(".", " ")  for a in rel_to_tails_likely.get('oReact',[])]
          if lst:
            lst.sort()
            txt += "The others won't feel " +  lst_to_sent(lst)+".\n"

          lst = [a[0].replace(".", " ")  for a in rel_to_tails_likely.get('isBefore',[])]
          if lst:
            lst.sort()
            txt += "Finally, none of the following will likely occur: "+ lst_to_sent(lst, ". ", ". And ")+".\n"
          txt = txt.replace(" hi ", " his ").replace("  ", " ").replace(" ,", ",")
          if len(txt) >= 100:
            if "PersonY" in txt:
              txt = txt.replace(" others ", " PersonY ").replace("the PersonY", "PersonY")
            output_story(txt, outf, dat)
        if 'HinderedBy' in rel_to_tails_likely: del rel_to_tails_likely['HinderedBy']

        if True:
          dat['relation'] = rel_to_tails_likely
          txt = '#### ' + head +"\n\n"
          lst =  [a[0].replace(".", " ")  for a in rel_to_tails_likely.get('xAttr',[])]
          xAttr = lst_to_sent(lst, ", ", " and ")
          if xAttr: txt += "PersonX is "+ xAttr+".\n"

          lst = [a[0].replace(".", " ")  for a in rel_to_tails_likely.get('isAfter',[])]
          if lst:
            lst.sort()
            txt += "First "+ lst_to_sent(lst, ". ", ". And ")+".\n"


          lst =  [a[0].replace(".", " ") for a in rel_to_tails_likely.get('xWant',[])]
          if lst:
            lst.sort()
            txt += "PersonX wants "+  lst_to_sent(lst)+".\n"



          #print (rel_to_tails_likely)
          lst =  [a[0].replace(" be ", " is ").replace("to ", "").replace(".", " ")  for a in rel_to_tails_likely.get('xNeed',[])]
          if lst:
            lst.sort()
            txt += "PersonX "+ lst_to_sent(lst)+".\n"
          ###
          txt += "So "+ head.replace(" a ", " the ").replace(" an ", " the ")+".\n"
          ###
          lst =  [a[0] for a in rel_to_tails_likely.get('xReason',[])]
          if lst:
            lst.sort()
            txt += "Because "+  lst_to_sent(lst)+".\n"

          lst =  [a[0] for a in rel_to_tails_likely.get('xReact',[])]
          if lst:
            lst.sort()
            txt += "Then PersonX feels "+  lst_to_sent(lst)+".\n"
          lst =  [a[0] for a in rel_to_tails_likely.get('xIntent',[])]
          if lst:
            lst.sort()
            txt += "Then PersonX intended to "+  lst_to_sent(lst)+".\n"
          lst =  [a[0].replace(".", " ")  for a in rel_to_tails_likely.get('xEffect',[])]
          if lst:
            lst.sort()
            txt += "Therefore, PersonX "+  lst_to_sent(lst)+".\n"
          lst =  ["want to "+a[0][len("to "):].replace(".", " ") if a[0].startswith("to ")  else a[0].replace(".", " ") for a in rel_to_tails_likely.get('oWant',[])]
          if lst:
            lst.sort()
            txt += "Others "+ lst_to_sent(lst)+".\n"
          lst =  [a[0].replace(".", " ")  for a in rel_to_tails_likely.get('oEffect',[])]
          if lst:
            lst.sort()
            txt += "This makes others "+  lst_to_sent(lst)+".\n"
          lst =  [a[0].replace(".", " ")  for a in rel_to_tails_likely.get('oReact',[])]
          if lst:
            lst.sort()
            txt += "This also makes others feel "+  lst_to_sent(lst)+".\n"

          lst = [a[0].replace(".", " ")  for a in rel_to_tails_likely.get('isBefore',[])]
          if lst:
            lst.sort()
            txt += "Finally "+  lst_to_sent(lst, ". ", ". And ")+".\n"


          txt = txt.replace(" hi ", " his ").replace("  ", " ").replace(" ,", ",")+"\n"
          found=False
          #"oEffect","oReact","oWant","xAttr" ,"xEffect" ,"xIntent" ,"xWant" ,"xReact" ,
          #"xNeed","HinderedBy","isBefore" ,"isAfter","xReason"
          for key in ['xAttr', 'isAfter', 'xWant', 'xNeed', 'xReason', 'xReact', 'xIntent', 'xEffect', 'oWant', 'oEffect', 'oReact', 'isBefore']:
            lst =  scenes[key]
            for rng in  range(0, len(lst), 3):
              lst2 = lst[rng:min(len(lst), rng+3)]
              txt2 = txt +  "\n".join(lst2)
              if "PersonY" in txt2:
                txt2 = txt2.replace(" others ", " PersonY ").replace("the PersonY", "PersonY")
              output_story(txt2, outf, dat)
              found = True

          if not found:
            if len(txt) < 100: continue
            if "PersonY" in txt:
              txt = txt.replace(" others ", " PersonY ").replace("the PersonY", "PersonY")
            output_story(txt, outf, dat)
            #if ( " rape" in txt or 'kidnap' in txt or ' incest ' in txt):
            #   print (txt)

        if  any(v for v in rel_to_tails_unlikely.values() if v):
          dat['relation'] = rels_to_tail_likely_unlikely
          if 'HinderedBy' in dat['relation'] : del dat['relation'] ['HinderedBy']

          txt = '#### ' + head +" with some suprising events\n\n"
          lst =  [a[0].replace(".", " ")  for a in rel_to_tails_unlikely.get('xAttr',[])]
          if not lst:
            if random.randint(0,1):
              lst =  [a[0].replace(".", " ")  for a in rel_to_tails_likely.get('xAttr',[])]
              xAttr = lst_to_sent(lst, ", ", " and ")
              if xAttr: txt += "PersonX is "+ xAttr+".\n"
          else:
            xAttr = lst_to_sent(lst, ", ", " and ")
            if xAttr: txt += "It wasn't apparatent but PersonX is "+ xAttr+".\n"

          lst =  [a[0].replace(".", " ")  for a in rel_to_tails_unlikely.get('isAfter',[])]
          if not lst:
            if random.randint(0,1):
              lst =  [a[0].replace(".", " ")  for a in rel_to_tails_likely.get('isAfter',[])]
              if lst:
                lst.sort()
                txt += "First "+ lst_to_sent(lst, ". ", ". And ")+".\n"
          else:
            lst.sort()
            txt += "Strangley, first "+ lst_to_sent(lst, ". ", ". And ")+".\n"


          lst =  [a[0].replace(".", " ")  for a in rel_to_tails_unlikely.get('xWant',[])]
          if not lst:
            if random.randint(0,1):
              lst =  [a[0].replace(".", " ")  for a in rel_to_tails_likely.get('xWant',[])]
              if lst:
                lst.sort()
                txt += "PersonX wants "+  lst_to_sent(lst)+".\n"
          else:
            lst.sort()
            txt += "Unknown to most people, PersonX wants "+  lst_to_sent(lst)+".\n"


          #print (rel_to_tails_likely)
          lst =  [a[0].replace(" be ", " is ").replace("to ", "").replace(".", " ")  for a in rel_to_tails_unlikely.get('xNeed',[])]
          if not lst:
            if random.randint(0,1):
              lst =  [a[0].replace(" be ", " is ").replace("to ", "").replace(".", " ")  for a in rel_to_tails_likely.get('xNeed',[])]
              if lst:
                lst.sort()
                txt += "PersonX "+ lst_to_sent(lst)+".\n"
          else:
            lst.sort()
            txt += "Suprisingly PersonX "+ lst_to_sent(lst)+".\n"
          ###
          txt += "So "+ head.replace(" a ", " the ").replace(" an ", " the ")+".\n"
          ###

          lst =  [a[0].replace(".", " ")  for a in rel_to_tails_unlikely.get('xReason',[])]
          if not lst:
            if random.randint(0,1):
              lst =  [a[0].replace(".", " ")  for a in rel_to_tails_likely.get('xReason',[])]
              if lst:
                lst.sort()
                txt += "Because "+  lst_to_sent(lst)+".\n"
          else:
            lst.sort()
            txt += "But because "+  lst_to_sent(lst)+".\n"
          lst =  [a[0].replace(".", " ")  for a in rel_to_tails_unlikely.get('xReact',[])]
          if not lst:
            if random.randint(0,1):
              lst =  [a[0].replace(".", " ")  for a in rel_to_tails_likely.get('xReact',[])]
              if lst:
                lst.sort()
                txt += "Then PersonX feels "+  lst_to_sent(lst)+".\n"
          else:
            lst.sort()
            txt += "But PersonX feels "+  lst_to_sent(lst)+".\n"
          lst =  [a[0].replace(".", " ")  for a in rel_to_tails_unlikely.get('xIntent',[])]
          if not lst:
            if random.randint(0,1):
              lst =  [a[0].replace(".", " ")  for a in rel_to_tails_likely.get('xIntent',[])]
              if lst:
                lst.sort()
                txt += "Then PersonX intended to "+  lst_to_sent(lst)+".\n"
          else:
            lst.sort()
            txt += "But then PersonX intended to "+  lst_to_sent(lst)+".\n"

          lst =  [a[0].replace(".", " ")  for a in rel_to_tails_unlikely.get('xEffect',[])]
          if not lst:
            if random.randint(0,1):
              lst =  [a[0].replace(".", " ")  for a in rel_to_tails_likely.get('xEffect',[])]
              if lst:
                lst.sort()
                txt += "Therefore, PersonX "+  lst_to_sent(lst)+".\n"
          else:
            lst.sort()
            txt += "Suprisingly, PersonX "+  lst_to_sent(lst)+".\n"
          lst =  ["want to "+a[0][len("to "):].replace(".", " ") if a[0].startswith("to ")  else a[0].replace(".", " ") for a in rel_to_tails_unlikely.get('oWant',[])]
          if not lst:
            if random.randint(0,1):
              lst =  ["want to "+a[0][len("to "):].replace(".", " ") if a[0].startswith("to ")  else a[0].replace(".", " ") for a in rel_to_tails_likely.get('oWant',[])]
              if lst:
                lst.sort()
                txt += "Others "+ lst_to_sent(lst)+".\n"
          else:
            lst.sort()
            txt += "And then suddently others "+ lst_to_sent(lst)+".\n"
          lst =  [a[0].replace(".", " ")  for a in rel_to_tails_unlikely.get('oEffect',[])]
          if not lst:
            if random.randint(0,1):
              lst =  [a[0].replace(".", " ")  for a in rel_to_tails_likely.get('oEffect',[])]
              if lst:
                lst.sort()
                txt += "This makes others "+  lst_to_sent(lst)+".\n"
          else:
            lst.sort()
            txt += "In a turn of events, this makes others "+  lst_to_sent(lst)+".\n"

          lst =  [a[0].replace(".", " ")  for a in rel_to_tails_unlikely.get('oReact',[])]
          if not lst:
            if random.randint(0,1):
              lst =  [a[0].replace(".", " ")  for a in rel_to_tails_likely.get('oReact',[])]
              if lst:
                lst.sort()
                txt += "This also makes others feel "+  lst_to_sent(lst)+".\n"
          else:
              lst.sort()
              txt += "For some reason, this also makes others feel "+  lst_to_sent(lst)+".\n"

          lst =  [a[0].replace(".", " ")  for a in rel_to_tails_unlikely.get('isBefore',[])]
          if not lst:
            if random.randint(0,1):
              lst =  [a[0].replace(".", " ")  for a in rel_to_tails_likely.get('isBefore',[])]
              if lst:
                lst.sort()
                txt += "Finally "+  lst_to_sent(lst, ". ", ". And ")+".\n"
          else:
            lst.sort()
            txt += "In a twist, finally "+  lst_to_sent(lst, ". ", ". And ")+".\n"

          txt = txt.replace(" hi ", " his ").replace("  ", " ").replace(" ,", ",")+"\n"
          if "PersonY" in txt:
            txt = txt.replace(" others ", " PersonY ").replace("the PersonY", "PersonY")
            #if idx < 1000:
            #  print (txt)
          if len(txt) >= 100:
            #if idx < 100:
            #    print (dat)
            #    print (txt)

            output_story(txt, outf, dat)

        #print (head, rel_to_tails_likely, rel_to_tails_unlikely)
        #if idx > 200: break

#create_stories_from_atomic()  

def llm_generate_stories_from_atomic():
  #@title generate stories from atomic seeds
  # make sure you have the latest version of transfomers and install wget
  from transformers import pipeline
  
  from transformers import AutoTokenizer, AutoModelForCausalLM
  try:
    if tokenizer is None: assert False
  except:
    tokenizer = AutoTokenizer.from_pretrained("UCLA-AGI/Gemma-2-9B-It-SPPO-Iter3")
    model = AutoModelForCausalLM.from_pretrained("UCLA-AGI/Gemma-2-9B-It-SPPO-Iter3").half().cuda()
  
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
  
  
  prompts = ["""Revise this story to make it compelling and more logical and detailed. Keep as much of the feelings and actions as possible, but remove anything that doesn't make sense. Make the story at least 10 paragraphs. Start with a title. %(warning)s The story should unfold through the characters interactions, decisions, and the consequences of their actions. Aim to weave in common sense lessons and social cues. The narrative should cater to a diverse age group, including at least one dialogue and presenting both positive and negative outcomes. Do not start with classic sentences like "Once upon a time", be creative:""",
            """Revise this story to make it compelling and more logical and detailed. Keep as much of the feelings and actions as possible, but remove anything that doesn't make sense. Make the story at least 10 paragraphs. Start with a title. %(warning)s Write as a real-life story shared by someone in a social media forum. The story should include:
  - Niche interests or humor: dive into specific hobbies, interests, or humorous situations
  - An unexpected plot twist or engaging conflict: introduce a relatable yet challenging situation or dilemma that the author faced.
  - Reflection and insight: end with a resolution that offers a new understanding, a sense of community, or a personal revelation, much like the conclusions drawn in forum discussions.
  Start the story right away. Do not start with sentences like  "Once upon a time" as this is a reddit post and not a novel, you should also avoid starting with classic sentences like "A few years ago" or "A few years back", be creative:""",
            """Revise this story to make it compelling and more logical and detailed. Keep as much of the feelings and actions as possible, but remove anything that doesn't make sense. Make the story at least 10 paragraphs. Start with a title. %(warning)s Write the story in the style of real-life situations that people share in forums. The story needs to include a compelling and unexpected plot twist. Your narrative should resonate with the authenticity and personal touch found in forum discussions. Include relatable events and emotional depth. Do not start with classic sentences like "Once upon a time", "A few years back" or "A few months ago", be creative:""",
            """Revise this story to make it compelling and more logical and detailed. Keep as much of the feelings and actions as possible, but remove anything that doesn't make sense. Make the story at least 10 paragraphs. Start with a title. %(warning)s The story should incorporate the following elements:
  - Dialogue: the story must feature at least one meaningful dialogue that reveals character depth, advances the plot, or unravels a crucial piece of the mystery
  - Interesting themes: explore themes resonant with a mature audience, such as moral ambiguity, existential queries, personal transformation, or the consequences of past actions.
  Do not start with classic sentences like "Once upon a time", "The sun hung low in the sky" or "In the dimly lit", be creative:"""]
  
  
  
  import wget, json, random, os
  
  url = 'https://huggingface.co/datasets/ontocord/atomic_2024/resolve/main/data/atomic_stories.jsonl'
  if not os.path.exists("atomic_stories.jsonl"):
    wget.download(url)
  with open("generated_atomic_stories.jsonl", "w") as outfile:
    with open("atomic_stories.jsonl") as infile:
      for l in infile:
        dat = json.loads(l)
        text = dat['text']
        prompt = random.choice(prompts)
        if 'assault' in text or 'robbery' in text or 'arson' in text or 'fellatio' in text or 'hand job' in text or 'prostitu' in text or 'handjob' in text or 'fucks' in text or 'blow job' in text or 'blowjob' in text or ' incest' in text or ' porn' in text or ' rape' in text or ' killer' in text or ' murder' in text or ' kidnap' in text or ' abduct' in text or ' sex ' in text  or ' torture' in text or ' kills ' in text:
          warning = "If this story contains themes of sex or violence, give a warning at the beginning of the story with an explanation."
        else:
          warning = ""
        prompt = prompt %{'warning': warning}
        #print (prompt)
        messages = [
          {"role": "user", "content": f"{prompt}\n\n{text}"}
        ]
        output_text = pipe(messages, max_length=2048, min_length=512, use_cache=True)[0]['generated_text'][1]['content']
        #print (output_text)
        outfile.write(json.dumps({'text': output_text, 'prompt': prompt, 'input': text, 'metadata': dat['metadata']})+"\n")
  
