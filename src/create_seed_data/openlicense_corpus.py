def load_olc():
  """%cd /content/
!GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/kernelmachine/open-license-corpus
%cd /content/open-license-corpus
!git lfs pull --include "data/sw_hackernews/*"
!git lfs pull --include "data/pd_arxiv_abstracts/*"
!git lfs pull --include "data/ccby_law/*"
!git lfs pull --include "data/ccby_wikinews/*"
!git lfs pull --include "data/pd_law/*"
!git lfs pull --include "data/pd_news/*"
!git lfs pull --include "data/sw_amps_math/*"
!git lfs pull --include "data/sw_dm_math/*"
!git lfs pull --include "data/sw_ubuntua_irc/*"
  """


from num2words import num2words

#dm math is deepmind/math which we can get on our own
%cd /content/open-license-corpus
import glob, json, gzip, random

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

from typing import List
import re
from huggingface_hub import hf_hub_download
try:
  import fasttext
except:
  !pip install fasttext
import fasttext
try:
  if textbook_model is None: assert False
except:
  textbook_model = fasttext.load_model(hf_hub_download("kenhktsui/llm-data-textbook-quality-fasttext-classifier-v2", "model.bin"))

try:
  if rpj_model is None: assert False
except:
  rpj_model = fasttext.load_model(hf_hub_download("ontocord/riverbed", "rj_model.bin"))


def predict(model, text, ltype):
  pred = model.predict(text.lower().replace("\n", " ")[:min(1000, len(text))].replace(" she ", " he ").replace(" her ", " his"))
  pred2 = model.predict(text.lower().replace("\n", " ")[-min(1000, len(text)):].replace(" she ", " he ").replace(" her ", " his"))
  label =  pred[0][0]
  label2 =  pred2[0][0]
  if ltype not in label and ltype not in label2:
    return max(1 - pred[1][0], 1 - pred2[1][0])
  if ltype not in label2:
    return 1 - pred[1][0]
  if ltype not in label:
    return 1 - pred2[1][0]
  ret =  max(pred[1][0],pred2[1][0])
  if ret < 0.1 and len(text) > 1000:
    text = " ".join(text[500:].split()[1:])
    pred = model.predict(text.lower().replace("\n", " ")[:min(1000, len(text))].replace(" she ", " he ").replace(" her ", " his"))
    return max(ret, pred)
  return ret

import string
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

def remove_non_ascii(text):
    return re.sub(r'[^\x00-\x7F]', ' ', text)

def txt2txtnum(txt):
  txt = txt.replace(")", " ) ").replace("(", " ( ").replace("/", " / ").replace("?", " ?").replace("<<", " << ").replace(">>", " >> ").replace("=", " = ").replace("+", " + ").replace("-", " -").replace("*", " * ").replace("^", " ^ ").replace("  ", " ").replace(" * * ", " ** ")
  txt_arr = txt.split(" ")
  txt_arr2 = []
  for w in txt_arr:
    if len(w) > 1:
      try:
        float(w.strip())
        txt_arr2.append (w.replace(w.strip(), w.strip() + " ("+("negative " if w.startswith("-0.") else "")+num2words(float(w))+")"))
      except:
        txt_arr2.append (w)
    else:
      txt_arr2.append (w)

  txt =  " ".join(txt_arr2).replace("(minus ", "(negative ").replace(" .", ".").replace(" ?", "?").replace(" )", ")").replace("( ", "(").replace("  ", " ")
  return (txt)

import random
%cd /content/open-license-corpus
for pat in [
             "data/pd_arxiv_abstracts/*",
             "data/sw_dm_math/*",
             #"data/sw_hackernews/*",
             #"data/sw_amps_math/*",
             #"data/pd_law/*",
              #"data/pd_news/*",
             #"data/ccby_law/*",
             #"data/ccby_wikinews/*",
              ]:
  with open("/content/drive/Shareddrives/ontocord_llc/olc/"+pat.split("/")[1]+".jsonl", "w") as outf:

    amps_skip = 0
    files = list(glob.glob("/content/open-license-corpus/"+pat))
    random.shuffle(files)
    for num_file, file in enumerate(files):
      #if num_file > 0: break
      #print (file)
      seen = {}
      try:
        f = gzip.open(file)
      except:
        print ("could not load", file)
        continue
      idx = 0
      prev_text = ""
      curr_dm_math = ""
      while True:
        try:
          l = next(f)
        except:
          print ('problem reading file', file)
          break #continue
        idx += 1
        names_list = []
        l = l.decode()
        dat = json.loads(l)

        # this is reddit which is not copyright permissive
        if pat == "data/ccby_law/*" and dat['text'].startswith("Title:"):
          continue

        dat['text']= dat['text'].replace("\r\n", "\n").replace("\r", "\n").replace("\t", "   ")
        if pat in {"data/pd_law/*", "data/ccby_wikinews/*", "data/pd_arxiv_abstracts/*", "data/ccby_law/*"}:
          dat['text']= dat['text'].replace("\n\n\n\n", "\n").replace("\n\n\n", "\n").replace("\n\n", "<n>").replace("\n", " ").replace("<n>", "\n")
        dat['text']= dat['text'].replace(".  ", ". ").replace(".  ", ". ")
        dat['text']= dat['text'].strip("\n_ ")
        dat['text']= dat['text'].replace("Wikinews", "Our news service")
        dat['meta'] = {'source': pat.split("/")[1]}
        if len(dat['text']) == 0: continue
        # cleanup hackernews and add names by each conversation thread
        if pat == "data/sw_hackernews/*":
          if dat['text'].count("~")/len(dat['text']) > 0.01:
            continue
          dat['text'] = dat['text'].replace("Thanks!", "")
          dat['text'] = dat['text'].replace("Thanks", "")
          dat['text'] = dat['text'].replace("Ask HN:", "Question:").replace("HN!", "")
          dat['text'] = dat['text'].replace("Show HN:", "Title:")
          dat['text'] = dat['text'].replace("Ask the Wizard:", "Title:")
          dat['text'] = dat['text'].replace("<p>", "\n\n").strip()
          dat['text'] = " ".join(a if "http" not in a else "\n\n" for a in dat['text'].split(" "))
          if  "-" in dat['text'][:100]:
            a, b  = dat['text'].split("-",1)
            poster = ""
            if "\n" in b:
              poster, b = b.split("\n",1)
              poster = poster.strip()
            dat['text'] = a.strip()+"\n"+b.strip()
            if poster:
              dat['text']  = dat['text'].replace(poster+"\n", "\n\nFollow-up: ").replace(poster, "\n\nFollow-up:")

          dat['text'] = dat['text'].replace("\n\n","<n>").replace("\n", " ").replace("<n>","\n")
          text1 = ""
          for a in dat['text'].split("\n"):
            a = a.strip()
            if a:
              if a == "====": continue
              arr = a.split()
              if a.startswith("~~~") and len(arr) > 1:
                arr = arr[1:]
                names_list.append(arr[0])
                arr[0] = arr[0]+"@:@"
              elif ":" not in arr[0] and len(arr) > 1 and (arr[1][0] == arr[1][0].upper()) and ")" not in arr[0] and "." not in arr[0] \
                  and "-" not in arr[0]  and ">" not in arr[0]  and "(" not in arr[0] and "," not in arr[0] and '"' not in arr[0] and \
                  arr[0].lower() not in {"but", "and", "or", "btw", "a1", "a2", "a3", "a4", "~~~", "we", "in", "on", "they", "this", "that", "what", "where", "why", "when", "i", "you", "aren't", "are", "is", "if", "of"}:
                names_list.append(arr[0])
                arr[0] = arr[0]+"@:@"
              a = " ".join(arr)
              if ">" in a:
                a, b = a.split(">",1)
                a = a+"\n  > "+ b.replace(">", " ").replace("  ", " ")
              text1 += "\n" + a
          text1 =  text1.strip("=*\n ")
          names_list = list(set(names_list))
          random.shuffle(names)
          for name1, name2 in zip(names_list, names):
            text1 = text1.replace(name1+"@:@", "\n"+name2+"@:@")
          if sum(len(a) for a in text1.split("\n"))/len(text1.split("\n")) < 30:
            continue
          dat['text'] = text1
          if len(dat['text']) < 200 or "@:@" not in dat['text']: continue
          dat['text'] = dat['text'].replace("@:@ ", ":\n").replace("@:@", ":\n").replace("\n\n  >", "\n  >").replace(" HN ", " Our Commmunity ").strip()
          dat['text'] = "### SOCIAL MEDIA DISCUSSION:\n" + dat['text']
        #now let's do basic text cleanup to get rid of template beginning text
        elif pat in {'data/pd_law/*', "data/ccby_law/*"}:
          for k in range(2):
            dat['text'] = dat['text'].strip()
            if len(dat['text']) < 100:
                break
            if dat['text'].startswith("DETAILED ACTION"):
              txt2 = dat['text'].split("\n")
              for idx, txt in enumerate(txt2):
                if len(txt) < 60 or "DETAILED ACTION" in txt or "Response to Arguments" in txt or "Applicant’s arguments" in txt or "Response to Amendment" in txt or "Conclusion" in txt or "Reasons for Allowance" in txt or "(where" in txt or ".jpg" in txt or ".png" in txt or not txt.strip("0123456789 ")  or "Greyscale" in txt or "PNG" in txt:
                  txt2[idx] = ''
              dat['text'] = "\n".join(txt2).strip().replace("\n\n", "\n")
              if len(dat['text']) < 100:
                break
            if "On appeal from"  in dat['text'][:200]:
              dat['text']= dat['text'].split("On appeal from",1)[-1].strip(", \n")
            if """the Securities Exchange Act of 1934 (the "Act")""" in dat['text'][:200]:
              dat['text']= dat['text'].split("""the Securities Exchange Act of 1934 (the "Act")""",1)[-1].strip(", \n")
            if """the Securities Exchange Act of 1934""" in dat['text'][:200]:
              dat['text']= dat['text'].split("""the Securities Exchange Act of 1934""",1)[-1].strip(", \n")
            if """THE SECURITIES EXCHANGE ACT OF 1934""" in dat['text'][:200]:
              dat['text']= dat['text'].split("""THE SECURITIES EXCHANGE ACT OF 1934""",1)[-1].strip(", \n")
            if """of the Sarbanes-Oxley Act of 2002""" in dat['text'][:200]:
              dat['text']= dat['text'].split("""of the Sarbanes-Oxley Act of 2002""",1)[-1].strip(", \n")
            if """OF THE SECURITIES EXCHANGE ACT OF 1934""" in dat['text'][:200]:
              dat['text']= dat['text'].split("""OF THE SECURITIES EXCHANGE ACT OF 1934""",1)[-1].strip(", \n")
            if """CERTIFICATE OF FORMATION OF""" in dat['text'][:200]:
              dat['text']= dat['text'].split("""CERTIFICATE OF FORMATION OF""",1)[-1].strip(", \n")
            if """ACT OF 2002""" in dat['text'][:200]:
              dat['text']= dat['text'].split("""ACT OF 2002""",1)[-1].strip(", \n")
            if """ACT OF 1934""" in dat['text'][:200]:
              dat['text']= dat['text'].split("""ACT OF 1934""",1)[-1].strip(", \n")
            if """OF THE SECURITIES EXCHANGE ACT""" in dat['text'][:200]:
              dat['text']= dat['text'].split("""OF THE SECURITIES EXCHANGE ACT""",1)[-1].strip(", \n")
            if """for SEC Filing""" in dat['text'][:200]:
              dat['text']= dat['text'].split("""for SEC Filing""",1)[-1].strip(", \n")
            if """FILED PURSUANT TO""" in dat['text'][:200]:
              dat['text']= dat['text'].split("""FILED PURSUANT TO""",1)[-1].strip(", \n")
            if dat['text'].startswith("EXHIBIT"):
              dat['text'] = dat['text'].split("EXHIBIT",1)[1].lstrip("()1234567890. \n")
            if dat['text'].startswith("Exhibit"):
              dat['text'] = dat['text'].split("Exhibit",1)[1].lstrip("()1234567890. \n")
            if "Dear Sir""" in dat['text'][:200]:
              dat['text']= "Dear Sir"+dat['text'].split("""Dear Sir""",1)[-1].strip(", \n")
          txt2 = dat['text'].split("\n")
          for idx, txt in enumerate(txt2):
            txt2[idx] = txt2[idx].rstrip(" \r\t")
            if ".com" in txt:
              txt2[idx] = ''
            if (idx < 20 or len(txt) > 20) and get_special_char_score(txt) > 0.2:
              txt2[idx] = ''
          dat['text'] = "\n".join([t.rstrip() for t in txt2 if t.strip()]).replace("\n\n", "\n")
          dat['text'] = dat['text'].strip()
          txt = dat['text'].lower()
          if " v.\n" in txt or " v. " in txt:
            dat['text'] = "### CASE LAW DOCUMENT:\n" + dat['text']
          elif "agreement" in txt or 'contract' in txt:
            dat['text'] = "### CONTRACTUAL AGREEMENT DOCUMENT:\n" + dat['text']
          elif "patent" in txt or 'invention' in txt:
            dat['text'] = "### PATENT RELATED DOCUMENT:\n" + dat['text']
          else:
            dat['text'] = "### LEGAL RELATED DOCUMENT:\n" + dat['text']
        elif pat in {"data/ccby_wikinews/*","data/pd_news/*"}:
          txt2 = dat['text'].split("\n")
          for idx, txt in enumerate(txt2):
            txt2[idx] = txt2[idx].rstrip(" \r\t")
            if (idx < 20 or len(txt) > 20) and get_special_char_score(txt) > 0.2:
              txt2[idx] = ''
            if "Listen: MP3 -->" in txt:
              txt2[idx] = ''
          txt2 = "\n".join([t.rstrip() for t in txt2 if t.strip()]).replace("\n\n", "\n")
          if 'Our news service' in txt2:
            idx = txt2.index('Our news service')
            txt2 = txt2[:idx+20] + txt2[idx+20:].replace('Our news service', 'We').replace('"Our news service"', 'We').replace('"We"', "We").replace(", We ", ", we ").replace("Our We", "We")
          txt2 = txt2.replace("Feel free to use the Wikimedia sites to solve our Our news service crossword (Please do not fill it out online as it would spoil it for other people, print it out and fill it in at you own leisure!) Answers tomorrow.", "").strip()
          dat['text'] = "### NEWS AND WORLD EVENTS:\n"+dat['text']
        elif pat in {"data/pd_arxiv_abstracts/*",}:
          dat['text'] = "### SCIENCE:\n"+dat['text'].strip()
        elif pat in {"data/sw_amps_math/*","data/sw_dm_math/*"}:
          found = False
          txt2 = dat['text'].split("\n")
          for idx, txt in enumerate(txt2):
            txt2[idx] = txt2[idx].rstrip(" \r\t")
            if len(txt) > 50:
              code = hash(txt[:100].replace(" ", "").lower())
              if code in seen:
                found = True
                break
              seen[code] = 1
          if found:
            amps_skip+=1
            continue
          if pat == "data/sw_dm_math/*":
            #dat['text'] = txt2txtnum(dat['text']).replace("?", "?\n")
            dat['text'] = dat['text'].replace("?", "?\n")
            curr_dm_math = curr_dm_math+"\n" + dat['text']
            if len(curr_dm_math) >= 1000:
              dat['text'] = curr_dm_math
              curr_dm_math = ""
            else:
              continue
          dat['text'] = "### MATH:\n"+dat['text']

        txt2 = dat['text'].split("\n")
        for idx, txt in enumerate(txt2):
          txt2[idx] = txt2[idx].rstrip(" \r\t")
          if len(txt) > 100:
            code = hash(txt[:100].replace(" ", "").lower())
            if code in seen:
              txt2[idx] = ''
            seen[code] = 1
        txt2 = "\n".join([t.rstrip() for t in txt2 if t.strip()]).replace("\n\n", "\n")
        dat['text'] = txt2.strip()

        if pat == "data/sw_dm_math/*":
          dat['meta']['textbook_score'] = 0
          dat['meta']['rpj_score'] = 0
          outf.write (json.dumps(dat)+"\n")
          continue

        if pat != "data/sw_dm_math/*" and len(dat['text']) < 100:
          continue

        dat['text'] = remove_non_ascii(dat['text']).replace("our Our news service", "our").\
                replace(", , ", ", ").replace(" ,", ",").replace(".  ", ". ").replace(". . ", ". ").replace(". \n", ".\n").replace(" s ", "'s ").strip()
        prev_text = hash(dat['text'][:100].strip().replace(" ", "").replace("\n","").lower())
        if prev_text in seen:
          continue
        seen[prev_text]= True

        score = predict(textbook_model, dat['text'], "HIGH")
        if score > 0:
          score2 = predict(rpj_model, dat['text'], "wiki")
          if pat in {'data/pd_law/*', "data/ccby_law/*"} and ((score2 < 0.0 and "  " in dat['text']) or  get_special_char_score(dat['text'][:500]) > 0.2):
            continue
          dat['meta']['textbook_score'] = score
          dat['meta']['rpj_score'] = score2
          outf.write (json.dumps(dat)+"\n")
      #!rm -rf $file
