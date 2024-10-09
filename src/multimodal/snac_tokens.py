#@title snac tokens
import random, json
#!wget https://zenodo.org/api/records/6517052/files-archive

try:
  from datasets import load_dataset
except:
  !pip -q install -q datasets

import locale
def getpreferredencoding(do_setlocale = True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding

try:
  from phonemizer import phonemize
  from phonemizer.separator import Separator
except:
  !pip -q install phonemizer
  !sudo apt-get install espeak-ng
  !pip -q install --upgrade protobuf

from phonemizer import phonemize
from phonemizer.separator import Separator
from datasets import load_dataset

dataset = load_dataset("blanchon/snac_llm_parler_tts", streaming=True)

with open("/content/drive/Shareddrives/ontocord_llc/safe_llm/snac_llm_parler_tts.jsonl", "w") as outf:

  for idx, dat in enumerate(dataset['train']):
    snac = "".join(["<AUD_"+a+">" for a in dat['snac24khz'].split()])
    text = dat['text'].replace(" i ", " I ")
    phonemes2 =  phonemize(
      text,
      language='en-us',
      backend='espeak',
      separator=Separator(phone=None, word=' ', syllable='|'),
      strip=True,
      preserve_punctuation=True,
      njobs=4)
    if random.randint(0,1):
      if random.randint(0,1):
        a = [text if  random.randint(0,1) else (dat['text_description'].strip(". ")+":\n" '"'+text+'"'),  "Audio for " +str(int(dat['audio_duration_right'])) + " seconds: <AUD>"+snac+"</AUD>", "Language: English.", "IPA phonemes: "+
            phonemes2 ]
        random.shuffle(a)
        instruct =  "## Convert between audio, text and phonemes.\n\n"+ "\n".join(a)
      else:
        a = [text if  random.randint(0,1) else (dat['text_description'].strip(". ")+":\n" '"'+text+'"'),  "Audio for " +str(int(dat['audio_duration_right'])) + " seconds: <AUD>"+snac+"</AUD>", ]
        random.shuffle(a)
        instruct =  "## Convert between audio and text.\n\n"+ "\n".join(a)
    elif random.randint(0,1):
      instruct = "## Convert from text to audio\n\n"+text +"\n"+"<AUD>"+snac+"</AUD>"
    else:
      instruct = "## Convert from audio to text\n\n"+"<AUD>"+snac+"</AUD>"+"\n"+text
    outf.write(json.dumps ({'instruct': instruct, 'text': text, 'metadata': {'source': "blanchon/snac_llm_parler_tts/"+str(idx)}})+"\n")
    if random.randint(0,1):
      instruct = "<AUD>"+snac+"</AUD>"
      outf.write(json.dumps ({'instruct': instruct, 'text': text, 'metadata': {'source': "blanchon/snac_llm_parler_tts/"+str(idx)}})+"\n")

