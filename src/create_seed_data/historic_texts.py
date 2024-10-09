#@title historical text
from datasets import load_dataset


from datasets import load_dataset
import os, json
with open("/content/drive/Shareddrives/ontocord_llc/safe_llm/historical_text.jsonl", "w") as outf:
  for lang in ['french',  'italian', 'german']:
    dataset = load_dataset("PleIAs/Post-OCR-Correction", lang)
    for idx, dat in enumerate(dataset['train']):
      sep = "Language:  Historical " + lang[0].upper() + lang[1:]+"\n"+('' if 'date' not in dat else ('Year:\n'+dat['date'].split("-")[0]))+"\n\nBelow is historical text that has been scanned in, and includes errors, typos, and formatting issues. Please correct the text to improve its quality."
      sep2 = "<|endoftext|>Language:  Historical " + lang[0].upper() + lang[1:]+"\nHistorical text can contain sexists, bias, or racist values and may include incorrect or outdated facts and principles about math and science. The historical text could be used for scholarship and references in creating fiction, but care should be taken to remove any illegal or bias content."
      outf.write(json.dumps ({'text': sep+"\n\n**\n\n"+dat['text']+sep2+"\n\n***\n\n"+dat['corrected_text']})+"\n")

      break
  #!rm -rf ~/.cache/hugging*/datasets
  german_18_books = [('Flöten und Dolche: Novellen', 'Heinrich Mann'),
    ('Flaubert und die Herkunft des modernen Romans', 'Heinrich Mann'),
    ('Der Vater', 'Heinrich Mann'),
    ('Professor Unrat, oder, Das Ende eines Tyrannen', 'Heinrich Mann'),
    ('Der Untertan', 'Heinrich Mann'),
    ('Die Ehrgeizige: Novelle', 'Heinrich Mann'),
    ('Gladius Dei; Schwere Stunde', 'Thomas Mann'),
    ('Der Tod in Venedig', 'Thomas Mann'),
    ('Tristan', 'Thomas Mann'),
    ('Tonio Kröger', 'Thomas Mann'),
    ('Buddenbrooks: Verfall einer Familie', 'Thomas Mann'),
    ('Königliche Hoheit: Roman', 'Thomas Mann'),
    ('Der kleine Herr Friedemann: Novellen', 'Thomas Mann'),
    ('Die Ermordung einer Butterblume und andere Erzählungen', 'Alfred Döblin'),
    ('Die Lobensteiner reisen nach Böhmen: Zwölf Novellen und Geschichten',
      'Alfred Döblin'),
    ('Wallenstein. 1 (of 2)', 'Alfred Döblin'),
    ('Wallenstein. 2 (of 2)', 'Alfred Döblin'),
    ('Die drei Sprünge des Wang-lun: Chinesischer Roman', 'Alfred Döblin')]

  dataset = load_dataset("BEE-spoke-data/gutenberg-en-v1-clean")
  for idx, a in enumerate(dataset['train']):

    text = a['text']
    j = 0
    max_rng = random.choice([1500, 1000, 500])
    for rng in range(0, len(text), max_rng):
      j+=1
      text2 = text[rng:min(len(text),rng+max_rng)]
      if rng == 0:
        text2 = "\n".join(text2.split("\n")[:-1])
      else:
        text2 = "\n".join(text2.split("\n")[1:-1])
      text2 = text2.strip()

      #TODO: detect science and put header
      outf.write(json.dumps({'text': text2, 'instruct': '', 'metadata': {'source': "BEE-spoke-data/gutenberg-en-v1-clean/"+str(idx)+"/"+str(j)}})+"\n")

  dataset = load_dataset("manu/project_gutenberg")

  dataset = load_dataset("biglam/bnl_newspapers1841-1879")

  dataset2 = load_dataset("biglam/europeana_newspapers")
