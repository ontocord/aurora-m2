#@title clean up law documents with citations
import glob, json
import re

from typing import List
import re
from huggingface_hub import hf_hub_download

def do_one(idx_file):
  idx, file = idx_file
  file2 = "pd_law_"+str(idx+1)+".jsonl"
  with open("/content/drive/Shareddrives/ontocord_llc/olc/"+file2, "w") as outf:
    for id, l in enumerate(open(file)):

      dat = json.loads(l)
      text = dat['text']
      if ":\n" not in text: continue
      doc_type, text = text.split(":\n",1)
      if "social security number" in text or "Official Form" in text: continue
      if "Claim(s) 1" in text or "Claim(s) 2" in text or "Claims 2" in text or "Claims 1" in text or "Claim 1" in text:
        doc_type = "### PATENT RELATED DOCUMENT"
      if "veteran" in text or "Veteran" in text or "defendant" in text or "Defendant" in text or "plaintiff" in text or "Plaintiff" in text:
        if "Claim(s) 1" in text or "Claim(s) 2" in text or "Claims 2" in text or "Claims 1" in text or "Claim 1" in text:
          continue
        doc_type = "### CASE LAW DOCUMENT"
      if len(text) < 200: continue
      doc_type = doc_type.strip()
      if doc_type.strip() not in {"### CASE LAW DOCUMENT", '### LEGAL RELATED DOCUMENT'}:
          dat['text'] = doc_type+"\n"+text
          outf.write(json.dumps(dat)+"\n")
          continue
      else:
        if len(text) < 550:
          continue
        text = text.split("\n")
        text = "".join(t.lstrip(".,?/;:)")+"\n" if len(t) > 200 or t[-1] in ".,!?:;" else " "+t for t in text if "" not in t and t.count("    ") < 3 and (len(t) > 200 or ("CASE NUMBER" not in t and 'Case Number' not in t and "CASE NO" not in t and 'Case No' not in t)))
        for _ in range(2):
          if "In this" in text[:500]:
            text = "In this " + text.split("In this",1)[-1]
          elif "OPINION" in text[:500]:
            text = "OPINION " + text.split("OPINION",1)[-1]
          elif "ISSUES" in text[:500]:
            text = "ISSUES " + text.split("ISSUES",1)[-1]
          elif "MEMORANDUM" in text[:500]:
            text = "MEMORANDUM " + text.split("MEMORANDUM",1)[-1]
          elif "PER CURIAM" in text[:500]:
            text = "PER CURIAM " + text.split("PER CURIAM",1)[-1]
          elif "SUMMARY" in text[:500]:
            text = "SUMMARY " + text.split("SUMMARY",1)[-1]
          elif "FACTS" in text[:500]:
            text = "FACTS " + text.split("FACTS",1)[-1]
          elif "This is a" in text[:500]:
            text = "This is a" + text.split("This is a",1)[-1]
          elif "The matter" in text[:500]:
            text = "The matter" + text.split("The matter",1)[-1]
          elif "The case" in text[:500]:
            text = "The case" + text.split("The case",1)[-1]
          elif "The issue" in text[:500]:
            text = "The issue" + text.split("The issue",1)[-1]
          elif "The veteran" in text[:500]:
            text = "The veteran" + text.split("The veteran",1)[-1]
          elif "The plaintiff" in text[:500]:
            text = "The plaintiff" + text.split("The plaintiff",1)[-1]
          elif "The defendants" in text[:500]:
            text = "The defendant" + text.split("The defendant",1)[-1]
          if "The veteran" in text[:500]:
            text = "The veteran" + text.split("The veteran",1)[-1]
          elif "The plaintiff" in text[:500]:
            text = "The plaintiff" + text.split("The plaintiff",1)[-1]
          elif "The defendants" in text[:500]:
            text = "The defendant" + text.split("The defendant",1)[-1]
          elif "The Plaintiff" in text[:500]:
            text = "The Plaintiff" + text.split("The Plaintiff",1)[-1]
          elif "The Defendant" in text[:500]:
            text = "The Defendant" + text.split("The Defendant",1)[-1]
          elif "The Petitioner" in text[:500]:
            text = "The Petitioner" + text.split("The Petitioner",1)[-1]
          elif "The Respondent" in text[:500]:
            text = "The Respondent" + text.split("The Respondent",1)[-1]
          elif "The Appellant" in text[:500]:
            text = "The Appellant" + text.split("The Appellant",1)[-1]
          elif "Plaintiff" in text[:500]:
            text = "Plaintiff" + text.split("Plaintiff",1)[-1]
          elif "Defendant" in text[:500]:
            text = "Defendant" + text.split("Defendant",1)[-1]
          elif "Petitioner" in text[:500]:
            text = "Petitioner" + text.split("Petitioner",1)[-1]
          elif "Respondent" in text[:500]:
            text = "Respondent" + text.split("Respondent",1)[-1]
          elif "Appellant" in text[:500]:
            text = "Appellant" + text.split("Appellant",1)[-1]
          elif "We " in text[:500]:
            text = "We " + text.split("We ",1)[-1]
          elif "This " in text[:500]:
            text = "This" + text.split("This",1)[-1]
          elif ". The" in text[:500]:
            text = "The" + text.split("The",1)[-1]
          elif "Argued" in text[:500]:
            text = "Argued" + text.split("Argued",1)[-1]
          elif "(1)" in text[:500]:
            text = "(1)" + text.split("(1)",1)[-1]
          elif "1." in text[:500]:
            text = "1." + text.split("1.",1)[-1]

        for _ in range(4):
          text = re.sub(r'(S\.\s+Ct\.\s+at\s+\d+)|(\d+\s+L\.Ed\.\d+d\s+at\d+)|(App\.?\s+\d+)|(U\.S\.[^,]+)|(\.\s+[\S]{2,10}\s+at\s+\d+)|(\d+\s+[\S+]{2,10}\s+\d+)|(\d+\s+[\S+]{2,10}\s+at\d+)|([^,]+\s+v\.?\s+[^,]+)|(\s+\d+\s+F.[\S]{2,5})|(\d+\s+\(\d+[\D]{3-10}\d{4}\))|(\d+\s+\(\d{4}\))', ' [CITATION] ', text)
          text = text.replace("[CITATION] [CITATION]", "[CITATION]").replace("[CITATION] TION]", "[CITATION]").replace("[CITATION] ,","[CITATION],").replace("] .", "].").replace(",  ", ", ").replace(", , ", ", ").replace("  ", " ")
          text = text.replace("[CITATION], [CITATION]", "[CITATION]")
          text = re.sub(r'(\[CITATION\]\s+[^\d]{3,10}\d+d\s+\d+)|(see\s+[^\[]{0,10}\[CITATION\])|(\[CITATION\]\s+d\s+[^\)]{2,10})|(\[CITATION\]\s+d\s+\d+)|(\[CITATION\]\,?[^\,]{1,20}\[CITATION\])|(\[CITATION\]\,?\s+S\.\s+Ct\.\s+at\s+\d+)|(\[CITATION\]\,?\s+U\.\s+S\.\s+\d+)|(\[CITATION\]\,?\sid\.?\s+\d+)|(\[CITATION\]\,?\s+at\s+\d+)|(\[CITATION\]\s+\([^\)]{1,35}\))|(\[CITATION\]\s+\-\d+)|(\[CITATION\]\,?\s+\S+\s+\[CITATION\])|(\[CITATION\]\,?\s+Inc\.?)|(IN\s+RE\s+[^\[]{4,10}\[CITATION\])|(In\s+Re\s+[^\[]{4,10}\[CITATION\])|(In\s+re\s+[^\[]{4,10}\[CITATION\])|(\[CITATION\]\,?\s+\d+\s+\([^\)]{4,20}\))|(\[CITATION\]\,?\s+\d+)|(\[CITATION\]\s+[\S]{1,3}\s+\[CITATION\])', ' [CITATION] ', text)
          text = text.replace("[CITATION] TION]", "[CITATION]").replace("[CITATION] ,","[CITATION],").replace(",  ", ", ").replace(", , ", ", ").replace("  ", " ")
          text = text.replace("[CITATION], [CITATION]", "[CITATION]")

        text = text.replace(" [CITATION] .", ".").replace(".  ", ". ").replace(",  ", ", ")
        text = text.replace("    [CITATION]", "\n[CITATION]").strip()
        if text.startswith("[CITATION]"):
          text = text[len("[CITATION]"):]
        text = text.lstrip(")1234567890~`.>,").strip()
        if text.startswith("[CITATION]"):
          text = text[len("[CITATION]"):]
        text = text.lstrip(")1234567890~`.>,").strip()
        text = text.split("\n")
        text = "\n".join(t.strip() for t in text)

        text = text.replace("$", " $").replace("( $", "($").replace(",.", ".").replace("  ", " ").replace("   ", " ")
        if not text: continue
        if text[0] != text[0].upper():
          text = text.split(".",1)[-1].strip()
        text = text.split("\n")
        text = "".join(" "+t if t[0] != t[0].upper() else "\n"+t for t in text if t.strip()).strip()

        if len(text) > 550:
          dat['text'] = doc_type+"\n"+text
          outf.write(json.dumps(dat)+"\n")

files = list(glob.glob("/content/drive/Shareddrives/ontocord_llc/olc/pd_lawa*"))
files.sort()
idx_files = list(enumerate(files))
idx_files = idx_files[0:4]
from multiprocessing import Pool
with Pool(4) as p:
  p.map(do_one, idx_files)
