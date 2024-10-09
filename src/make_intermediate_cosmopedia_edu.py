#@title make intermediate cosmopedia edu file
from datasets import load_dataset
import json, random

dataset = load_dataset("HuggingFaceTB/cosmopedia_2B_annotated_edu_score")
with open("/content/drive/Shareddrives/ontocord_llc/scored_textbook.jsonl", "w") as outf:
  for idx, dat in enumerate(dataset['train']):
      topic0 = dat['text'].split("\n")[0].split("Course Unit:")[-1].strip()
      text = dat['completion']
      topic = text.split(".")[0].\
            split("field of ", 1)[-1].\
            split("covering ")[-1].\
            split("focusing on ")[-1].\
            split("contributes to ")[-1].\
            split("in relation to ")[-1].\
            split("comes from " ,1)[-1].\
            split("list ",1)[-1].\
            split("presents ",1)[-1].\
            split("discuss ",1)[-1].\
            split("describe ",1)[-1].\
            split("explains ",1)[-1].\
            split("describes ",1)[-1].\
            split("discusses ",1)[-1].\
            split("covers ",1)[-1].\
            split("analysis of ",1)[-1].\
            split("related to ",1)[-1].\
            split("about ",1)[-1].\
            split("involves ",1)[-1].\
            split("involve ",1)[-1].\
            strip()
      if topic == text.split(".")[0].strip():
          topic = text.split(".")[1].\
            split("field of ", 1)[-1].\
            split("covering ")[-1].\
            split("focusing on ")[-1].\
            split("contributes to ")[-1].\
            split("in relation to ")[-1].\
            split("comes from " ,1)[-1].\
            split("list ",1)[-1].\
            split("presents ",1)[-1].\
            split("discuss ",1)[-1].\
            split("describe ",1)[-1].\
            split("explains ",1)[-1].\
            split("describes ",1)[-1].\
            split("discusses ",1)[-1].\
            split("covers ",1)[-1].\
            split("analysis of ",1)[-1].\
            split("related to ",1)[-1].\
            split("about ",1)[-1].\
            split("involves ",1)[-1].\
            split("involve ",1)[-1].\
            strip()
      topic = topic.split(", which")[0].split(";")[0].split(", it ")[0].split(".")[0]
      text = text.replace("score: 1", "score: low.")
      text = text.replace("score: 2", "score: medium.")
      text = text.replace("score: 3", "score: medium.")
      text = text.replace("score: 4", "score: high.")
      text = text.replace("score: 5", "score: high.")
      text = text.replace("Score: 1", "score: low.")
      text = text.replace("Score: 2", "score: medium.")
      text = text.replace("Score: 3", "score: medium.")
      text = text.replace("Score: 4", "score: high.")
      text = text.replace("Score: 5", "score: high.")
      sents = [a.strip() for a in text.split(".") if a.strip()]
      sents = [a.strip() for a in sents if "score:" not in a] + [a.strip().replace("Educational score", "Score").strip('"') for a in sents if "score:"  in a]
      if topic in sents or len(topic) > 250 or len(topic) < 20 or " score " in topic or "evaluation" in topic:
        topic = topic0
      else:
        topic = topic[0].upper()+topic[1:]
        topic = topic0+" : " + topic
      if not topic.strip(): continue
      text = ". ".join(sents).replace("\n\n", "\n").replace(". -", ".\n-").replace(". *", ".\n*").\
          replace(". 2", ".\n2").replace(". 3", ".\n3").replace(". 4", ".\n4").replace(".5", ".\n5").replace(". 6", ".\n6").replace(". 7", ".\n7")
      if "Score:" not in text: continue
      if "Score: medium" in text and len(dat['text']) < 3000: continue
      if "Score: medium" in text  and random.randint(0,30) > 0: continue
      if "Score: low" in text  and random.randint(0,2) > 0: continue
      outf.write (json.dumps({'topic': topic, 'response': text, 'textbook': dat['text'], 'seed_text': dat['seed_text']})+"\n")
      #if idx > 100000: break
