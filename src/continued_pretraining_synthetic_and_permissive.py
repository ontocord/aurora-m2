#@title make continued_pretraining_synthetic_and_permissive
import json, random, math
try:
  from datasets import load_dataset
except:
  !pip install -q datasets
from datasets import load_dataset


def get_ngram(text, lang="en", window_size=3, ):
  if lang_is_cjk(lang):
    tokens = text
    ret= ["".join(tokens[i : i + window_size])   for i in range(len(tokens) - window_size)]
  else:
    tokens = text.split()
    ret= [" ".join(tokens[i : i + window_size])   for i in range(len(tokens) - window_size)]
  return Counter(ret)

def get_ngram_score(text, lang="en", window_size=3, ):
  if not text: return 1
  aHash = get_ngram(text, lang=lang, window_size=window_size)
  text_len = text.count(" ")+1
  for key in list(aHash.keys()):
    aHash[key] = aHash[key]/text_len
  if not aHash: return 0.0
  return aHash.most_common(1)[0][1]
def process_1():
  with open("/content/drive/Shareddrives/ontocord_llc/continued_pretraining_synthetic_and_permissive_part3.jsonl", "w") as outf:
    if False:
      #'auto_math_text', 'khanacademy', 'openstax', 'stanford',  'web_samples_v1',
        for config in  ['web_samples_v2', 'wikihow']:
          !rm -rf ~/.cache/hugging*/*
          dataset = load_dataset("kenhktsui/cosmopedia_quality_score_v2", config, streaming=True)
          for idx, dat in enumerate(dataset['train']):
            if dat['quality_score_v2'] > 0.9:
              outf.write(json.dumps({'text': dat['text'].strip(), 'instruct': '', 'metadata': {'source': "cosmopedia/"+config}})+"\n")
  
    #!rm -rf ~/.cache/hugging*/*
    dataset = load_dataset("nampdn-ai/tiny-strange-textbooks")
    for idx, a in enumerate(dataset['train']):
      outf.write(json.dumps({'text': a['text'], 'instruct': '', 'metadata': {'source': 'nampdn-ai/tiny-strange-textbooks'}})+"\n")
    #!rm -rf ~/.cache/hugging*/*
    dataset = load_dataset("nampdn-ai/mini-peS2o")
    for idx, a in enumerate(dataset['train']):
      outf.write(json.dumps({'text': a['text'], 'instruct': '', 'metadata': {'source': 'nampdn-ai/mini-peS2o'}})+"\n")
  
  
  
    #!rm -rf ~/.cache/hugging*/*
    dataset = load_dataset("nampdn-ai/tiny-codes")
    for idx, dat in enumerate(dataset['train']):
      topic = dat['common_sense_topic']
      response = dat['response'].strip()
      programming_language = dat['programming_language']
      instruct = f"This is {programming_language} pseudocode to teach {topic}. It is not meant to be executable code, and is only meant to describe the topic using code.\n===\n{response}"
      if random.randint(0,1):
        instruct = instruct.replace("This is", "Below you will find")
      if random.randint(0,1):
        instruct = instruct.replace("This is", "Below is")
      if random.randint(0,1):
        instruct = instruct.replace("This is", "Here is")
      if random.randint(0,1):
        instruct = instruct.replace("to teach", "to explore")
      if random.randint(0,1):
        instruct = instruct.replace("to teach", "to educate")
      if random.randint(0,1):
        instruct = instruct.replace("to teach", "to demonstrate")
      if random.randint(0,1):
        instruct = instruct.replace("It is not meant to be", "It is not")
      if random.randint(0,1):
        instruct = instruct.replace("It is not meant to be", "WARNING: This is not")
      if random.randint(0,1):
        instruct = instruct.replace("and is only meant to", "and is to")
      if random.randint(0,1):
        instruct = instruct.replace("describe the", "demonstrate the")
      if random.randint(0,1):
        instruct = instruct.replace("describe the", "show the")
      if random.randint(0,1):
        instruct = instruct.replace("the topic", "the topic of " +topic)
      if random.randint(0,1):
        instruct = instruct.replace("the topic", "ideas")
      if random.randint(0,1):
        instruct = instruct.replace("using code", "using " +programming_language)
      outf.write(json.dumps({'text': response, 'instruct': instruct, 'metadata': {'source': "nampdn-ai/tiny-codes"}})+"\n")
  
  
  
    !rm -rf ~/.cache/hugging*/*
    dataset = load_dataset("kenhktsui/simple_wikipedia_LM_quality_score_v1")
  
    for idx, dat in enumerate(dataset['train']):
      text = ""
      if "List of" in dat['title'] or 'album' in dat['title']: continue
      if len(dat['text']) < 1000: continue
      if 'Storyline\n' in dat['text']:
        text = "Fiction: "+dat['title']+"\n"+dat['text'].split("Storyline\n",1)[-1].split(":")[0]
      elif 'Backstory' in dat['text']:
        text = "Fiction: "+dat['title']+"\n"+dat['text'].split("Backstory\n",1)[-1].split(":")[0]
      elif 'Story\n' in dat['text']:
        text = "Fiction: "+dat['title']+"\n"+dat['text'].split("Story\n",1)[-1].split(":")[0]
      elif 'Plot\n' in dat['text']:
        text = "Fiction: "+dat['title']+"\n"+dat['text'].split("Plot\n",1)[-1].split(":")[0]
      if len(text) > 1000:
        outf.write(json.dumps({'text': text.replace(" () ", ""), 'metadata': {'source': 'simlewiki'}})+"\n")
        continue
      if text: continue
      if (' player ' in dat['text'] or ' actor ' in dat['text']) and random.randint(0,10) > 0:
        continue
      if ") is an American" in text or ") is an English" in text:
          continue
      if dat['quality_score_v1'] > 0.8:
        text = dat['text']
        text = text.replace("(,", "(").replace(", ,", ",").replace("( ,", "(").replace("(; ", "(").replace("( ;", "(").replace(" ()", "").replace(" ( )", "").replace(" ,", ",")
        if ") is an American" in text or ") is an English" in text:
          continue
  
        outf.write(json.dumps({'text': dat['title']+"\n"+text, 'instruct': '', 'metadata': {'source': 'simlewiki'}})+"\n")
  
def process_2():
  import json, random, math
  try:
    from datasets import load_dataset
  except:
    !pip install -q datasets
  from datasets import load_dataset
  try:
    if dataset is None: assert False
  except:
    dataset = load_dataset("HuggingFaceTB/cosmopedia", "stories")
  openhermes_stories = None
  try:
    if openhermes_stories is None: assert False
  except:
    openhermes_stories = {}
    ultrachat_stories = {}
  
    for idx, dat in enumerate(dataset['train']):
      story = dat['text'].strip()
      story = story.replace("redditor", "netizen").replace("Redditor", "Netizen").replace("Reddit", "An Online Forum").replace("subreddit", "online sub-discussion").replace("reddit", "online discussion").replace(" r/", " @").replace(" u/", " @")
      audience = dat['audience']
      if "children" in audience:
        story = "Story appropriate for "+ audience.replace("_", " ")+"\n" + story
      source = dat['seed_data']
      instruction = dat['prompt'].split("“",1)[-1].split("”")[0].replace("\n", " ")
      instruction = instruction.strip(". ,?")
  
      if len(instruction) > 50:
        instruction = instruction[:50].strip(". ,?")
  
      if "ultra" in source:
        ultrachat_stories[instruction] = story
      else:
        openhermes_stories[instruction] = story
  
  
      if "?" in instruction:
        instruction2, _ = instruction.split("?",1)
        instruction2 = instruction2.strip(". ,?")
        if "ultra" in source:
          ultrachat_stories[instruction2] = story
        else:
          openhermes_stories[instruction2] = story
      if "." in instruction:
        instruction2, _ = instruction.split(".",1)
        instruction2 = instruction2.strip(". ,?")
  
        if "ultra" in source:
          ultrachat_stories[instruction2] = story
        else:
          openhermes_stories[instruction2] = story
  
  
  #dataset = load_dataset("NickyNicky/Iker-Colossal-Instruction-Translation-EN-ES_deduplicated")
  
  with open("/content/drive/Shareddrives/ontocord_llc/continued_pretraining_synthetic_and_permissive_part5.jsonl", "w") as outf:
    seen = {}
    seen_story = {}
    def process_hermes(instruction, story, dat):
        chosen = dat['chosen'][1]['content'].strip()
        prompt = dat['prompt'].strip()
        prompt = prompt.replace("You are an AI ", "You are an expert ")
        for p in prompt.split():
          if "http" in p:
            prompt = prompt.replace("url: "+p, "").replace(" "+p, "").replace(p+" ", "").replace("\n"+p, "").replace(p+"\n", "")
        hermes = [s for policy, s in zip(dat['candidate_policies'], dat['candidates_completions']) if "2.5" in policy][0]
        for p in hermes.split():
          if "http" in p:
            hermes = hermes.replace("url: "+p, "").replace(" "+p, "").replace(p+" ", "").replace("\n"+p, "").replace(p+"\n", "")
        for p in chosen.split():
          if "http" in p:
            chosen = chosen.replace("url: "+p, "").replace(" "+p, "").replace(p+" ", "").replace("\n"+p, "").replace(p+"\n", "")
  
        if "2.5" not in dat['chosen_policy'] and abs(len(hermes) - len(chosen)) > 100:
          if len(hermes) > len(chosen):
            instruction= f"<|user|>{prompt}<|endoftext|>\n<|assistant|>{hermes}<|endoftext|>\n<|user|>Please make the answer more concise?<|endoftext|>\n<|assistant|>{chosen}<|endoftext|>"
          else:
            if  "=" in chosen or"=" in prompt:
              instruction= f"<|user|>{prompt}<|endoftext|>\n<|assistant|>{hermes}<|endoftext|>\n<|user|>Can you give me the answer with more steps and explanations?<|endoftext|>\n<|assistant|>{chosen}<|endoftext|>"
            else:
              instruction= f"<|user|>{prompt}<|endoftext|>\n<|assistant|>{hermes}<|endoftext|>\n<|user|>Please make the answer more detailed?<|endoftext|>\n<|assistant|>{chosen}<|endoftext|>"
        else:
          candidate_arr = [s for policy, s in zip(dat['candidate_policies'], dat['candidates_completions']) if policy != dat['chosen_policy']]
          candidates = "\n".join(["### Answer "+str(idx+1)+".\n"+(s.strip())  for idx, s in enumerate(candidate_arr)])
          if "=" in chosen or "=" in candidates or "=" in prompt or len(candidates) < 100 or len(chosen) < 100:
            if random.randint(0,1):
              instruction= f"""<|user|>{prompt}\n\nBelow are potential answers to the above question. Create a better answer by combining the best of each answers and adding or correcting any details. Improve anything that does not make sense:\n\n{candidates}\n<|endoftext|>\n<|assistant|>{chosen}<|endoftext|>"""
            elif random.randint(0,1):
              instruction= f"""<|user|>Q: {prompt}\nA:\nBelow are potential answers to the above question. Create a better answer by combining the best of each answers and adding or correcting any details. Improve anything that does not make sense:\n\n{candidates}\n<|endoftext|>\n<|assistant|>{chosen}<|endoftext|>"""
            else:
              instruction= f"""<|user|>### Context: {prompt}\n### Question:\nBelow are potential answers to the above context. Create a better answer by combining the best of each answers and adding or correcting any details. Improve anything that does not make sense:\n\n{candidates}\n<|endoftext|>\n<|assistant|>{chosen}<|endoftext|>"""
  
          else:
            if random.randint(0,4) == 0:
              instruction= f"""<|user|>{prompt}<|endoftext|>\n<|assistant|>{chosen}<|endoftext|>"""
            elif random.randint(0,4) == 0:
              choice1 = random.choice([s for policy, s in zip(dat['candidate_policies'], dat['candidates_completions']) if policy != dat['chosen_policy']])
              choice2 = [s for policy, s in zip(dat['candidate_policies'], dat['candidates_completions']) if policy == dat['chosen_policy']][0]
              if random.randint(0,1):
                candidates = "\n".join(["###Answer "+str(idx+1)+".\n"+(s.strip())  for idx, s in enumerate([choice2, choice1])])
                instruction= f"""<|user|> {prompt}\n\nBelow are potential answers to the above question. Tell me which one is better. Do not explain why.\n\n{candidates}\n<|endoftext|>\n<|assistant|>1<|endoftext|>"""
              else:
                candidates = "\n".join(["### Answer "+str(idx+1)+".\n"+(s.strip())  for idx, s in enumerate([choice1, choice2])])
                instruction= f"""<|user|>{prompt}\n\nBelow are potential answers to the above question. Tell me which one is better. Do not explain why.\n\n{candidates}\n<|endoftext|>\n<|assistant|>2<|endoftext|>"""
            elif random.randint(0,1):
              instruction= f"""<|user|>Below are potential answers to a question. Create a better answer by combining the best of each answers and adding or correcting any details. Improve anything that does not make sense. Then provide a possible instruction that would result in this answer.\n\n{candidates}\n<|endoftext|>\n<|assistant|>### Here is an improved answer:\n\n{chosen}\n\n### A possible instruction for this answer could be:\n{prompt}<|endoftext|>"""
            elif random.randint(0,4)==0:
              instruction= f"""<|user|>{candidates}\n\Above are potential answers to a question. Create a better answer by combining the best of each answers and adding or correcting any details. Improve anything that does not make sense. Then provide a possible instruction that would result in this answer.\n<|endoftext|>\n<|assistant|>### Here is an improved answer:\n\n{chosen}\n\n### A possible instruction for this answer could be:\n{prompt}<|endoftext|>"""
            elif random.randint(0,4)==0:
              instruction= f"""<|user|>{candidates}\n\Above are potential answers to a question. First provide a possible question that would result in these answers. Then create a better answer by combining the best of each answers and adding or correcting any details. Improve anything that does not make sense.\n<|endoftext|>\n<|assistant|>Q: {prompt}\nA: {chosen}\n<|endoftext|>"""
            else:
              instruction= f"""<|user|>Below are potential answers to a question. First provide a possible question that would result in these answers. Then create a better answer by combining the best of each answers and adding or correcting any details. Improve anything that does not make sense.\n\n{candidates}\n<|endoftext|>\n<|assistant|>Q: {prompt}\nA: {chosen}\n<|endoftext|>"""
  
        if len(chosen) > 200:
          text = chosen
        else:
          text = "Q: " + prompt + " A: " + chosen
        found = False
        if story and instruction not in seen:
          seen[instruction] = 1
          seen_story[story] = 1
          if len(prompt) < 100 and random.randint(0,1):
            if "Story" not in story[:20] and "Title" not in story[:20]:
              story = "Fiction: " + story
            else:
              if "Title:" in story[:100]:
                if 'appropriate for' in story[:50] and 'children' in story[:50]:
                  story = "<|user|>Write a children's story entitled '"+story.split("Title:",1)[-1].split("\n",1)[0].strip()+"'<|endoftext|>\n<|assistant|>"+story
                else:
                  story = "<|user|>Write a story entitled '"+story.split("Title:",1)[-1].split("\n",1)[0].strip()+"'<|endoftext|>\n<|assistant|>"+story
            if random.randint(0,1):
              instruction = instruction.replace("<|user|>", "<|user|>Inspired by the story, please answer the below:\n")
              instruction = story+"<|endoftext|>"+instruction
            else:
              if random.randint(0,1):
                instruction = story+"<|endoftext|>"+instruction
              else:
                instruction = story+"<|endoftext|>"+instruction
            if "Title:" in story: found=True
          else:
            if 'appropriate for' in story[:50] and 'children' in story[:50]:
              instruction = instruction+"<|user|>Write a children's story inspired by the prior text.<|endoftext|>\n<|assistant|>"+story+"<|endoftext|>"
            else:
              instruction = instruction+"<|user|>Write a story inspired by the prior text.<|endoftext|>\n<|assistant|>"+story+"<|endoftext|>"
        if random.randint(0,1):
          instruction = instruction.replace("Tell me which", "Which")
        if random.randint(0,1):
          instruction = instruction.replace("You are a", "Take on a role of a")
        if random.randint(0,1):
          instruction = instruction.replace("You are a", "Pretend to be a")
        if random.randint(0,1):
          instruction = instruction.replace("Do not explain why.", "Don't explain")
        if random.randint(0,1):
          instruction = instruction.replace("Do not explain why.", "No need for an explanation.")
        if random.randint(0,1):
          instruction = instruction.replace("Below are", "Here are")
        if random.randint(0,1):
          instruction = instruction.replace("Here is", "I have provided")
        if random.randint(0,1):
          instruction = instruction.replace("Here is an ", "The ")
        if random.randint(0,1):
          instruction = instruction.replace("an improved", "a better")
        if random.randint(0,1):
          instruction = instruction.replace("instruction", "question")
        if random.randint(0,1):
          instruction = instruction.replace("question", "instruction")
        if random.randint(0,1):
          instruction = instruction.replace("instruction", "query")
        if random.randint(0,1):
          instruction = instruction.replace("answers to ", "responses to ")
        if random.randint(0,1):
          instruction = instruction.replace("potential", "possible")
        if random.randint(0,1):
          instruction = instruction.replace("potential answers to ", "AI generated text to ")
        if random.randint(0,1):
          instruction = instruction.replace("a question ", "a user provided query ")
        if random.randint(0,1):
          instruction = instruction.replace("a question ", "a user's conversation ")
        if random.randint(0,1):
          instruction = instruction.replace("provide a possible question", "give me an instruction")
        if random.randint(0,1):
          instruction = instruction.replace("Create a better ", "Improve the ")
        if random.randint(0,1):
          instruction = instruction.replace("Create a better answer by combining the best of each answers", "Combine the best of each of the above answers")
        if random.randint(0,1):
          instruction = instruction.replace("adding or correcting any details", "otherwise improve the answer")
        if random.randint(0,1):
          instruction = instruction.replace("Improve", "Fix")
        if random.randint(0,1):
          instruction = instruction.replace("anything that does not make sense", "the answer")
        if random.randint(0,1):
          instruction = instruction.replace("Please make", "Make")
        if random.randint(0,1):
          instruction = instruction.replace("the answer", "the reply")
        if random.randint(0,1):
          instruction = instruction.replace("the answer", "your response")
        if random.randint(0,1):
          instruction = instruction.replace("detailed", "complete")
        if random.randint(0,1):
          instruction = instruction.replace("detailed", "informative")
        if random.randint(0,1):
          instruction = instruction.replace("concise", "precise")
        if random.randint(0,1):
          instruction = instruction.replace("concise", "succinct")
        if random.randint(0,1):
          instruction = instruction.replace("Can you give me", "Provide")
        if random.randint(0,1):
          instruction = instruction.replace("Can you give me", "Responsed with")
        if random.randint(0,1):
          instruction = instruction.replace("more steps", "more analysis")
        if random.randint(0,1):
          instruction = instruction.replace("explanations", "thoughtfulness")
        if random.randint(0,1):
          instruction = instruction.replace("inspired by", "based on")
        if random.randint(0,1):
          instruction = instruction.replace("Inspired by", "Based on")
        if random.randint(0,1):
          instruction = instruction.replace("inspired by", "with elements of")
        if random.randint(0,1):
          instruction = instruction.replace("Write a", "Please draft a")
        if random.randint(0,1):
          instruction = instruction.replace("Write a", "Provide me with a")
        if random.randint(0,1):
          instruction = instruction.replace("prior text", "your answer")
        if random.randint(0,1):
          instruction = instruction.replace("prior text", "what you just said")
        if random.randint(0,1):
          instruction = instruction.replace("prior text", "what you wrote above")
        instruction = instruction.replace("a instruction", "an instruction").replace("the your", "your").replace("the what you", "what you")
        if story:
          outf.write(json.dumps({'text': text, 'instruct': instruction, 'metadata': {'source': "HuggingFaceTB/cosmopedia/stories/openhermes+argilla/OpenHermesPreferences/"+('' if not dat['source'] else dat['source'])+"/"+("" if not dat['category'] else dat['category'])}})+"\n")
        else:
          outf.write(json.dumps({'text': text, 'instruct': instruction, 'metadata': {'source': "argilla/OpenHermesPreferences/"+('' if not dat['source'] else dat['source'])+"/"+("" if not dat['category'] else dat['category'])}})+"\n")
  
    try:
      if open_hermes_preferences is None: assert False
    except:
      open_hermes_preferences = load_dataset("argilla/OpenHermesPreferences")
    not_found = 0
    if True:
      openhermes_dataset = []
      for idx, dat in enumerate(open_hermes_preferences['train']):
        instruction = dat['prompt'].strip()
        instruction = instruction.replace("\n", " ")
        for _ in range(3):
          if instruction.startswith("You are knowledgeable"):
            instruction = instruction.split(".",1)[-1].strip()
          if instruction.startswith("You are an AI assis"):
            instruction = instruction.split(".",1)[-1].strip()
          if instruction.startswith("You are an Ar"):
            instruction = instruction.split(".",1)[-1].strip()
          if instruction.startswith("You are a help"):
            instruction = instruction.split(".",1)[-1].strip()
          if instruction.startswith("You are a world class trivia AI"):
            instruction = instruction.split(".",1)[-1].strip()
          if instruction.startswith("You are a teacher"):
            instruction = instruction.split(".",1)[-1].strip()
          if instruction.startswith("Given a task"):
            instruction = instruction.split(".",1)[-1].strip()
          if instruction.startswith("You solve"):
            instruction = instruction.split(".",1)[-1].strip()
          if instruction.startswith("Provide a detailed"):
            instruction = instruction.split(".",1)[-1].strip()
          if instruction.startswith("User will you give"):
            instruction = instruction.split(".",1)[-1].strip()
          if instruction.startswith("You will be given a task"):
            instruction = instruction.split(".",1)[-1].strip()
          if instruction.startswith("Help as much as you can"):
            instruction = instruction.split(".",1)[-1].strip()
          if instruction.startswith("Think like you are answer"):
            instruction = instruction.split(".",1)[-1].strip()
          if instruction.startswith("Your goal"):
            instruction = instruction.split(".",1)[-1].strip()
          if instruction.startswith("Your job"):
            instruction = instruction.split(".",1)[-1].strip()
          if instruction.startswith("While answer"):
            instruction = instruction.split(".",1)[-1].strip()
          if instruction.startswith("While perform"):
            instruction = instruction.split(".",1)[-1].strip()
          if instruction.startswith("You must gener"):
            instruction = instruction.split(".",1)[-1].strip()
          if instruction.startswith("You should des"):
            instruction = instruction.split(".",1)[-1].strip()
          if instruction.startswith("Your task"):
            instruction = instruction.split(".",1)[-1].strip()
  
        instruction = instruction.strip(". ,?")
        if len(instruction) > 50:
          instruction = instruction[:50].strip(". ,?")
        if instruction in openhermes_stories:
          process_hermes(instruction, openhermes_stories[instruction], dat)
          continue
        if "?" in instruction:
          instruction2, _ = instruction.split("?",1)
          instruction2 = instruction2.strip(". ,?")
          if instruction2 in openhermes_stories:
            process_hermes(instruction, openhermes_stories[instruction2], dat)
            continue
        if "." in instruction:
          instruction2, _ = instruction.split(".",1)
          instruction2 = instruction2.strip(". ,?")
          if instruction2 in openhermes_stories:
            process_hermes(instruction, openhermes_stories[instruction2], dat)
            continue
  
        process_hermes(instruction, '', dat)
        not_found += 1
  
  
      for idx, dat in enumerate(dataset['train']):
        story = dat['text'].strip()
        story = story.replace("redditor", "netizen").replace("Redditor", "Netizen").replace("Reddit", "An Online Forum").replace("subreddit", "online sub-discussion").replace("reddit", "online discussion").replace(" r/", " @").replace(" u/", " @")
        audience = dat['audience']
        if "children" in audience:
          story = "Story appropriate for "+ audience.replace("_", " ")+"\n" + story
        source = dat['seed_data']
        if "ultra" not in source and story not in seen_story:
          outf.write(json.dumps({'text': story, 'instruct': '', 'metadata': {'source': "HuggingFaceTB/cosmopedia/stories/openhermes"}})+"\n")
  
    
