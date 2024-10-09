#@title RAG
with open("/content/drive/Shareddrives/ontocord_llc/selfrag.jsonl", "w") as outf:
  #!rm -rf ~/.cache/hugging*/datasets
  dataset = load_dataset("selfrag/selfrag_train_data")
  for idx, a in enumerate(dataset['train']):
    if len(a['output']) < 500: continue
    if random.randint(0,1):
      instruct = "<|user|>"+a['instruction']+"<|endoftext|>\n<|assistant|>"+a['output']+"<|endoftext|>"
    elif random.randint(0,1):
      instruct = "Q: "+a['instruction']+"\nA: "+a['output']
    elif random.randint(0,1):
      instruct = "Question: "+a['instruction']+"\nAnswer: "+a['output']
    elif random.randint(0,1):
      instruct = "### Answer:\nGiven this answer, give me a possible instruction or question: "+a['output']+"\n### Instruction/Question:\n"+a['instruction']
    if random.randint(0,1):
      instruct = instruct.replace('Given this', 'Provided a')
    if random.randint(0,1):
      instruct = instruct.replace('Given this answer', 'I have the below text')
    if random.randint(0,1):
      instruct = instruct.replace('give me a possible', 'find a')
    if random.randint(0,1):
      instruct = instruct.replace('instruction or question', 'instruction')
    if random.randint(0,1):
      instruct = instruct.replace('instruction or question', 'question')
    if random.randint(0,1):
      instruct = instruct.replace('Instruction/Question', 'Instruction')
    if random.randint(0,1):
      instruct = instruct.replace('Instruction/Question', 'Question')
    outf.write(json.dumps({'text': ("Q: "+a['instruction']+"\nA: "+a['output']) if len(a['output']) < 100 else a['output'], 'instruct': instruct, 'metadata': {'source': "selfrag/selfrag_train_data/"+str(idx)}})+"\n")


