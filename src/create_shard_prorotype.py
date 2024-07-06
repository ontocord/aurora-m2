# make sure you have the latest version of transfomers and install wget
import json
import random
from typing import List

from datasets import load_dataset
from transformers import pipeline, Pipeline

from torch.utils.data import DataLoader

from utils import postprocess_results


def process_request(text, prompts):
    prompt = random.choice(prompts)
    if 'assault' in text or 'robbery' in text or 'arson' in text or 'fellatio' in text or 'hand job' in text or 'prostitu' in text or 'handjob' in text or 'fucks' in text or 'blow job' in text or 'blowjob' in text or ' incest' in text or ' porn' in text or ' rape' in text or ' killer' in text or ' murder' in text or ' kidnap' in text or ' abduct' in text or ' sex ' in text or ' torture' in text or ' kills ' in text:
        warning = "If this story contains themes of sex or violence, give a warning at the beginning of the story with an explanation."
    else:
        warning = ""
    prompt = prompt % {'warning': warning}
    text = {"role": "user", "content": f"{prompt}\n\n{text}"}
    return text, prompt


def process_requests(texts, metatadata, prompts):
    p_texts, prompts = [], []
    for txt, meta in zip(texts, metatadata):
        p_text, prompt = process_request(text=txt, prompts=prompts)
        p_texts.append(p_text), prompts.append(prompt)
    return p_texts, metatadata


def create_shard(llm: Pipeline, stories_per_shard: int, src_file: str, shard_path: str, prompts: List[str], batch_size: str) -> None:
    dataset = load_dataset('json', data_files=src_file)['train']
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4)
    with open(shard_path, "w") as outfile:
        total_stories: bool = 0
        while total_stories < stories_per_shard:
            for texts, metadatas in loader:
                if total_stories >= stories_per_shard:
                    break
                total_stories += len(texts)
                messages = process_requests(texts=texts, metatadata=metadatas, prompts=prompts)
                prompt = random.choice(prompts)
                output = llm(messages, max_length=2048, min_length=512, use_cache=True)
                output_texts = postprocess_results(output)
                for output_text, metadata in zip(output_texts, metadatas):
                    outfile.write(
                        json.dumps(
                            {
                                'text': output_text,
                                'prompt': prompt,
                                'input': output_text,
                                'metadata': metadata
                            }) + "\n"
                    )
