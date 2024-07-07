# make sure you have the latest version of transfomers and install wget
import json
import random
from multiprocessing import Value
from typing import List

from datasets import load_dataset, Features, Value, tqdm
from transformers import pipeline, Pipeline

from torch.utils.data import DataLoader

from src.utils import postprocess_results


def process_request(text, prompts):
    prompt = random.choice(prompts)
    if 'assault' in text or 'robbery' in text or 'arson' in text or 'fellatio' in text or 'hand job' in text or 'prostitu' in text or 'handjob' in text or 'fucks' in text or 'blow job' in text or 'blowjob' in text or ' incest' in text or ' porn' in text or ' rape' in text or ' killer' in text or ' murder' in text or ' kidnap' in text or ' abduct' in text or ' sex ' in text or ' torture' in text or ' kills ' in text:
        warning = "If this story contains themes of sex or violence, give a warning at the beginning of the story with an explanation."
    else:
        warning = ""
    prompt = prompt % {'warning': warning}
    text = [{"role": "user", "content": f"{prompt}\n\n{text}"}]
    return text, prompt


def process_requests(texts, metatadata, prompts):
    p_texts, used_prompts = [], []
    for txt, meta in zip(texts, metatadata):
        p_text, prompt = process_request(text=txt, prompts=prompts)
        p_texts.append(p_text), used_prompts.append(prompt)
    return p_texts, used_prompts


def create_shard(llm: Pipeline, stories_per_shard: int, src_file: str, shard_path: str, prompts: List[str], batch_size: int) -> None:
    print(src_file)
    dataset = load_dataset('json', data_files=src_file)['train']
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4)
    with open(shard_path, "w") as outfile:
        total_stories: bool = 0
        while total_stories < stories_per_shard:
            for texts, metadatas in loader:
                if total_stories >= stories_per_shard:
                    break
                total_stories += len(texts)
                messages, used_prompts = process_requests(texts=texts, metatadata=metadatas, prompts=prompts)
                output = llm(messages, max_length=2048, min_length=512, use_cache=True)
                output_texts = postprocess_results(output)
                for output_text, used_prompt, metadata in zip(output_texts, used_prompts, metadatas):
                    outfile.write(
                        json.dumps(
                            {
                                'text': output_text,
                                'prompt': used_prompt,
                                'input': output_text,
                                'metadata': metadata
                            }) + "\n"
                    )
