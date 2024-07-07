# make sure you have the latest version of transfomers and install wget
import json
import os
import random
from typing import List
from tqdm import tqdm
from transformers import Pipeline
from src.utils import postprocess_results


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
    p_texts, used_prompts = [], []
    for txt, meta in zip(texts, metatadata):
        p_text, prompt = process_request(text=txt, prompts=prompts)
        p_texts.append(p_text), used_prompts.append(prompt)
    return p_texts, metatadata


def create_shard(llm: Pipeline, stories_per_shard: int, src_file: str, shard_path: str, prompts: List[str], batch_size: int) -> None:
    print(src_file)
    # FIXME: This breaks on leonardo for some reason, would be faster to do it this way though
    #dataset = load_dataset('json', data_files=src_file)['train']
    #loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4)
    print("Loading datapoints")
    # FIXME: This line is broken right now
    with open(src_file, "r", os.O_NONBLOCK | os.O_RDONLY) as fp:
        rows = []
        for row in fp:
            print(row)
            rows.append(row)
        print(rows)
        json_files = [(json.loads(x)['text'], json.loads(x)['metadata']) for x in tqdm(rows, 'loading_samples')]
        print("Loading completed, extracting metadata")
        all_text, all_metadata = [x[0] for x in json_files], [x[1] for x in json_files]
        print("write file")
    with open(shard_path, "w", os.O_NONBLOCK | os.O_RDONLY) as outfile:
        total_stories: bool = 0
        while total_stories < stories_per_shard:
            for i in range(0, len(all_text), batch_size):
                texts = all_text[i:min(i+batch_size, len(all_text))]
                metadatas = all_metadata[i:min(i+batch_size, len(all_metadata))]
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
