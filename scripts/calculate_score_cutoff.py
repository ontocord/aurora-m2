import os
import json 

score_cutoffs = ["0.12", "0.14", "0.16", "0.18", "0.20", "0.22", "0.24", "0.26","0.28", "0.30", "0.32", "0.34", "0.36", "0.38"]

score_cutoffs_and_cos_scores = []
for score_cutoff in score_cutoffs:
    print("score_cutoff:", score_cutoff)
    data = []
    with open(f"/leonardo_work/EUHPC_E03_068/safellm/data/re_caption-out-{score_cutoff}.jsonl", "r") as file:
        for line in file:
            data.append(json.loads(line))
    
    cos_scores = []
    for datum in data:
        if len(datum["metadata"]["related"]) == 3:
            cos_score = datum["metadata"]["related"][-1][1]
        if len(datum["metadata"]["related"]) == 4:
            cos_score = max(datum["metadata"]["related"][-1][1], datum["metadata"]["related"][-2][1])
        cos_scores.append(cos_score)    
    # print("cos_scores:", cos_scores)
    print("avg_cos_score:", sum(cos_scores)/len(cos_scores))
    print()
    score_cutoffs_and_cos_scores.append((score_cutoff, sum(cos_scores)/len(cos_scores)))

max_ele = max(score_cutoffs_and_cos_scores, key=lambda x: x[1])
print(f"MAX CUTOFF OF {max_ele[0]} AT WITH COS SIM OF {max_ele[1]}")


## SCORE CUTOFFS:
# 1. MAX CUTOFF OF 0.14 AT WITH COS SIM OF 0.33921077439022546
# "{\"input_path\": \"data/multimodal/step-1-test.jsonl\", \"batch_size\": 4, \"cache_dir\": \"/leonardo_scratch/fast/EUHPC_E03_068/.cache\", \"purpleteam_generative_model_path\": \"teknium/OpenHermes-2.5-Mistral-7B\", \"cos_score_model_path\": \"openai/clip-vit-base-patch32\", \"caption_generator_model_path\": \"multimodalart/Florence-2-large-no-flash-attn\", \"image_generator_model_path\": \"black-forest-labs/FLUX.1-schnell\", \"llamaguard_path\": \"meta-llama/Llama-Guard-3-8B\", \"output_path\": \"data/multimodal/step-2-test.jsonl\"}"



