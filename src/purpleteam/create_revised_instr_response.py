import argparse
import einops
import random
import itertools
import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from src.purpleteam.blueteam import blueteam_classify_conversation, llamaguard_classifier_categories, llamaguard_category2name
from src.purpleteam.utils import chatml_format_instructions, generate_with_batching, remove_quotes, generate_image_aware_instruction, tokenize_with_assistant_continuation
from src.purpleteam.templates.rule import rule_templates
from src.purpleteam.templates.seed import *
from src.accelerator import accelerator

torch.cuda.empty_cache()


def setup(args):
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=False,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.float16
    # )

    # Setup for PurpleTeam generative model
    purpleteam_generative_tokenizer = AutoTokenizer.from_pretrained("teknium/OpenHermes-2.5-Mistral-7B", cache_dir=args.cache_dir)
    purpleteam_generative_model = AutoModelForCausalLM.from_pretrained("teknium/OpenHermes-2.5-Mistral-7B", low_cpu_mem_usage=True, device_map="auto", cache_dir=args.cache_dir).eval() # quantization_config=bnb_config
    purpleteam_generative_tokenizer.pad_token = purpleteam_generative_tokenizer.eos_token
    purpleteam_generative_model = accelerator.prepare(purpleteam_generative_model)

    return purpleteam_generative_model, purpleteam_generative_tokenizer

def main():
    parser = argparse.ArgumentParser(description="Set up models with quantization and specific configurations.")
    parser.add_argument("--input_path", type=str, help="Path to the input file.")
    parser.add_argument("--cache_dir", type=str, default="", help="Path to cache directory.")
    parser.add_argument("--purpleteam_generative_model", type=str, default="teknium/OpenHermes-2.5-Mistral-7B", help="Purpleteam generative model hf path.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--output_path", type=str, help="Path to the output file.")

    args = parser.parse_args()

    purpleteam_generative_model, purpleteam_generative_tokenizer = setup(args)

    with open(args.input_path, "r") as infile: 
        with open(args.output_path, "w") as outfile:
            rng = 0
            while True:
                # Read a batch of lines from the input file
                lines = list(itertools.islice(infile, args.batch_size))
                if not lines:
                    break  # Exit the loop if no lines are left

                # Apply the algo over the batched data
                all_data = [json.loads(l) for l in lines]
                captions = [remove_quotes(data['caption']) for data in all_data]
                instructions = [data['instruction'] for data in all_data]
                responses = [data['chosen_response'] for data in all_data]

                revised_instructions = generate_image_aware_instruction(captions, instructions, purpleteam_generative_model, purpleteam_generative_tokenizer)

                # if random.randint(0,1):
                #     prompts = [purpleteam_generative_tokenizer.apply_chat_template([{"role": "user", "content": f"Given the below image:\n{caption}\n===\n{revised_instruction} If the question cannot be answered by the image state so politely and state why."}], tokenize=False)
                #                 for caption, revised_instruction in zip(captions, revised_instructions)]
                #     revised_responses = generate_with_batching(purpleteam_generative_model, purpleteam_generative_tokenizer, prompts, accelerator.device)
                # else:
                prompts = [purpleteam_generative_tokenizer.apply_chat_template([{"role": "user", "content": instruction}, {"role": "assistant", "content": response},
                                                            {"role": "user", "content": f"Given the below image:\n{caption}\n===\n{revised_instruction}\nIf instruction cannot be answered based on the image alone, refer to the prior conversation, and explain why. Do not refer to any context document in your answer. If the question cannot be answered by the image state so politely and state why."}], tokenize=False)
                            for caption, instruction, revised_instruction, response in zip(captions, instructions, revised_instructions, responses)]
                revised_responses = generate_with_batching(purpleteam_generative_model, purpleteam_generative_tokenizer, prompts, accelerator.device)
                revised_responses = [revised_response[0].replace("\n\n", "\n").strip() for revised_response in revised_responses]

                for idx, (instruction, response, revised_instruction, revised_response) in enumerate(zip(instructions, responses, revised_instructions, revised_responses)):
                    all_data[idx]["metadata"]['revised_instruction'] = revised_instruction
                    all_data[idx]["metadata"]['revised_response'] = revised_response
                    all_data[idx]["metadata"]['instruction'] = instruction
                    all_data[idx]["metadata"]['response'] = response
                    all_data[idx]["metadata"]["create_revised_instr_response-params"] = json.dumps(vars(args))
                    prompt = f"User: {all_data[idx]['revised_instruction']}\nAssistant: {all_data[idx]['chosen_response']}"
                    outfile.write(json.dumps({'prompt': prompt,
                                                'is_pairwise': True,
                                                'captions': [all_data[idx]['caption']], 
                                                'chosen_response': all_data[idx]['chosen_response'], 
                                                'rejected_responses': all_data[idx]['rejected_responses'],
                                                'caption_media_scores': [all_data[idx]["metadata"]["caption_media_score"]], 
                                                'medias': all_data[idx]['images'], 
                                                'media_coordinates': [[0, 0, 0, 0]], # dummy
                                                'media_types': ["image"],
                                                'metadata': {'source': all_data[idx]['metadata']["source"], 
                                                            'params': json.dumps(all_data[idx]['metadata'])}})+"\n")

                    # outfile.write(json.dumps(all_data[i])+"\n")
                rng += args.batch_size


if __name__ == "__main__":
    main()
    print("Completed!!")