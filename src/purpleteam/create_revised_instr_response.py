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
            while True:
                # Read a batch of lines from the input file
                lines = list(itertools.islice(infile, args.batch_size))
                if not lines:
                    break  # Exit the loop if no lines are left

                # Apply the algo over the batched data
                all_data = [json.loads(l) for l in lines]
                text_array = [remove_quotes(data['caption']) for data in all_data]
                instr_array = [data['instruction'] for data in all_data]
                response_array = [data['chosen_response'] for data in all_data]

                revised_instructions, revised_responses = [], []
                for rng in range(0, len(text_array), args.batch_size):
                    captions = text_array[rng:min(len(text_array), rng+args.batch_size)]
                    instructions = instr_array[rng:min(len(instr_array), rng+args.batch_size)]
                    responses = response_array[rng:min(len(response_array), rng+args.batch_size)]

                    revised_instruction_batch = generate_image_aware_instruction(captions, instructions, purpleteam_generative_model, purpleteam_generative_tokenizer)
                    revised_instructions += revised_instruction_batch

                    if random.randint(0,1):
                        prompts = [purpleteam_generative_tokenizer.apply_chat_template([{"role": "user", "content": f"Given the below image:\n{caption}\n===\n{revised_instruction} If the question cannot be answered by the image state so politely and state why."}], tokenize=False)
                                    for caption, revised_instruction in zip(captions, revised_instruction_batch)]
                        revised_responses += generate_with_batching(purpleteam_generative_model, purpleteam_generative_tokenizer, prompts, accelerator.device)
                    else:
                        prompts = [purpleteam_generative_tokenizer.apply_chat_template([{"role": "user", "content": instruction}, {"role": "assistant", "content": response},
                                                                  {"role": "user", "content": f"Given the below image:\n{caption}\n===\n{revised_instruction}\nIf instruction cannot be answered based on the image alone, refer to the prior conversation, and explain why. Do not refer to any context document in your answer."}], tokenize=False)
                                    for caption, instruction, revised_instruction, response in zip(captions, instructions, revised_instruction_batch, responses)]
                        revised_responses += generate_with_batching(purpleteam_generative_model, purpleteam_generative_tokenizer, prompts, accelerator.device)

                revised_responses = [revised_response[0].replace("\n\n", "\n").strip() for revised_response in revised_responses]

                for i, (revised_instruction, revised_response) in enumerate(zip(revised_instructions, revised_responses)):
                    all_data[i]['revised_instruction'] = revised_instruction
                    all_data[i]['revised_response'] = revised_response
                    all_data[i]["metadata"]["step3_params"] = json.dumps(vars(args))
                    outfile.write(json.dumps(all_data[i])+"\n")


if __name__ == "__main__":
    main()
    print("Completed!!")