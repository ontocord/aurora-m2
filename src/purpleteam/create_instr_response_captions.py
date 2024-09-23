import argparse
import einops
import random
import itertools
import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from src.purpleteam.blueteam import blueteam_classify_conversation, llamaguard_classifier_categories, llamaguard_category2name
from src.purpleteam.utils import chatml_format_instructions, generate_with_batching, remove_quotes
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

    # # Setup for LlamaGuard model
    # llamaguard_tokenizer = AutoTokenizer.from_pretrained(args.llamaguard_path, cache_dir="/leonardo_scratch/fast/EUHPC_E03_068/.cache")
    # llamaguard_model = AutoModelForCausalLM.from_pretrained(args.llamaguard_path, low_cpu_mem_usage=True, device_map="auto", cache_dir="/leonardo_scratch/fast/EUHPC_E03_068/.cache").eval() # quantization_config=bnb_config
    # llamaguard_tokenizer.pad_token = llamaguard_tokenizer.eos_token
    # llamaguard_model = accelerator.prepare(llamaguard_model)

    # Setup for PurpleTeam generative model
    purpleteam_generative_tokenizer = AutoTokenizer.from_pretrained("teknium/OpenHermes-2.5-Mistral-7B", cache_dir=args.cache_dir)
    purpleteam_generative_model = AutoModelForCausalLM.from_pretrained("teknium/OpenHermes-2.5-Mistral-7B", low_cpu_mem_usage=True, device_map="auto", cache_dir=args.cache_dir).eval() # quantization_config=bnb_config
    purpleteam_generative_tokenizer.pad_token = purpleteam_generative_tokenizer.eos_token
    purpleteam_generative_model = accelerator.prepare(purpleteam_generative_model)

    # # Setup for target model
    # target_tokenizer = AutoTokenizer.from_pretrained(args.target_model_path, cache_dir="/leonardo_scratch/fast/EUHPC_E03_068/.cache")
    # target_model = AutoModelForCausalLM.from_pretrained(args.target_model_path, low_cpu_mem_usage=True, device_map="auto", cache_dir="/leonardo_scratch/fast/EUHPC_E03_068/.cache").eval() # quantization_config=bnb_config
    # target_tokenizer.pad_token = target_tokenizer.eos_token
    # target_model = accelerator.prepare(target_model)

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
            all_data = [json.loads(l) for l in infile]
            text_array = [remove_quotes(data['caption']) for data in all_data]
            instr_array = [data['instruction'] for data in all_data]
            outputs = []
            for rng in range(0, len(text_array), args.batch_size):
                captions = text_array[rng:min(len(text_array), rng+args.batch_size)]
                instructions = instr_array[rng:min(len(instr_array), rng+args.batch_size)]
                # instructions = [text.split("### Response:",1)[0].split("### Instruction:")[-1] for text in instructions] 
                prompts = [purpleteam_generative_tokenizer.apply_chat_template([{"role": "user", "content": f"You are given the below image:\n{caption}\n===\nRevise the below question such that events, physical conditions, attributes, color, actions, feelings, objects, people or other information from the image are removed from the question, and the question refers to the image instead. Do not refer to any context document. Do not refer to the 'description' of the image. Retain the theme of the question. Do not repeat this instruction or the information from the image in your revised question. Do not answer the question, but simply provide the revised question. The question is:\n{instruction}"}], tokenize=False) for caption, instruction in zip(captions, instructions)]
                outputs += generate_with_batching(purpleteam_generative_model, purpleteam_generative_tokenizer, prompts, accelerator.device)
            for i, output in enumerate(outputs):
                if "answer:" in output.lower(): continue
                revised_instruction = output.split("Revised question:",1)[-1].split("Revised Question:",1)[-1].split("Question:",1)[-1].split("question:",1)[-1].strip()
                revised_instruction = revised_instruction.replace("caption", "image")
                all_data[i]['instruction'] = revised_instruction
                all_data[i]["metadata"]["step3_params"] = json.dumps(vars(args))
                outfile.write(json.dumps(all_data[i])+"\n")


if __name__ == "__main__":
    main()
    print("Completed!!")