import argparse
import einops
import random
import itertools
import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from src.purpleteam.blueteam import blueteam_classify_conversation, llamaguard_classifier_categories, llamaguard_category2name
from src.purpleteam.utils import chatml_format_instructions, generate_with_batching
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
    purpleteam_generative_tokenizer = AutoTokenizer.from_pretrained(args.purpleteam_generative_model_path, cache_dir="/leonardo_scratch/fast/EUHPC_E03_068/.cache")
    purpleteam_generative_model = AutoModelForCausalLM.from_pretrained(args.purpleteam_generative_model_path, low_cpu_mem_usage=True, device_map="auto", cache_dir="/leonardo_scratch/fast/EUHPC_E03_068/.cache").eval() # quantization_config=bnb_config
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
    parser.add_argument("--input_path", type=str, default="data/instructions.jsonl", help="Path to the input file.")
    parser.add_argument("--purpleteam_generative_model_path", type=str, default="teknium/OpenHermes-2.5-Mistral-7B", help="Purpleteam generative model hf path.")
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size")
    parser.add_argument("--output_path", type=str, default="data/multimodal/step-1.jsonl", help="Path to the output file.")

    args = parser.parse_args()

    purpleteam_generative_model, purpleteam_generative_tokenizer = setup(args)

    # TODO: load jsonl till batch_size
    with open(args.input_path, "r") as infile:
        with open(args.output_path, "w") as outfile:
            all_data = [json.loads(l) for l in infile]
            text_array = [data['text'] for data in all_data]
            outputs = []
            for rng in range(0, len(text_array), args.batch_size):
                d = text_array[rng:min(len(text_array), rng+args.batch_size)]
                instructions = [text.split("### Response:",1)[0].split("### Instruction:")[-1] for text in d] 
                responses = [text.split("### Response:",1)[1].strip() for text in d] 
                # prompts = [purpleteam_generative_tokenizer.apply_chat_template([{"role": "user", "content": f"Create an image caption that would be useful for answering this instruction, including topics, people, places, things and details as necessary. Generate **three** captions on each line:\n\n{instruction}"}], tokenize=False) for instruction in instructions]
                prompts = [purpleteam_generative_tokenizer.apply_chat_template([{"role": "user", "content": f"Create an image caption that would be useful for answering this instruction, including topics, people, places, things and details as necessary.\n\n{instruction}"}], tokenize=False) for instruction in instructions]
                outputs += generate_with_batching(purpleteam_generative_model, purpleteam_generative_tokenizer, prompts, accelerator.device)
            for i, output in enumerate(outputs):
                caption = output.replace("Caption:", "").replace("caption:", "").replace("caption", "").replace("Caption", "").strip()
                caption = caption.replace("1.","").replace("2.","").replace("3.","").replace("1)","").replace("2)","").replace("3)","").strip('-\' "').strip()
                instr = all_data[i]['text'].split("### Response:",1)[0].split("### Instruction:")[-1]
                chosen_response = all_data[i]['text'].split("### Response:",1)[1].strip()
                rejected_responses = [all_data[i]['text2'].split("### Response:",1)[1].strip(), all_data[i]['text3'].split("### Response:",1)[1].strip()]
                all_data[i]["metadata"]["step1_params"] = json.dumps(vars(args))

                # change the key names
                for key in list(all_data[i]['metadata'].keys()):
                    key1 = key.replace("text1", "chosen")
                    key1 = key1.replace("text2", "rejection1")
                    key1 = key1.replace("text3", "rejection2")
                    if key != key1:
                        all_data[i]['metadata'][key1] = all_data[i]['metadata'][key]
                        del all_data[i]['metadata'][key]
                    
                outfile.write(json.dumps({'instruction': instr, 'caption': caption, 'chosen_response': chosen_response, 'rejected_responses': rejected_responses, 'images': [], 'metadata': all_data[i]['metadata']})+"\n")


if __name__ == "__main__":
    main()