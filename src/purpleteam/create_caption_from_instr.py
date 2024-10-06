import argparse
import einops
import random
import itertools
import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from src.purpleteam.blueteam import blueteam_classify_conversation, llamaguard_classifier_categories, llamaguard_category2name
from src.purpleteam.utils import chatml_format_instructions, generate_with_batching, tokenize_with_assistant_continuation
from src.purpleteam.templates.rule import rule_templates
from src.purpleteam.templates.seed import *
from src.accelerator import accelerator

torch.cuda.empty_cache()


def setup(args):

    # Setup for PurpleTeam generative model
    purpleteam_generative_tokenizer = AutoTokenizer.from_pretrained(args.purpleteam_generative_model_path, cache_dir=args.cache_dir)
    purpleteam_generative_model = AutoModelForCausalLM.from_pretrained(args.purpleteam_generative_model_path, low_cpu_mem_usage=True, device_map="auto", torch_dtype=torch.bfloat16, cache_dir=args.cache_dir).eval() # quantization_config=bnb_config
    purpleteam_generative_tokenizer.pad_token = purpleteam_generative_tokenizer.eos_token
    purpleteam_generative_model = accelerator.prepare(purpleteam_generative_model)


    return purpleteam_generative_model, purpleteam_generative_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Set up models with quantization and specific configurations.")
    parser.add_argument("--input_path", type=str, help="Path to the input file.")
    parser.add_argument("--cache_dir", type=str, default="", help="Path to cache directory.")
    parser.add_argument("--purpleteam_generative_model_path", type=str, default="teknium/OpenHermes-2.5-Mistral-7B", help="Purpleteam generative model hf path.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--output_path", type=str, help="Path to the output file.")

    args = parser.parse_args()

    purpleteam_generative_model, purpleteam_generative_tokenizer = setup(args)

    # TODO: load jsonl till batch_size
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
                text_array = [data['text'] for data in all_data]

                instructions = [text.split("### Response:",1)[0].split("### Instruction:")[-1] for text in text_array] 
                # responses = [text.split("### Response:",1)[1].strip() for text in text_array] 
                prefix = random.choice(["an image of", "a photo of", "a photograph of", "a picture of", "a screenshot of", "a screen shot of"])
                prompts = [tokenize_with_assistant_continuation(purpleteam_generative_tokenizer, [{"role": "user", "content": f"Create an image caption that would be useful for answering this instruction, including topics, people, places, things and details as necessary.\n\n{instruction}"},
                                                                {"role": "assistant", "content": f"Caption: {prefix}"}]) for instruction in instructions]
                outputs = generate_with_batching(purpleteam_generative_model, purpleteam_generative_tokenizer, prompts, accelerator.device, max_new_tokens=400, batch_size=args.batch_size)
                for i, output in enumerate(outputs):
                    caption = output.replace("Caption:", "").replace("caption:", "").replace("caption", "").replace("Caption", "").strip()
                    caption = caption.replace("1.","").replace("2.","").replace("3.","").replace("1)","").replace("2)","").replace("3)","").strip('-\' "').strip()
                    instr = all_data[i]['text'].split("### Response:",1)[0].split("### Instruction:", 1)[-1]
                    chosen_response = all_data[i]['text'].split("### Response:",1)[1].strip()
                    rejected_responses = [all_data[i]['text2'].split("### Response:",1)[1].strip(), all_data[i]['text3'].split("### Response:",1)[1].strip()]
                    all_data[i]["metadata"]["create_caption_from_instr-params"] = json.dumps(vars(args))

                    # change the key names
                    for key in list(all_data[i]['metadata'].keys()):
                        key1 = key.replace("text1", "chosen")
                        key1 = key1.replace("text2", "rejection1")
                        key1 = key1.replace("text3", "rejection2")
                        if key != key1:
                            all_data[i]['metadata'][key1] = all_data[i]['metadata'][key]
                            del all_data[i]['metadata'][key]
                        
                    outfile.write(json.dumps({'instruction': instr, 'caption': caption, 'chosen_response': chosen_response, 'rejected_responses': rejected_responses, 'images': [], 'metadata': all_data[i]['metadata']})+"\n")
                rng += args.batch_size


if __name__ == "__main__":
    main()
    print("Completed!!")