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
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=False,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.float16
    # )

    # Setup for PurpleTeam generative model
    purpleteam_generative_tokenizer = AutoTokenizer.from_pretrained(args.purpleteam_generative_model_path, cache_dir=args.cache_dir)
    purpleteam_generative_model = AutoModelForCausalLM.from_pretrained(args.purpleteam_generative_model_path, low_cpu_mem_usage=True, device_map="auto", cache_dir=args.cache_dir).eval() # quantization_config=bnb_config
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
            while True:
                # Read a batch of lines from the input file
                lines = list(itertools.islice(infile, args.batch_size))
                if not lines:
                    break  # Exit the loop if no lines are left

                # Apply the algo over the batched data
                all_data = [json.loads(l) for l in lines]
                text_array = [data['caption'] for data in all_data]
                outputs = []
                for rng in range(0, len(text_array), args.batch_size):
                    d = text_array[rng:min(len(text_array), rng+args.batch_size)]
                    if randint(0, 1):
                        prompts = [tokenize_with_assistant_continuation(purpleteam_generative_tokenizer, [{"role": "system", "content": f"""You are an AI visual assistant that can analyze a single image. You receive five sentences, each describing the same image you are observing. In addition, specific object locations within the image are given, along with detailed coordinates. These coordinates are in the form of bounding boxes, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.\n\nThe task is to use the provided caption and bounding box information, create a plausible question about the image, and provide the answer in detail.\n\nCreate complex questions beyond describing the scene.\nTo answer such questions, one should require first understanding the visual content, then based on the background knowledge or reasoning, either explain why the things are happening that way, or provide guides and help to user's request.  Make the question challenging by not including the visual content details in the question so that the user needs to reason about that first.\n\nInstead of directly mentioning the bounding box coordinates, utilize this data to explain the scene using natural language. Include details like object counts, position of the objects, relative position between the objects.\n\nWhen using the information from the caption and coordinates, directly explain the scene, and do not mention that the information source is the caption or the bounding box. Always answer as if you are directly looking at the image."""},
                                                                                                          {"role": "user", "content": f"Caption: {caption}"}]) for caption in captions]
                    else:
                        prompts = [tokenize_with_assistant_continuation(purpleteam_generative_tokenizer, [{"role": "system", "content": f"""You are an AI visual assistant that can analyze a single image. You receive five sentences, each describing the same image you are observing. In addition, specific object locations within the image are given, along with detailed coordinates. These coordinates are in the form of bounding boxes, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.\n\nUsing the provided caption and bounding box information, describe the scene in a detailed manner.\n\nInstead of directly mentioning the bounding box coordinates, utilize this data to explain the scene using natural language. Include details like object counts, position of the objects, relative position between the objects.\n\nWhen using the information from the caption and coordinates, directly explain the scene, and do not mention that the information source is the caption or the bounding box.  Always answer as if you are directly looking at the image."""},
                                                                                                          {"role": "user", "content": f"Caption: {caption}"}]) for caption in captions]
                    outputs += generate_with_batching(purpleteam_generative_model, purpleteam_generative_tokenizer, prompts, accelerator.device)
                for i, output in enumerate(outputs):
                    instr = all_data[i]['text'].split("### Response:",1)[0].split("### Instruction:")[-1]
                    chosen_response = all_data[i]['text'].split("### Response:",1)[1].strip()
                    rejected_responses = [all_data[i]['text2'].split("### Response:",1)[1].strip(), all_data[i]['text3'].split("### Response:",1)[1].strip()]
                    all_data[i]["metadata"]["step2_params"] = json.dumps(vars(args))
                        
                    outfile.write(json.dumps({'instruction': instr, 'caption': caption, 'chosen_response': chosen_response, 'rejected_responses': rejected_responses, 'images': [], 'metadata': all_data[i]['metadata']})+"\n")


if __name__ == "__main__":
    main()
    print("Completed!!")