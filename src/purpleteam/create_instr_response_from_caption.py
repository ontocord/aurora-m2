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
            rng = 0
            while True:
                # Read a batch of lines from the input file
                lines = list(itertools.islice(infile, args.batch_size))
                if not lines:
                    break  # Exit the loop if no lines are left

                # Get instr for the batched data
                all_data = [json.loads(l) for l in lines]
                captions = [data['caption'] for data in all_data]
                prompts = [tokenize_with_assistant_continuation(purpleteam_generative_tokenizer, [{"role": "system", "content": f"""You are an AI visual assistant that can analyze an image. You receive a caption, describing the image you are observing.\n\nThe task is to use the provided caption, create a plausible question about the image, and provide the answer in detail.\n\nCreate complex questions beyond describing the scene.\nTo answer such questions, one should require first understanding the visual content, then based on the background knowledge or reasoning, either explain why the things are happening that way, or provide guides and help to user's request. Make the question challenging by not including the visual content details in the question so that the user needs to reason about that first. Include details like object counts, position of the objects, relative position between the objects.\n\nWhen using the information from the caption, directly explain the scene, and do not mention that the information source is the caption. Always answer as if you are directly looking at the image."""},
                                                                                                  {"role": "user", "content": f"Caption: {caption}"},
                                                                                                  {"role": "assistant", "content": "Question:"}]) for caption in captions]
                outputs = generate_with_batching(purpleteam_generative_model, purpleteam_generative_tokenizer, prompts, accelerator.device)
                for idx, output in enumerate(outputs):
                    instr, chosen_response = output.split("Answer:", 1)[0], output.split("Answer:", 1)[-1]
                    instr = instr.split("Question:", 1)[-1]
                    chosen_response = chosen_response.split("Question:", 1)[0]
                    prompt = f"User: {instr}\nAssistant: {chosen_response}"
                    rejected_responses = [""]
                    all_data[idx]["metadata"]["create_instr_from_caption-params"] = json.dumps(vars(args))
                    outfile.write(json.dumps({'prompt': prompt,
                                                'is_pairwise': False,
                                                'captions': [all_data[idx]['caption']], 
                                                'chosen_response': chosen_response, 
                                                'rejected_responses': rejected_responses,
                                                'caption_media_scores': [all_data[idx]["metadata"]["caption_media_score"]], 
                                                'medias': all_data[idx]['images'], 
                                                'media_coordinates': [[0, 0, 0, 0]], # dummy
                                                'media_types': ["image"],
                                                'metadata': {'source': all_data[idx]['metadata']["source"], 
                                                            'params': json.dumps(all_data[idx]['metadata'])}})+"\n")


if __name__ == "__main__":
    main()
    print("Completed!!")