import argparse
import einops
import random
import itertools
import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from src.purpleteam.blueteam import blueteam_classify_conversation, llamaguard_classifier_categories, llamaguard_category2name
from src.purpleteam.utils import chatml_format_instructions, generate_with_batching, remove_quotes, add_img_context_to_question
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

def create_multiturn(captions, instructions, responses, num_turns=5):
    user_history_batch = [[{"role": "system", "content": f"""You have to act as a user which asks an AI assistant some questions based on an image and the conversation history. Your question should strictly adhere to the image and the context.
Your task is to ONLY ask questions but not answer them. Below is the image caption and the question:\n\nImage Caption: {caption}\n=====\nQuestion: {instruction}"""},
{"role": "assistant", "content": f"Answer: {response}"}] for caption, instruction, response in zip(captions, instructions, responses)]
    assistant_history_batch = [[{"role": "system", "content": f"""You have to act as an AI assistant who answers user's question. Your answer should strictly adhere to the image and the context.
Your task is to ONLY answer the provided question."""},
{"role": "user", "content": f"Image: {caption}\n=====\nQuestion: {instruction}"},
{"role": "assistant", "content": f"Answer: {response}"}] for caption, instruction, response in zip(captions, instructions, responses)]
    for _ in range(num_turns):
        user_history_batch = [uu + [{"role": "assistant", "content": "Question:"}] for uu in user_history_batch]
        prompts = [tokenize_with_assistant_continuation(purpleteam_generative_tokenizer, uu) for uu in user_history_batch]
        outputs = generate_with_batching(purpleteam_generative_model, purpleteam_generative_tokenizer, prompts, accelerator.device)
        # add img context to question
        for uu, oo in zip(user_history_batch, outputs):
            uu[-1]['content'] += " " + (add_img_context_to_question(oo) if random.randint(0, 1) else oo) 
        assistant_history_batch = [aa + [{"role": "user", "content": uu[-1]['content']}] for aa, uu in zip(assistant_history_batch, user_history_batch)]

        assistant_history_batch = [aa + [{"role": "assistant", "content": "Answer:"}] for aa in assistant_history_batch]
        prompts = [tokenize_with_assistant_continuation(purpleteam_generative_tokenizer, uu) for uu in assistant_history_batch]
        outputs = generate_with_batching(purpleteam_generative_model, purpleteam_generative_tokenizer, prompts, accelerator.device)
        for aa, oo in zip(assistant_history_batch, outputs): aa[-1]['content'] += " " + oo       
        user_history_batch = [uu + [{"role": "user", "content": aa[-1]['content']}] for uu, aa in zip(user_history_batch, assistant_history_batch)]
    assistant_history_batch = [aa[1:] for aa in assistant_history_batch]
    for assistant_history in assistant_history_batch: 
        for aa in assistant_history:
            aa['content'] = aa['content'].split("Question:", 1)[-1].split("question:", 1)[-1].split("Answer:", 1)[-1].split("answer:", 1)[-1].strip()
    return assistant_history_batch

def main():
    parser = argparse.ArgumentParser(description="Set up models with quantization and specific configurations.")
    parser.add_argument("--input_path", type=str, help="Path to the input file.")
    parser.add_argument("--cache_dir", type=str, default="", help="Path to cache directory.")
    parser.add_argument("--num_turns", type=int, default=6, help="Max number of turns in conversation.")
    parser.add_argument("--purpleteam_generative_model", type=str, default="teknium/OpenHermes-2.5-Mistral-7B", help="Purpleteam generative model hf path.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--output_path", type=str, help="Path to the output file.")

    args = parser.parse_args()

    purpleteam_generative_model, purpleteam_generative_tokenizer = setup(args)
    num_turns = random.randint(args.num_turns-3, args.num_turns)

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
                chosen_responses = [data['chosen_responses'] for data in all_data]

                conversations, revised_instructions, revised_conversations = [], [], []
                for rng in range(0, len(text_array), args.batch_size):
                    captions = text_array[rng:min(len(text_array), rng+args.batch_size)]
                    instructions = instr_array[rng:min(len(instr_array), rng+args.batch_size)]
                    responses = chosen_responses[rng:min(len(chosen_responses), rng+args.batch_size)]

                    conversations += create_multiturn(captions, instructions, responses, num_turns)

                    instr_revision_prompts = [tokenize_with_assistant_continuation(purpleteam_generative_tokenizer, [{"role": "user", "content": f"You are given the below image:\n{caption}\n===\nRevise the below instruction such that events, physical conditions, attributes, color, actions, feelings, objects, people or other information from the image are removed from the question, and the instruction refers to those things in the image instead. Do not refer to proper names in the instruction if those names are already in the image. Do not refer to any context document. Do not refer to the 'description' of the image. Retain the theme of the instruction. Do not repeat this instruction or the information from the image in your revised instruction. The instruction is:\n{instruction}"}, 
                                                                                                                    {"role": "assistant", "content": "Revised Instruction: Using this image"}]) for caption, instruction in zip(captions, instructions)]
                    revised_instruction_batch = generate_with_batching(purpleteam_generative_model, purpleteam_generative_tokenizer, prompts, accelerator.device)
                    revised_instruction_batch = [
                        revised_instruction.split("instruction:", 1)[-1]
                        .split("Instruction:", 1)[-1]
                        .split("Revised Instruction:", 1)[-1]
                        .split("Revised instruction:", 1)[-1]
                        .split("assistant\n", 1)[-1]
                        .replace("caption", "image").strip()
                        for revised_instruction in revised_instructions_batch
                        if "answer:" not in revised_instruction.lower()
                    ]
                    revised_instructions += revised_instruction_batch

                    revised_conversations += create_multiturn(captions, revised_instruction_batch, responses, num_turns)
                    # # instructions = [text.split("### Response:",1)[0].split("### Instruction:")[-1] for text in instructions] 
                    # prompts = [purpleteam_generative_tokenizer.apply_chat_template([{"role": "user", "content": f"You are given the below image:\n{caption}\n===\nRevise the below question such that events, physical conditions, attributes, color, actions, feelings, objects, people or other information from the image are removed from the question, and the question refers to the image instead. Do not refer to any context document. Do not refer to the 'description' of the image. Retain the theme of the question. Do not repeat this instruction or the information from the image in your revised question. Do not answer the question, but simply provide the revised question. The question is:\n{instruction}"}], tokenize=False) for caption, instruction in zip(captions, instructions)]
                    # outputs += generate_with_batching(purpleteam_generative_model, purpleteam_generative_tokenizer, prompts, accelerator.device)
                for i, (conversation, revised_instruction, revised_conversation) in enumerate(zip(conversations, revised_instructions, revised_conversations)):
                    all_data[i]['conversation'] = conversation
                    all_data[i]['revised_instruction'] = revised_instruction
                    all_data[i]['revised_conversation'] = revised_conversation
                    all_data[i]["metadata"]["step3_params"] = json.dumps(vars(args))
                    outfile.write(json.dumps(all_data[i])+"\n")


if __name__ == "__main__":
    main()
    print("Completed!!")