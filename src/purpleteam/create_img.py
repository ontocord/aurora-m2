import json
import time
import argparse
import spacy
import itertools
import random
import numpy as np
from src.purpleteam.utils import get_element_to_img, pil_image_to_base64

import torch
from torch.nn.functional import cosine_similarity
from PIL import Image
from diffusers import FluxPipeline
from transformers import pipeline
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel, AutoModel, AutoTokenizer, AutoModelWithLMHead
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from src.accelerator import accelerator
from src.purpleteam.utils import chatml_format_instructions, generate_with_batching, assign_uuid, tokenize_with_assistant_continuation, augment_for_quotes, replace_color_rectangles_with_text

from src.frcnn.visualizing_image import SingleImageViz
from src.frcnn.processing_image import Preprocess
from src.frcnn.modeling_frcnn import GeneralizedRCNN
from src.frcnn.utils import Config
from src.frcnn.utils import decode_image


# Load necessary data
digits_to_words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
                  'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen',
                  'nineteen', 'twenty']

spacy_nlp = spacy.load('en_core_web_sm')
max_detections = 36

def setup(args):
  flux_pipe = FluxPipeline.from_pretrained(args.image_generator_model_path, torch_dtype=torch.bfloat16, cache_dir=args.cache_dir)
  flux_pipe = flux_pipe.to(accelerator.device)

  return flux_pipe


def generate_images(prompt_array: list, suffix: str = ""):
    # Modify the original prompt by appending adversarial suffix
    # prompt_array = [f"{prompt} {suffix}".strip() for prompt in prompt_array]

    # remove quotation text from the prompt
    # detected_and_cleaned_texts = augment_for_quotes(prompt_array)

    # prompt_array = [obj[1] if len(obj[1]) > 10 else prompt for obj, prompt in zip(detected_and_cleaned_texts, prompt_array)]

    time0 = time.time()
    # Generate image with Flux pipeline
    images = flux_pipe(
        prompt_array,
        guidance_scale=0.0,
        num_inference_steps=4,
        max_sequence_length=256,
        generator=torch.Generator(accelerator.device).manual_seed(0)
    ).images
    print(f"@TIME |'create_img' | Image Gen time: {time.time() - time0}")
    return images


def main():
    parser = argparse.ArgumentParser(description="Set up models with quantization and specific configurations.")
    parser.add_argument("--input_path", type=str, help="Path to the input file.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--cache_dir", type=str, default="", help="Path to cache directory.")
    parser.add_argument("--image_generator_model_path", type=str, default="black-forest-labs/FLUX.1-schnell")
    parser.add_argument("--output_path", type=str, help="Path to save output for this step.")
    parser.add_argument("--output_dir", type=str, help="Dir to save images.")

    args = parser.parse_args()
    global flux_pipe
    flux_pipe = setup(args)

    base_path = args.output_path.split("/")[-1].split(".jsonl")[0]
    # TODO: load jsonl till batch_size
    with open(args.output_path, "w") as outfile: 
      with open(args.input_path, "r") as infile:
        rng = 0
        while True:
          # Read a batch of lines from the input file
          lines = list(itertools.islice(infile, args.batch_size))
          if not lines:
              break  # Exit the loop if no lines are left

          # Apply the algo over the batched data
          all_data = [json.loads(l) for l in lines]
          captions = [data['caption'] for data in all_data]
          images = generate_images(captions,)

          # save
          for (idx, image) in enumerate(images):
            data = all_data[idx]
            if "metadata" not in data:
              data["metadata"] = {}
            image.save(f"{args.output_dir}/{base_path}-{rng+idx}.png")
            data["images"] = [f"{args.output_dir}/{base_path}-{rng+idx}.png"]
            data["metadata"]["source"] = args.output_dir
            data["metadata"]["create_img-params"] = json.dumps(vars(args))
            outfile.write(json.dumps(data)+"\n")
          rng += args.batch_size


if __name__ == "__main__":
    main()
    print("Completed!!")
