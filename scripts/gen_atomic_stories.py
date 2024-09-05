from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import json
import random
import os
import wget

SYSTEM_PROMPT = ["""Revise this story to make it compelling and more logical and detailed. Keep as much of the feelings and actions as possible, but remove anything that doesn't make sense. Make the story at least 10 paragraphs. Start with a title. %(warning)s The story should unfold through the characters interactions, decisions, and the consequences of their actions. Aim to weave in common sense lessons and social cues. The narrative should cater to a diverse age group, including at least one dialogue and presenting both positive and negative outcomes. Do not start with classic sentences like "Once upon a time", be creative:""",
          """Revise this story to make it compelling and more logical and detailed. Keep as much of the feelings and actions as possible, but remove anything that doesn't make sense. Make the story at least 10 paragraphs. Start with a title. %(warning)s Write as a real-life story shared by someone in a social media forum. The story should include:
- Niche interests or humor: dive into specific hobbies, interests, or humorous situations
- An unexpected plot twist or engaging conflict: introduce a relatable yet challenging situation or dilemma that the author faced.
- Reflection and insight: end with a resolution that offers a new understanding, a sense of community, or a personal revelation, much like the conclusions drawn in forum discussions.
Start the story right away. Do not start with sentences like  "Once upon a time" as this is a reddit post and not a novel, you should also avoid starting with classic sentences like "A few years ago" or "A few years back", be creative:""",
          """Revise this story to make it compelling and more logical and detailed. Keep as much of the feelings and actions as possible, but remove anything that doesn't make sense. Make the story at least 10 paragraphs. Start with a title. %(warning)s Write the story in the style of real-life situations that people share in forums. The story needs to include a compelling and unexpected plot twist. Your narrative should resonate with the authenticity and personal touch found in forum discussions. Include relatable events and emotional depth. Do not start with classic sentences like "Once upon a time", "A few years back" or "A few months ago", be creative:""",
          """Revise this story to make it compelling and more logical and detailed. Keep as much of the feelings and actions as possible, but remove anything that doesn't make sense. Make the story at least 10 paragraphs. Start with a title. %(warning)s The story should incorporate the following elements:
- Dialogue: the story must feature at least one meaningful dialogue that reveals character depth, advances the plot, or unravels a crucial piece of the mystery
- Interesting themes: explore themes resonant with a mature audience, such as moral ambiguity, existential queries, personal transformation, or the consequences of past actions.
Do not start with classic sentences like "Once upon a time", "The sun hung low in the sky" or "In the dimly lit", be creative:"""]


def do_generation(prompts, pipeline, savepath):
    # Generate stories
    with open(savepath, "w") as outfile:
        with open(json_path) as infile:
            for line in infile:
                dat = json.loads(line)
                text = dat['text']

                # Choose a random prompt
                prompt = random.choice(prompts)

                # Add warning if certain themes are detected in the text
                if any(keyword in text.lower() for keyword in ['assault', 'robbery', 'arson', 'fellatio', 'hand job', 'prostitu', 'handjob', 'fucks', 'blow job', 'blowjob', 'incest', 'porn', 'rape', 'killer', 'murder', 'kidnap', 'abduct', 'sex', 'torture', 'kills']):
                    warning = "If this story contains themes of sex or violence, give a warning at the beginning of the story with an explanation."
                else:
                    warning = ""

                # Format the prompt
                prompt = prompt % {'warning': warning}

                # Generate output text based on prompt and input text
                messages = [{"role": "user", "content": f"{prompt}\n\n{text}"}]
                output_text = pipeline(messages, max_length=2048, min_length=512, use_cache=True)[0]['generated_text']

                # Write to output JSON file
                outfile.write(json.dumps({'text': output_text, 'prompt': prompt, 'input': text, 'metadata': dat['metadata']}) + "\n")


# Function to generate stories
def generate_stories(model_path, tokenizer_path, json_path, prompts, output_file):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).half().cuda()
    # Set up text generation pipeline
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

    # Download JSON file if not exists
    if not os.path.exists(json_path):
        print("file not found downloading")
        url = 'https://huggingface.co/datasets/ontocord/atomic_2024/resolve/main/data/atomic_stories.jsonl'
        wget.download(url, json_path)

    do_generation(prompts=prompts, pipeline=pipe, savepath=output_file)




# Example usage
if __name__ == "__main__":
    # Specify paths and parameters
    model_path = "UCLA-AGI/Gemma-2-9B-It-SPPO-Iter3"
    model_path = "facebook/opt-125m"
    tokenizer_path = "UCLA-AGI/Gemma-2-9B-It-SPPO-Iter3"
    tokenizer_path = "facebook/opt-125m"
    json_path = "atomic_stories.jsonl"
    output_file = "generated_atomic_stories.jsonl"

    # Generate stories
    generate_stories(model_path, tokenizer_path, json_path, prompts=SYSTEM_PROMPT, output_file=output_file)
