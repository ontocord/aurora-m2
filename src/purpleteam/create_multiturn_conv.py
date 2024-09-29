import argparse
import einops
import random
import itertools
import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from src.purpleteam.blueteam import blueteam_classify_conversation, llamaguard_classifier_categories, llamaguard_category2name
from src.purpleteam.utils import chatml_format_instructions, generate_with_batching, remove_quotes, generate_image_aware_instruction, tokenize_with_assistant_continuation, tokenize_with_user_continuation
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

def create_multiturn(captions, instructions, responses, revised_instructions, revised_responses,
                     model, tokenizer, num_turns=4,  max_new_tokens=2048, min_new_tokens=1028, **args):
    # Randomly replace instructions and responses with revised versions
    if random.randint(0, 1):
        instructions = revised_instructions
    if random.randint(0, 1):
        responses = revised_responses
    question_prefixes = [random.choice(["Using the context above,", "Based on the context,", "Keep focusing on the context, ", "With respect to the context,", "Reflect on the context,",]) for i in range(len(captions))]
    captions = [f"An image of {caption}" if "image" not in caption else caption for caption in captions]
    themes = [random.choice(["", "", "", "", "", "", "", "", "", "", "", "", "", "", 'AI safety', 'human ecology', 'human anatomy', 'asian history', 'african history', 'south american history', 'middle eastern history', 'comparative religion', 'comparative law', 'organizational behavior', 'atomic data', 'cognitive-behavioral therapy', 'data structures', 'political ideologies', 'world religions', 'number theory', 'biomedical engineering', 'elementary particles', 'branding strategies', 'aesthetics', 'international trade law', 'topology', 'classical antiquity', 'socioeconomic aspects of aging', 'social change', 'fluid dynamics', 'justice theory', 'genomics', 'environmental issues', 'gravitation', 'electricity and magnetism', 'data collection', 'moral disputes', 'reinforcement learning', 'programming basics', 'confidence intervals', 'inorganic chemistry', 'forensic computing', 'circuit analysis', 'geometry', 'conceptual physics', 'cold war', 'mechanical waves','project management', 'viral genetics', 'plasmas', 'business ethics', 'evolution', 'human body', 'real analysis', 'arithmetic', 'environmental chemistry', 'clinical psychology', 'immunology', 'consumer behavior', 'longevity', 'forecasting', 'data encryption', 'relativity', 'energy conservation', 'philosophical debates', 'managerial accounting', 'reproductive system', 'renaissance art', 'world systems theory', 'media relations', 'critical legal studies', 'correlation', 'statistics', 'surgery', 'algorithm', 'networks', 'ageism', 'algebra of random variables', 'internet basics', 'high school european history', 'monetary policy', 'human reproductive systems', 'decision-making processes', 'economic cycles', 'bronze age', 'nervous system', 'jurisprudence', 'organic chemistry', 'epidemiology', 'brand management', 'case law',  'marketing', 'post hoc ergo propter hoc', 'neutrinos', 'national policy', 'college medicine', 'metaphysics', 'epistemology', 'human geography', 'vaccine development', 'emergency medicine', 'jurisprudence theories', 'orthopedics', 'computer network', 'respiratory anatomy', 'family planning', 'electrical engineering', 'geopolitical analysis', 'industrialization', 'robotics', 'psychiatry', 'atomic processes', 'human rights', 'stoichiometry', 'nutritional epidemiology', 'population genetics', 'biogerontology', 'fluid mechanics', 'blood brain', 'abstract algebra', 'miscellaneous', 'high school psychology', 'newtonian', 'endocrine system', 'cell biology', 'astrochemistry', 'judicial systems', 'learning theories', 'medical imaging', 'survival analysis', 'world war i', 'cardiovascular anatomy', 'college biology', 'religious philosophy', 'moral scenarios', 'nuclear physics', 'dermatology', 'cognitive dissonance', 'mitosis', 'computer graphics', 'symbolic logic', 'international alliances', 'developmental stages', 'cardiovascular system', 'problem-solving skills', 'formal logic', 'utilitarianism', 'trade agreements', 'galois theory', 'regression analysis', 'experimental design',  'management', 'materials chemistry', 'religious conflicts', 'set theory', 'diplomatic strategies', 'environmental statistics', 'prehistoric europe', 'globalization', 'diet-related diseases', 'sales and marketing', 'cultural diversity', 'internet security', 'hypothesis testing', 'straw man', 'fascism', 'ethical conflicts', 'healthcare for elderly', 'high school physics', 'market segmentation', 'professional accounting', 'acceleration of',  'economic indicators', 'unsupervised learning', 'human genome project', 'childhood nutrition', 'sexual health education', 'stakeholder engagement', 'internal medicine', 'conflict resolution', 'ethics', 'time series analysis', 'economic theories', 'software development', 'DNA replication', 'world conflicts', 'fiscal policy', 'production theory', 'renaissance', 'cryptography', 'black hole', 'clinical trials', 'colonialism', 'infant mortality', 'european union', 'whistleblowing', 'compliance and risk management', 'neural networks', 'world economies', 'general practice', 'corporate communications', 'theology', 'developmental biology', 'foreign aid', 'algorithms', 'senior health', 'professional psychology', 'sexual development',  'enlightenment', 'trigonometry', 'sociology',  'circulatory system', 'market structures', 'geology', 'public relations', 'ecology', 'acids and bases', 'population studies', 'socialism', 'stone age', 'spectroscopy', 'market analysis', 'totalitarianism', 'human-computer interaction', 'statistical mechanics', 'bayesian inference', 'antiviral treatments', 'peace negotiations', 'labor economics', 'artificial intelligence', 'business law', 'major world religions', 'physical geography', 'newtonian mechanics', 'food safety', 'political parties', 'unemployment', 'climate zones', 'confirmation bias', 'national security', 'data interpretation', 'criminal law', 'mathematical reasoning', 'electrodynamics', 'historical demographics', 'diagnostic methods', 'power systems', 'periodic table', 'basic geometry', 'enzymology', 'modern era', 'cognitive psychology', 'democratic process', 'tax accounting', 'trade economics', 'atomic physics', 'labor ethics', 'exploration', 'prehistoric culture', 'high school computer science', 'behavioral therapy', 'obstetrics and gynecology', 'econometrics', 'social psychology', 'sexual ethics', 'operating systems', 'nutrition', 'cosmology', 'augmented reality', 'threat models', 'forensic accounting', 'civil war', 'supramolecular chemistry', 'semiconductors', 'molecular processes',  'map reading', 'deontology', 'philosophy',  'college chemistry', 'war crimes', 'social structures', 'ancient civilizations', 'statistical methods',  'databases', 'reproductive anatomy', 'industrial revolution', 'us foreign policy', 'public policies', 'social theory', 'algebra i', 'space race', 'astroparticle physics', 'skeletal system', 'analytical chemistry', 'quantum mechanics', 'high school government and politics', 'medieval period', 'data mining', 'panel data', 'algebra', 'international advocacy', 'patient care', 'prehistory', 'risk management', 'genetic disorders', 'epigenetics', 'human sexuality', 'sustainability', 'auditing', 'internet age', 'property law', 'conflict resolution strategies', 'descriptive statistics', 'treatment planning', 'multivariate analysis', 'photosynthesis', 'computational thinking', 'experimental psychology', 'natural law', 'high school chemistry', 'ethical hacking', 'galaxies',  'logical fallacies','clinical knowledge', 'correlation coefficient', 'virtue ethics', 'logic', 'contemporary history', 'control systems', 'cultural norms', 'high school biology', 'prehistoric art', 'chemical reactions', 'optics', 'econometric models', 'quick sort', 'syllogistic logic', 'module theory', 'photonics', 'virology', 'management science', 'intelligence analysis', 'urinary system', 'biochemistry', 'public health nutrition', 'elasticity', 'lymphatic system', 'conflict theory', 'radiology', 'specialized medicine', 'radiochemistry', 'software engineering', 'american revolution', 'electromagnetism', 'game theory', 'theorem', 'college physics', 'inflation', 'physical chemistry', 'hydrodynamics', 'respiratory system', 'numerical analysis', 'meiosis', 'deductive reasoning',   'microelectronics', 'resource allocation', 'election processes', 'anatomy', 'dark matter', 'health systems', 'reproductive rights', 'constitutional principles', 'virtual reality', 'sexual health', 'global facts', 'civic duties', 'population statistics', 'legal studies', 'exoplanets', 'legal positivism', 'epidemiology of viruses', 'iron age', 'abnormal psychology', 'optics', 'international sanctions', 'middle ages', 'complex analysis', 'machine learning', 'STDs prevention', 'high school mathematics', 'nuclear chemistry', 'topography', 'high school geography', 'data analysis techniques', 'gene therapy', 'progressive era', 'oncology', 'world capitals', 'strategic management', 'space exploration', 'aging process', 'high school microeconomics', 'imperialism', 'geopolitical facts',  'international law', 'digestive anatomy', 'moral judgments', 'differential equations', 'contract law', 'neuropsychology', 'chemical bonds', 'jurisprudence ethics', 'cloud computing', 'mechanics', 'equation of state', 'decimals', 'proteomics', 'quantum chemistry', 'gender identity', 'pediatrics', 'magnetohydrodynamics', 'physiology', 'stakeholder theory', 'age-related diseases', 'oceanography', 'pre-calculus', 'nebulae', 'practical ethics', 'legal theory', 'tort law', 'anesthesiology', 'professional medicine', 'mathematical proof', 'cybersecurity', 'digital marketing', 'feudalism', 'high school macroeconomics', 'constitutional law', 'comparative law', 'molecular biology', 'defense strategies', 'astronomy', 'preventive medicine', 'nuclear reactions', 'high school us history', 'particle physics', 'maritime laws', 'personality theories', 'homomorphisms', 'planets', 'ad hominem', 'international courts', 'polynomials', 'modal logic', 'welfare economics', 'medicinal chemistry', 'operations management', 'integumentary system', 'crisis management', 'fallacies', 'viral diseases', 'treaties', 'world war ii', 'chemical kinetics', 'nanotechnology', 'relativistic', 'psychoanalysis', 'fair trade', 'professional law', 'causal inference', 'government branches', 'cultural sociology', 'space age', 'probability', 'computer security', 'analysis of variance', 'pharmacology', 'elementary mathematics', 'calculus', 'world wars', 'security protocols', 'genetics', 'truth tables', 'supervised learning', 'psychological assessment', 'magnetic fields', 'humanistic psychology', 'economic policy', 'communism',  'cardiac', 'gerontology', 'taxonomy', 'philosophical logic', 'great depression', 'quantum concepts', 'cross-border laws', 'sample survey methods', 'consumer theory', 'meteorology', 'disease management', 'corporate responsibility', 'linear algebra', 'gastrointestinal', 'intestinal', 'corporate governance', 'digestive system', 'limbic system', 'democracy', 'moral reasoning', 'civil rights movement', 'religious practices', 'thermodynamics', 'microbiology', 'cardiology', 'psychological disorders', 'high school world history', 'biological classification', 'modern physics',  'financial accounting', 'dietary guidelines', 'muscular system', 'simple algebra', 'dark energy', 'medical genetics', 'statistical inference', 'normal distribution', 'security studies', 'sexual orientation', 'cognitive aging', 'human aging', 'algebraic structures', 'logical connectives', 'high school statistics',  'diplomacy', 'pathology', 'market failure', 'solid state physics', 'cartography', 'quantifiers', 'ethical dilemmas', 'network security', 'college mathematics', 'civil rights', 'malware analysis', 'nationalism', 'deep learning',  'combinatorics', 'black holes', 'group theory', 'college computer science', "human rights", "law", "politics", "history", "litearture", "humor", "math", "science", "reasoning"]) for i in range(len(captions))]
    conversation =  [[
                #{"role": "system", "content": "You will act as a helpful and polite assistant engaging in a conversation about an image with a user who asks simple questions. The conversation always refers to the image and the theme " + random.choice(["", "", "litearture", "humor", "math", "science", "reasoning"])+" and is conherent with the conversation history."},
                {"role": "user", "content":  f"You will act as a helpful, knowledgeable and truthful assistant who is an expert in {theme}, programming, math, art, science, literature, humanities and other fields, engaging in a conversation with a user about a context. You will always start your response with 'Assistant:'. And I will always start my instruction with 'User:'.  The conversation always refers to the context and the theme {theme} and is coherent with the conversation history. I will sometimes give you instructions that are unsafe, tricky, playful, difficult or not answerable to test you. If an instruction cannot be answered, please state so politely and explain why.\n"+
                          f"Context: {caption}\n"+
                          f"===\nUser: {question_prefix} {instruction}"},
                {"role": "assistant", "content": f'Assistant: {response}'},
                ] for theme, caption, question_prefix, instruction, response in zip(themes, captions, question_prefixes, instructions, responses)]
    #print (conversation)
    img_turn = random.choice(range(num_turns+1)) # to chose an idx for inserting the "Describe the image"
    instr_turn = random.choice(range(num_turns+1))  # to chose an idx for inserting the (instruction, response)
    if img_turn == instr_turn: 
        instr_turn += 1
    for turn in range(num_turns):
        if turn == img_turn:
            conversation = [conv + [{"role": "user", "content": "User: Can you please describe the given image?"},
                                    {"role": "assistant", "content": f"Assistant: Sure, this is an image of {caption}" if "image" not in caption else f"Assistant: {caption}"}]
                            for caption, conv in zip(captions, conversation)]
        elif turn == instr_turn:
            conversation = [conv + [{"role": "user", "content": f"User: {instruction}"},
                                    {"role": "assistant", "content": f"Assistant: {response}"}]
                            for instruction, response, conv in zip(instructions, responses, conversation)]
        else:
            #"Do", "Can", "What", "Why", "Where", "When", "How", "How much",
            question_prefixes = [random.choice([ "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "Reflect on each of your previous responses, and explain step-by-step in detail how you might improve your response and why.",  "Using the context above,", "Let's turn to something different.", "Keep focusing on the context, ", "Answer this thinking step-by-step.", "Reflect on your previous answer.", "Be creative.", "Using the above, create an image cation", "Act as a program that can call functions to answer. Using the context, please", "Using the context above again,", "Referring to the context above again,", "Based on the context above again,", "Given these choices derived from the context", "Multiple choice question based on context:", "Choose one", "What is better","Write me", "Write my", "Write a", "Write a textbook based on the above", "Write a story based on our conversation", "Tell me", "Write a textbook", "Write a story", "Tell me", "Provide", "Write", "Improve the instruction and answer", "Imagine", "Role-play", "Play act", "Pretend", "Solve", "Write", "Improve the instruction and answer", "Imagine", "Role-play", "Play act", "Pretend", "Solve", "Create a math problem", "Revise the instructions", "Please reflect and try answering again my first question", "Provide an answer that is more responsive, and explain step-by-step why it is better", "Using the context above again,", "Let's turn to something different.", "Answer this thinking step-by-step.", "Reflect on your previous answer.", "Be creative.", "Create a math problem", "Revise the instructions", "Please reflect and try answering again my first question", "Provide an answer that is more responsive, and explain step-by-step why it is better", "Using the context above again,", "Let's turn to something different.", "Answer this thinking step-by-step.", "Reflect on your previous answer.", "Be creative."]) for i in range(len(captions))]
            user_history_batch = [
                tokenize_with_user_continuation(tokenizer, conv +  [{"role": "user", "content": f"User: {question_prefix}",}]) for conv, question_prefix in zip(conversation, question_prefixes)
            ]
            #print (user_history_batch[0])
            outputs = [o.replace("<|im_end|>", " ").replace("<s>", " ").replace("</s>", " ") for o in generate_with_batching(model, tokenizer, user_history_batch, "cuda", max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens, **args)]
            for question_prefix, conv, h in zip(question_prefixes, conversation, outputs):
              h = h.strip().replace("\n\n", "\n").replace("\n\n", "\n").replace("User\n", "\nUser:\n").replace("Assistant\n", "\nAssitant:\n").replace("#### human", "User:").replace("#### ai", "Assistant:").replace("<|im_start|> assistant", "Assistant:").replace(tokenizer.user_ending, "Assistant:").replace(tokenizer.assistant_ending, "User:").replace("Penelope:", "Assistant:").replace("### Instruction:", "User:").replace("### Response:", "Assistant:").replace("Assistent","Assistant").replace("Asststant", "Assistant").replace("assistant:", "Assistant:").replace("user:", "User:").replace("'Assistant", "'AI").replace("'User", "'Friend").replace("\"Assistant", "\"AI").replace("\"User", "\"Friend").replace("Assistant:\nAssistant:", "Assistant:").replace("Assistant: Assistant:", "Assistant:").replace("User:\nUser:", "User:").replace("User: User:", "User:").replace(" :", ":").replace(" :", ":")
            #   print (h)
              h = (question_prefix+" "+h.replace("\n\n", "\n")).split("User:")
              for idx, h2 in enumerate(h):
                pair =  [a.strip("\n :").rstrip("\n- :") for a in h2.split("Assistant:")]
                pair = [a for a in  pair if a.strip()]
                if not pair: continue
                if len(pair) < 2:
                  pair = pair[0].split('\n')
                  if len(pair) < 2: continue
                # we don't want the assistant to answer a question with a question. there could be a case where the assistant is instructed to create a question though.
                if 'question' not in pair[0] and  'instruction' not in pair[0] and "?" in pair[1][:20]: continue
                conv.append({"role": "user", "content": "User: "+pair[0]})
                # print (conv[-1])
                pair[1] = pair[1].split("I hope this",)[0].split("Let me know")[0].split("Have a great day")[0]
                if "As an AI" in pair[1] or pair[1].startswith('As a'):
                  print ('AS AN AI')
                conv.append({"role": "assistant", "content": "Assistant: "+pair[1]})
                # print (conv[-1])
            print (len(conversation[0]))

    if img_turn >= num_turns:
        conversation = [conv + [{"role": "user", "content": "User: Can you please describe the given image?"},
                                {"role": "assistant", "content": f"Assistant: Sure, this is an image of {caption}" if "image" not in caption else f"Assistant: {caption}"}]
                        for caption, conv in zip(captions, conversation)]
    if instr_turn >= num_turns:
        conversation = [conv + [{"role": "user", "content": f"User: {instruction}"},
                                {"role": "assistant", "content": f"Assistant: {response}"}]
                        for instruction, response, conv in zip(instructions, responses, conversation)]
                        
    # clean the conversation 
    for conv in conversation:
        for question_prefix in question_prefixes:
            if question_prefix in conv[0]['content']:
                conv[0]['content'] = conv[0]['content'].split(question_prefix, 1)[-1]
                break
        for cc in conv: cc["content"] = cc["content"].replace("Assistant:", "").replace("User:", "").strip()
    return conversation

def create_multiturn2(captions, instructions, responses, revised_instructions, revised_responses,
                     model, tokenizer, num_turns=5):
    # Randomly replace instructions and responses with revised versions
    if random.randint(0, 1):
        instructions = revised_instructions
    if random.randint(0, 1):
        responses = revised_responses

    # Create initial user and assistant history batches
    user_history_batch = [
        [
            {
                "role": "system",
                "content": (
                    "You have to act as a user which asks an AI assistant some questions based on an image and the conversation history. "
                    "Your question should strictly adhere to the image and the context.\n"
                    "Your task is to ONLY ask questions but not answer them. Below is the image caption and the question:\n\n"
                    f"Image Caption: {caption}\n=====\nQuestion: {instruction}"
                )
            },
            {"role": "user", "content": f"Answer: {response}"}
        ]
        for caption, instruction, response in zip(captions, instructions, responses)
    ]

    assistant_history_batch = [
        [
            {
                "role": "system",
                "content": (
                    "You have to act as an AI assistant who answers the user's question. "
                    "Your answer should strictly adhere to the image and the context.\n"
                    "Your task is to ONLY answer the provided question."
                )
            },
            {"role": "user", "content": f"Image: {caption}\n=====\nQuestion: {instruction}"},
            {"role": "assistant", "content": f"Answer: {response}"}
        ]
        for caption, instruction, response in zip(captions, instructions, responses)
    ]

    rand_turn = random.choice(range(num_turns))

    for i in range(num_turns):
        if i == rand_turn:
            # Add a specific message at a random turn
            user_history_batch = [
                uu + [{"role": "assistant", "content": "Can you please describe the image attached?"}]
                for uu in user_history_batch
            ]
            assistant_history_batch = [
                aa + [{"role": "user", "content": uu[-1]['content']}]
                for aa, uu in zip(assistant_history_batch, user_history_batch)
            ]
            assistant_history_batch = [
                aa + [{"role": "assistant", "content": f"Sure, the image consists of {caption}"}]
                for aa, caption in zip(assistant_history_batch, captions)
            ]
            user_history_batch = [
                uu + [{"role": "user", "content": aa[-1]['content']}]
                for uu, aa in zip(user_history_batch, assistant_history_batch)
            ]
        else:
            # User adds "Question:"
            user_history_batch = [
                uu + [{"role": "assistant", "content": "Question:"}]
                for uu in user_history_batch
            ]

            # Generate prompts
            prompts = [
                tokenize_with_assistant_continuation(tokenizer, uu)
                for uu in user_history_batch
            ]

            # Generate outputs
            outputs = generate_with_batching(model, tokenizer, prompts, accelerator.device)

            # Process outputs
            outputs = [oo.split("Question:", 1)[-1] for oo in outputs]

            # Add image context to question
            for uu, oo, caption in zip(user_history_batch, outputs, captions):
                new_content = (
                    generate_image_aware_instruction(caption, oo, model, tokenizer)
                    if random.randint(0, 1) else oo
                )
                uu[-1]['content'] += " " + new_content

            # Update assistant history batch
            assistant_history_batch = [
                aa + [{"role": "user", "content": uu[-1]['content']}]
                for aa, uu in zip(assistant_history_batch, user_history_batch)
            ]

            # Assistant adds "Answer:"
            assistant_history_batch = [
                aa + [{"role": "assistant", "content": "Answer:"}]
                for aa in assistant_history_batch
            ]

            # Generate assistant prompts
            prompts = [
                tokenize_with_assistant_continuation(tokenizer, aa)
                for aa in assistant_history_batch
            ]

            # Generate assistant outputs
            outputs = generate_with_batching(model, tokenizer, prompts, accelerator.device)

            # Process outputs
            outputs = [oo.split("Answer:", 1)[-1] for oo in outputs]

            # Update assistant history batch
            for aa, oo in zip(assistant_history_batch, outputs):
                aa[-1]['content'] += " " + oo

            # Update user history batch
            user_history_batch = [
                uu + [{"role": "user", "content": aa[-1]['content']}]
                for uu, aa in zip(user_history_batch, assistant_history_batch)
            ]

    # Remove the initial system prompt from assistant history batch
    assistant_history_batch = [aa[1:] for aa in assistant_history_batch]

    # Clean up the content by removing "Question:" and "Answer:" prefixes
    for assistant_history in assistant_history_batch:
        for message in assistant_history:
            content = message['content']
            for prefix in ["Question:", "question:", "Answer:", "answer:"]:
                content = content.split(prefix, 1)[-1]
            message['content'] = content.strip()

    return assistant_history_batch

def main():
    parser = argparse.ArgumentParser(description="Set up models with quantization and specific configurations.")
    parser.add_argument("--input_path", type=str, help="Path to the input file.")
    parser.add_argument("--cache_dir", type=str, default="", help="Path to cache directory.")
    parser.add_argument("--num_turns", type=int, default=5, help="Max number of turns in conversation.")
    parser.add_argument("--purpleteam_generative_model", type=str, default="teknium/OpenHermes-2.5-Mistral-7B", help="Purpleteam generative model hf path.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--output_path", type=str, help="Path to the output file.")

    args = parser.parse_args()

    purpleteam_generative_model, purpleteam_generative_tokenizer = setup(args)
    num_turns = random.randint(args.num_turns-3, args.num_turns)
    tokenize_with_user_continuation(purpleteam_generative_tokenizer, [])
    tokenize_with_assistant_continuation(purpleteam_generative_tokenizer, [])

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
                revised_instr_array = [data['revised_instruction'] for data in all_data]
                revised_response_array = [data['revised_response'] for data in all_data]

                conversations = []
                for rng in range(0, len(text_array), args.batch_size):
                    captions = text_array[rng:min(len(text_array), rng+args.batch_size)]
                    instructions = instr_array[rng:min(len(instr_array), rng+args.batch_size)]
                    responses = response_array[rng:min(len(response_array), rng+args.batch_size)]
                    revised_instructions = revised_instr_array[rng:min(len(revised_instr_array), rng+args.batch_size)]
                    revised_responses = revised_response_array[rng:min(len(revised_response_array), rng+args.batch_size)]

                    conversations += create_multiturn(captions, instructions, responses, revised_instructions, revised_responses, 
                                                      purpleteam_generative_model, purpleteam_generative_tokenizer, num_turns=num_turns)

                for i, conversation in enumerate(conversations):
                    all_data[i]['conversation'] = conversation
                    all_data[i]["metadata"]["step4_params"] = json.dumps(vars(args))
                    outfile.write(json.dumps(all_data[i])+"\n")


if __name__ == "__main__":
    main()
    print("Completed!!")