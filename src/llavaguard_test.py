from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import requests

model = LlavaForConditionalGeneration.from_pretrained('AIML-TUDA/LlavaGuard-v1.1-7B-hf')
processor = AutoProcessor.from_pretrained('AIML-TUDA/LlavaGuard-v1.1-7B-hf')

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": policy},
            ],
    },
]

text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

url = "https://www.ilankelman.org/stopsigns/australia.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=text_prompt, images=image, return_tensors="pt")
model.to('cuda:0')
inputs = {k: v.to('cuda:0') for k, v in inputs.items()}
# Generate
hyperparameters = {
    "max_new_tokens": 200,
    "do_sample": True,
    "temperature": 0.2,
    "top_p": 0.95,
    "top_k": 50,
    "num_beams": 2,
    "use_cache": True,
}
output = model.generate(**inputs, **hyperparameters)
print(processor.decode(output[0], skip_special_tokens=True))
