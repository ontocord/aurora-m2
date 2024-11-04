"""Encoding text for training LLM.
Copyright 2024, Ontocord, LLC. All rights reserved.
Licensed under Apache 2.0

=== This code is meant to create different encoding to teach an LLM
from raw text. In this way we can teach multi-way encoding based on
eglish/text as a pivot. The theory is that if we force an LLM to learn
different representation of text, it can infer new meaning from the
text.

This interleaved format is similar to the UL2 objective and
approrpaite for teaching an autoregressive model.

We could also do different types of encoding pairs, like
mel-spec->base64, embedding->brialle etc.

"""

#TODO: do case of mel-spec image->clip embedding->random mapping.
#TODO: normalize embeddings and clip embeddings

import base64
import io
import random
import torch
import matplotlib.pyplot as plt
import librosa
import librosa.display
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from g2p_en import G2p # this uses CPU/numpy to do some processing. we can use GPU if we want faster speed.
from collections import Counter
import re

import random
import base64
from g2p_en import G2p

# Initialize the G2p converter
g2p = G2p()
#tts_model = ...  # Assume TTS model to convert text to speech is initialized here
#consider styletts

# Define the ASCII Braille dictionary
braille_map = {
    'a': '⠁', 'b': '⠃', 'c': '⠉', 'd': '⠙', 'e': '⠑', 'f': '⠋', 'g': '⠛', 'h': '⠓',
    'i': '⠊', 'j': '⠚', 'k': '⠅', 'l': '⠇', 'm': '⠍', 'n': '⠝', 'o': '⠕', 'p': '⠏',
    'q': '⠟', 'r': '⠗', 's': '⠎', 't': '⠞', 'u': '⠥', 'v': '⠧', 'w': '⠺', 'x': '⠭',
    'y': '⠽', 'z': '⠵', ' ': '⠀', '.': '⠲', ',': '⠂', '?': '⠦', '!': '⠖'
    # Extend as needed for more characters
}
#TODO: change to stopwords
common_words = {'the', 'is', 'and', 'a', 'to', 'of', 'in', 'it', 'for', 'that', 'with', 'on', 'as'}  # Common words to exclude

model_name = "sentence-transformers/LaBSE"  # Can also use m-clip model name if preferred
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name) #TODO - cuda, half, etc.

def text_to_mel_spectrogram(text, sr=16000):
    # Use TTS model to convert text to audio
    speech_audio = tts_model(text).cpu().numpy()

    # Generate Mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(y=speech_audio, sr=sr)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Plot and encode Mel-spectrogram to base64
    fig, ax = plt.subplots(figsize=(6, 4))
    librosa.display.specshow(mel_spec_db, sr=sr, hop_length=512, x_axis="time", y_axis="mel", cmap="magma")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    return img_str

def extract_keywords(text, num_keywords=5):
    # Basic keyword extraction using word frequency
    words = re.findall(r'\b\w+\b', text.lower())  # Extract words
    filtered_words = [word.strip("~!@#$%^&*()-_=+<>,.?/:;'\"") for word in words if word.strip("~!@#$%^&*()-_=+<>,.?/:;'\"").lower() not in common_words]
    word_counts = Counter(filtered_words) # upweight words with captializations and longer words
    keywords = [word for word, count in word_counts.most_common(num_keywords)]
    return ', '.join(keywords)

def encode_segment(segment, encoding_type):
    if encoding_type == "mel_spec":
        return text_to_mel_spectrogram(segment)
    elif encoding_type == "base64":
        return base64.b64encode(segment.encode("utf-8")).decode("utf-8")
    elif encoding_type == "braille":
        return ''.join(braille_map.get(char, '') for char in segment.lower())
    elif encoding_type == "phoneme":
        phonemes = g2p(segment)
        return " ".join(phonemes)
    elif encoding_type == 'embedding':
        inputs = tokenizer(segment, return_tensors="pt", truncation=True)
        with torch.no_grad():
            embeddings = model(**inputs).last_hidden_state.mean(dim=1)
        embeddings = embeddings.cpu().numpy()
        return base64.b64encode(embeddings)
    else:
        raise ValueError("Unsupported encoding type. Use 'mel_spec', 'embedding', 'base64', 'braille', or 'phoneme'.")

def add_boundaries(text):
    text = text.replace(". ", ". <SENT_BOUNDARY> ").replace("? ", "? <SENT_BOUNDARY> ").replace("! ", "! <SENT_BOUNDARY> ").replace("<p> ", "<p> <PARA_BOUNDARY> ")
    return text


def random_chunk(text_with_boundaries):
    segments = text_with_boundaries.split(" ")
    
    chunks = []
    i = 0
    while i < len(segments):
        chunk_type = random.choice(["word", "word", "word", "word", "word", "sentence", "sentence", "paragraph"])
        
        if chunk_type == "paragraph" and "<PARA_BOUNDARY>" in segments[i:]:
            end_index = segments.index("<PARA_BOUNDARY>", i) + 1
            raw_chunk = " ".join(segments[i:end_index]).replace(" <PARA_BOUNDARY> ", "").replace("<PARA_BOUNDARY> ", "").replace(" <PARA_BOUNDARY>", "")
            chunks.append(raw_chunk.replace(" <SENTENCE_BOUNDARY> ", "").replace(" <SENTENCE_BOUNDARY>", "").replace("<SENTENCE_BOUNDARY> ", "").strip())
            i = end_index
        elif chunk_type == "sentence" and "<SENT_BOUNDARY>" in segments[i:]:
            end_index = segments.index("<SENT_BOUNDARY>", i) + 1
            raw_chunk = " ".join(segments[i:end_index]).replace(" <SENT_BOUNDARY> ", "").replace("<SENT_BOUNDARY>", "").replace(" <SENT_BOUNDARY>", "")
            chunks.append(raw_chunk.replace(" <PARA_BOUNDARY> ", "").replace(" <PARA_BOUNDARY>", "").replace("<PARA_BOUNDARY>", "").strip())
            i = end_index
        else:
            word_count = random.randint(1, 5)
            raw_chunk = " ".join(segments[i:min(len(segments), i + word_count)])
            chunks.append(raw_chunk.strip())
            i += word_count
    return chunks

def create_interleaved_text(text, encoding_type="base64"):
    """
    Creates interleaved text with the specified encoding type for each chunk.
    
    encoding_type: str, one of ['base64', 'braille', 'phoneme', 'TTS-mel-spec', 'embedding']
    """
    text_with_boundaries = add_boundaries(text)
    chunks = random_chunk(text_with_boundaries)
    
    interleaved_output = []
    for i, chunk in enumerate(chunks):
        encoded_chunk = encode_segment(chunk, encoding_type)
        keywords = extract_keywords(chunk)
        if encoding_type == "mel_spec":
            interleaved_output.append(f'<image src="data:image/png;base64,{encoded_chunk}" alt="TTS-mel-spec, keywords: {keywords}" />')
        else:
            interleaved_output.append(f'<span data-encoding="{encoding_type}" alt="keywords: {keywords}">{encoded_chunk}</span>')
        interleaved_output.append(chunk)

    final_output = "".join(interleaved_output).strip()
    return final_output

# Example usage
text = "Hello, world! This is a test to convert text segments into mel-spectrogram images."
encoding_type = "mel_spec"  # Use "base64", "braille", or "phoneme" for other encodings
output = create_interleaved_text(text, encoding_type)

sample_text = """This is a sample text for testing. It contains multiple sentences and paragraphs.
Here is a new paragraph to add variation.
<p>
This text is meant to be encoded in different ways for Base64 testing."""

print (create_interleaved_text(sample_text))



    
