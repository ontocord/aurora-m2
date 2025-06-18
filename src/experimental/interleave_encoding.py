import random
import base64
from g2p_en import G2p

# Initialize the G2p converter
g2p = G2p()

# Define the ASCII Braille dictionary
braille_map = {
    'a': '⠁', 'b': '⠃', 'c': '⠉', 'd': '⠙', 'e': '⠑', 'f': '⠋', 'g': '⠛', 'h': '⠓',
    'i': '⠊', 'j': '⠚', 'k': '⠅', 'l': '⠇', 'm': '⠍', 'n': '⠝', 'o': '⠕', 'p': '⠏',
    'q': '⠟', 'r': '⠗', 's': '⠎', 't': '⠞', 'u': '⠥', 'v': '⠧', 'w': '⠺', 'x': '⠭',
    'y': '⠽', 'z': '⠵', ' ': '⠀', '.': '⠲', ',': '⠂', '?': '⠦', '!': '⠖'
    # Extend as needed for more characters
}

def encode_segment(segment, encoding_type):
    """
    Encodes the segment based on the specified encoding type.
    
    encoding_type: str, one of ['base64', 'braille', 'phoneme']
    """
    if encoding_type == 'base64':
        return base64.b64encode(segment.encode("utf-8")).decode("utf-8")
    elif encoding_type == 'braille':
        return ''.join(braille_map.get(char, '') for char in segment.lower())
    elif encoding_type == 'phoneme':
        phonemes = g2p(segment)
        return " ".join(phonemes)
    else:
        raise ValueError("Unsupported encoding type. Choose from 'base64', 'braille', or 'phoneme'.")

def add_boundaries(text):
    text = text.replace(". ", ". <SENT_BOUNDARY> "). \
        replace("? ", "? <SENT_BOUNDARY> ").\
        replace("! ", "! <SENT_BOUNDARY> ").\
        replace("<p> ", "<p> <PARA_BOUNDARY> ")
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
    
    encoding_type: str, one of ['base64', 'braille', 'phoneme']
    """
    text_with_boundaries = add_boundaries(text)
    chunks = random_chunk(text_with_boundaries)
    
    interleaved_input = []
    interleaved_output = []
    
    for i, chunk in enumerate(chunks):
        encoded_chunk = encode_segment(chunk, encoding_type)
        if i % 2 == 0:
            interleaved_input.append(chunk)
            interleaved_output.append(f" <span data-encoding=\"{encoding_type}\">"+encoded_chunk+"</span> ")
        else:
            interleaved_input.append(f" <span data-encoding=\"{encoding_type}\">"+encoded_chunk+"</span> ")
            interleaved_output.append(chunk)
    
    final_input = "".join(interleaved_input).strip()
    final_output = "".join(interleaved_output).strip()
    
    return final_input, final_output
# Sample text
sample_text = """This is a sample text for testing. It contains multiple sentences and paragraphs.
Here is a new paragraph to add variation.
<p>
This text is meant to be encoded in different ways for Base64 testing."""

print (create_interleaved_text(sample_text))



    
