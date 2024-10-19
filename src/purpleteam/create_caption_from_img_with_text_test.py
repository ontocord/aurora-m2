import json
import time
import argparse
import spacy
import glob
import itertools
import random
import numpy as np

import torch
from torch.nn.functional import cosine_similarity
from PIL import Image
from diffusers import FluxPipeline
from transformers import pipeline
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel, AutoModel, AutoTokenizer, AutoModelWithLMHead
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from src.accelerator import accelerator
from src.purpleteam.utils import *

from src.frcnn.visualizing_image import SingleImageViz
from src.frcnn.processing_image import Preprocess
from src.frcnn.modeling_frcnn import GeneralizedRCNN
from src.frcnn.utils import Config
from src.frcnn.utils import decode_image

import pyarrow
from pyarrow import parquet
from io import BytesIO
from PIL import Image
from collections import Counter

# Load necessary data
digits_to_words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
                  'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen',
                  'nineteen', 'twenty']

spacy_nlp = spacy.load('en_core_web_sm')
max_detections = 36

import cv2
import numpy as np
from matplotlib import colors
import random
from collections import OrderedDict


hsv_color_ranges = {
    "red": [(0, 50, 50), (10, 255, 255)],  # Expanded to include more shades of red
    "red_alt": [(170, 50, 50), (180, 255, 255)],  # Wraparound red for hue near 0/180
    "orange": [(10, 50, 50), (25, 255, 255)],  # Expanded for different shades of orange
    "yellow": [(25, 50, 50), (35, 255, 255)],  # Expanded yellow range
    "lime green": [(35, 50, 50), (70, 255, 255)],  # Expanded bright green (lime) range
    "green": [(35, 50, 50), (85, 255, 255)],  # Combined lime and green ranges
    "cyan": [(80, 50, 50), (95, 255, 255)],  # Broadened cyan range
    "blue": [(80, 50, 50), (140, 255, 255)],  # Combined cyan, blue, and indigo ranges
    "indigo": [(115, 50, 50), (140, 255, 255)],  # Broadened range for indigo (between blue and violet)
    "purple": [(130, 50, 50), (160, 255, 255)],  # Expanded to include various shades of purple
    "pink": [(140, 50, 50), (170, 255, 255)],  # Combined pink and magenta ranges
    "magenta": [(140, 50, 50), (170, 255, 255)],  # Similar to pink, but more intense
    "brown": [(10, 50, 20), (20, 255, 200)],  # Broadened range for brown
    "black": [(0, 0, 0), (180, 255, 50)],  # Broadened black range
    "white": [(0, 0, 230), (180, 30, 255)],  # Adjusted to capture all shades of white
    "gray": [(0, 0, 40), (180, 20, 255)]  # Combined dark gray, gray, and light gray ranges
}

numbering_list = ['3', '7)', '7.', '4', 'iii.', 'iii-', '8.', '4-', 'v:', 'I:', 'ii.', 'i.', 'V)', 'E)', 'I)', 'III.', 'III)', '2-', '1)', 'v-', 'III', 'I.', 'c)', '1.', 'V-', 'iv)', 'A)', 'v)', 'IV', 'C.', 'ii)', 'I', 'IV.', 'C)', 'II-', '2.', 'III-', 'IV)', 'd)', 'iii', 'i-', 'iii:', 'A.', 'B.', '1', '6)', 'ii', '8)', '3)', 'e)', 'ii-', '5-', 'II)', 'iv-', '2)', 'e.', 'IV:', 'III:', 'i)', '10.', 'V', 'V.', 'v.', 'D)', 'E.', 'iv:', 'B)', 'II', 'ii:', 'V:', 'a.', '5.', 'IV-', '9.', 'D.', '3.', '4:', '2:', 'i', 'II.', '3-', '2', 'c.', 'a)', '3:', '10)', 'd.', 'i:', 'iv.', '1-', '4.', '5', 'iv', 'iii)', 'b.', '1:', 'II:', 'v', '5:', '6.', 'b)', 'I-', '9)', '4)', '5)']
stopwords_list = ['es', 'ing', 'ed', 'include', 'includes', 'also', 'haven', 'are', 'why', 'most', "won't", 'against', 'with', 'needn', 'couldn', 'now', 'mustn', 'who', 'under', 'doing', 'am', 'aren', 'they', "didn't", 'd', 'doesn', 'if', 'he', 'her', "haven't", 'isn', 'own', 'does', 'such', 'until', 'into', 'had', 'again', 'over', "hadn't", "you'll", 't', 'by', 'be', "wasn't", 'so', 'yours', 'both', 'any', 'did', "you've", 'these', 'myself', 'o', 'hasn', "isn't", 'you', 'other', 'shan', 'being', 'yourselves', 'was', 'no', 'm', 'those', 'will', 'its', 'itself', 'have', 'down', 'weren', 'having', 'wouldn', 'herself', "mustn't", 'very', 'do', "should've", 'him', "you'd", 'below', 'just', 'that', 'for', 'which', 'but', 'nor', 'all', 'then', 'i', 'whom', 'it', 'once', 'here', 've', "you're", 'ours', "that'll", 'a', 'won', 'himself', 'where', 'this', 'your', "hasn't", 'same', 'when', 'ourselves', 'because', "needn't", 'theirs', 'from', 'mightn', 'my', 'while', 'yourself', "she's", 'each', "doesn't", 'only', 'at', 's', 'their', "wouldn't", 'shouldn', 'and', 'themselves', 'hers', 'has', 'up', 'ma', 'in', 'll', 'we', 're', 'y', 'of', 'after', 'our', "shan't", 'before', 'wasn', 'can', 'should', 'been', 'through', 'as', 'further', 'during', 'between', 'there', 'me', 'on', 'don', "shouldn't", 'more', 'out', "don't", 'the', "weren't", "aren't", "it's", 'what', 'or', "couldn't", 'hadn', "mightn't", 'his', 'above', 'to', 'how', 'few', 'off', 'them', 'didn', 'ain', 'not', 'she', 'an', 'than', 'too', 'is', 'some', 'were', 'about']

common_title_words_set = {'introduction', 'conclusion', 'section', 'chapter', 'works', 'notes', 'note', 'further', 'see', 'references', 'reference', 'section', 'title', 'conclusion', 'intro', 'introduction', 'executive', 'summary', 'key', 'plot', 'theme'}
stopwords_set = set(stopwords_list + numbering_list)


def find_quotes(text):
  accum = []
  text = text.replace("'s ", " @s@ ").replace("'ve ", " @ve@ ").replace("'m ", " @m@ ").replace("'t ", " @t@ ")
  for idx, segment in enumerate(text.split("'")):
    if idx % 2 != 0:
      accum.append(segment)
  accum = [a.replace(" @s@ ", "'s ").replace(" @ve@ ", "'ve ").replace( " @m@ ", "'m ").replace(" @t@ ", "'t ").replace("  ", " ").replace("  ", " ").strip() for a in accum]
  accum.sort(key=lambda a: len(a), reverse=True)
  return accum

def strip_left_stopwords(e_text):
  e_text2 = []
  add_rest = False
  for et in e_text.split():
      if add_rest or ((et.lower() not in stopwords_set and et.lower not in common_title_words_set) or et.lower().strip(".") in {"a", "an", "united", "the", "new", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",  "asian", "american", "african", "european", }):
        add_rest = True
        e_text2.append(et)
  return " ".join(e_text2)


def strip_right_stopwords(e_text):
  e_text2 = []
  add_rest = False
  e_text_arr = e_text.split()
  e_text_arr.reverse()
  for et in e_text_arr:
      if add_rest or (et.lower() not in stopwords_set or et.lower().strip(".") in {"act", "code", "statute", "regulation", "regulations", "percent", "feet", "foot", "square", "barrells", "hour", "hours", "people", "asian", "american", "african", "european", "act", "law", "facilities", "facility", "center", "square", "rd", "street", "way", "blvd", "ave", "avenue", "states", "kingdom", "court", "corp", "corporation", "co", "company", "ltd", "llc", "llp", "incorp.", "incorporated"}):
        add_rest = True
        e_text2.append(et)
  return " ".join(reversed(e_text2))

default_sides = ["top", "top", "top", "top", "top",
                 "bottom", "bottom", "bottom", "bottom",
                 "left", "left",
                 "right", "right",
                 "upper left", "lower left",
                 "upper right", "lower right", "center"]

base_colors = [ 'orange', 'cyan',
 'yellow',
 'lime green',
 'green',
 'blue',
 'indigo',
 'purple',
 'pink',
 'magenta',
 'brown',
 'black',
 'white',
 'gray']

discuss_phrases = [
    "document containing", "translate", "named", "states", "reads", "translating",
    "naming", "stating", "reading", "explanation", "labeled", "label", "calls for",
    "advertise", "advertising", "title", "titled", "information", "info", "explaining",
    "mentioned", "explained", "described", "mention", "explain", "describe",
    "emphasiz", "emphasize", "emphasized", "details the", "detailing the", "noting",
    "discuss", "discussed", "discussing", "quotes", "quotation", "speaks about",
    "talks about", "communicates", "message", "description", "paragraph", "sentence",
    "reference to", "referred to", "defining", "clarifies", "clarified", "informs",
    "presents", "presents details of", "recounts", "narrates", "elaborates on",
    "details", "highlighting that", "highlights in text", "shows in writing", "depicts in writing",
    "in words", "textual explanation", "verbal description", "summary of",
    "report on", "documented", "corresponds to", "mentions", "statement",
    "articulated", "provides details", "addresses", "suggests", "indicates",
    "written account", "lecture", "written depiction", "tells about", "annotated",
    "remarks", "notes", "defines", "specifies", "proposes", "conveys", "outlines",
    "clarifying", "summarizing", "documenting", "footnote", "annotation",
    "analyzes", "breaks down", "examines", "the passage", "the text indicates",
    "reports", "concludes", "observes", "elucidates", "delves into", "references",
    "interprets", "glossary", "analyzing", "refers to", "overview", "expounds on",
    "written explanation", "verbalizes", "further details", "outlines the key points",
    "restates", "contextualizes", "assesses", "reflects on", "summarized",
    "reviews", "offers insights", "an investigation of", "evaluates", "opinion",
    "sheds light on", "supports the idea", "expresses", "inscribed", "inscribing",
]

discuss_phrases.sort(key=lambda a: len(a), reverse=True)
delete_phrases = [
    "states", "stating", "reads", "reading", "words", "written", "text", "entitled", "titled", "title", "font", "caption",
    "subtitles", "heading", "label", "wording", "written word",
    "print", "typing", "typography", "annotations", "inscription", "motto",
    "slogan",  "written description", "subheading",
    "chapter", "line of text", "dialogue",  "font size",
    "printed", "words on", "tagline", "message written", "footnote", "header",
    "watermark", "quotation marks", "headline", "byline", "text formatting",
    "bullet points", "italicized", "bolded", "text placement",
    "footer", "annotation", "inline text", "typeface", "typed", "phrase",
    "textual", "quote marks",  "signage", "document title", "label text"
]

delete_phrases.sort(key=lambda a: len(a), reverse=True)
def augment_for_quotes(prompt_array, color="pink", text_cutoff=20):
    # Modify the original prompt by appending adversarial suffix
    prompt_array2 = []
    found = False

    prompt_array = [text.replace("\"", "'") for text in prompt_array]
    for prompt in prompt_array:
      prompt = prompt.replace("infographic", "image").replace("slides", "image").replace("document", "image")
      for _ in range(5):
        color = random.choice(base_colors)
        if color in prompt:
          continue
        break
      ret = []
      accum = find_quotes(prompt)
      accum2 = []
      prompt2 = []
      prompt3 =[]
      for sentence in prompt.split(". "):
        add = False
        for s in accum:
          if s not in sentence: continue
          if len(s) > text_cutoff:
            sentence = sentence.replace(s, '')
            side = random.choice(default_sides)
            for side2 in ["top", "bottom", "lower left", "upper left", "lower right", "upper right", "left", "right", "center", ]:
                if side2 in sentence.lower() or (side2 == "top" and ("above" in sentence.lower() or "upper" in sentence.lower())) \
                 or (side2 == "bottom" and ("lower" in sentence.lower() or "below" in sentence.lower())) or \
                  (side2 == "center" and "middle" in sentence.lower()):
                  side = side2
                  break
            sentence = sentence.replace("the words ''", f"a large {color} solid rectangle "+random.choice(["at", "in", "on", "by", "to"])+f" the {side}")
            sentence = sentence.replace("title ''",  f"a large {color} solid rectangle "+random.choice(["at", "in", "on", "by", "to"])+f" the {side}")
            sentence = sentence.replace("titled ''",  f"a large {color} solid rectangle "+random.choice(["at", "in", "on", "by", "to"])+f" the {side}")
            sentence = sentence.replace("named ''",  f"a large {color} solid rectangle "+random.choice(["at", "in", "on", "by", "to"])+f" the {side}")
            sentence = sentence.replace("states ''",  f"a large {color} solid rectangle "+random.choice(["at", "in", "on", "by", "to"])+f" the {side}")
            sentence = sentence.replace("reads ''",  f"a large {color} solid rectangle "+random.choice(["at", "in", "on", "by", "to"])+f" the {side}")
            sentence = sentence.replace("stating ''",  f"a large {color} solid rectangle "+random.choice(["at", "in", "on", "by", "to"])+f" the {side}")
            sentence = sentence.replace("reading ''",  f"a large {color} solid rectangle "+random.choice(["at", "in", "on", "by", "to"])+f" the {side}")
            sentence = sentence.replace("which translates to ''", "")
            for word in delete_phrases:
              sentence = sentence.replace(word, " ")
            sentence = sentence.replace("  ", " ")
            if 'solid rectangle'  in sentence:
              ret.append((color, side, s))
              for _ in range(5):
                color = random.choice(base_colors)
                if color in prompt:
                  continue
                break
            elif s not in accum2:
              accum2.append(s.strip(",.")+".")
          break
        # when there is no quote, there may be phrases that denotes discussions or explanations
        if any(b for b in discuss_phrases if b in sentence):
          s = ""
          for word in discuss_phrases:
            if word not in sentence: continue
            _, info = sentence.split(word,1)
            for word2 in discuss_phrases:
              info = info.replace(word2, " ")
            info = strip_left_stopwords(info)
            if len(info) > 10:
              s = info[0].upper()+info[1:]
              s = s.strip(",.") + "."
          if "'" not in sentence:
            for word in delete_phrases:
              sentence = sentence.replace(word, " ")
          for word2 in discuss_phrases:
            sentence = sentence.split(word2, 1)[0]
          if s:
            if random.randint(0,5) != 0:
              accum2.append(s)
            else:
              side = random.choice(default_sides)
              for side2 in ["top", "bottom", "lower left", "upper left", "lower right", "upper right", "left", "right", "center", ]:
                  if side2 in sentence.lower() or (side2 == "top" and ("above" in sentence.lower() or "upper" in sentence.lower())) \
                  or (side2 == "bottom" and ("lower" in sentence.lower() or "below" in sentence.lower())) or \
                    (side2 == "center" and "middle" in sentence.lower()):
                    side = side2
                    break
              sentence = sentence +f" with a large {color} solid rectangle in the {side}"
              if len(sentence) < 20: continue
              ret.append((color, side, s))
              for _ in range(5):
                  color = random.choice(base_colors)
                  if color in prompt:
                    continue
                  break
        for word in delete_phrases:
            sentence = sentence.replace(word, " ")
        prompt2.append(sentence)
      prompt2 = ". ".join(prompt2)
      prompt2 = prompt2.replace("''", " ")
      prompt2 = prompt2.strip(".")+"."
      prompt2 = ".".join(s for s in prompt2.split(".") if s.count("rectangle") + s.count("solid")  < 4)
      prompt2 = prompt2.replace("  ", " ").replace("  ", " ")
      if accum2:
        if len(accum2) > 10:
          sides = ["left", "right", "top", "bottom", "upper left", "upper right", "lower left", "lower right", "center"]
        else:
          sides = ["top", "bottom", "left", "right", "upper left", "upper right", "lower left", "lower right", "center"]
        for side2 in sides:
          if side2 not in prompt2:
            if (side2 == "top" and ("above" in sentence.lower() or "upper" in sentence.lower())) \
                 or (side2 == "bottom" and ("lower" in sentence.lower() or "below" in sentence.lower())) or \
                  (side2 == "center" and "middle" in sentence.lower()): continue
            side = side2
            break
        ret.append((color, side, "\n".join(accum2)))
        for _ in range(5):
          color = random.choice(base_colors)
          if color in prompt:
            continue
          break
      accumHash = {}
      colorHash = {}
      for color, side, text in ret:
        colorHash[side] = color
        accumHash[side] = accumHash.get(side, '')
        for t in text.split("\n"):
            if t in accumHash[side]: continue
            accumHash[side] = accumHash[side] + "\n" + t
        accumHash[side] = accumHash[side].strip()
      ret = {}
      for side, text in list(accumHash.items()):
        color = colorHash[side]
        ret[color] = ret.get(color, [])
        ret[color].append((side, text))
        if not text.strip(): continue
        if side not in prompt2:
          prompt2 = random.choice([f"There is a large {color} solid rectangle "+random.choice(["at", "in", "on", "by", "to"])+f" the {side}.",
                            f"The image is mostly on one side, and there is a large {color} solid rectangle "+random.choice(["at", "in", "on", "by", "to"])+f" the {side}.",]) + " " + prompt2

        f"There is a large {color} solid rectangle in the {side}. " + prompt2
      prompt2 = " "+prompt2+" "
      prompt2 = prompt2.replace("that a", "a").replace(",.", ".").replace("It also.", "").replace(" the.", ".").replace("The image.", "").replace("image is an image", "image").replace("The.", "").replace("The the ", "The ").replace("The a ", "A ").replace("The an ", "An ").replace(" the an ", " an ").replace(" the a ", " a ").replace("The of ", "The ").replace(" the of ", " the ").replace(" the , ", ", ").replace(" a , ", ",").replace(" .", ".").strip()
      prompt2 = prompt2.replace("that a", "a").replace(",.", ".").replace("It also.", "").replace(" the.", ".").replace("The image.", "").replace("image is an image", "image").replace("The.", "").replace("The the ", "The ").replace("The a ", "A ").replace("The an ", "An ").replace(" the an ", " an ").replace(" the a ", " a ").replace("The of ", "The ").replace(" the of ", " the ").replace(" the , ", ", ").replace(" a , ", ",").replace(" .", ".").strip()
      prompt_no_textbox = prompt2
      for color, sides in ret.items():
        for side in sides:
          for prep in ["at", "in", "on", "by", "to"]:
            prompt_no_textbox = prompt_no_textbox.replace(f"There is a large {color} solid rectangle {prep} the {side[0]}.", "").\
              replace(f"The image is mostly on one side, and there is a large {color} solid rectangle {prep} the {side[0]}.", "").\
              replace(f"a large {color} solid rectangle {prep} the {side[0]}", "")
      prompt_no_textbox = prompt_no_textbox.replace(" with .", ".").replace(" with.", ".").replace("  ", " ").replace(" .", ".").strip()
      prompt_no_textbox = prompt_no_textbox.replace("that a", "a").replace(",.", ".").replace("It also.", "").replace(" the.", ".").replace("The image.", "").replace("image is an image", "image").replace("The.", "").replace("The the ", "The ").replace("The a ", "A ").replace("The an ", "An ").replace(" the an ", " an ").replace(" the a ", " a ").replace("The of ", "The ").replace(" the of ", " the ").replace(" the , ", ", ").replace(" a , ", ",").replace(" .", ".").strip()
      prompt_no_textbox = prompt_no_textbox.replace("that a", "a").replace(",.", ".").replace("It also.", "").replace(" the.", ".").replace("The image.", "").replace("image is an image", "image").replace("The.", "").replace("The the ", "The ").replace("The a ", "A ").replace("The an ", "An ").replace(" the an ", " an ").replace(" the a ", " a ").replace("The of ", "The ").replace(" the of ", " the ").replace(" the , ", ", ").replace(" a , ", ",").replace(" .", ".").strip()
      prompt_array2.append ((prompt_no_textbox, prompt2, list(ret.items())))
    return prompt_array2



# Function to convert a color name to an HSV tuple directly for color detection
def get_hsv_from_name(color_name):
    upper, lower =  hsv_color_ranges.get(color_name)
    return [int(upper[0]+lower[0]/2), int(upper[1]+lower[1]/2), int(upper[2]+lower[2]/2) ]

# Function to map rectangles to relative positions
def get_position(x, y, w, h, img_width, img_height):
    cx, cy = x + w // 2, y + h // 2  # Center of the rectangle

    # Determine if the rectangle is primarily aligned on the left, right, top, or bottom
    if cx < img_width // 3:
        if cy < img_height // 3:
            return 'upper left'
        elif cy > 2 * img_height // 3:
            return 'lower left'
        else:
            return 'left'
    elif cx > 2 * img_width // 3:
        if cy < img_height // 3:
            return 'upper right'
        elif cy > 2 * img_height // 3:
            return 'lower right'
        else:
            return 'right'
    elif cy < img_height // 3:
        return 'top'
    elif cy > 2 * img_height // 3:
        return 'bottom'
    else:
        return "center"

def create_default_rectangles(img_width, img_height):
    # Create the OrderedDict with the specified order of rectangles
    default_rectangles = OrderedDict([
        ("top", (int(0.01 * img_width), int(0.01 * img_height), int(0.98 * img_width), int(0.2 * img_height))),
        ("bottom", (int(0.01 * img_width), int(0.75 * img_height), int(0.98 * img_width), int(0.2 * img_height))),
        ("left", (int(0.01 * img_width), int(0.01 * img_height), int(0.48 * img_width), int(0.98 * img_height))),
        ("right", (int(0.51 * img_width), int(0.01 * img_height), int(0.48 * img_width), int(0.98 * img_height))),
        ("upper left", (int(0.01 * img_width), int(0.01 * img_height), int(0.48 * img_width), int(0.2 * img_height))),
        ("upper right", (int(0.51 * img_width), int(0.01 * img_height), int(0.48 * img_width), int(0.2 * img_height))),
        ("lower left", (int(0.01 * img_width), int(0.75 * img_height), int(0.48 * img_width), int(0.2 * img_height))),
        ("lower right", (int(0.51 * img_width), int(0.75 * img_height), int(0.48 * img_width), int(0.2 * img_height))),
        ("center", (int(0.25 * img_width), int(0.25 * img_height), int(0.5 * img_width), int(0.5 * img_height))),

    ])
    return default_rectangles

# Main function to detect rectangles and assign text
def replace_color_rectangles_with_text(image, text_list, detection_color="pink", clear_background=True):
    # Convert to PIL to cv2, BGR
    original_image = np.array(image)
    original_image = original_image[:, :, ::-1]
    image = original_image.copy()

    # Get the HSV range for the specified detection color
    lower_bound, upper_bound = hsv_color_ranges.get(detection_color, hsv_color_ranges["white"])
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_mean = np.mean(hsv[:, :, 0])  # Average Hue
    s_mean = np.mean(hsv[:, :, 1])  # Average Saturation
    v_mean = np.mean(hsv[:, :, 2])  # Average Value (Brightness)

    # Return the average HSV value
    replace_color =  (h_mean, s_mean, v_mean)

    # Threshold the image to get only colors in the range
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = list(contours)
    if detection_color == "pink":
      lower_bound, upper_bound = hsv_color_ranges.get("magenta", [(0, 0, 200), (180, 30, 255)])
      mask = cv2.inRange(hsv, lower_bound, upper_bound)
      contours2, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
      contours.extend(list(contours2))  
    elif detection_color == "blue":
      lower_bound, upper_bound = hsv_color_ranges.get("cyan", [(0, 0, 200), (180, 30, 255)])
      mask = cv2.inRange(hsv, lower_bound, upper_bound)
      contours2, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
      contours.extend(list(contours2))  
    elif detection_color == "red":
      lower_bound, upper_bound = hsv_color_ranges.get("red_alt", [(0, 0, 200), (180, 30, 255)])
      mask = cv2.inRange(hsv, lower_bound, upper_bound)
      contours2, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
      contours.extend(list(contours2)) 
    elif detection_color == "purple":
      lower_bound, upper_bound = hsv_color_ranges.get("indigo", [(0, 0, 200), (180, 30, 255)])
      mask = cv2.inRange(hsv, lower_bound, upper_bound)
      contours2, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
      contours.extend(list(contours2))  
    elif detection_color == "green":
      lower_bound, upper_bound = hsv_color_ranges.get("lime greeen", [(0, 0, 200), (180, 30, 255)])
      mask = cv2.inRange(hsv, lower_bound, upper_bound)
      contours2, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
      contours.extend(list(contours2))    

    # Assign text to positions
    position_map = OrderedDict()  # Maps positions like 'upper right' to contours
    remaining_text = []  # Stores text that doesn't have a specific position

    img_height, img_width, _ = image.shape

    # Iterate through contours and check for rectangles
    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.05 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check if the contour has 4 points (suggests a rectangle) and is convex
        if len(approx) == 4 and cv2.isContourConvex(approx):
            # Check if the approximated polygon is close to a rectangle
            rect = cv2.boundingRect(approx)
            x, y, w, h = rect
            if w > img_width - 10 and h > img_height - 10:
              continue
            # Filter out very small rectangles
            if w > 150 and h > 60:

                position = get_position(x, y, w, h, img_width, img_height)
                if not position: continue
                position_map[position] = rect  # Store the bounding box for this position
    if not position_map:
      clear_background = False
      position_map = create_default_rectangles(img_height, img_width)
    default_position = create_default_rectangles(img_height, img_width)
    # Process the text list
    for item in text_list:
        if isinstance(item, tuple):  # Tuple containing (position, text)
            position, text = item
            if position in position_map:
                rect = position_map.pop(position)
                del default_position[position]
                x, y, w, h = rect
                replace_color = original_image[max(0,x-10), max(0,y-10)].tolist()
                draw_text_in_rectangle_bgr(image, rect, text, replace_color,  clear_background=clear_background)
            else:
                remaining_text.append(item)
        else:
            remaining_text.append(item)

    for item in remaining_text:
        if isinstance(item, tuple):  # Tuple containing (position, text)
            position, text = item
            remaining_text.pop(0)
            if position in default_position:
                rect = default_position.pop(position)
                x, y, w, h = rect
                replace_color = original_image[max(0,x-10), max(0,y-10)].tolist()
                draw_text_in_rectangle_bgr(image, rect, text, replace_color, clear_background=False)

    for position in list(position_map.keys()):
        rect = position_map[position]
        if remaining_text:
            item = remaining_text.pop(0)
            if isinstance(item, tuple):  # Tuple containing (position, text)
              position, text = item
            else:
              text = item
            del default_position[position]
            x, y, w, h = rect
            replace_color = original_image[max(0,x-10), max(0,y-10)].tolist()
            draw_text_in_rectangle_bgr(image, rect, text, replace_color, clear_background=clear_background)
        else:
          break

    for position in list(default_position.keys()):
        rect = default_position[position]
        if remaining_text:
            item = remaining_text.pop(0)
            if isinstance(item, tuple):  # Tuple containing (position, text)
              position, text = item
            else:
              text = item
            del default_position[position]
            x, y, w, h = rect
            replace_color = image[max(0,x-10), max(0,y-10)].tolist()
            draw_text_in_rectangle_bgr(image, rect, text, replace_color,  clear_background=False)
        else:
          break
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)

color_table_bgr = [('red', (0, 0, 255)),
      ('green', (0, 255, 0)),
      ('blue', (255, 0, 0)),
      ('white', (255, 255, 255)),
      ('black', (0, 0, 0)),
      ('gray', (211, 211, 211)),
      ('yellow', (0, 255, 255)),
      ('blue', (255, 255, 0)),  # Initially was blue (now cyan)
      ('pink', (255, 0, 255)),
      ('orange', (0, 165, 255)),
      ('purple', (128, 0, 128)),
      ('pink', (203, 192, 255)),
      ('brown', (42, 42, 165)),
      ('purple', (238, 130, 238)),  # Same as original
      ('purple', (130, 0, 75))]  # Indigo purple

light_colors_bgr = {
    "white": [255, 255, 255],
    "pink": [200, 200, 255],
    "yellow": [0, 255, 255],
    "blue": [230, 216, 173],
    "brown": [140, 230, 240],
    "green": [144, 238, 144],
    "coral": [193, 182, 255]
}

light_colors_bgr_keys = list(light_colors_bgr.keys())

dark_colors_bgr = {
    "black": [0, 0, 0],
    "gray": [105, 105, 105],
    "blue": [139, 0, 0],  # Dark red as blue
    "red": [0, 0, 139],  # Dark blue as red
    "green": [0, 100, 0],
}

dark_colors_bgr_keys = list(dark_colors_bgr.keys())

def get_color_name(rgb_value, tolerance=100):
    # Define the color table
  
    def is_within_range(color1, color2, tolerance):
        """Check if two RGB colors are within the specified tolerance."""
        return all(abs(c1 - c2) <= tolerance for c1, c2 in zip(color1, color2))

    # Loop through the color table to find the closest matching RGB value within the tolerance
    for color_name, color_value in color_table_bgr:
        if is_within_range(color_value, rgb_value, tolerance):
            return color_name
    return "Unknown color"
# Function to draw text inside a rectangle, using random fonts, random justification, random line types, and larger text size
# NOTE: image is in BGR not RGB
def draw_text_in_rectangle_bgr(image, rect, text, replace_color, clear_background=True):
    x, y, w, h = rect
    # Function to determine if the background color is dark
    def is_color_dark(b, g, r):
        luminance = (0.299 * r + 0.587 * g + 0.114 * b)
        return luminance < 128

    if clear_background:
      replace_color_name = get_color_name(replace_color)
    else:
      replace_color_name = None
    # Choose the font color based on the brightness of the replace_color
    if is_color_dark(replace_color[0], replace_color[1], replace_color[2]):
        for _ in range(5):
          if random.randint(0,1):
            font_color = random.choice(light_colors_bgr_keys)  # Use a light color for dark backgrounds
          else:
            font_color = "white"
          if font_color == replace_color_name: continue
          break
        font_color = light_colors_bgr[font_color]
    else:
        for _ in range(5):
          if random.randint(0,1):
            font_color = random.choice(dark_colors_bgr_keys)  # Use a dark color for light backgrounds
          else:
            font_color = "black"
          if font_color == replace_color_name: continue
          break
        font_color = dark_colors_bgr[font_color]
    # Draw the background rectangle
    if clear_background:
        cv2.rectangle(image, (x, y), (x + w, y + h),replace_color , -1)

    # Split the text into lines based on newline characters
    lines = text.split("\n")

    # Randomly select a font from OpenCV fonts (adding more options)
    fonts = [
        cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_COMPLEX_SMALL, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
        cv2.FONT_HERSHEY_SCRIPT_COMPLEX, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_ITALIC
    ]
    font = random.choice(fonts)

    # Start with a larger font scale
    font_scale = (random.random() + 2)*2
    font_thickness = max(2, int(font_scale * 2))  # Use a larger font thickness

    # Randomly select a line type
    line_types = [cv2.LINE_AA, cv2.LINE_8, cv2.LINE_4]
    line_type = random.choice(line_types)

    # Find the longest line to fit within the width of the rectangle
    longest_line = max(lines, key=len)
    text_size = cv2.getTextSize(longest_line, font, font_scale, font_thickness)[0]

    # Adjust font_scale to fit the longest line within the rectangle's width
    max_text_width = w * 0.9  # Allow text to take up 90% of the rectangle's width
    max_text_height = h * 0.9  # Allow text to take up 90% of the rectangle's height
    step_size = int(text_size[1] + font_thickness * 1.2)  # Step size for each line

    while text_size[0] > max_text_width and font_scale > 0.5:
        font_scale -= 0.1
        font_thickness = max(1, int(font_scale * 2))
        text_size = cv2.getTextSize(longest_line, font, font_scale, font_thickness)[0]

    step_size = int(text_size[1] + font_thickness * 1.2)  # Step size for each line

    # Calculate total text height to ensure it fits within the rectangle's height
    total_text_height = len(lines) * step_size  # Height for all lines

    # Now that the font scale fits, draw each line of text
    y_offset = y + (h - total_text_height) // 2  # Center the text vertically

    # Randomly choose the text alignment (left, center, right)
    justifications = ["left", "center", "right"]
    justification = random.choice(justifications)

    for i, line in enumerate(lines):
        line_size = cv2.getTextSize(line, font, font_scale, font_thickness)[0]

        # Determine the x-coordinate based on the justification
        if justification == "left":
            text_x = x + int(0.05 * w)  # Left-aligned (5% padding)
        elif justification == "right":
            text_x = x + w - line_size[0] - int(0.05 * w)  # Right-aligned (5% padding)
        else:
            text_x = x + (w - line_size[0]) // 2  # Center-aligned

        text_y = y_offset + step_size  # Move down for each line
        y_offset = text_y + (line_size[1] + font_thickness)

        # Put each line of text on the image with random line type
        cv2.putText(image, line, (text_x, text_y), font, font_scale, font_color, font_thickness, line_type)

#     --output_dir $directory \

def setup(args):
  clip_model = CLIPModel.from_pretrained(args.cos_score_model_path, cache_dir=args.cache_dir, device_map="auto")
  clip_model = accelerator.prepare(clip_model)
  clip_processor = CLIPProcessor.from_pretrained(args.cos_score_model_path, cache_dir=args.cache_dir)

  fluo_model = AutoModelForCausalLM.from_pretrained(args.caption_generator_model_path, trust_remote_code=True, cache_dir=args.cache_dir).to(accelerator.device).eval()
  fluo_processor = AutoProcessor.from_pretrained(args.caption_generator_model_path, trust_remote_code=True, cache_dir=args.cache_dir)
  fluo_model = accelerator.prepare(fluo_model)

  purpleteam_generative_tokenizer = AutoTokenizer.from_pretrained(args.purpleteam_generative_model_path, cache_dir=args.cache_dir)
  purpleteam_generative_model = AutoModelForCausalLM.from_pretrained(args.purpleteam_generative_model_path, low_cpu_mem_usage=True, device_map="auto", cache_dir=args.cache_dir).eval()
  purpleteam_generative_tokenizer.pad_token = purpleteam_generative_tokenizer.eos_token
  purpleteam_generative_model = accelerator.prepare(purpleteam_generative_model)

  frcnn_config = json.load(open("src/frcnn/config.jsonl"))
  frcnn_config = Config(frcnn_config)
  image_preprocessor= Preprocess(frcnn_config).half().cuda()
  box_segmentation_model= GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned",frcnn_config,  cache_dir="/leonardo_scratch/fast/EUHPC_E03_068/.cache").half().cuda()

  return image_preprocessor, box_segmentation_model, clip_processor, clip_model, fluo_model, fluo_processor, purpleteam_generative_tokenizer, purpleteam_generative_model



def remove_quotes(text):
  text = text = text.replace("\"", "'")
  text = text.replace("'s ", " @s@ ").replace("'ve ", " @ve@ ").replace("'m ", " @m@ ").replace("'t ", " @t@ ")
  ret_text = []
  text_split = text.split("'")
  len_text_split = len(text_split)
  for idx, segment in enumerate(text_split):
    if idx % 2 == 0:
      if idx == len_text_split-1:
        ret_text.append(segment + " ")
      else:
        ret_text.append(segment + " '' ")        
  text = ''.join(ret_text).strip()
  text = text.replace(" @s@ ", "'s ").replace(" @ve@ ", "'ve ").replace( " @m@ ", "'m ").replace(" @t@ ", "'t ").strip()
  return text


def cosim_eval(images, texts):
    # evaluate the generated text by comparing its similarity with flux generated image 
    inputs = clip_processor(images=images, return_tensors="pt")
    clip_vision_output = clip_model.vision_model(**inputs)
    image_features = clip_model.visual_projection(clip_vision_output["pooler_output"])

    inputs = clip_processor(texts, padding=True, truncation=True, max_length=76, return_tensors="pt").to(accelerator.device)
    text_features = clip_model.get_text_features(**inputs)
    cos_scores = cosine_similarity(image_features, text_features, dim=1)

    return cos_scores

def get_element_to_img(matched_sentence, img, box_segmentation_model,\
  image_preprocessor, clip_processor, clip_model, ignore_from_box=[], other_element_arr=[],\
  get_box_images=True, num_boxes=5, box_add_factor=0.65, box_detect_verbs=True, use_longest_subsuming_text=True,\
                         score_cutoff=0.2, ignore_digits=True, ignore_quotes=True):
  global spacy_nlp

  if ignore_digits:
    matched_sentence = " " + matched_sentence + " "
    for word in digits_to_words: 
      matched_sentence = matched_sentence.replace(" " + word + " ", " ")
  matched_sentence = remove_quotes(matched_sentence)
  width, height = img.size
  doc = spacy_nlp(matched_sentence)
  noun_chunks = [strip_left_stopwords(e.text)  for e in doc.noun_chunks if len(e.text) > 4 and e.text.lower() not in stopwords_set]
  verbs = [strip_left_stopwords(e.text) for e in doc if len(e.text) > 4 and e.tag_.startswith('VB') and e.text.lower() not in stopwords_set] + \
          [a for a in noun_chunks if a.endswith("ed") or a.endswith("ing")]
  ents = [strip_left_stopwords(e.text) for e in doc.ents if len(e.text) > 4 and e.text.lower() not in stopwords_set]
  noun_chunks = [a for a in noun_chunks if not a.endswith("ed") and not a.endswith("ing")]
  ner_and_verbs = dict([((e.lower() if len(e) < 5 else e.lower()[:5]), e)  for e in (ents + verbs + noun_chunks)])
  text4 = list(set([a.strip("()[]0123456789-:,.+? ") for a in (list(ner_and_verbs.values()) + other_element_arr) if a.strip()]))
  text4 = [a for a in text4 if a.strip()]
  if use_longest_subsuming_text: #to get ony longest subsuming text
    text5 = []
    text4.sort(key=lambda a: len(a), reverse=True)
    for atext in text4:
      if any(a for a in text5 if atext in a): continue
      text5.append(atext)
    text4 = text5
  text4 = [" "+a+" " for a in text4]
  text4 = [a.split("''")[0].strip() for a in text4 if  " corner" not in a and "foregr" not in a and "backgr" not in a and " word " not in a and "picture" not in a and "illustration" not in a and\
           " words" not in a and "photo" not in a and  "drawing" not in a and "portrait" not in a and " left " not in a and " right " not in a and \
           a.strip().lower() not in {"some", "more", "others", "other", "the type", "a type", "a color", "the color", "the middle", "the center", "the left", "the center", "the right", "the top", "the bottom", "an image", "the image", "image", "the images", "place", "location"}]
  if text4:
    with torch.no_grad():
      if get_box_images:
        clip_output = clip_image_to_multitext_score(clip_model, clip_processor, img, text4, decompose_image=True, ignore_from_box=([] if box_detect_verbs else verbs) + ignore_from_box, box_add_factor=box_add_factor, num_boxes=num_boxes, box_segmentation_model=box_segmentation_model, image_preprocessor=image_preprocessor)
      else:
        clip_output = clip_image_to_multitext_score(clip_model, clip_processor, img, text4, decompose_image=True, ignore_from_box=([] if box_detect_verbs else verbs) + ignore_from_box, box_add_factor=box_add_factor)
      box_images = clip_output['box_images']
      if clip_output is not None:
        # now get relationship between things as a sentence.
        if clip_output['box2element']:
          box2element = [(a[0], a[1], a[2], box_images[idx], a[3]) for idx, a in clip_output['box2element'].items()]
        else:
          box2element = None
        ent2score =  dict([(a, [b.item(), []]) for a, b in zip(text4, clip_output['scores']) ])
        if box2element:
          for element, score, coord, img, attr in box2element:
            if " corner" not in element and "foregr" not in element and "backgr" not in element:
              rec = ent2score.get(element, [0, []])
              rec[0] = max(rec[0], score)
              rec[1].append((score, img, attr, coord))
              ent2score[element] = rec
              
        sents = []
        if box2element:
          background_element = None
          prev_small_element = None
          for element, score, coord, img, attr in box2element:
            if  (element.endswith("ed") or element.endswith("ing")) and box_detect_verbs: continue
            if score >= score_cutoff and " corner" not in element and "foregr" not in element and "backgr" not in element:
              if attr:
                attr = attr.split(",")[0]
                if attr == "black":
                  attr = "dark colored"
                elif attr == "white":
                  attr = "light colored"
                # print (f"the {element} is also {attr}.", score)
                sents.append(f"the {element} is also {attr}.")
              if coord[0]/width <= 0.03 and coord[1]/height <= 0.03 and  coord[2]/width >= 0.20 and coord[3]/height >= 0.10 and coord[3]/height < 0.30:
                sents.append(f"the {element} is in the background.")
                background_element = element
                continue
              x_center = (coord[0] + (coord[2] - coord[0])/2.0)
              y_center  = (coord[1] + (coord[3] - coord[1])/2.0)
              if ((coord[2] - coord[0])/width <= 0.3 or (coord[3] - coord[1])/height <= 0.3) and prev_small_element:
                prev_element, prev_score, prev_coord = prev_small_element
                if (x_center   - (prev_coord[0] + (prev_coord[2] - prev_coord[0])/2.0))/width > 0.2:
                  if random.randint(0,1) == 0:
                    sents.append(f"the {prev_element} is to the left of the {element}.")
                  else:
                    sents.append(f"the {element} is to the right of the {prev_element}.")
                  prev_small_element = None
                  continue
                elif (x_center   - (prev_coord[0] + (prev_coord[2] - prev_coord[0])/2.0))/width > 0.05:
                  sents.append(f"the {prev_element} is beside the {element}.")
                  prev_small_element = None
                  continue
                elif (y_center   - (prev_coord[1] + (prev_coord[3] - prev_coord[1])/2.0))/height > 0.05:
                  if random.randint(0,1) == 0:
                    sents.append(f"the {prev_element} is above the {element}")
                  else:
                    sents.append(f"the {element} is in front of the {prev_element}")
                  prev_small_element = None
                  continue
                elif (x_center   - (prev_coord[0] + (prev_coord[2] - prev_coord[0])/2.0))/width <= 0.05 and \
                  (y_center   - (prev_coord[1] + (prev_coord[3] - prev_coord[1])/2.0))/height <= 0.05:
                  if (prev_coord[2] - prev_coord[0]) < (coord[2] - coord[0]):
                    sents.append(f"the {prev_element} is on the {element}.")
                  elif (prev_coord[3] - prev_coord[1]) < (coord[3] - coord[1]):
                    sents.append(f"the {prev_element} is on the {element}.")
                  else:
                    sents.append(f"the {element} is on the {prev_element}.")
                  prev_small_element = None
                  continue

              #print (x_center, element)
              if x_center/height < .2:
                sents.append(f"the {element} is on the left.")
              if x_center/width > .8:
                sents.append(f"the {element} is on the right.")
              if (coord[2] - coord[0])/width <= .3 or (coord[3] - coord[1])/height <= .4:
                prev_small_element = (element, score, coord)
        sents = [" "+s+" " for s in sents]
        sents = [s.replace(" the the ", " the ").replace(" the a ", " the ").replace(" the an ", " the ").strip() for s in sents]
        return ent2score, sents
    return {}, []


  
def generate_captions(images, suffix: str = "", batch_size: int = 4, score_cutoff: int = 0.2):

    # Process the image with fluorence and generate caption
    fluo_prompt = '<MORE_DETAILED_CAPTION>'
    try:
      inputs = fluo_processor(text=[fluo_prompt]*len(images), images=images, return_tensors="pt").to(accelerator.device)
      generated_ids = fluo_model.generate(
        **inputs,
        max_new_tokens=1024,
        early_stopping=True, # is this false or true?
        do_sample=False,
        num_beams=3,
      )
    except e:
      print(f"Skipping the corrupted image batch!! Error: {e}")      
      return None
    generated_texts = fluo_processor.batch_decode(generated_ids, skip_special_tokens=True)
    #create working batches
    return_text = []
    images_idxs = []

    # save away a reference of the image->various text  
    for generated_text, image_idx in zip(generated_texts, range(len(images))):
      return_text.append(generated_text)
      images_idxs.append(image_idx)
    # Remove digits as words
    _working_prompt = []
    for prompt in generated_texts:
      prompt = " "+ prompt +" "
      for word in digits_to_words: 
          prompt = prompt.replace(" " + word + " ", " ")
      _working_prompt.append(prompt)

    working_prompt = []
    elements = []
    elements_ = []
    for prompt, image in zip(_working_prompt, images):
      # for working_prompt
      aHash, rel_sents = get_element_to_img(prompt, image, box_segmentation_model,\
                                            image_preprocessor, clip_processor, clip_model, score_cutoff=score_cutoff)
      print (aHash)

      for element, val in list(aHash.items()):
          # if we don't detect an actual image but clip thinks there is the element SOMEWHERE in the picture, then we want a higher cutoff
          if element not in prompt or ((val[1] and val[0] < score_cutoff) or (not val[1] and val[0] < score_cutoff + 0.05)):
              del aHash[element]
              prompt = prompt.replace(element+"es ", " ")
              prompt = prompt.replace(element+"s ", " ")                            
              prompt = prompt.replace(element+" ", " ")
              prompt = prompt.replace(" "+ element, " ")
              prompt = prompt.replace(element, " ")
      for element, val in list(aHash.items()):
          if not val[1]: continue
          all_detected_imgs = val[1]
          count = len([a for a in all_detected_imgs if a[0] >= score_cutoff])
          if count > 1 and not element.endswith("ing"):
              if element.split()[0].lower() in {"the", "an", "a",}:
                  element = " ".join(element.split()[1:])
              if prompt.count(element) == 1:
                prompt = prompt.replace(" " + element, " " + digits_to_words[count] + " " + element)
              else:
                prompt = prompt.strip(".") + ". There are " + digits_to_words[count] + " " + element+"."
      prompt = prompt.replace(" es ", " ").replace(" ed ", " ").replace(" ly ", " ").replace(" ing ", " ").replace("  ", " ").strip()
      print ("PROMPT:", prompt)
      working_prompt.append(prompt)
      element_arr = []
      for element in aHash.keys():
        if element.endswith("ing"): continue
        if element.split()[0].lower() in {"the", "an", "a",}:
           element = " ".join(element.split()[1:])
        element_arr.append(element)
      element_arr = list(set(element_arr))     
      elements.append(", ".join(element_arr))
      elements_.append(elements[-1].strip(".") + ". " + " ".join(rel_sents))

    # upsample the caption and correct the count of elements
    up_prompt = []
    prefix = random.choice(["an image of", "a photo of", "a photograph of", "a picture of", "a screenshot of", "a screen shot of"])
    for prompt, e1, e2 , image_idx in zip(working_prompt, elements, elements_, range(len(images))):
      e1 = e1.strip().replace("  ", " ").strip()
      e2 = e2.strip().replace("  ", " ").strip()
      print ("E1" + e1)
      print ("E2" + e2)
      up_prompt.append(tokenize_with_assistant_continuation(purpleteam_generative_tokenizer, [{"role": "user", "content": f"Modify this image caption to make it grammatical and depicting a matter-of-fact scenary. Do not add new color, objects or people. Do not make up details about the image and stick strictly to the caption given. DO NOT add any comments, just give the modified caption. Caption:\n {prompt}.\n\n=====\n\nRemember to include these elements:\n{e1}"},
                                                                                              {"role": "assistant", "content": f"Modified Caption: {prefix}"}]))
      images_idxs.append(image_idx)
      if e1 != e2:
        up_prompt.append(tokenize_with_assistant_continuation(purpleteam_generative_tokenizer, [{"role": "user", "content": f"Modify this image caption to make it grammatical and depicting a matter-of-fact scenary. Do not add new color, objects or people. Do not make up details about the image and stick strictly to the caption given. DO NOT add any comments, just give the modified caption. Caption:\n {prompt}.\n\n=====\n\nRemember to include these elements:\n{e2}"}, 
                                                                                                {"role": "assistant", "content": f"Modified Caption: {prefix}"}]))
        images_idxs.append(image_idx)
    outputs = generate_with_batching(purpleteam_generative_model, purpleteam_generative_tokenizer, up_prompt, accelerator.device,  use_cache=True, repetition_penalty=1.2, no_repeat_ngram_size=4, max_new_tokens=400 ,batch_size=batch_size)
    outputs = [o.split("Modified Caption:",1)[-1] for o in outputs]
    outputs = [o.replace("Caption:", "").replace("caption:", "").replace("Modified Caption:", "").replace("Modified caption:", "").replace("modified caption:", "").strip() for o in outputs]
    return_text.extend(outputs)

    # # Get LlamaGuard safety score
    # safety_tags = lguard_pipe([[{"role": "user", "content": text}] for text in return_text])
    # safety_tags = ["unsafe" if "unsafe" in tag else "safe" for tag in safety_tags]
    
    # evaluate the generated text by comparing its similarity with flux generated image 
    ret = []
    cosine_batch = {}
    for image_idx, text in zip(images_idxs, return_text):
      cosine_batch[image_idx] = cosine_batch.get(image_idx, [])+ [text]
      
    for image_idx, texts in cosine_batch.items():
      cos_scores = cosim_eval([images[image_idx]], texts)
      ret.extend([(image_idx,text, score.item(), list(zip(texts, [ss.item() for ss in cos_scores]))) for text, score in zip(texts, cos_scores)])
    #detected_and_cleaned_texts = augment_for_quotes(generated_texts)
    #prompt_array = [obj[0] if len(obj[1]) > 10 else prompt for obj, prompt in zip(detected_and_cleaned_texts, generated_texts)]
    #prompt_drawing_array = [obj[1] if len(obj[1]) > 10 else prompt for obj, prompt in zip(detected_and_cleaned_texts, generated_texts)]
    #print (detected_and_cleaned_texts)
      
    return ret


def main():
    parser = argparse.ArgumentParser(description="Set up models with quantization and specific configurations.")
    parser.add_argument("--input_dir", type=str, default="", help="Path to the input file.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--score_cutoff", type=float, default=0.14, help="score cutoff")
    parser.add_argument("--cache_dir", type=str, default="", help="Path to cache directory.")
    parser.add_argument("--purpleteam_generative_model_path", type=str, default="teknium/OpenHermes-2.5-Mistral-7B", help="Purpleteam generative model hf path.")
    parser.add_argument("--cos_score_model_path", type=str, default="openai/clip-vit-base-patch32", help="Model used to get the image-text cosine similarity.")
    parser.add_argument("--caption_generator_model_path", type=str, default='multimodalart/Florence-2-large-no-flash-attn', help="Model used for generating caption of an image.")
    parser.add_argument("--output_path", type=str, default="", help="Path to save output for this step.")

    args = parser.parse_args()
    global clip_processor, clip_model, fluo_model, fluo_processor
    global purpleteam_generative_tokenizer, purpleteam_generative_model
    global flux_pipe, image_preprocessor, box_segmentation_model
    image_preprocessor, box_segmentation_model, clip_processor, clip_model, fluo_model, fluo_processor, purpleteam_generative_tokenizer, purpleteam_generative_model = setup(args)
    
    # TODO: load jsonl till batch_size
    with open(args.output_path, "w") as outfile: 
      for file in glob.glob(args.input_dir.rstrip("/")+"/*/*/*/*"):
        df = parquet.read_table(file)
        idx = 0
        all_data = []
        for image, caption, blip_text, title, usertags, url in zip(df['jpg'], df['caption'], df['blip2_caption'], df['title'], df['usertags'], df['downloadurl']):
          all_data.append({'image': image, 'orig_caption': caption.as_py(), 'blip2_text': blip_text.as_py(), 'title': title.as_py(), 'usertags': usertags.as_py(), 'url': url.as_py(), 'source': file})
        image_array = [Image.open(BytesIO(data['image'].as_py())) for data in all_data]

        for rng in range(0, len(image_array), args.batch_size):
          images = image_array[rng:min(len(image_array), rng+args.batch_size)]
          tmp = generate_captions(images, batch_size=args.batch_size, score_cutoff=args.score_cutoff)
          if tmp is None: continue



          # add batch_id to idx
          for idx, tmpp in enumerate(tmp):
            tmp[idx] = (rng + tmpp[0],) + tmpp[1:]
          idx_text_score_related = tmp
          for (idx, text, score, related) in idx_text_score_related:
            metadata = all_data[idx]
            data = {'caption': text, 'metadata': metadata}
            data["metadata"]["caption_media_score"] = score
            data["metadata"]["related"] = related
            if "image" in data["metadata"]:
              del data["metadata"]["image"]
            else:
              print("'image' key not in data['metadata']")
            data["metadata"]["create_caption_from_img-params"] = json.dumps(vars(args))
            
            outfile.write(json.dumps(data)+"\n")
          if rng >= 20*args.batch_size: break


if __name__ == "__main__":
    main()
    print("Completed!!")
    
