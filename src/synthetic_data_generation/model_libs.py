#@title model libs
import copy
from typing import Optional, Tuple, Union
import os
import re
import subprocess
from typing import List
import torch
import torch.nn as nn
from torch import LongTensor
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.models.deberta_v2.modeling_deberta_v2 import (
    DebertaV2PreTrainedModel,
    DebertaV2Model,
    SequenceClassifierOutput
)
import spacy
import fasttext
import whoosh.index as whoosh_index
from whoosh.qparser import QueryParser
from whoosh.analysis import StemmingAnalyzer, Filter
from huggingface_hub import hf_hub_download
from fastembed import TextEmbedding
from json_repair import repair_json
import os, json
import fasttext
import random
from FlagEmbedding import FlagReranker
import langid
from .general_libs import spell, stopwords_set, ner_rel_template_extract
from .prompt_config import (
    evolv_doc_starter,
    instruction_starter,
    first_instruction_starter,
    reasoning_methods,
    step_1_preprocess_prompts,
    step_2_enhance_prompts,
    step_3_task_based_prompts,
    step_4_subject_matter_prompts,
    step_5_safety_prompts
)


subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
if not os.path.exists("./riverbed"):
    subprocess.run(["git", "clone", "https://huggingface.co/ontocord/riverbed"])
if not os.path.exists("fasttext_model.bin"):
    subprocess.run(["wget", "http://dl.turkunlp.org/register-labeling-model/fasttext_model.bin"])


# tested on these models
#generative_model = "microsoft/Phi-3.5-mini-instruct"
#generative_model = "microsoft/Phi-3-medium-128k-instruct"

generative_model = "microsoft/Phi-3-small-128k-instruct"
generative_model = "BAAI/Infinity-Instruct-7M-Gen-mistral-7B"

generative_model = "UCLA-AGI/Gemma-2-9B-It-SPPO-Iter3"
generative_model = "Qwen/Qwen2-7B-Instruct"
generative_model = "openbmb/MiniCPM3-4B" # "openbmb/MiniCPM-MoE-8x2B"
generative_model = "openbmb/Eurus-7b-kto"
generative_model = "teknium/OpenHermes-2.5-Mistral-7B"
generative_model = "01-ai/Yi-Coder-9B-Chat"


spacy_nlp = spacy.load('en_core_web_sm')


bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )


edu_model = fasttext.load_model(hf_hub_download("kenhktsui/llm-data-textbook-quality-fasttext-classifer-v2", "model.bin"))
oh_eli5 = fasttext.load_model(hf_hub_download("mlfoundations/fasttext-oh-eli5", "openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin"))
red_pajama_model = fasttext.load_model(hf_hub_download("ontocord/riverbed", "rj_model.bin"))
pile_class_model = fasttext.load_model(hf_hub_download("ontocord/riverbed", "expert_classify.ftz"))
domain_model = fasttext.load_model("fasttext_model.bin")



device = "cuda"
model = AutoModelForCausalLM.from_pretrained(generative_model,  quantization_config=bnb_config, low_cpu_mem_usage=True,  trust_remote_code=True, device_map={"":0}).eval()
tokenizer = AutoTokenizer.from_pretrained(generative_model, trust_remote_code=True)


tokenizer.pad_token = tokenizer.eos_token


# from FlagEmbedding import FlagReranker
#
# try:
# if reranker is None: assert False
# except:
# reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
#
# def example_reranker():
# # You can map the scores into 0-1 by set "normalize=True", which will apply sigmoid function to the score
# scores = reranker.compute_score([['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']], normalize=True)
# print(scores) # [0.00027803096387751553, 0.9948403768236574]



def striphtml(data):
    p = re.compile(r'<.*?>')
    return p.sub('', data)


special_char = "”\“,\",.,;:'\"/<>?~`!@#$%^&*[]()*-_\n "

def simplify(word):
    """ helper function to index paragraphs"""
    word = word.strip(special_char).replace("a", "").replace("e", "").replace("i", "").replace("o", "").replace("u", "").replace("y", "").\
    replace("qq", "q").replace("ww", "w").replace("tt", "t").replace("nn", "n").replace("ss", "s").replace("pp", "p").replace("rr", "r").replace("gg", "g").\
    replace("xx", "x").replace("ll", "l").\
    replace("A", "").replace("E", "").replace("I", "").replace("O", "").replace("U", "").replace("Y", "").\
    replace("191", "1**").replace("181", "1**").replace("171", "1**").replace("161", "1**").\
    replace("190", "1**").replace("180", "1**").replace("170", "1**").replace("160", "1**").\
    replace("19o", "1**").replace("18o", "1**").replace("17o", "1**").replace("16o", "1**").\
    replace("19O", "1**").replace("18O", "1**").replace("17O", "1**").replace("16O", "1**").\
    replace("1O", "1*").replace("1o", "1*").replace("10", "1*").replace("19", "1*").replace("18", "1*").replace("17", "1*").replace("16", "1*").\
    replace("200", "2**").replace("21o", "2**").replace("2oo", "2**").replace("2OO", "2**").replace("21O", "2**").\
    replace("20", "2*").replace("2O", "2*").replace("21", "2*").replace("2o", "2*").\
    replace("30", "3*").replace("3O", "3*").replace("31", "3*").replace("3o", "3*")
    # sometimes numbers get messed up when copying from source documents. let's guess the best answer.
    # these simplified patterns are only used after we do other checks like approximate sequence match, spacey, and/or embedding

    return word

# consider de-biasing
# deal with stop words.
# add option to turn unknown words into ___,***, ?? or [___] etc.

def cleanup_generated_based_on_ref_doc(text, ref_document):
  """ This spell checks a document and uses a reference document to fill in typos """
  #TODO: fix numbering and do approximate sequence match, spacy_ner, and maybe add in embedding match
  cleanedup_text2 = text.replace("  ", " <@@> ").replace("\n", " <**>\n").replace("'s ", " 's ").replace("'nt ", " 'nt ").replace("'d ", " 'd ")
  aHash = dict((a.strip(special_char).lower(),a.strip(special_char)) for a in ref_document.split() if len(a.strip(special_char))> 2)
  for word in ref_document.split():
    new_word = simplify(word)
    if not new_word.strip(): continue
    if new_word not in aHash and len(word.strip(special_char)) > 3:
      aHash[new_word] = word.strip(special_char)
    if new_word.lower() not in aHash and len(word.strip(special_char)) > 3:
      aHash[new_word.lower()] = word.strip(special_char)
  new_cleaned_text = [a for a in cleanedup_text2.split() if len(a) < 20]
  spell_checked = [spell(a.strip(special_char).strip("1234567890")) for a in new_cleaned_text]
  simplified = [simplify(a) for a in new_cleaned_text]
  no_special_char = [a.strip(special_char) for a in new_cleaned_text]
  new_cleaned_text = " ".join([a                                                 if no_special_char_a.lower() in stopwords_set or  no_special_char_a.lower() in aHash else
                              (a.replace(no_special_char_a, aHash[simplified_a]) if spell_checked_a.lower() != spell_checked_a.lower() and  len(no_special_char_a) > 3 and simplified_a in aHash  else
                              (a.replace(no_special_char_a.strip("1234567890"), spell_checked_a)     if len(no_special_char_a) > 4 and spell_checked_a.lower() in aHash else
                              a))
                          for simplified_a, no_special_char_a, spell_checked_a, a in zip(simplified, no_special_char, spell_checked, new_cleaned_text)])
  new_cleaned_text = new_cleaned_text.replace("  ", " ").replace(" <@@> ", "  ").replace(" <@@> ", "  ").replace("<@@> ", "  ").replace(" <@@>", "  ").replace("<**> ", "\n").replace(" ' s ", "'s ").replace(" 's ", "'s ").replace(" 'nt ", "'nt ").replace(" 'd ", "'d ")
  new_cleaned_text = new_cleaned_text.replace(" .", ". ").replace("..", ".").replace("..", ".").\
    replace(". 1", ".1").replace(". 2", ".2").replace(". 3", ".3").replace(". 4", ".4").replace(". 5", ".5").\
    replace(". 6", ".6").replace(". 7", ".7").replace(". 8", ".8").replace(". 9", ".9").replace(". 10", ".10")
  return new_cleaned_text



class DebertaV2PairRM(DebertaV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.n_tasks = config.n_tasks
        self.drop_out = config.drop_out

        # LM
        self.pretrained_model = DebertaV2Model(config)
        self.hidden_size = config.hidden_size

        self.sep_token_id = config.sep_token_id # to add
        self.source_prefix_id = config.source_prefix_id # to add
        self.cand_prefix_id = config.cand_prefix_id
        self.cand1_prefix_id = config.cand1_prefix_id
        self.cand2_prefix_id = config.cand2_prefix_id

        self.head_layer = nn.Sequential(
            nn.Dropout(self.drop_out),
            nn.Linear(2*self.hidden_size, 1*self.hidden_size),
            nn.Tanh(),
            nn.Dropout(self.drop_out),
            nn.Linear(1 * self.hidden_size, self.n_tasks),
        )
        self.sigmoid = nn.Sigmoid()

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        #  <source_prefix_id>...<sep><cand1_prefix_id>...<sep><cand2_prefix_id> ... <sep>
        assert all([self.source_prefix_id in input_ids[i] for i in range(input_ids.shape[0])]), "<source> id not in input_ids"
        assert all([self.cand1_prefix_id in input_ids[i] for i in range(input_ids.shape[0])]), "<candidate1> id not in input_ids"
        assert all([self.cand2_prefix_id in input_ids[i] for i in range(input_ids.shape[0])]), "<candidate2> id not in input_ids"

        keep_column_mask = attention_mask.ne(0).any(dim=0)
        input_ids = input_ids[:, keep_column_mask]
        attention_mask = attention_mask[:, keep_column_mask]
        outputs = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=return_dict,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
        )
        encs = outputs.hidden_states[-1]
        source_idxs = torch.where(input_ids == self.source_prefix_id)
        source_encs = encs[source_idxs[0], source_idxs[1], :]
        cand1_idxs = torch.where(input_ids == self.cand1_prefix_id)
        cand1_encs = encs[cand1_idxs[0], cand1_idxs[1], :]
        cand2_idxs = torch.where(input_ids == self.cand2_prefix_id)
        cand2_encs = encs[cand2_idxs[0], cand2_idxs[1], :]

        # reduce
        source_cand1_encs = torch.cat([source_encs, cand1_encs], dim=-1)
        source_cand2_encs = torch.cat([source_encs, cand2_encs], dim=-1)
        left_pred_scores = self.head_layer(source_cand1_encs)
        right_pred_scores = self.head_layer(source_cand2_encs)

        loss = None
        if labels is not None:
            loss = self.compute_loss(left_pred_scores, right_pred_scores, labels)


        preds = (left_pred_scores - right_pred_scores).mean(dim=-1)
        return SequenceClassifierOutput(
            loss=loss, logits=preds,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions
        )

    def compute_loss(self, left_pred_scores, right_pred_scores, labels):
        """
        Args:
            left_pred_scores: [n_candidates, n_task]
            right_pred_scores: [n_candidates, n_task]
            labels: [n_candidates, n_task], 1/0/-1 for left/right/both is better
        """

        device = left_pred_scores.device
        loss = torch.tensor(0.0).to(left_pred_scores.device)

        dif_scores = labels
        left_pred_scores = left_pred_scores * dif_scores.sign()
        right_pred_scores = - right_pred_scores * dif_scores.sign()
        cls_loss = torch.tensor(0.0, device=device)
        cls_loss += - torch.log(torch.sigmoid(left_pred_scores+right_pred_scores)).mean()
        loss += cls_loss
        return loss

def compute_better_candidate(sources:List[str], candidate1s:List[str], candidate2s:List[str], source_max_length=1224, candidate_max_length=412):
    source_prefix = "<|source|>"
    cand1_prefix = "<|candidate1|>"
    cand2_prefix = "<|candidate2|>"
    ids = []
    assert len(sources) == len(candidate1s) == len(candidate2s)
    max_length = source_max_length + 2 * candidate_max_length
    for i in range(len(sources)):
        source_ids = pairrm_tokenizer.encode(source_prefix + sources[i], max_length=source_max_length, truncation=True)
        candidate_max_length = (max_length - len(source_ids)) // 2
        candidate1_ids = pairrm_tokenizer.encode(cand1_prefix + candidate1s[i], max_length=candidate_max_length, truncation=True)
        candidate2_ids = pairrm_tokenizer.encode(cand2_prefix + candidate2s[i], max_length=candidate_max_length, truncation=True)
        ids.append(source_ids + candidate1_ids + candidate2_ids)
    encodings = pairrm_tokenizer.pad({"input_ids": ids}, return_tensors="pt", padding="max_length", max_length=max_length)

    encodings = {k:v.to(pairrm.device) for k,v in encodings.items()}
    outputs = pairrm(**encodings)
    logits = outputs.logits.tolist()
    comparison_results = outputs.logits > 0
    return comparison_results.cpu().tolist()

if False:
  try:
    if pairrm is None: assert False
  except:
    pairrm = DebertaV2PairRM.from_pretrained("llm-blender/PairRM-hf", device_map="cuda:0", torch_dtype=torch.bfloat16).eval()
    pairrm_tokenizer = AutoTokenizer.from_pretrained('llm-blender/PairRM-hf')

  import torch
  from transformers import AutoTokenizer, AutoModelForSequenceClassification

  device = 'cuda'
  try:
    if reward_tokenizer is None: assert False
  except:
    # load model and tokenizer
    reward_tokenizer = AutoTokenizer.from_pretrained('Ray2333/GRM-Gemma-2B-sftreg')
    reward_model = AutoModelForSequenceClassification.from_pretrained(
                  'Ray2333/GRM-Gemma-2B-sftreg', torch_dtype=torch.float16,  trust_remote_code=True,
                  device_map=device,
                  ).eval()

  def example_grm_reward():
    message = [
      {'role': 'user', 'content':"What is 1 + 1?"},
      {'role': 'assistant', 'content': "2"}]

    message2 = [
      {'role': 'user', 'content':"What is 1 + 1?"},
      {'role': 'assistant', 'content': "3"}
    ]
    message_template = reward_tokenizer.apply_chat_template(message, tokenize=False)
    message_template2 = reward_tokenizer.apply_chat_template(message2, tokenize=False)

    kwargs = {"padding": 'max_length', "truncation": True, "return_tensors": "pt"}
    tokens = reward_tokenizer([message_template, message_template2], **kwargs)
    reward_model = reward_model.eval()
    with torch.no_grad():
      _, _, reward_tensor = reward_model(**tokens.to(device))
      reward = reward_tensor.cpu().detach().tolist()
      print (reward)

class MyFilter(Filter):
  def __call__(self, tokens):

    for t in tokens:
        t.text = t.text.lower()
        if len(t.text) > 5:
          yield t
          t.text = t.text[:5]
        yield t

try:
  if qp is None: assert False
except:
  bm25_dir = "./riverbed"
  index = whoosh_index.open_dir(bm25_dir)
  searcher = index.searcher()
  qp = QueryParser("content", schema=index.schema)


def get_tokens_as_list(word_list):
    "Converts a sequence of words into a list of tokens"
    tokens_list = []
    for word in word_list:
        tokenized_word = tokenizer([word], add_special_tokens=False).input_ids[0]
        tokens_list.append(tokenized_word)
    return tokens_list


bad_words_ids = get_tokens_as_list(word_list=["<issue_comment>", "<|im_end|>", "\n<|im_end|>", "</s>", '(Continue', '<pr_in_reply_to_review_id>', 'CHATGPT', '...\n', '\n...', '<file_sep>', 'openai', '\n<|endoftext|>', '<jupyter_output>', '<|endofgeneration|question|>(', \
                                              '<jupyter_code>', '<pr>', '<jupyter_text>', '<pr_event_id>', '<EMAIL>', '<|endof generation|>', '[Continue', '<pr_comment>', '<|endofgeneration|>.', '[...]', \
                                              '<pr_status>', '<pr_file>', '<pr_review_state>', '<fim_prefix>', '... ', 'GPT-4', "OpenAI's", '<empty_output>', '< |endofgeneration||>', '\n<|im_end|>', \
                                              '<fim_pad>', '<pr_diff>', '<NAME>', '<|>', 'openAI', '<|question_end|>', '[Continued', '<|endofgeneration|question|>', '(Continued', '...', ' ...', \
                                              '[END OF RESPONSE]', '  ...\n', '  ...', '<|end|>', '<|im_end|>', '<pr_in_reply_to_comment_id>', 'GPT-3', '<pr_diff_hunk_comment_line>', '<|endofgeneration|>', '<|EOT|>', "Please let me know if there is anything else",\
                                              '<fim_middle>', '<PASSWORD>', 'OpenAI', '<|endoftempate|>', '<fim_suffix>', '<|endoftext|>', '<pr_is_merged>', '   ...', "Due to the character limitations", "Due to space constraints",\
                                              "GPT-", "GPT-2", "GPT-3", "GPT-4", "GPT-5", "GPT-6", "GPT-III", "OpenAI's", "ChatGPT-3", "<|im_end|>"])




def generate(model, tokenizer, data, use_cache=True, repetition_penalty=1.2, no_repeat_ngram_size=4, max_new_tokens=2048, return_response_only=False,  **args):
  global bad_words_ids
  if type(data) != list:
    data = [data]
  with torch.no_grad():
    torch.cuda.empty_cache()
    gen_input = tokenizer(data, truncation=True, padding=True,  max_length=1000000, return_tensors="pt", add_special_tokens=False).to(device)
    output = model.generate(**gen_input,
                          use_cache=use_cache, repetition_penalty=repetition_penalty, no_repeat_ngram_size=no_repeat_ngram_size, max_new_tokens=max_new_tokens, bad_words_ids=bad_words_ids, **args)
    torch.cuda.empty_cache()
  if return_response_only:
    ret = []
    for prompt, out in zip(data, output):
      prompt_len = len(tokenizer(prompt, add_special_tokens=False).input_ids)
      ret.append(tokenizer.decode(out[prompt_len:]))
    return ret
  else:
    return [o1.strip() for o1 in tokenizer.batch_decode(output)]

def create_separators(command_response_words, separators, section_beginning):
  """
  Create many variations of the spearators and store in command_response_words array
  """
  for section in separators:
    for beg in section_beginning:
      if beg == '\n' and len(section) >=4:
        command_response_words.append(beg+section[0].upper()+section[1:].lower()+":")
        continue
      if beg == "[":
        command_response_words.append(beg+" "+section[0].upper()+section[1:].lower()+" ]")
        command_response_words.append(beg+" "+section+" ]")
        command_response_words.append(beg+" "+section.upper()+" ]")
        continue
      if beg == "#":
        if len(section) >=4:
          command_response_words.append(beg+" "+section[0].upper()+section[1:].lower())
          command_response_words.append(beg+" "+section)
          command_response_words.append(beg+" "+section.upper())
        command_response_words.append(beg+beg+" "+section[0].upper()+section[1:].lower())
        command_response_words.append(beg+beg+" "+section)
        command_response_words.append(beg+beg+" "+section.upper())
        command_response_words.append(beg+beg+beg+" "+section[0].upper()+section[1:].lower())
        command_response_words.append(beg+beg+beg+" "+section)
        command_response_words.append(beg+beg+beg+" "+section.upper())
        continue
      if beg == "<|":
        command_response_words.append(beg+section[0].upper()+section[1:].lower()+"|>")
        command_response_words.append(beg+section+"|>")
        command_response_words.append(beg+section.upper()+"|>")
        continue
      if len(beg) == 1:
        command_response_words.append(beg+" "+section[0].upper()+section[1:].lower()+" "+beg)
        command_response_words.append(beg+" "+section+" "+beg)
        command_response_words.append(beg+" "+section.upper()+" "+beg)
        command_response_words.append(beg+beg+" "+section[0].upper()+section[1:].lower()+" "+beg+beg)
        command_response_words.append(beg+beg+" "+section+" "+beg+beg)
        command_response_words.append(beg+beg+beg+" "+section.upper()+" "+beg+beg+beg)
        command_response_words.append(beg+beg+beg+" "+section[0].upper()+section[1:].lower()+" "+beg+beg+beg)
        command_response_words.append(beg+beg+beg+" "+section+" "+beg+beg+beg)
        command_response_words.append(beg+beg+beg+" "+section.upper()+" "+beg+beg+beg)
        continue
      if len(beg) == 2:
        command_response_words.append(beg+" "+section[0].upper()+section[1:].lower()+" "+beg.replace("[", "]").replace("(", ")").replace("<", ">").replace("{", "}"))
        command_response_words.append(beg+" "+section+" "+beg.replace("[", "]").replace("(", ")").replace("<", ">").replace("{", "}"))
        command_response_words.append(beg+" "+section.upper()+" "+beg.replace("[", "]").replace("(", ")").replace("<", ">").replace("{", "}"))
        continue
      end = "".join(list(reversed(beg)))
      command_response_words.append(beg+" "+section[0].upper()+section[1:].lower()+" "+end.replace("[", "]").replace("(", ")").replace("<", ">").replace("{", "}"))
      command_response_words.append(beg+" "+section+" "+end.replace("[", "]").replace("(", ")").replace("<", ">").replace("{", "}"))
      command_response_words.append(beg+" "+section.upper()+" "+end.replace("[", "]").replace("(", ")").replace("<", ">").replace("{", "}"))
      command_response_words.append(beg[0]+beg+" "+section[0].upper()+section[1:].lower()+" "+end.replace("[", "]").replace("(", ")").replace("<", ">").replace("{", "}")+beg[0])
      command_response_words.append(beg[0]+beg+" "+section+" "+end.replace("[", "]").replace("(", ")").replace("<", ">").replace("{", "}")+beg[0])
      command_response_words.append(beg[0]+beg+" "+section.upper()+" "+end.replace("[", "]").replace("(", ")").replace("<", ">").replace("{", "}")+beg[0])
      command_response_words.append(beg[0]+beg[0]+beg+" "+section[0].upper()+section[1:].lower()+" "+end.replace("[", "]").replace("(", ")").replace("<", ">").replace("{", "}")+beg[0]+beg[0])
      command_response_words.append(beg[0]+beg[0]+beg+" "+section+" "+end.replace("[", "]").replace("(", ")").replace("<", ">").replace("{", "}")+beg[0]+beg[0])
      command_response_words.append(beg[0]+beg[0]+beg+" "+section.upper()+" "+end.replace("[", "]").replace("(", ")").replace("<", ">").replace("{", "}")+beg[0]+beg[0])



### Global Variables

command_response_words = ["", "", "", "\n", "\n\n", "\n***\n", "== Data == ", "-- Data -- ", "[[ Data ]]", "[[Data]]", "Q:", "A:", "<start_of_turn>user","<start_of_turn>assistant", "<start_of_turn>system","<start_of_turn>model", "<end_of_turn>",
               '<im_end>\n<im_start>assistant','<im_end>\n<im_start>user', '<im_end>\n<im_start> assistant','<im_end>\n<im_start> user','<im_start> assistant','<im_start> user', '<im_start>assistant','<im_start>user', "<|end|>\n<|assistant|>", "<|end|> <|assistant|>", "<|end|><|assistant|>", "<|end|>\n<|user|>","<|end|><|user|>","<|end|> <|user|>",]

command_separator = ['im_start', 'start_of_turn', 'system', 'user 3', 'user 2', 'user','user 1', 'start_sequence', \
                      'instruction', 'begin', 'input', 'instructions', 'start',  'msg', 'message', \
                      'followup', 'follow-up', 'follow up', 'assignment', 'instruct', 'question', 'request', \
                      'prompt', 'directive', ]

advanced_response_separator = {'sub-goal', 'evaluation', 'goal', 'action', 'observation', 'plan', 'reflection', 'tool_call', 'tool_response', 'scratch_pad',\
                               'methodology', 'explanation', 'method', 'protocol', 'processes', 'plan', 'process', 'procedure', \
                               'analysis', 'purpose', 'goal', 'issue', 'solutions', 'answers', \
                               'solution', 'rule', 'use case', 'step-by-step solution', 'rules', 'real world example', 'guide', 'strategy',\
                               'resolution', 'approach', 'steps', 'tactic', 'result', 'search results', 'exception', \
                               'trigger', 'interpreter', 'plugin', 'thought', 'action', 'reaction', 'observation', \
                               'final response', 'wrap up', 'finalization', 'new chapter', 'new section', 'answer key', 'dialog between teacher and student', \
                               'new instruction', 'new input', 'new instructions',  'new message', \
                               'new assignment', 'new question', 'new request', 'discussion', 'advanced topics', \
                               'new prompt', 'new directive', 'real world usage', 'step-by-step reasoning', 'simplified topics', 'textbook', 'lesson book', \
                               'modified instruction', 'modified input', 'modified instructions',  'modified message', \
                               'modified assignment', 'modified question', 'modified request', \
                               'modified prompt', 'modified directive', \
                               'revised instruction', 'revised input', 'revised instructions',  'revised message', \
                               'revised assignment', 'revised question', 'revised request', \
                               'revised prompt', 'revised directive', }

basic_response_separator = {'AI', 'im_end','end_of_turn', 'end_sequence', 'assistant 1', 'answer', 'output', 'response', \
                                              'expert', 'assistant', 'agent', 'bot', 'agent 2', 'assistant 3', 'assistant 2', \
                                               'agent 1', 'end', 'agent 3',  'chatbot', }
basic_response_separator_list = list(basic_response_separator)
response_separator = list(advanced_response_separator) + list(basic_response_separator)
section_separator = command_separator + response_separator
section_beginning = ["'''", "\n", '#', "-<<",  "=<<", "- <<",  "= <<",  "-{{",  "={{", "- {{",  "= {{",  "-((",  "=((", "- ((",  "= ((", "-[[",  "=[[", "- [[",  "= [[", "**", "==", '===', "--", '---', "[[", "<<", "{{","((", "##", "<|", "["]
extra_separator_items =  ['<|end|>', 'Phi"];', '</s>', '<<SYS>>', '<context>', '=END=', '<bos>', '<｜', '｜>', '-- [[', '--[[', '==[[', '== [[', '-- [[', '==[[', '--[[', '== ((', '-- ((', '==((', '--((', '== {{', '-- {{', '=={{', '--{{', '== <<', \
                          '-- <<', '==<<', '--<<', 'solution:**', '[End Response]', '--Followup', '-- Followup', '-- End', '--End', '--response', '--Response', '--ASSIST', '--ASSIS', '--ASSI', '[[ASSIST', '[[ASSIS', '[[ASSI', 'Update:', 'prompt}}', \
                          'INSTRUCTION]', 'INSTRUCTION"]', '[--]', '[/SYS]', '<end_of_turn>', '<|im_end|>', '<start_of_turn>', '<s>', 'Word Problem #', '|user|', '|User|', '|USER|', '|assistant|', '|Assistant|', '|ASSISTANT|', 'Problem #', '--ASSI', \
                          'User 1', 'User 2', 'User 3', 'User 4', 'User 5', 'User 6', 'User 7', 'User 8', 'User 9', '== Human', '## Human', '-- Human', '==Human', '##Human', '--Human', '== human', '## human', '-- human', '==human', '##human', '--human', \
                          '((END))', 'end|>', '<|end', '== END', '==END', '{{ASSIST', 'ASSISTANT}}', '=(END', '_ctxt_', '--END', '-- END', '##RESPON', \
                          '## Instruciton', '-- Instruciton', '== Instruciton', '##Instruciton', '--Instruciton', '==Instruciton', ]
extra_separator_items.sort(key=lambda a: len(a), reverse=True)
start_basic_command = ["--->", "<---", "A:", "<start_of_turn>assistant",  '<im_end>\n<im_start> tool','<im_start> tool', '<im_end>\n<im_start>tool','<im_start>tool', '<im_end>\n<im_start>assistant','<im_end>\n<im_start> assistant', '<im_start> assistant', '<im_start>assistant', "<|end|>\n<|assistant|>", "<|end|> <|assistant|>", "<|end|><|assistant|>", ]
create_separators(start_basic_command, list(basic_response_separator), section_beginning)

start_command = copy.copy(command_response_words)
start_response = copy.copy(command_response_words)
create_separators(start_command, command_separator, section_beginning)
create_separators(start_response, response_separator, section_beginning)
command_response_words = start_command + start_response
command_response_words = list(set(command_response_words))
command_response_words.sort(key=lambda a: len(a), reverse=True)

start_command = list(set(start_command))
start_command.sort(key=lambda a: len(a), reverse=True)

start_response = list(set(start_response))
start_response.sort(key=lambda a: len(a), reverse=True)

### Prompts
step_1_preprocess_prompts_list = list(step_1_preprocess_prompts.items())
step_2_enhance_prompts_list = list(step_2_enhance_prompts.items())
step_3_task_based_prompts_list =  list(step_3_task_based_prompts.items())
step_4_subject_matter_prompts_list =  list(step_4_subject_matter_prompts.items())
step_5_safety_prompts_list =  list(step_5_safety_prompts.items())



def diversify_prompt(prompt, params, add_instruction_evolution=True, \
                   prefer_coding=False):
  """
  Given a prompt, diversify and evolve the prompt.
  """
  global command_response_words, command_separator, advanced_response_separator, response_separator, section_separator, section_beginning
  global start_command, start_response, command_response_words, start_basic_command
  global step_1_preprocess_prompts_list, step_2_enhance_prompts_list, step_3_task_based_prompts_list, step_4_subject_matter_prompts_list, step_5_safety_prompts_list

  if random.randint(0,5) == 0 and not add_instruction_evolution:
    command = random.choice(command_response_words)
    if command and command[0] in "<-[=#({":
      response = random.choice([s for s in command_response_words if s and s[0] == command[0]])
    elif command and command[-1] in ":>-]=#)}":
      response = random.choice([s for s in command_response_words if s and s[-1] == command[-1]])
    else:
      response = random.choice(command_response_words)
  else:
    command = random.choice(start_command)
    if not add_instruction_evolution:
      if command and command[0] in "<-[=#({":
        response = random.choice([s for s in start_basic_command if s and s[:2] == command[:2]])
        if not response:
          response = random.choice([s for s in start_basic_command if s and s[0] == command[0]])
        if not response:
          response = random.choice(start_basic_command)
      else:
        response = random.choice(start_basic_command)
    else:
      if command and command[0] in "<-[=#({":
        response = random.choice([s for s in start_response if s and s[:2] == command[:2]])
        if not response:
          response = random.choice([s for s in start_response if s and s[0] == command[0]])
        if not response:
          response = random.choice(start_response)
      else:
        response = random.choice(start_response)

  if random.randint(0,3) != 0:
    command = command.replace(" ", "")
    response = response.replace(" ", "")
  orig_response = response
  extra_requirements = ""

  response = "@#@"
  if random.randint(0,5) == 0:
    response = random.choice(["", "", " ", "\n", "**\n"]) + response
    response = "Answer the question completely and do not truncate for brevity or write placeholders like '...', '[Content continues]' or '[Continued]'. "+ response
  response = random.choice(["", "", " ", "\n", "**\n"]) + response
  reasoning_method = ""
  reasoning_choice = random.randint(0,15)
  if prefer_coding:
    response = "Include programming language code or psuedo-code in the response.\n" + response
  elif add_instruction_evolution:
    if random.randint(0,5) == 0:
      response = evolv_doc_starter(all_personas=[params['assistant_persona'], "helpful and respectful AI assistant"], stakeholders=[params['stakeholder']]).replace("document", random.choice(["document", "response", "answer", "solution", "reply"])) + " " + response
    if random.randint(0,1):
      reasoning_method = random.choice(reasoning_methods)
      extra_requirements = " to "+random.choice(["showcase", "test", "teach", "use"]) +" "+ reasoning_method
      if "(" in reasoning_method and reasoning_choice <= 5 and random.randint(0,1):
        choice = random.randint(0,5)
        if choice == 0:
          response = "Do not mention the words '"+reasoning_method.split("(")[0].strip()+"' in the response. " + response
        elif choice == 1:
          response = reasoning_method.split("(")[0].strip()+" should be implied by your response, but do not explicitly mention '"+reasoning_method.split("(")[0].strip()+"' in the response. " + response
        elif choice == 2:
          response = "Show " + reasoning_method.split("(")[0].strip()+" but do not mention the words '"+reasoning_method.split("(")[0].strip()+"' in the response. " + response
        elif choice == 3:
          response = "Show " + reasoning_method.split("(")[0].strip()+" but don't tell what '"+reasoning_method.split("(")[0].strip()+"' means. " + response
        else:
          response = "Do not mention the words '"+reasoning_method.split("(")[0].strip()+"'. " + response

    if reasoning_choice==0:
      response = f"\n***\nRevise the above instruction by adding details and more complexity to the instruction{extra_requirements}. Remove all contradictions. Then reflect on the instruction to confirm the new instruction turly exhibits enhanced complexity, and not merely having the original scope of the original instruction. Finally respond to the instruction. {response}"
    elif reasoning_choice==1:
      response = f"\n###\nEvolve the above instructions{extra_requirements} by adding details and more complexity to all aspects of the questions and commands. Remove everything that doesn't make sense and ignore references not in the document. Make sure the new instruction is not so hard that it can't be answered. Then respond to the instructions. {response}\nRevised"
    elif reasoning_choice==2:
      response = f"\n###\nImprove the above instructions by adding more necessary details and more complexity to all aspects of the questions and commands, but remove duplication or unencessary details. Make sure the instructions are meaningful and truthful{extra_requirements}. The new instruction should have necessary qualifications, necessitating additional inquiries for generating a meaningful response. Then respond to the instructions. {response}"
    elif reasoning_choice==3:
      response = f"\n***\nLet's add details and more complexity to the instruction{extra_requirements}. Vary any context document by changing the scenario and people. Remove all contradictions or references not in the document. Then reflect on the instruction, especially on its clarity and consistency with the original instruction. Finally respond to the instruction. {response}"
    elif reasoning_choice==4:
      response = f"\n###\nPlease first revise the above instructions{extra_requirements} by adding details and more complexity to all aspects of the questions and commands. Diversify any context document by changing the scenario and people. Remove everything that doesn't make sense amd references not in the document. Then respond to the instructions. {response}\nRevised"
    elif reasoning_choice==5:
      response = f"\n###\nI want you to revise the above instructions by adding details and more complexity to all aspects of the questions and commands to make it more complete, coherent and answerable, but remove verbosity and redundancy. Change any context document by changing the scenario and people. Make sure the instructions are meaningful and truthful{extra_requirements}. Then respond to the instructions. {response}"
    elif reasoning_choice==6:
      response = f"\n***\nYou are an articulate expert starting with the above instruction. Please create a new instruction{extra_requirements}. Create a new context that is in the same subject domain as the current context but is different in details. Remove all contradictions. The language should be clear and concise. Then reflect on the instruction, and finally respond to the instruction. {response}"
    elif reasoning_choice==7:
      response = f"\n###\nInspired by the above instruction, evolve the above instructions{extra_requirements} by adding details and more complexity to all aspects of the questions and commands. The language should be clear and concise. Create a new context that is in the same subject domain as the current context but is different in details. Remove everything that doesn't make sense and ignore references not in the document. Then respond to the instructions. {response}\nRevised"
    elif reasoning_choice==8:
      response = f"\n***\nLoosly based on the above instruction, create a new instruction{extra_requirements}. The new instruction should be a contrasting instruction to the current one. Create a new context that is in the same subject domain as the current context but is different in people, places, things and other details. Remove all contradictions. The language should be clear and concise. Then reflect on the instruction, and finally respond to the instruction. {response}"
    elif reasoning_choice==9:
      response = f"\n###\nStarting with the above instruction, evolve the above instructions{extra_requirements} so that the response is opposite of what it would otherwise be in the original instruction. The language should be clear and concise. Create a new context that is in the same subject domain as the current context but is different in people, places, things and other details. Remove everything that doesn't make sense and ignore references not in the document. Then respond to the instructions. {response}\nRevised"

  response = response.strip()
  response = response.replace("complexity", random.choice(["depth", "difficulty", "nuance", "complexity"]))
  response = response.replace("Inspired by", random.choice(["Inspired by", "Slightly based on", "Guided by", "With focus on"]))
  response = response.replace("instruction", random.choice(["instruction", "command", "question", "request"]))
  params['evolve_instruction'] = response.split("@#@")[0].strip()
  response = response.replace("@#@", orig_response)
  params['start_command'] = command
  params['start_response'] = response
  params['reasoning_requirement'] = extra_requirements

  prompt = prompt % params
  params['start_response'] =orig_response

  if random.randint(0,1):
    prompt = prompt.replace(response+"\n", response+" ")
  prompt = prompt.replace(" \n", "\n").strip()
  if random.randint(0,1):
    prompt = prompt.replace("\n\n", "\n")
  if random.randint(0,1):
    prompt = prompt.replace("\n\n", "\n")
  if random.randint(0,1):
    prompt = prompt.replace("###", "---")
    prompt = prompt.replace("##", "--")
  if random.randint(0,1):
    prompt = prompt.replace("###", "===")
    prompt = prompt.replace("##", "==")
  if random.randint(0,1):
    prompt = prompt.replace("###", "***")
    prompt = prompt.replace("##", "**")
  if random.randint(0,1):
    prompt = prompt.replace("### ", "###")
    prompt = prompt.replace("] ", "]")
    prompt = prompt.replace("> ", ">")
    prompt = prompt.replace("]-- ", "]--")
  elif random.randint(0,1):
    prompt = prompt.replace(">", "> ")
    prompt = prompt.replace("]--", "]-- ")
  prompt = prompt.replace("\nRevised\n", "\nRevised ").replace("> >", ">>")
  choice = random.randint(0,4)
  if choice == 0:
    prompt = prompt.replace("at the beginning, end or inside", "at the beginning of")
  elif choice == 1:
    prompt = prompt.replace("at the beginning, end or inside", "at the end of")
  elif choice == 2:
    prompt = prompt.replace("at the beginning, end or inside", "inside")

  if random.randint(0,1):
    prompt = prompt.replace("If applicable, add images", "Add images")

  if random.randint(0,4)==0:
    prompt = prompt.replace(":  --- Context Document:", ", Context:")
    prompt = prompt.replace(":  ### Context Document:", " ===Context===")
    prompt = prompt.replace(":  === Context Document:", ": ---Context--- ")
    prompt = prompt.replace(":  *** Context Document:", ": ***Conttext***")
  elif random.randint(0,4)==0:
    prompt = prompt.replace(":  --- Context Document:", ": ------")
    prompt = prompt.replace(":  ### Context Document:", ": ######")
    prompt = prompt.replace(":  === Context Document:", ": ======")
    prompt = prompt.replace(":  *** Context Document:", ": ******")
  if prefer_coding:
    prompt = prompt+"\n```"
  #print (prompt)
  context_document = params['context_document']
  params['context_word'] = 'Context'
  if random.randint(0,5) == 0:
    context_word = random.choice(["Context", "Search Results", "Database Lookup Result", "Chat History", "Dataset Entry"])

    orig_response = orig_response.replace("Context", context_word)
    orig_response = orig_response.replace("context", context_word.lower())

    prompt = prompt.replace("Context", context_word)
    prompt = prompt.replace("context", context_word.lower())

    context_document = context_document.replace("Context", context_word)
    context_document = context_document.replace("context", context_word.lower())
    params['context_word'] = context_word
  params['prompt'] = prompt
  params['start_response'] = orig_response
  params['context_document'] = context_document
  return prompt, params

def split_output_to_docs(output):
  """
  Given an output of an LLM intended to create multiple responses, split the output into docs.
  """
  global command_response_words, command_separator, advanced_response_separator, response_separator, section_separator, section_beginning
  global start_command, start_response, command_response_words
  global step_1_preprocess_prompts_list, step_2_enhance_prompts_list, step_3_task_based_prompts_list, step_4_subject_matter_prompts_list, step_5_safety_prompts_list

  output = output.replace("[End Date]", "(End Date)")
  for assistant in ["ASSISTENT", "ASSISANT", "ASS ISTANT", "ASSI STANT", "ASSIS TANT", "ASSIST ANT", "ASSITANT", "AssiSTANT", "AssisTANT", "AssiSTANT", "AssistANT", "AssistaNT", "AssistanT"]:
    output = output.replace(assistant, "ASSISTANT")
    output = output.replace(assistant.upper(), "ASSISTANT")
    output = output.replace(assistant.lower(), "assistant")
    output = output.replace(assistant[0].upper()+assistant[1:].lower(), "Assistant")
  for key in command_response_words:
    if key == "A:" or "answer" in key.lower():
        continue
    if len(key) >= 1 and len(key.strip(" \n~!@#$%^&*()_+=-<>,.?/:;[{}]")) > 0:
      if len(key) <= 3:
        key = "\n"+key
      output = output.replace(key, "@#@")
      output = output.replace(key.upper(), "@#@")
      output = output.replace(key.lower(), "@#@")
      output = output.replace(key.replace(" ", ""), "@#@")
      output = output.replace(key.replace(" ", "\n"), "@#@")
      key = key.strip()
      if len(key) > 4:
        output = output.replace("\n"+key.rstrip(":")+": ", "@#@")
      key = key.rstrip(" \n~!@#$%^&*()_+=-<>,.?/:;[{}]")
      if len(key) < 4 or key[0] not in "!@#$%^&*()_+=-<>,.?/:;[{}]": continue
      output = output.replace(key, "@#@")
      output = output.replace(key.upper(), "@#@")
      output = output.replace(key.lower(), "@#@")
      output = output.replace(key.replace(" ", ""), "@#@")
      output = output.replace(key.replace(" ", "\n"), "@#@")
  for word in extra_separator_items:
    output = output.replace(word, "@#@")
  output = output.replace("(End Date)", "[End Date]")
  return output.split("@#@")

def cleanup_individual_doc(doc, prev_doc, params):
  """cleanup an individual document. standardize the formatting. """
  # TODO: add a python, javascript, markdown, html etc. fixer and detector

  global command_response_words, command_separator, advanced_response_separator, response_separator, section_separator, section_beginning
  global start_command, start_response, command_response_words
  global step_1_preprocess_prompts_list, step_2_enhance_prompts_list, step_3_task_based_prompts_list, step_4_subject_matter_prompts_list, step_5_safety_prompts_list
  proto_answer_separator= params['proto_answer_separator']
  doc = doc.replace("'''javascript", "```javascript").replace("'''python", "```python").replace("'''markdown", "```markdown").replace("'''html", "```html").replace("'''json", "```json")
  doc = doc.replace(",o0", ",00").replace(",oo0", ",000").replace("0o", "00").replace("1o", "10").replace("2o", "20").replace("3o", "30").replace("4o", "40").replace("5o", "50").replace("6o", "60").replace("7o", "70").\
          replace("8o", "80").replace("9o", "90")
  doc = doc.replace(",o0", ",00").replace(",oo0", ",000").replace("0o", "00").replace("1o", "10").replace("2o", "20").replace("3o", "30").replace("4o", "40").replace("5o", "50").replace("6o", "60").replace("7o", "70").\
          replace("8o", "80").replace("9o", "90")
  doc = doc.replace(",o0", ",00").replace(",oo0", ",000").replace("0o", "00").replace("1o", "10").replace("2o", "20").replace("3o", "30").replace("4o", "40").replace("5o", "50").replace("6o", "60").replace("7o", "70").\
          replace("8o", "80").replace("9o", "90")
  doc = doc.replace("&", " & ").replace("  &", " &").replace("&  ", "& ")
  doc = doc.replace("***", "**").replace('^^^', '^^').replace('///', '//').replace('%%%', '%%').replace('$$$', '$$')
  doc = doc.replace("***", "**").replace('^^^', '^^').replace('///', '//').replace('%%%', '%%').replace('$$$', '$$')
  doc = doc.replace("***", "**").replace('^^^', '^^').replace('///', '//').replace('%%%', '%%').replace('$$$', '$$')
  doc = doc.split("[Back to",1)[0]
  doc = doc.split("[Return to",1)[0]
  doc = doc.split("©",1)[0]
  doc = doc.split("All rights reserved",1)[0]
  doc = doc.replace("Open Assistant", "")
  doc = doc.rstrip(":;-#=_|\n <|{([-=*").lstrip(":;-#=_|>|})]")

  o1 = doc.split("\n")
  if len(o1[0]) <= 20 and "```" not in o1[0]:
      o1 = o1[1:]
  o2 = []
  for s in o1:
     if any(b for b in s.split() if len(b.strip(":~!@#$%^&*()_+-=<>,.?/"))/(1+b.count("http")) > 30):
      # remove lines with words that are too long
      continue
     o2.append(s)
  for _ in range(6):
    if not o2: continue
    if "USER 1" in o2[0] or "ASSISTANT 1" in o2[0]:
      o = o[1:]
      continue
    if "USER 1" in o2[-1] or "ASSISTANT 1" in o2[-1]:
      o2 = o2[:-1]
      continue
    if len(o2[-1]) <= 3:
      o2 = o2[:-1]
      continue
    if len(o2[0]) <= 20 and o2[0].endswith("--") and "--" not in o2[0][:-10]:
      o2 = o2[1:]
      continue
    if len(o2[0]) <= 20 and o2[0].endswith("##") and "##" not in o2[0][:-10]:
      o2 = o2[1:]
      continue
    if len(o2[0]) <= 20 and o2[0].startswith("##") and not o2[0].endswith("##") :
      o2 = o2[1:]
      continue
    if len(o2[0]) <= 20 and o2[0].endswith("==") and "==" not in o2[0][:-10]:
      o2 = o2[1:]
      continue
    if len(o2[0]) <= 20 and o2[0].endswith("--") and "--" not in o2[0][:-10]:
      o2 = o2[1:]
      continue
    if len(o2[0]) <= 20 and o2[0].endswith("->") and "<-" not in o2[0][:-10]:
      o2 = o2[1:]
      continue
    if len(o2[0]) <= 20 and o2[0].endswith("|>") and "<|" not in o2[0][:-10]:
      o2 = o2[1:]
      continue
    if o2[0] in {"s:", "==", "---", "===", "###", "##", "**", "***"}:
      o2 = o2[1:]
      continue
    if o2[-1] in {"==", "---",  "===", "###", "##", "**", "***"}:
      o2 = o2[:-1]
      continue
    if ("let me know" in o2[-1].lower() or "good luck" in o2[-1].lower() or ("due to" in o2[-1].lower() and "constraints" in o2[-1].lower() or "(continu" in o2[-1].lower() or "[continu" in o2[-1].lower())):
      o2 = o2[:-1]
      continue
    if len(o2[0]) <= 25 and ("<|" in o2[0] or "|>" in o2[0]): #  or "start" in o2[0].lower() or " end " in o2[0].lower() or "assistant" in o2[0].lower() or "user" in o2[0].lower() or "instruction" in o2[0].lower() or "response" in o2[0].lower() or "new" in o[0].lower() or "revised" in o[0].lower() or "agent" in o[0].lower()):
      o2 = o2[1:]
      continue
    if len(o2[-1]) <= 25 and ("<|" in o2[-1] or "|>" in o2[-1]): #  or "start" in o[-1].lower() or " end " in o[-1].lower() or "assistant" in o[-1].lower() or "user" in o[-1].lower() or "instruction" in o[0].lower() or "response" in o[0].lower() or "new" in o[-1].lower() or "revised" in o[-1].lower() or "agent" in o[-1].lower()):
      o2 = o2[:-1]
      continue
    break
  o3 = []
  for s in o2:
    if len(s) > 500 and (s.count(";") + s.count(".") + s.count("?") + s.count("!"))/len(s) < 0.001:
      break
    o3.append(s)
  if not o3:
    return None
  doc = "\n".join(o3)
  doc = doc.rstrip(":;-#=_|\n <|{([-=*").lstrip(":;-#=_|>|})]")
  #print (doc)
  if "]]" in doc[:100] and "[[" not in doc[:100]:
    doc = doc.split("]]",1)[-1]
  if "--" in doc[:100] and doc[:100].count("--") == 1:
    doc = doc.split("--",1)[-1].split("--",1)[-1]
  if "==" in doc[:100] and doc[:100].count("==") == 1:
    doc = doc.split("--",1)[-1].split("==",1)[-1]
  doc = doc.rstrip(":;-#=_|\n <|{([-=*").lstrip(".,:;-#=_| \n>|})]")
  doc = doc.replace("\n\n\n", "\n\n")
  if len(doc) < 20:
    return None
  if (doc.split()[0].lower() not in instruction_starter) and len(doc) < 100:
    return None
  if "```" not in doc and 'python' not in doc and 'Python' not in doc and '\ndef ' not in doc and sum(len(s) for s in o2)/len(o2) < 20:
    return None
  if "```" not in doc and 'python' not in doc and 'Python' not in doc and '\ndef ' not in doc and any(s for s in doc.split() if len(s.strip("~!@#$%^&*()<>,.?/=-_+")) > 30 and "http" not in s):
    return None

  doc = doc.replace("\n\n\n", "\n\n")
  if random.randint(0,1):
    doc = doc.replace("\n\n", "\n")
  if random.randint(0,1):
    doc = doc.replace("\n\n", "\n")
  if "USER" in doc[:20]:
    doc = doc.split("USER",1)[-1]
  if "ASSISTANT" in doc[:20]:
    doc = doc.split("ASSISTANT",1)[-1]
  doc  = doc.replace("As an AI language model, ", "").replace("As an AI model, ", "").replace("As a language model, ", "").replace("As an AI, ", "").strip("：:\n").rstrip("：:\n ").replace("As Phi,", "").replace(" Phi ", " an Assistant ").replace(" PHI ", " an Assistant ").replace("I understand! ", "").replace("I understand now! ", "").replace("Certainly!", "").replace("Absolutely!", "").replace("Okay, great! ", "")
  doc = doc.rstrip(":;-#=_|\n <|{([-=*").lstrip(".,:;-#=_| \n>|})]")
  if doc.startswith("s "):
    doc = doc[2:]
  doc = doc[0].upper()+doc[1:]
  if doc.startswith("```") and "```" not in doc[5:]:
    doc= doc+"\n```"
  doc = doc.replace("```-jsonl", "```json").replace("```jsonl", "```json")
  if "```json" in doc:
    for idx, s in enumerate(doc.split("```json")):
      if idx % 2 != 0:
        s = s.split("```")[0].strip()
        try:
          new_json  =json.dumps(json.loads(repair_json(s)), indent=4)
          doc = doc.replace(s, new_json)
        except:
          #print ('json error')
          pass
  if "\n" in doc and doc[-1].lower() in "qwertyuiopasdfghjklzxcvbnm,:-=+" and doc.count(".") >= 3:
    last = doc.split(".")[-1].split("?")[-1].split("!")[-1].split("`")[-1].split("'")[-1].split("\"")[-1].split(".")[-1].split("\n")[-1]
    last = doc.split(".")[-1].split("?")[-1].split("!")[-1].split("`")[-1].split("'")[-1].split("\"")[-1].split(".")[-1].split("\n")[-1]
    last = last.strip()
    o = doc.split("\n")
    o[-1] = o[-1].replace(last, "").strip()
    doc = "\n".join(o).strip()
  # if there are any typical responses, turn this into a q/a
  for word in basic_response_separator_list:
    if word.lower() not in proto_answer_separator.lower():
      doc = doc.replace("\n"+word+":\n", proto_answer_separator)
      doc = doc.replace("\n"+word[0].upper()+word[1:]+":\n", proto_answer_separator)
      doc = doc.replace("\n"+word.upper()+":\n", proto_answer_separator)
  has_qa = False
  if prev_doc and "\n" not in prev_doc and ("\n" in doc or len(doc) > 200) and proto_answer_separator not in doc and "A:" not in doc and "Answer:" not in doc and \
    "==Answer" not in doc and "--Answer" not in doc and  "#Answer" not in doc and \
    "== Answer" not in doc and "-- Answer" not in doc and  "# Answer" not in doc and \
    "*Answer" not in doc:
    last_doc = prev_doc.lower().strip()
    if "?" in last_doc or any (s for s in instruction_starter if last_doc.startswith(s+" ") or last_doc.startswith("Please "+s)):
      doc= prev_doc+proto_answer_separator+doc
      has_qa = True
  doc = cleanup_generated_based_on_ref_doc(doc, params['context_document'])
  return doc, has_qa


def batch_create_prompts(context_documents_list=[], add_rl_pair_at_end=True, add_anonymization_and_ner=True, permute_context_document=True, \
                         prompt_choice_list=None, format_list=None, min_new_tokens_list=None, params_list=None, add_instruction_evolution=True, \
                   prefer_coding=False, eos="<|endoftext|>", proto_answer_separator = "\n===\nAnswer:\n", use_context_document_in_training=True,
    ):
  """
  Create a varied prompt related to a context document. Extract documents from the response of an LLM to the prompt. Do cleanup.
  Returns a record for trainig.
  TODO: do batching
  """
  assert context_documents_list or params_list, "You need to either pass in a context document, or the params which includes a context document"
  if not context_documents_list:
    context_documents_list = [params['context_document'] for params in params_list]
  if not params_list:
    params_list = [None]*len(context_documents_list)
  if not prompt_choice_list:
    prompt_choice_list = [None]*len(context_documents_list)
  if not  min_new_tokens_list:
    min_new_tokens_list = [None]*len(context_documents_list)
  if not format_list:
    genre = random.choice(genres)
    format_list = [genre +" in Markdown format"] *len(context_documents_list)
  new_params_list = []
  ents = []
  rels = []
  for context_document, params, prompt_choice, min_new_tokens, format in zip(context_documents_list, params_list, prompt_choice_list, min_new_tokens_list, format_list):
    # let's figure out the quality of the seed. the quality will determine which prompts we might use
    seed_edu_pred = edu_model.predict(context_document.lower().replace("\n", " ")[:min(1000, len(context_document))].replace(" she ", " he ").replace(" her ", " his"))
    seed_oh_eli5_pred = oh_eli5.predict(context_document.lower().replace("\n", " ")[:min(1000, len(context_document))].replace(" she ", " he ").replace(" her ", " his"))
    seed_rpj_pred = red_pajama_model.predict(context_document.lower().replace("\n", " ")[:min(1000, len(context_document))].replace(" she ", " he ").replace(" her ", " his"))

    orig_context_document = context_document = context_document.strip()
    if add_anonymization_and_ner and "{PERSON" not in context_document and "{LOC" not in context_document and "{REGION" not in context_document and "{ORG" not in context_document:
      ents, context_document, rels = ner_rel_template_extract(context_document)
      #print (rels)
    if permute_context_document and len(context_document) > 1000 and "```" not in context_document and "\ndef " not in context_document:
      if "\n" in context_document:
        o = context_document.split("\n")
        pivot = random.randint(1, len(o)-1)
        o = o[pivot:] + o[:pivot]
        context_document = "\n".join(o)
      elif ". " in context_document:
        context_document = context_document.rstrip(".")
        o = context_document.split(". ")
        pivot = random.randint(1, len(o)-1)
        o = o[pivot:] + o[:pivot]
        context_document = ". ".join(o) + "."
    if rels:
      rel_summary = "\n* ".join(" ".join(rel) for rel in rels)
      if rel_summary.strip():
        context_document =  "* "+rel_summary + "\n\n"+ context_document
        context_document = context_document.strip()
      #print ('NEW CONTEXT', context_document)
    # create longer generated text for longer inputs
    # TODO: override with output length, add_instruction_evolution and prefer_coding params from the prompt specs
    context_document = context_document.replace("..",".").replace(". .", ".").replace(";", ".").replace(":.", ":").strip()
    if len(context_document) < 1000:
      min_new_tokens = 256
    elif len(context_document) < 2000:
      min_new_tokens = 512
    elif len(context_document) < 3000:
      min_new_tokens = 1024
    else:
      min_new_tokens = 2048


    if prompt_choice is None:
      prompt_choice = random.randint(1,5)

    if type(prompt_choice) is str:
      # get a specific prompt template
      prompt_name = prompt_choice
      prompt = step_1_preprocess_prompts.get(prompt_name)
      if not prompt:
        prompt = step_2_enhance_prompts.get(prompt_name)
      if not prompt:
        prompt = step_3_task_based_prompts.get(prompt_name)
      if not prompt:
        prompt = step_4_subject_matter_prompts.get(prompt_name)
      if not prompt:
        prompt = step_5_safety_prompts.get(prompt_name)
      if not prompt:
        prompt_choice = 1
        prompt_name, prompt = random.choice(step_1_preprocess_prompts_list)
        add_instruction_evolution = prefer_coding= False
    else:
      # prefer step 1 preprocessing prompts for long documents that might not be of high quality
      if len(context_document) >= 2000 and not ((seed_rpj_pred[0][0] == '__label__wiki'  and seed_rpj_pred[1] > 0.60 ) or \
        (seed_edu_pred[0][0] == '__label__High'  and seed_edu_pred[1] > 0.60 ) or \
        (seed_oh_eli5_pred[0][0] == '__label__hq' and  seed_oh_eli5_pred[1] > 0.60)):
          if random.randint(0,1):
            prompt_choice = 1

      # get a random prompt tepmplates
      if prompt_choice==1:
        prompt_name, prompt = random.choice(step_1_preprocess_prompts_list)
        add_instruction_evolution = prefer_coding= False
      elif prompt_choice==2:
        prompt_name, prompt = random.choice(step_2_enhance_prompts_list)
      elif prompt_choice==3:
        prompt_name, prompt = random.choice(step_3_task_based_prompts_list)
      elif prompt_choice==4:
        prompt_name, prompt = random.choice(step_4_subject_matter_prompts_list)
      elif prompt_choice==5:
        prompt_name, prompt = random.choice(step_5_safety_prompts_list)
        add_instruction_evolution = prefer_coding= False
      else:
        prompt_name, prompt = random.choice(step_1_preprocess_prompts_list)
        add_instruction_evolution = prefer_coding= False

    prompt = prompt['prompt']

    if params is None:
      tools = label = span_text = purpose = questions = "[To be determined]"

      stakeholders=["high school student", "grade school student", "five year old", "college student", "graduate student"]
      stakeholder = random.choice(stakeholders)
      #TODO - get randomized audience, etc.
      params =  {'start_system': random.choice(['<|system|>', "", "_system_", "== SYSTEM == ", "-- SYSTEM -- ", "[SYSTEM]", "[[SYSTEM]]"]), 'tools': tools,\
            'span_text': span_text, 'quality': 'high', 'tone': 'helpful', 'target_language': 'English', \
            'assistant_persona': 'You are a helpful and respectful assistant', \
            'format': format, 'question_starter': first_instruction_starter(), \
            'audience': stakeholder, 'stakeholder': stakeholder, 'data_type': 'table', \
            'questions': questions, 'purpose': purpose, 'chosen': '', 'rejected': '', \
            'label': label, 'prompt_name': prompt_name, \
            'add_instruction_evolution':  add_instruction_evolution,\
            'ents': ents, 'rels': rels, 'add_rl_pair_at_end': add_rl_pair_at_end, \
            'prefer_coding': prefer_coding, 'use_context_document_in_training': use_context_document_in_training, \
            }
    params['context_document'] =  context_document
    prompt, params = diversify_prompt(prompt, params, add_instruction_evolution=add_instruction_evolution, \
                    prefer_coding=prefer_coding)
    params['prompt'] = prompt
    #print (prompt)
    #params['context_document'] =  orig_context_document
    #params['prompt_template'] = prompt_template.replace(params['context_document'], '%(context_document)s')
    params['eos'] =  eos
    params['proto_answer_separator'] = proto_answer_separator
    params['seed_predictions'] = [seed_edu_pred, seed_oh_eli5_pred, seed_rpj_pred]
    params['min_new_tokens'] = min_new_tokens
    new_params_list.append(params)
  return new_params_list

def batch_generate_from_prompt(params_list, min_new_tokens):
  # batch the generation by length
  # now do the batch generation
  prompts = [params['prompt'] for params in params_list]
  output_list = generate(model, tokenizer, prompts, return_response_only=True, min_new_tokens=min_new_tokens, max_new_tokens=int(min_new_tokens*2))
  for params, output in  zip(params_list, output_list):
    params['output'] = output
    params['generative_model'] = generative_model
  return params_list

def batch_create_synthetic_data_from_llm_output(params_list):
  for params in params_list:
    output = params['output']
    eos = params['eos']
    use_context_document_in_training = params['use_context_document_in_training']
    seed_edu_pred, seed_oh_eli5_pred, seed_rpj_pred = params['seed_predictions']
    proto_answer_separator = params['proto_answer_separator']
    all_docs = split_output_to_docs(output)
    all_filtered_docs = []
    all_pred = []
    prev_doc = None
    idx = 0
    qa_idx = -1
    for doc in all_docs:

      #
      result = cleanup_individual_doc(doc, prev_doc, params)
      if not result: continue
      doc, has_qa = result
      code_portion = ""
      # if this is code, we will just save it away
      if "```" in doc:
        doc_type = doc.split("```",1).split("\n",1)[0]
        doc_type = doc_type[:min(4,len(doc_type))]
        if doc_type in  {'pyth', 'java', 'type', 'c', 'c++', 'ruby' 'perl'}: # todo - add others
          code_idx = doc.indexof("```")
          doc, code_portion = doc[:code_idx], doc[code_idx:]
      lang = langid.classify(doc)
      #TODO: if not english, we should do a basic kenlm score for multilingual
      edu_pred = edu_model.predict(doc.lower().replace("\n", " ")[:min(1000, len(doc))].replace(" she ", " he ").replace(" her ", " his"))
      oh_eli5_pred = oh_eli5.predict(doc.lower().replace("\n", " ")[:min(1000, len(doc))].replace(" she ", " he ").replace(" her ", " his"))
      ##((pred[0][0] == '__label__Mid' and pred[1] > 0.60 ) and (oh_eli5_pred[0][0] == '__label__hq' and  oh_eli5_pred[1] > 0.40)) or \
      rpj_pred = red_pajama_model.predict(doc.lower().replace("\n", " ")[:min(1000, len(doc))].replace(" she ", " he ").replace(" her ", " his"))
      pile_pred =  pile_class_model.predict(doc.lower().replace("\n", " ")[:min(1000, len(doc))].replace(" she ", " he ").replace(" her ", " his"))
      domain_pred = domain_model.predict(doc.lower().replace("\n", " ")[:min(1000, len(doc))].replace(" she ", " he ").replace(" her ", " his"))
      if has_qa:
          #save away q/a regardless of scoring. we can use "bad" pairs for DPO
          all_filtered_docs[-1] = doc
          all_pred[-1] = [edu_pred, oh_eli5_pred, pile_pred, rpj_pred, domain_pred]
          prev_doc = None
          qa_idx = idx-1
          continue
      if "?" in doc or any (s for s in instruction_starter if doc.startswith(s+" ") or doc.startswith("Please "+s)):
        # save away the  question or instruction regardless of scoring. this is a query and can be 'low' quality esp. if we are simulating a real user request
        all_filtered_docs.append(doc)
        all_pred.append([edu_pred, oh_eli5_pred, pile_pred, rpj_pred, domain_pred])
        prev_doc = doc
        idx += 1
        continue

      # these are regular documents or information extractions. do some filtering for quality.
      # information extractions are kept for the next rounds of data generations.
      if ((edu_pred[0][0] == '__label__High'  and edu_pred[1] > 0.55 ) and (oh_eli5_pred[0][0] == '__label__hq' and  oh_eli5_pred[1] > 0.55)) or  \
          ((edu_pred[0][0] == '__label__Mid'  and edu_pred[1] > 0.70 ) and (oh_eli5_pred[0][0] == '__label__hq' and  oh_eli5_pred[1] > 0.80)) or  \
            (oh_eli5_pred[0][0] == '__label__hq' and  oh_eli5_pred[1] > 0.70) or \
            (edu_pred[0][0] == '__label__High' and edu_pred[1] > 0.70) or \
            (pile_pred[0][0] not in {'__label__Pile-CC', '__label__OpenWebText', '__label__HackerNews'} and pile_pred[1] > 0.70) or \
            (rpj_pred[0][0] == '__label__wiki' and rpj_pred[1] > 0.70):
          all_filtered_docs.append(doc)
          all_pred.append([edu_pred, oh_eli5_pred, pile_pred, rpj_pred, domain_pred])
          prev_doc = doc
          idx += 1
          continue

      #TODO: do hallucination correction via KG and NLI
      if len(doc) >= 1000 and len(params['context_document']) >= 1000:
        # these are long context documents and response documents. We want to keep non low edu documents that are better than the context documents.
        if (seed_edu_pred[0][0] == edu_pred[0][0] and seed_edu_pred[0] < edu_pred[0]) or \
          (seed_oh_eli5_pred[0][0] == oh_eli5_pred[0][0] and seed_oh_eli5_pred[0] < oh_eli5_pred[0]) or \
          (seed_oh_eli5_pred[0][0] == '__label__cc' and oh_eli5_pred[0][0] == '__label__hq') or \
          (seed_edu_pred[0][0] == '__label__Low' and edu_pred[0][0] != '__label__Low') or \
          (seed_edu_pred[0][0] == '__label__Mid' and edu_pred[0][0] ==  '__label__High'):
          if edu_pred[0][0] == '__label__Low': continue
          all_filtered_docs.append(doc)
          all_pred.append([edu_pred, oh_eli5_pred, pile_pred, rpj_pred, domain_pred])
          prev_doc = doc
          idx += 1
          continue

      # documents that are near the first response and have lots of ':'  are probably an information extraction result
      if idx <= 1 and doc.count("\n") >= 5 and doc.count(":")/doc.count("\n") >= 0.1:
        if edu_pred[0][0] == '__label__Low': continue
        #print (('information extraction', doc))
        all_filtered_docs.append(doc)
        all_pred.append([edu_pred, oh_eli5_pred, pile_pred, rpj_pred, domain_pred])
        prev_doc = doc
        idx += 1
        continue

      prev_doc = None
    if params['add_rl_pair_at_end'] and qa_idx >= 0:
      qa = all_filtered_docs[qa_idx]
      q, a = qa.split(proto_answer_separator, 1)
      all_filtered_docs = all_filtered_docs[:qa_idx-1]+ all_filtered_docs[qa_idx+1:] + [q]
      params['chosen'] = a.strip()
    params['all_filtered_docs'] = all_filtered_docs
    params['seed_predictions'] = [seed_edu_pred, seed_oh_eli5_pred], seed_rpj_pred
    params['all_predictions'] = all_pred
    text = eos.join(all_filtered_docs).strip()
    if not use_context_document_in_training:
      text = text + eos
    else:
      # the training text will include the context document
      if len(params['context_document'] ) < 3000 or qa_idx >= 0:
        text = params['context_document'] + eos + text + eos
      else:
        choice = random.randint(0,10)
        if choice <= 5:
          if random.randint(0,1):
            text = "The following is related to a "+params['context_word']+". The task is to infer the "+params['context_word']+" after answering the questions and/or reading the documents."+eos+text + eos + "Inferred "+params['context_word'] + ":\n"+params['context_document'] + eos
          elif random.randint(0,1):
            text = "[[The following is related to a "+params['context_word']+". The task is to infer the "+params['context_word']+" after answering the questions and/or reading the documents.]]"+eos+text + eos +  "Inferred "+params['context_word'] + ":\n"+params['context_document'] + eos
          elif random.randint(0,1):
            text = "The following is related to a "+params['context_word']+". The task is to infer the "+params['context_word']+" after answering the questions and/or reading the documents."+eos+text + eos + "Inferred "+params['context_word'] + ":\n"+params['context_document'] + eos
          else:
            text = "The following is related to a "+params['context_word']+". The task is to infer the "+params['context_word']+" after answering the questions and/or reading the documents."+eos+text + eos + "Inferred "+params['context_word'] + ":\n"+params['context_document'] + eos
          if random.randint(0,1):
            text = text.replace("The following", "Please read the text below which")
          if random.randint(0,1):
            text = text.replace("The following", "This")
          if random.randint(0,1):
            text = text.replace("The following", "Here")
          if random.randint(0,1):
            text = text.replace("The task is", "Please")
          if random.randint(0,1):
            text = text.replace("after answering the questions and/or reading the", "based on the following")
          if random.randint(0,1):
            text = text.replace("after answering", "after reading")
          if random.randint(0,1):
            text = text.replace("after answering", "based on examining")
          if random.randint(0,1):
            text = text.replace("after answering", "upon analyzing")
          if random.randint(0,1):
            text = text.replace("the questions and/or", "the chat history and")
          if random.randint(0,1):
            text = text.replace("the questions and/or", "the discussion and")
          if random.randint(0,1):
            text = text.replace("after answering the questions and/or reading the", "based on the following")
          text = text.replace("documents", random.choice(["documents", "documents", "data", "text", "examples", "segments", "discussions"]))
        elif choice == 6:
          text = params['context_document'] + eos + text + eos
        elif choice == 7:
          text = text + eos + params['context_word'] + ":\n"+params['context_document'] + eos
        elif choice == 8:
          text = params['context_word'] + ":\n"+params['context_document'] + eos + text + eos
        elif choice == 9:
          text = params['context_document'] + "\n\n***\n\nLet's conisder the above further ..." + eos + text + eos
        else:
          text = params['context_document'] + "\n\n===\n\nAnalysis of the above "+params['context_word'] +"." +eos + text + eos
    text = text.strip()
    params['text'] = text
  return params_list

def one_generation_synthetic_data(context_document, add_anonymization_and_ner=True, permute_context_document=True, prompt_choice=None,  add_instruction_evolution=True, \
                   add_rl_pair_at_end=True, prefer_coding=False, eos="<|endoftext|>", proto_answer_separator = "\n===\nAnswer:\n", use_context_document_in_training=True,
                                  min_new_tokens=None,
    ):
  params_list = batch_create_prompts([context_document], add_rl_pair_at_end=add_rl_pair_at_end, add_anonymization_and_ner=add_anonymization_and_ner, prompt_choice_list= None if not prompt_choice else [prompt_choice],
                                     min_new_tokens_list= None if not min_new_tokens else [min_new_tokens],
                                     add_instruction_evolution=add_instruction_evolution, prefer_coding=prefer_coding, eos=eos, \
                                     proto_answer_separator = proto_answer_separator, use_context_document_in_training=use_context_document_in_training,)
  #print (params_list[0]['prompt'])
  params_list = batch_generate_from_prompt (params_list, params_list[0]['min_new_tokens'])
  params_list = batch_create_synthetic_data_from_llm_output(params_list)
  return params_list[0]
def batch_generation_synthetic_data(context_documents_list=[], add_rl_pair_at_end=True, add_anonymization_and_ner=True, permute_context_document=True, \
                         prompt_choice_list=None, format_list=None, min_new_tokens_list=None, params_list=None, add_instruction_evolution=True, \
                   prefer_coding=False, eos="<|endoftext|>", proto_answer_separator = "\n===\nAnswer:\n", use_context_document_in_training=True,
    ):
  params_list = batch_create_prompts(context_documents_list=context_documents_list, \
                                     add_rl_pair_at_end=add_rl_pair_at_end, add_anonymization_and_ner=add_anonymization_and_ner,\
                                     permute_context_document=permute_context_document, \
                                     prompt_choice_list=prompt_choice_list, format_list=format_list,
                                     min_new_tokens_list=min_new_tokens_list, params_list=params_list, add_instruction_evolution=add_instruction_evolution, \
                                     prefer_coding=prefer_coding, eos=eos, proto_answer_separator = proto_answer_separator,
                                     use_context_document_in_training=use_context_document_in_training)
  params_list_by_buckets = {}
  for param in params_list:
    bucket = int(param['min_new_tokens'] //1000)
    params_list_by_buckets[bucket] = params_list_by_buckets.get(bucket, []) + [param]
  # we should batch this into buckets of similar min_new_tokens
  for bucket, params in params_list_by_buckets.items():
    # generate will modify the param itself
    batch_generate_from_prompt (params_list, max([param['min_new_tokens'] for param in params]))
  params_list = batch_create_synthetic_data_from_llm_output(params_list)
  return params_list

#TODO: extract info into params
