#@title general libs

import locale
from collections import Counter
import subprocess
import fasttext
from autocorrect import Speller
import wn
from wn.morphy import Morphy
import spacy
import scispacy


en_wn = wn.Wordnet('oewn:2023', lemmatizer=Morphy())
spacy_nlp = spacy.load('en_core_web_sm')
spacy_nlp = spacy.load('en_core_web_sm')    
#sci_spacy = spacy.load("en_ner_bc5cdr_md")



def getpreferredencoding(do_setlocale=True):
    return "UTF-8"


locale.getpreferredencoding = getpreferredencoding

spell = Speller(lang='en')

numbering_list = ['3', '7)', '7.', '4', 'iii.', 'iii-', '8.', '4-', 'v:', 'I:', 'ii.', 'i.', 'V)', 'E)', 'I)', 'III.', 'III)', '2-', '1)', 'v-', 'III', 'I.', 'c)', '1.', 'V-', 'iv)', 'A)', 'v)', 'IV', 'C.', 'ii)', 'I', 'IV.', 'C)', 'II-', '2.', 'III-', 'IV)', 'd)', 'iii', 'i-', 'iii:', 'A.', 'B.', '1', '6)', 'ii', '8)', '3)', 'e)', 'ii-', '5-', 'II)', 'iv-', '2)', 'e.', 'IV:', 'III:', 'i)', '10.', 'V', 'V.', 'v.', 'D)', 'E.', 'iv:', 'B)', 'II', 'ii:', 'V:', 'a.', '5.', 'IV-', '9.', 'D.', '3.', '4:', '2:', 'i', 'II.', '3-', '2', 'c.', 'a)', '3:', '10)', 'd.', 'i:', 'iv.', '1-', '4.', '5', 'iv', 'iii)', 'b.', '1:', 'II:', 'v', '5:', '6.', 'b)', 'I-', '9)', '4)', '5)']

stopwords_list = ['by', 'er', 'to', 'concerning', 'specifying', 'thoughts', 'list', 'whatve', 'won', 'ky', 'widely', 'minus', 'g', 'farther', 'sides', 'did', 'off', 'differently', 'comprises', 'directly', 'sup', 'whats', 'sent', 'mill', 'lets', 'respectively', "what'd", 'interests', 'interested', 'works', 'ru', "c'mon", 'looks', 'dz', 'meet', 'sub', 'cr', 'most', 'despite', "where's", 'omitted', 'werent', 'vols', 'really', 'noted', 'are', 'tp', 'k', 'lost', 'beginnings', 'days', 'resulting', 'willing', 'whole', 'ordering', 'probably', 'has', 'sh', 'wherever', "ain't", 'detail', 'll', 'room', "that's", 't', 'yet', 'ord', 'whoever', 'youd', "what've", "'ve", 'thatve', 'find', 'welcome', 'howbeit', 'each', 'resulted', 'successfully', 'accordingly', 'already', 'int', 'site', 'inner', 'vs', 'dk', 'if', 'keeps', 'ts', 'immediate', 'insofar', 'longer', 'mn', 'likewise', 'described', 'dead', 'hes', 'quite', 'ml', 'unlike', 'kh', 'ie', 'early', 'clearly', 'your', 'very', "there's", 'com', 'therefore', 'test', 'consider', 'theres', 'seconds', 'faces', 'briefly', '’re', 'someday', 'begin', 'ne', 'md', 'actually', 'fk', 'hardly', 'gn', 'what', 'jp', 'own', 'included', 'downs', 'even', 'miss', 'inside', 'hn', 'whilst', '’s', 'use', 'gu', 'primarily', 'presumably', 'allow', 'cz', "would've", 'seemed', 'alone', 'day', 'anyways', 'none', 'pp', 'ci', 'ec', 'newer', 'followed', 'goes', 'do', 'didn', 'next', 'woman', "shouldn't", 'ex', 'whom', 'tw', 'yu', 'interest', 'lot', '’ll', 'e', 'so', 'looking', 'whereby', 'co', 'qa', 'four', 'u', "he'd", 'parts', 'along', 'gb', 'mc', 'gp', 'slightly', 'wasn', 'done', 'ht', 'brief', "isn't", 'men', 'specifically', 'both', 'gotta', 'ed', 'doesnt', 'affecting', 'his', 'containing', 'ii', 'much', 'accordance', 'yeah', 'went', 'kill', 'other', 'furthers', 'fj', 'non', 'generally', 'ff', 'happens', 'we', 'nothing', 'couldn', 'example', "a's", 'indeed', 'twas', 'course', 'not', 'wrong', 'usefulness', 'true', 'anyhow', 'like', 'place', 'stop', 'via', 'could', 'cs', 'appropriate', 'aside', 'little', 'thereupon', 'want', 'io', 'kw', 'adopted', 'wanna', 'know', 'om', 'sk', 'doubtful', 'since', 'sz', 'least', 'plus', 'felt', 'as', 'inc', 'kn', 'taken', 'another', "she'll", 'ending', 'us', 'join', 'better', 'former', 'noone', 'ae', "we're", 'pm', 'up', '’nt', 'mrs', 'giving', 'doesn', 'ill', 'sn', 'every', 'fc', 'high', 'gy', 'arent', 'others', 'per', 'bs', 'hither', 'hold', 'owing', 'it', "i'm", 'me', 'these', 'wheres', 'kg', 'include', 'youngest', 'head', 'se', 'im', 'why', 'nowhere', 'anybody', 'interesting', 'until', 'regarding', 'anywhere', 'older', 'sc', 'webpage', 'and', 'edu', 'presented', 'pointing', 'ups', 'seven', 'before', 'theirs', 'means', 'end', 'az', 'thousand', 'ir', 'anything', 'ought', "one's", 'recently', "there've", 'cu', 'neverf', 'became', 'predominantly', 'sm', 'five', 'shes', 'help', 'uy', 'rd', "didn't", 'dm', 'gf', 'problems', "what'll", 'perhaps', 'son', 'uz', 'liked', 'section', 'ug', 'thered', 'bn', 'affects', 'able', 'inasmuch', 'downwards', 'anyway', 'whim', 'ones', 'dad', 'ph', 'bj', 'behind', 'hard', 'right', 'co.', 'years', 'shouldnt', "you've", 'serious', 'somebody', 'yes', 'nobody', 'best', 'comprising', 'beside', 'itll', 'parted', 'contain', 'significant', 'grouped', 'vn', 'mk', 'got', 'unto', 'overview', 'bit', 'guess', 'information', 'p', 'billion', 'mw', "wouldn't", 'eg', 'tried', "where'll", 'ch', 'ro', 'said', 'hereby', 'substantially', 'somethan', 'whose', 'care', 'worked', 'great', 'hu', 'among', "they're", "when'll", 'b', 'known', 'think', 'long', 'sufficiently', 'ready', 'become', 'believe', 'tv', 'call', "where'd", 'kz', 'viz', 'pmid', "there'll", 'sb', 'about', 'furthermore', 'being', 'results', 'nine', 'synopsis', 'fm', 'pt', 'open', 'thanks', 'vol', 'work', 'instead', 'ah', 'pointed', 'sensible', 'an', 'certainly', 'was', 'finds', 'hell', 'haven', 'whenever', 'cg', 'tf', 'width', 'never', 'without', 'she', 'pa', 'whereafter', 'hm', 'sl', 'otherwise', 'took', 'ended', 'cf', 'nc', 'wed', 'couldnt', 'ring', 'act', 'gotten', 'grouping', 'case', 'saw', 'certain', 'cl', 'get', 'things', 'de', 'showing', 'okay', 'had', 'may', 'neither', 'during', 'everywhere', 'selves', 'trillion', 'tc', 'began', 'them', 'amoungst', 'ordered', 'hope', 'myse”', 'mustn', 'copy', 'last', 'tg', 'be', 'hadnt', 'although', 'showed', 'wasnt', 'hers', 'h', 'seventy', 'give', 'everyone', 'her', 'backs', 'vc', 'thereby', "he'll", "there're", 'some', 'mu', 'under', 'related', 'vi', 'potentially', 'everything', 'bb', 'see', 'lb', 'um', 'mostly', 'that', 'old', 'mh', 'r', 'within', 'group', "let's", 'np', 'htm', 'becomes', 'around', 'take', 'ltd', 'outside', "i'd", 'alongside', 'pe', 'affected', 'mg', 'might', 'especially', 'notwithstanding', 'length', 'fairly', 'fx', 'given', 'needs', 'associated', 'et', 'then', 'something', 'in', "when's", 'adj', 'ever', 'thorough', 'areas', 'big', 'no-one', 'too', 'eighty', 'into', 'asks', 'approximately', 'd', "could've", 'hello', 'third', 'ignored', 'but', 'whence', 'thirty', "shan't", 'backing', "we've", 'ma', 'low', 'www', 'whos', "why'd", 'con', 'longest', 'doing', 'small', 'au', 'good', 'ok', 'il', 'msie', 'ninety', 'ke', 'higher', 'click', 'mean', 'contains', 'underneath', 'almost', 'a', 'state', 'cy', 'nz', 'wont', 'seeing', 'from', "i've", 'mustnt', 'opposite', 'particular', 'gs', 'zero', "needn't", 'young', 'backed', 'my', 'dj', 'hereafter', 'now', 'mom', "mightn't", 'kind', 'exactly', 'thereof', 'rather', 'toward', 'tj', "'twas", 'for', "i'll", 'put', 'vu', 'will', "who'd", 'am', 'neednt', 'whereas', 'z', 'nor', 'itse”', 'value', 'causes', 'fifteen', 'different', 'turns', 'again', 'overall', 'oldest', 'km', 'hadn', 'however', "should've", 'themselves', 'first', 'pages', 'poorly', 'life', 'ls', 'awfully', 'sec', 'go', 'importance', 's', 'apparently', "that've", 'thereto', 'usually', 'weve', 'hasn', "hasn't", 'any', 'bh', 'thatll', 'fify', 'second', 'hear', 'opens', 'below', 'whether', 'same', 'eh', 'lk', 'zm', 'seems', 'cd', 'further', 'nos', "here's", 'gonna', 'ca', 'fine', 'free', 'wf', 'sometime', 'ly', 'turning', 'self', 'hr', 'previously', 'indicate', 'show', "they'll", 'huh', 'sees', 'ends', 'how', 'making', 'greatest', 'sd', 'job', '10', 'two', 'lest', 'weren', 'backward', 'gw', 'theyre', 'nu', 'evermore', 'tis', 'start', 'w', 'reserved', 'ableabout', 'abst', 'cases', 'thinks', 'ourselves', "why's", "aren't", 'well', 'more', 'kp', 'isnt', 'bring', 'aint', 'n', 'fewer', 'look', 'presenting', 'side', 'gone', 'biol', "haven't", 'six', 'ive', 'asked', 'somewhere', 'heard', 'becoming', 'proud', 'unless', 'puts', 'cause', 'left', 'this', 'meanwhile', 'try', 'etc', "who'll", 'shown', 'groups', "it's", 'when', 'on', 'ahead', 'buy', 'ms', 'research', 'significantly', 'live', 'there', 'saying', 'thats', 'hk', 'allows', 'gl', 'bd', 'name', 'says', 'somehow', 'forward', 'gm', "you'd", 'abroad', 'smallest', 'za', 'un', 'necessary', 'which', 'needing', 'thru', 'tries', 'knew', 'herself', 'tm', 'girl', 'year', 'cx', 'out', 'mz', 'new', 'shell', "there'd", 'line', 'jo', 'tends', 'myself', 'mv', 'him', 'herse”', 'specify', 'i.e.', "'tis", 'recent', 'have', 'does', 'various', 'c', 'sv', 'twice', 'lr', 'date', 'appreciate', 'forever', "can't", 'going', 'q', 'ng', 'ye', 'thanx', 'at', 'fill', 'opening', 'he', 'dare', 'ba', 'thence', 'no', 'apart', 'till', 'came', 'readily', 'et-al', 'ck', 'moreover', 'either', 'due', "doesn't", 'newest', 'keep', 'together', 'because', 'away', 'bill', 'hey', "'ll", 'just', 'secondly', 'jm', 'obviously', 'cry', 'sorry', 'someone', 'face', 'oughtnt', 'down', 'mt', 'greetings', 'til', "she'd", 'mm', 'useful', 'tt', 'sure', 'unlikely', 'nr', 'upon', 'backwards', 'versus', 'gt', 'following', 'than', 'himse”', 'mp', 'where', 'text', 'wanted', 'placed', 'inward', 'th', 'evenly', 'cv', 'cm', 'hereupon', 'undoing', 'problem', 'soon', 'their', 'begins', 'mine', 'mind', 'states', 'definitely', 'youve', 'round', 'whereupon', 'wholl', 'sa', 'therell', 'theyd', 'would', 'all', 'youre', 'tz', 'leave', 'hopefully', 'downing', 'large', 'those', 'afterwards', 'refs', 'hed', 'play', 'amidst', 'cannot', 'number', 'indicates', 'sixty', 'guys', 'sg', 'bottom', 'guy', 'id', 'rooms', 'mq', 'entirely', 'important', 'tk', 'possibly', 'kept', 'somewhat', "might've", 'top', "mayn't", 'home', 'showns', 'lu', 'summary', 'namely', 'over', 'less', 'i', 'uucp', 'car', 'darent', 'wherein', 'past', "what's", 'l', 'found', 'therein', 'full', 'getting', 'wonder', 'throughout', 'appear', 'http', 'still', "oughtn't", 'can', 'sincere', 'house', 'die', 'heres', 'immediately', 'thoroughly', 'latest', 'thereafter', 'ask', 'gd', "t's", 'fully', 'talk', 'across', 'cant', "they'd", 'beginning', 'night', 'parting', 'of', 'its', 'seen', 'quickly', 'tn', 'net', "he's", 'nay', 'sr', 'ago', 'highest', 'greater', 'between', 'pn', 'mr', 'ai', 'havent', 'run', 'itd', 'cn', 'oh', 'feel', 'above', 'isn', 'pg', 'que', 'ki', 'comes', 'gh', 'f', 'ad', 'ga', 'thin', 'ways', 'words', 'formerly', "weren't", 'general', 'nonetheless', 'eight', 'similar', 'also', 'provides', 'knows', 'wouldn', 'herein', 'ran', 'currently', 'mainly', 'j', 'tip', 'follows', 'neverless', 'thus', 'similarly', 'near', 'ni', 'seeming', 'hid', 'necessarily', "wasn't", 'facts', 'inc.', 'fo', 'one', 'normally', 'usefully', 'lt', 'thought', 'happy', 'didnt', 'mil', 'turned', 'using', 'several', 'ua', 'having', 'includes', 'fix', 'the', 'or', 'make', 'shan', 'available', 'wife', 'three', 'es', 'wait', 'few', 'anymore', "don't", 'v', "daren't", 'enough', 'onto', 'beyond', 'bi', 'including', 'else', 'phone', 'uses', 'ours', 'la', 'half', 'ain', 'homepage', 'merely', 'gmt', 'aren', 'amid', "you're", 'nl', 'pl', 'part', 'web', 'gr', 'shows', 'wells', 'keys', 'himself', 'member', 'needed', 'need', 'whomever', "she's", 'here', 'mx', 'once', 'regardless', 'seriously', 'bf', 'reasonably', 'nevertheless', 'redir', 'world', 'after', 'idea', 'though', 'aq', 'pw', 'shall', 'possible', 'arise', 'seem', 'trying', 'against', 'front', 'beings', 'places', 'everybody', 'come', 'effect', 'added', "when'd", 'furthered', 'wanting', 'point', "it'll", 'likely', 'upwards', 'lc', 'sir', 'while', 'made', 'stay', 'su', 'cmon', 'vg', 'changes', 'je', 'pr', 'obtained', 'na', 'pk', 'consequently', 'bw', 'ten', 'website', 'dear', 'theyll', 'fi', 'ge', "couldn't", 'bv', 'thank', 'af', 'gov', 'regards', 'thoughh', 'forth', 'turn', 'don', 'yt', 'amount', "how'll", 'with', 'yours', "hadn't", 'hundred', 'members', 'told', "why'll", 'page', 'many', 'relatively', 'smaller', 'o', 'makes', 'amongst', 'bz', 'youll', 'often', 'st', 'later', 'who', 'is', 'lower', 'through', 'elsewhere', 'org', 'whichever', 'used', 'back', 'ee', 'furthering', 'yourselves', 'bo', 'been', 'such', 'mo', 'nearly', 'mightnt', 'null', 'pf', 'order', 'thing', 'besides', 'gives', "mustn't", 'taking', 'nd', 'auth', 'twenty', "must've", 'si', 'maynt', 'clear', "won't", 'novel', 'gave', 'area', 'maybe', 'iq', 'nf', 'bt', 'ref', "that'll", 'arpa', 'only', 'ao', 'sometimes', 're', 'caption', 'downed', 'y', 'itself', 'asking', 'hence', 'describe', 'considering', 'must', 'twelve', 'let', 'presents', 'shed', 'particularly', 'dont', 'specified', "how'd", 'gets', 'obtain', 'mightn', 'far', "you'll", 'cc', 'suggest', 'you', 'way', 'rw', 'unfortunately', 'ag', "we'll", 've', 'latter', 'fr', 'bg', "it'd", 'corresponding', 'whod', 'eleven', 'indicated', 'qv', 'tr', 'should', 'index', 'say', 'm', 'largely', 'li', 'empty', 'provided', 'zr', 'towards', 'thou', 'fire', 'shouldn', "c's", 'our', "they've", 'yourself', 'truly', 'today', 'anyone', 'td', 'strongly', 'please', 'gi', 'ar', 'they', 'were', 'wish', 'tell', 'gg', 'x', 'nice', '39', 'wouldnt', 'kr', 'whatll', 'move', 'hasnt', "how's", 'va', 'latterly', 'lately', 'beforehand', 'announce', 'lv', 'promptly', 'whatever', 'except', 'gq', 'according', 'throug', 'needn', 'differ', 'br', "we'd", 'aw', 'whither', 'theyve', 'shant', 'hi', "who's", 'bm', 'sy', 'py', 'thereve', 'wants', 'present', 'al', 'always', 'orders', 'fact', 'therere', 'html', 'meantime', 'sj', 'ws']
common_title_words_set = {'works', 'notes', 'note', 'further', 'see', 'references', 'reference', 'section', 'title', 'conclusion', 'intro', 'introduction', 'executive', 'summary', 'key', 'plot', 'theme'}
stopwords_set = set(stopwords_list + numbering_list)


# some of this is based on https://github.com/amazon-science/RefChecker which is under Apache 2
def sentencize(nlp, text):
    """Split text into sentences"""
    doc = nlp(text)
    return [sent for sent in doc.sents]


def split_text(nlp, text, segment_len=200):
    """Split text into segments according to sentence boundaries."""
    segments, seg = [], []
    sents = [[token.text for token in sent] for sent in sentencize(nlp, text)]
    for sent in sents:
        if len(seg) + len(sent) > segment_len:
            segments.append(" ".join(seg))
            seg = sent
            # single sentence longer than segment_len
            if len(seg) > segment_len:
                # split into chunks of segment_len
                seg = [
                    " ".join(seg[i:i+segment_len])
                    for i in range(0, len(seg), segment_len)
                ]
                segments.extend(seg)
                seg = []
        else:
            seg.extend(sent)
    if seg:
        segments.append(" ".join(seg))
    return segments


def is_num(n):
  try:
    float(n)
    return True
  except:
    return False


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


def guess_ner_label(ent, label=""):
  hype_lemma = []
  exact_match = False
  # basic pattern matching to override wn which doesn't always match
  if "@" in ent and (".co" in ent or ".org" in ent or ".gov" in ent or ".edu" in ent):
    label = "EMAIL"
    return label, hype_lemma, exact_match
  if "www" in ent or "http" in ent or ".co" in ent or ".org" in ent or ".gov" in ent or ".edu" in ent:
    label = "URL"
    return label, hype_lemma, exact_match
  last_word = ent.split()[-1].lower().strip(".")
  if last_word in {"act", "law",}:
    label = "TRAIT"
    return label, hype_lemma, exact_match
  if last_word in {"people", "asian", "american", "african", "european"}:
    label = "TRAIT"
    return label, hype_lemma, exact_match
  if last_word in {"states", "kingdom",}:
    label = "REGION"
    return label, hype_lemma, exact_match
  if last_word in {"school", "facilities", "facility", "center", "square", "rd", "street", "way", "blvd", "ave", "avenue"}:
    label = "LOC"
    return label, hype_lemma, exact_match
  if last_word in {"court", "corp", "corporation", "co", "company", "ltd", "llc", "llp", "incorp.", "incorporated"}:
    label = "ORG"
    return label, hype_lemma, exact_match
  if last_word in {"feet", "barrells", "hours", "hour"}:
    label = "QUANTITY"
    return label, hype_lemma, exact_match
  if last_word in {"percent"} or ent.endswith("%"):
    label = "PERCENT"
    return label, hype_lemma, exact_match
  if (ent[0] in "1234567890" and "U.S" in ent) or ("U.S.C." in ent) or last_word in {"act", "code", "statute", "regulation", "regulations"}:
    label = "LAW"
    return label, hype_lemma, exact_match
  if ent.lower().strip() in {"death", "disease", "symptom", "deaths", "diseases", "symptoms", "disorder", "disorders"}:
    return label, hype_lemma, exact_match
  ent = ent.replace("Symptom", "Disease").replace("symptom", "disease").replace("syndrome", "disease").replace("Syndrome", "Disease").replace("Death", "Disease").replace("death", "disease").replace("Cancer", "Disease").replace("cancer", "disease").replace("Disorder", "Disease").replace("disorder", "disease")

  #now do wordnet checking
  ss_list = list(en_wn.synsets(ent, pos='n'))
  if ss_list:
    exact_match = True
  if not ss_list and "'s " in ent:
    ss_list = list(en_wn.synsets(ent.split("'s ")[-1].strip(), pos='n'))
  if not ss_list and ent[-1] not in "1234567890":
    ss_list = list(en_wn.synsets(ent.split()[-1], pos='n'))
  if not ss_list and ent[-1] in "1234567890":
    ss_list = list(en_wn.synsets(ent.split()[0], pos='n'))
  if not ss_list:
      ent2 = ent.replace("'s", " 's").replace("(", " ( ").replace(")", " ) ").replace("-", " - ").replace(":", " : ")
      for s in ["(", ")", ":", "-", "of", "for", "in", "on", "from", "at"]:
        ent2 = ent2.replace(" "+s+" ", "@#@")
        ent2 = ent2.replace(" "+s.upper()+" ", "@#@")
      if "@#@" in ent2:
        first_word = ent2.strip().split("@#@")[0]
        ss_list = list(en_wn.synsets(first_word, pos='n'))
  orig_label = label
  if ss_list:
    ss = ss_list[0]
    hype = ss.hypernyms()
    hype_lemma = []
    if hype:
      hype_lemma = hype[0].lemmas()
      hype_hype = hype[0].hypernyms()
      while hype_hype:
        hype_lemma.extend(hype_hype[0].lemmas())
        hype_hype = hype_hype[0].hypernyms()
    if ent[0] in "1234567890" and ('unit' in hype_lemma or 'property' in hype_lemma):
      label = "QUANTITY"
    elif 'way' in hype_lemma or 'land' in hype_lemma or 'construction' in hype_lemma or 'route' in hype_lemma or 'road' in hype_lemma or 'facility' in hype_lemma or 'location' in hype_lemma:
      label = "LOCATION"
    elif 'period' in hype_lemma:
      label = "DATE"
    elif 'rule' in hype_lemma or 'duty' in hype_lemma or 'law' in hype_lemma:
      label = "LAW"
    elif 'region' in hype_lemma:
      label = "REGION"
    elif 'natural language' in hype_lemma or "religion" in hype_lemma or 'denizen' in hype_lemma:
      label = "TRAIT"
    elif 'person' in hype_lemma or 'worker' in hype_lemma or 'professional' in hype_lemma:
      label = "PERSON"
    elif 'event' in hype_lemma or 'phenomenon' in hype_lemma:
      label = "EVENT"
    elif 'organization' in hype_lemma or 'social group' in hype_lemma or 'group action' in hype_lemma:
      label = "ORG"
    elif 'creation' in hype_lemma or 'art' in hype_lemma or 'agreement' in hype_lemma or 'content' in hype_lemma or 'legal document' in hype_lemma or 'writing' in hype_lemma or 'record' in hype_lemma:
      label = "DOCUMENT_OR_ARTIFACT"
    elif 'time unit' in hype_lemma:
      label = "TIME"
    elif 'number' in hype_lemma or 'rank' in hype_lemma:
      label = "NUMBER"
    elif 'quantity' in hype_lemma or 'rate' in hype_lemma:
      label = "QUANTITY"
    elif 'microorganism' in hype_lemma or 'harm' in hype_lemma or  'death' in hype_lemma or 'unhealthiness' in hype_lemma or \
          'disorder' in hype_lemma or 'disease' in hype_lemma or 'illness' in hype_lemma or \
          'pathogen' in hype_lemma or 'symptom' in hype_lemma or 'mental condition' in hype_lemma or 'enlargement' in hype_lemma:
      label = "DISEASE_OR_HARM"
    elif 'chemical' in hype_lemma or 'drug' in hype_lemma or 'substance' in hype_lemma or 'molecule' in hype_lemma:
      label = "CHEMICAL"
    elif 'monetary unit' in hype_lemma:
      label = "MONEY"
    elif 'commodity' in hype_lemma and exact_match:
      label = "PRODUCT"
    elif 'animal' in hype_lemma and exact_match:
      label = "ANIMAL"
    elif 'plant' in hype_lemma and exact_match:
      label = "PLANT"

  return label, hype_lemma, exact_match

def templatize(text2, key, label):
  text2 = " "+text2+" "
  key= key.strip("{} ")
  text2 = text2.replace(" "+key,' {'+label+'} ').strip()
  text2 = text2.replace(key+" ",' {'+label+'} ').strip()
  return text2

# do reverb like extraction NER obj verb subj
def get_verb_relation(text):
  #TODO: do RB and "-" and ":" before verb
  #TODO: do basic relationships such as "known as", "defined as", "included in", "is a", "known for"
  doc = spacy_nlp(text)
  verb_relationship = ""
  orig_verb = ""
  prev_be = ""
  for token in doc:
    if token.lemma_ in {'do', 'be', 'have'}:
        prev_be = token.lemma_+"_"
        continue
    if token.tag_.startswith("VB") and token.tag_ not in {"VBZ", }:
      orig_verb = token.text
      verb_relationship = prev_be+str(token.lemma_)
      prev_be = ""
      continue
    if verb_relationship:
      if token.tag_ in {"RB","IN"}:
        orig_verb += " "+token.text
        verb_relationship += "_"+str(token.lemma_)
        prev_be = ""
        break
      else:
        break
  obj = None
  if orig_verb:
    obj = text.split(orig_verb,1)[-1]
    obj = strip_right_stopwords(obj)
    obj = strip_left_stopwords(obj)
  return verb_relationship, orig_verb, obj

strip_chars = ',~!@#^&*()-_=+" \n<>\/|:[]'

def basic_cleanup_word(ent):
  ent = ent.strip(strip_chars)
  ent = strip_right_stopwords(ent)
  ent = strip_left_stopwords(ent)
  ent = ent.strip(strip_chars)
  if ent.startswith("s ") or ent.startswith("'s ") or ent.startswith("’s ") or ent.startswith(". "):
    ent = ent[2:].strip()
  if ent.endswith("’s") or ent.endswith("'s"):
    ent = ent[:-2].strip()
  for word in ["A", "An", "The", "Mr.", "Mrs.", "Dr.",]:
    if ent.startswith(word+" "):
      ent = ent.split(word+" ",1)[-1].strip()
      break
  for word in numbering_list:
    if ent.startswith(word+" "):
      ent = ent.split(word+" ",1)[-1].strip()
      break
  return ent

def ner_rel_template_extract(text, min_ner_len=3, length_for_rel=200):
  global spacy_nlp, sci_spacy

  orig_text = text
  text2 = text.replace("{", "-lbracket-").replace("}", "-rbracket-")
  text = "\n"+text+"\n"
  ner_cnt = {}
  ents = {}
  # spacy doesn't chatch some NER
  #todo - do dictionary
  #text = text.replace("The People\'s Republic of China", "China").replace("The People\'s Republic", "China").replace("People\'s Republic", "China")


  # gather disease and chemical NER
  text_lower = text.lower()
  doc =sci_spacy(text)
  total_ents = len(doc.ents)
  if 'disease' in text_lower or 'bio' in text_lower or 'medic' in text_lower or total_ents > 3:
    chunks0 = dict([(ent.text, ent.label_) for ent in doc.ents])
  else:
    chunks0 = {}

  # gather NER
  doc =spacy_nlp(text)
  chunks = dict([(ent.text, ent.label_) for ent in doc.ents if ent.text not in chunks0])
  # gather other noun chunks
  chunks2 = dict([(ent.text.strip(), "") for ent in doc.noun_chunks if ent.text not in chunks0 and ent.text not in chunks])
  chunks0 =  list(chunks0.items())
  chunks =  list(chunks.items())
  chunks2 =  list(chunks2.items())
  chunks0.sort(key=lambda a: len(a[0]), reverse=True)
  chunks.sort(key=lambda a: len(a[0]), reverse=True)
  chunks.sort(key=lambda a: len(a[0]), reverse=True)
  cnt_chunks = Counter([c[0].strip() for c in chunks0 + chunks + chunks2])

  for ent_label in  chunks0 + chunks + chunks2:
    ent, label = ent_label
    if ent.count("_") > 5: continue
    subset = [a for a in ents if a.lower() in ent.lower()]
    ent = " "+ent+" "
    if subset:
      for s in subset:
        ent = ent.replace(" "+s+" ", "@#@")
        ent = ent.replace(" "+s.upper()+" ", "@#@")
        ent = ent.replace(" "+s.lower()+" ", "@#@")
    all_ents = ent.strip().split("@#@")
    if len(all_ents) == 1:
      ent2 = ent.replace("'s", " 's").replace("(", " ( ").replace(")", " ) ").replace("-", " - ").replace(":", " : ")
      for s in ["(", ")", ":", "-", "'s", "of", "for", "in", "on", "from", "at"]:
        ent2 = ent2.replace(" "+s+" ", "@#@")
        ent2 = ent2.replace(" "+s.upper()+" ", "@#@")
      if "@#@" in ent2:
        all_ents = all_ents + ent2.strip().split("@#@")
    orig_label = label
    for ent in all_ents:
      label = orig_label
      ent = ent.strip()
      cnt = cnt_chunks.get(ent.strip(), 0)
      #print (ent, label )
      ent = basic_cleanup_word(ent)
      if not ent: continue
      if is_num(ent): continue
      if ent in ents or ent.lower()  in stopwords_set or any(a for a in ents if a.lower() in ent.lower() or ent.lower() in a.lower()): continue

      # cleanup label names
      if label == "DISEASE":
        label = "DISEASE_OR_HARM"
      elif not label and '"' in ent:
          label = "DOCUMENT_OR_ARTIFACT"
      elif label in {'ORDINAL', 'CARDINAL'}:
        label = "NUMBER"
      elif label in {'LOC', 'FAC'}:
        label = "LOCATION"
      elif label == 'GPE':
        label = "REGION"
      elif label in {'LANGUAGE','NORP'}:
        label = "TRAIT"
      elif label == "WORK_OF_ART":
        label = "DOCUMENT_OR_ARTIFACT"
      last_word = ent.split()[-1]
      if (cnt > 4 and len(ent.strip(" .")) > 4 and ((" " in ent and ent[0] == ent[0].upper() and last_word[0] == last_word[0].upper()) or label in {"NUMBER", "PERCENT", "QUANTITY", "DATE", "TIME", "MONEY"})) or \
          (len(ent) >= min_ner_len  and ((" " in ent and ent[0] == ent[0].upper() and last_word[0] == last_word[0].upper()) or label in {"NUMBER", "PERCENT", "QUANTITY", "DATE", "TIME", "MONEY"})) or \
          (ent == ent.upper() and cnt > 3) or \
          (ent == ent.upper() and len(ent.strip(" .")) > 2):
        label2, hype_lemma, exact_match =  guess_ner_label(ent)
        if (label in {"NUMBER", "PERCENT", "QUANTITY", "DATE", "TIME", "MONEY"} and label2 in {"NUMBER", "PERCENT", "QUANTITY", "DATE", "TIME", "MONEY"}) or \
          (label in {"ORG", "EVENT", "DOCUMENT_OR_ARTIFACT"} and label2 in {"ORG", "EVENT", "DOCUMENT_OR_ARTIFACT"}) or \
          (label in {"LOCATION", "PERSON", "REGION"} and label2 in {"LOCATION", "PERSON", "REGION"})  or \
          (label in {"DISEASE_OR_HARM", "CHEMICAL"} and label2 in {"DISEASE_OR_HARM", "CHEMICAL"})  or \
          (label in {"ORG", "LOCATION", "REGION"} and label2 in {"ORG", "LOCATION", "REGION"}) or \
          (label in {"TRAIT", "PERSON", "ORG"} and label2 in {"TRAIT", "PERSON", "ORG"}) or \
          (label in {"TRAIT", "LOCATION", "REGION"} and label2 in {"TRAIT", "LOCATION", "REGION"}) or \
          (label in {"DOCUMENT_OR_ARTIFACT", "LAW"} and label2 in {"DOCUMENT_OR_ARTIFACT", "LAW"}) or \
          (label in {"ORG", "PERSON"} and label2 in {"ORG", "PERSON"} and ent[-1] == 's') or \
          (label in {"ORG", "DOCUMENT_OR_ARTIFACT"} and label2 in {"ORG", "DOCUMENT_OR_ARTIFACT"}) or \
          (not label2 and label and label not in {"DISEASE_OR_HARM", "CHEMICAL"} ) or \
          (label == label2 and label2) or \
          not hype_lemma:
          pass
        elif (not label and label2):
          label = label2
        elif (exact_match and label2 and (" " in ent or ent[0] == ent[0].upper())):
          print(ent, label, label2, "->",hype_lemma)
          label = label2
        else:
          print ('NO MATCH', ent, label, label2, "->",hype_lemma)
          continue
        if label == "WORK_OF_ART" and (" " not in ent or "\"" not in ent or "." not in ent): continue
        if label:
          ents[ent] = label


  #create labels with numbers at the end (e.g., PERSON_1)
  ents = [[a[0], a[1]] for a in ents.items()]
  ents.sort(key=lambda a: len(a[0]), reverse=True)
  ret_items = []
  idx = 0
  for st_label in ents:
    st, label = st_label
    st = st.replace("'s", "")
    #we are not doing NER for code
    if "->" in st or "{" in st or "}" in st:
      text = text.replace(st,' ')
      continue
    if st in text:
      label2 = ""
      matched = [a for a in ret_items if st.lower() in a[0].lower()]
      if matched:
        label3 = matched[0][1]
        if label in label3:
          label2 = label3
      if not label2:
        ner_cnt[label] = ner_cnt.get(label, 0)
        ner_cnt[label] += 1
        label2 = label+'_'+str(ner_cnt[label])
      text2 = text2.replace(st,' {'+str(idx)+'} ')
      ret_items.append((st,label2))
      idx += 1
      text = text.replace(st,' ')

  #word parts -> LABEL. for things like first names, NER might not catch these
  idx = len(ret_items)
  ret = dict(ret_items)
  for idx0, key_label in enumerate(ret_items):
    key, label = key_label
    text2 = text2.replace(" "+key,' {'+str(idx0)+'} ')
    text2 = text2.replace(key+" ",' {'+str(idx0)+'} ')
    if " " in key:
      for st in key.split():
        st = st.strip("123456789")
        if not st: continue
        if st in ret or st.lower() in stopwords_set: continue
        if len(st.strip(".")) > 3:
          if " "+st in text:
            text2 = text2.replace(" "+st,' {'+str(idx)+'} ')
            ret_items.append((st, label))
            idx += 1
          elif st+" " in text:
            text2 = text2.replace(st+" ",' {'+str(idx)+'} ')
            ret_items.append((st, label))
            idx += 1
          elif " "+st.upper() in text:
            text2 = text2.replace(" "+st.upper(),' {'+str(idx)+'} ')
            ret_items.append((st.upper(), label))
            idx += 1
          elif st.upper()+" " in text:
            text2 = text2.replace(st.upper()+" ",' {'+str(idx)+'} ')
            ret_items.append((st.upper(), label))
            idx += 1
  for idx, key_label in enumerate(ret_items):
    text2 = text2.replace("{"+str(idx)+"}", "{"+key_label[-1]+"}")

  # cleanup the templatized text
  text2 = text2.replace("  {", " {").replace("}  ", "} ")
  text2 = text2.replace("\n {", "\n{").strip()
  text2 = text2.replace("} s ", "} ").replace("} es ", "} ").replace("} ies ", "} ").\
    replace("} s, ", "} ,").replace("} es, ", "} ,").replace("} ies, ", "} ,").\
    replace("} s. ", "} .").replace("} es. ", "} .").replace("} ies. ", "} .")
  text3 = []
  prev_word = ""
  prev_prev_word = ""
  #let's maximaize the words that are not slots.
  for word in text2.split(" "):
    if (word and word[-1] == "}" and word == prev_word):
      continue
    if word and word[-1] == "}" and word == prev_prev_word:
      test3 = text3[:-1]
      continue
    if word and prev_word and word[-1] == "}"  and prev_word[-1] == "}":
      continue
    if word and prev_word and prev_prev_word and word[-1] == "}" and prev_word in "-:," and prev_prev_word[-1] == "}":
      test3 = text3[:-1]
      continue
    text3.append(word)
    prev_prev_word = prev_word
    prev_word = word
  text2 = " ".join(text3)

  ret = dict(ret_items)
  rels =[]

  # now gather potential relationships
  args = dict([(b, "{"+a+"}") for a, b in ret_items ])
  if ret_items:
    text3 = text2.format(**args)
    for entity, label in ret.items():
      if "DATE" in label or "NUMBER" in label or "TRAIT" in label or "QUANTITY" in label or "PERCENT" in label or "TIME" in label: continue
      if "{"+entity+"}" not in text3:
        continue
      for idx, text5 in enumerate(text3.split("{"+entity+"}")):
        if idx == 0: continue
        text5 = entity+" " + text5
        text5 = text5.replace("{", " ").replace("}", " ").replace("  ", " ")
        if len(text5) > length_for_rel+len(entity)+2:
          text5 = text5[:length_for_rel+len(entity)+2]
        text5 = text5.replace("\n", " ")
        text5 = text5.replace(".\"", "\" . ")
        if ". " in text5[len(entity):]:
          text5 = text5[:len(entity)] + text5[len(entity):].split(". ",1)[0]+" "
        if ", " in text5[len(entity):]:
          text5 = text5[:len(entity)] + text5[len(entity):].split(", ",1)[0]+" "
        text5 = text5.replace("\"", " \" ")
        quote = ""
        if text5.count("\"") ==2:
          quote = text5.split("\"")[1]
          text5 = text5 = text5.replace(text5.split("\"")[1], "{TEXT}")
        else:
          text5 = text5.split("\"")[0]
        if text5[-1] == '"':
          text5 = text5+" "
        if text5[-1] != ' ':
          text5 = " ".join(text5.split(" ")[:-1])
        text5 = text5.strip()
        if text5[len(entity):].count(" ") < 2: continue
        rel, orig_verb, obj = get_verb_relation(text5)
        if quote:
          obj = obj.replace("{TEXT}", quote)
        if obj and obj.strip() and rel not in entity and orig_verb not in entity:
          rels.append ([entity, rel, obj])
  for idx, rec in enumerate(rels):
    for key, label in list(ret.items()):
      rec[0] = templatize(rec[0], key, label)
    for key, label in list(ret.items()):
      rec[-1] = templatize(rec[-1], key, label)


  return ret, text2.replace("-lbracket-", "{").replace("-rbracket-", "}"), rels
