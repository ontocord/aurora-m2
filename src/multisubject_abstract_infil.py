#@title multi subject abstract infil


import os
if not os.path.exists("/content/drive"):
  from google.colab import drive
  drive.mount('/content/drive')

if not os.path.exists("/content/drive/Shareddrives/ontocord_llc/unified/unified_abstract_infill.jsonl"):
    !mkdir -p /content/drive/Shareddrives/ontocord_llc/unified
    !mkdir -p /content/drive/Shareddrives/ontocord_llc/unified_out/
    !cp /content/drive/Shareddrives/LAION_archived/OIG/unified_abstract_infill.jsonl /content/drive/Shareddrives/ontocord_llc/unified
%cd /content/drive/Shareddrives/ontocord_llc/unified_out/

if False:

  !rm -- *.jsonl
  mmlu_list = ['CHEBI', 'Austrian history', 'Belgian history', 'Bulgarian history', 'Croatian history', 'Cypriot history', 'Czech history', 'Danish history', 'Estonian history', 'Finnish history', 'French history', 'German history', 'Greek history', 'Hungarian history', 'Irish history', 'Italian history', 'Latvian history', 'Lithuanian history', 'Luxembourg history', 'Maltan history', 'Dutch history', 'Polish history', 'Portugese history', 'Romanian history', 'Slovakian history', 'Slovenian history', 'Spanish history', 'Swedish history',
               'human ecology', 'human anatomy', 'asian history', 'african history', 'south american history', 'middle eastern history', 'comparative religion', 'comparative law', 'organizational behavior', 'atomic data', 'cognitive-behavioral therapy', 'data structures', 'political ideologies', 'world religions', 'number theory', 'biomedical engineering', 'elementary particles', 'branding strategies', 'aesthetics', 'international trade law', 'topology', 'classical antiquity', 'socioeconomic aspects of aging', 'social change', 'fluid dynamics', 'justice theory', 'genomics', 'environmental issues', 'gravitation', 'electricity and magnetism', 'data collection', 'moral disputes', 'reinforcement learning', 'programming basics', 'confidence intervals', 'inorganic chemistry', 'forensic computing', 'circuit analysis', 'geometry', 'conceptual physics', 'cold war', 'mechanical waves','project management', 'viral genetics', 'plasmas', 'business ethics', 'evolution', 'human body', 'real analysis', 'arithmetic', 'environmental chemistry', 'clinical psychology', 'immunology', 'consumer behavior', 'longevity', 'forecasting', 'data encryption', 'relativity', 'energy conservation', 'philosophical debates', 'managerial accounting', 'reproductive system', 'renaissance art', 'world systems theory', 'media relations', 'critical legal studies', 'correlation', 'statistics', 'surgery', 'algorithm', 'networks', 'ageism', 'algebra of random variables', 'internet basics', 'high school european history', 'monetary policy', 'human reproductive systems', 'decision-making processes', 'economic cycles', 'bronze age', 'nervous system', 'jurisprudence', 'organic chemistry', 'epidemiology', 'brand management', 'case law',  'marketing', 'post hoc ergo propter hoc', 'neutrinos', 'national policy', 'college medicine', 'metaphysics', 'epistemology', 'human geography', 'vaccine development', 'emergency medicine', 'jurisprudence theories', 'orthopedics', 'computer network', 'respiratory anatomy', 'family planning', 'electrical engineering', 'geopolitical analysis', 'industrialization', 'robotics', 'psychiatry', 'atomic processes', 'human rights', 'stoichiometry', 'nutritional epidemiology', 'population genetics', 'biogerontology', 'fluid mechanics', 'blood brain', 'abstract algebra', 'miscellaneous', 'high school psychology', 'newtonian', 'endocrine system', 'cell biology', 'astrochemistry', 'judicial systems', 'learning theories', 'medical imaging', 'survival analysis', 'world war i', 'cardiovascular anatomy', 'college biology', 'religious philosophy', 'moral scenarios', 'nuclear physics', 'dermatology', 'cognitive dissonance', 'mitosis', 'computer graphics', 'symbolic logic', 'international alliances', 'developmental stages', 'cardiovascular system', 'problem-solving skills', 'formal logic', 'utilitarianism', 'trade agreements', 'galois theory', 'regression analysis', 'experimental design',  'management', 'materials chemistry', 'religious conflicts', 'set theory', 'diplomatic strategies', 'environmental statistics', 'prehistoric europe', 'globalization', 'diet-related diseases', 'sales and marketing', 'cultural diversity', 'internet security', 'hypothesis testing', 'straw man', 'fascism', 'ethical conflicts', 'healthcare for elderly', 'high school physics', 'market segmentation', 'professional accounting', 'acceleration of',  'economic indicators', 'unsupervised learning', 'human genome project', 'childhood nutrition', 'sexual health education', 'stakeholder engagement', 'internal medicine', 'conflict resolution', 'ethics', 'time series analysis', 'economic theories', 'software development', 'DNA replication', 'world conflicts', 'fiscal policy', 'production theory', 'renaissance', 'cryptography', 'black hole', 'clinical trials', 'colonialism', 'infant mortality', 'european union', 'whistleblowing', 'compliance and risk management', 'neural networks', 'world economies', 'general practice', 'corporate communications', 'theology', 'developmental biology', 'foreign aid', 'algorithms', 'senior health', 'professional psychology', 'sexual development',  'enlightenment', 'trigonometry', 'sociology',  'circulatory system', 'market structures', 'geology', 'public relations', 'ecology', 'acids and bases', 'population studies', 'socialism', 'stone age', 'spectroscopy', 'market analysis', 'totalitarianism', 'human-computer interaction', 'statistical mechanics', 'bayesian inference', 'antiviral treatments', 'peace negotiations', 'labor economics', 'artificial intelligence', 'business law', 'major world religions', 'physical geography', 'newtonian mechanics', 'food safety', 'political parties', 'unemployment', 'climate zones', 'confirmation bias', 'national security', 'data interpretation', 'criminal law', 'mathematical reasoning', 'electrodynamics', 'historical demographics', 'diagnostic methods', 'power systems', 'periodic table', 'basic geometry', 'enzymology', 'modern era', 'cognitive psychology', 'democratic process', 'tax accounting', 'trade economics', 'atomic physics', 'labor ethics', 'exploration', 'prehistoric culture', 'high school computer science', 'behavioral therapy', 'obstetrics and gynecology', 'econometrics', 'social psychology', 'sexual ethics', 'operating systems', 'nutrition', 'cosmology', 'augmented reality', 'threat models', 'forensic accounting', 'civil war', 'supramolecular chemistry', 'semiconductors', 'molecular processes',  'map reading', 'deontology', 'philosophy',  'college chemistry', 'war crimes', 'social structures', 'ancient civilizations', 'statistical methods',  'databases', 'reproductive anatomy', 'industrial revolution', 'us foreign policy', 'public policies', 'social theory', 'algebra i', 'space race', 'astroparticle physics', 'skeletal system', 'analytical chemistry', 'quantum mechanics', 'high school government and politics', 'medieval period', 'data mining', 'panel data', 'algebra', 'international advocacy', 'patient care', 'prehistory', 'risk management', 'genetic disorders', 'epigenetics', 'human sexuality', 'sustainability', 'auditing', 'internet age', 'property law', 'conflict resolution strategies', 'descriptive statistics', 'treatment planning', 'multivariate analysis', 'photosynthesis', 'computational thinking', 'experimental psychology', 'natural law', 'high school chemistry', 'ethical hacking', 'galaxies',  'logical fallacies','clinical knowledge', 'correlation coefficient', 'virtue ethics', 'logic', 'contemporary history', 'control systems', 'cultural norms', 'high school biology', 'prehistoric art', 'chemical reactions', 'optics', 'econometric models', 'quick sort', 'syllogistic logic', 'module theory', 'photonics', 'virology', 'management science', 'intelligence analysis', 'urinary system', 'biochemistry', 'public health nutrition', 'elasticity', 'lymphatic system', 'conflict theory', 'radiology', 'specialized medicine', 'radiochemistry', 'software engineering', 'american revolution', 'electromagnetism', 'game theory', 'theorem', 'college physics', 'inflation', 'physical chemistry', 'hydrodynamics', 'respiratory system', 'numerical analysis', 'meiosis', 'deductive reasoning',   'microelectronics', 'resource allocation', 'election processes', 'anatomy', 'dark matter', 'health systems', 'reproductive rights', 'constitutional principles', 'virtual reality', 'sexual health', 'global facts', 'civic duties', 'population statistics', 'legal studies', 'exoplanets', 'legal positivism', 'epidemiology of viruses', 'iron age', 'abnormal psychology', 'optics', 'international sanctions', 'middle ages', 'complex analysis', 'machine learning', 'STDs prevention', 'high school mathematics', 'nuclear chemistry', 'topography', 'high school geography', 'data analysis techniques', 'gene therapy', 'progressive era', 'oncology', 'world capitals', 'strategic management', 'space exploration', 'aging process', 'high school microeconomics', 'imperialism', 'geopolitical facts',  'international law', 'digestive anatomy', 'moral judgments', 'differential equations', 'contract law', 'neuropsychology', 'chemical bonds', 'jurisprudence ethics', 'cloud computing', 'mechanics', 'equation of state', 'decimals', 'proteomics', 'quantum chemistry', 'gender identity', 'pediatrics', 'magnetohydrodynamics', 'physiology', 'stakeholder theory', 'age-related diseases', 'oceanography', 'pre-calculus', 'nebulae', 'practical ethics', 'legal theory', 'tort law', 'anesthesiology', 'professional medicine', 'mathematical proof', 'cybersecurity', 'digital marketing', 'feudalism', 'high school macroeconomics', 'constitutional law', 'comparative law', 'molecular biology', 'defense strategies', 'astronomy', 'preventive medicine', 'nuclear reactions', 'high school us history', 'particle physics', 'maritime laws', 'personality theories', 'homomorphisms', 'planets', 'ad hominem', 'international courts', 'polynomials', 'modal logic', 'welfare economics', 'medicinal chemistry', 'operations management', 'integumentary system', 'crisis management', 'fallacies', 'viral diseases', 'treaties', 'world war ii', 'chemical kinetics', 'nanotechnology', 'relativistic', 'psychoanalysis', 'fair trade', 'professional law', 'causal inference', 'government branches', 'cultural sociology', 'space age', 'probability', 'computer security', 'analysis of variance', 'pharmacology', 'elementary mathematics', 'calculus', 'world wars', 'security protocols', 'genetics', 'truth tables', 'supervised learning', 'psychological assessment', 'magnetic fields', 'humanistic psychology', 'economic policy', 'communism',  'cardiac', 'gerontology', 'taxonomy', 'philosophical logic', 'great depression', 'quantum concepts', 'cross-border laws', 'sample survey methods', 'consumer theory', 'meteorology', 'disease management', 'corporate responsibility', 'linear algebra', 'gastrointestinal', 'intestinal', 'corporate governance', 'digestive system', 'limbic system', 'democracy', 'moral reasoning', 'civil rights movement', 'religious practices', 'thermodynamics', 'microbiology', 'cardiology', 'psychological disorders', 'high school world history', 'biological classification', 'modern physics',  'financial accounting', 'dietary guidelines', 'muscular system', 'simple algebra', 'dark energy', 'medical genetics', 'statistical inference', 'normal distribution', 'security studies', 'sexual orientation', 'cognitive aging', 'human aging', 'algebraic structures', 'logical connectives', 'high school statistics',  'diplomacy', 'pathology', 'market failure', 'solid state physics', 'cartography', 'quantifiers', 'ethical dilemmas', 'network security', 'college mathematics', 'civil rights', 'malware analysis', 'nationalism', 'deep learning',  'combinatorics', 'black holes', 'group theory', 'college computer science']

  for topic in mmlu_list:
    if 'Us ' in topic:
      topic = topic.replace("Us ", "US ")
    if topic.endswith("es"):
      topic = topic[:-2]
    elif  topic.endswith("s"):
      topic = topic[:-1]
    elif  topic.endswith("y"):
      topic = topic[:-1]
    topic = topic.strip("- ")
    print (topic)
    file = topic.replace(" ", "_")
    file = f"{file}.jsonl"
    topic2 = "'"+topic+"'"
    !grep $topic2 /content/drive/Shareddrives/ontocord_llc/unified/unified_abstract_infill.jsonl > $file
    if " and " not in topic:
      topic = " ".join([t[0].upper()+t[1:] for t in topic.split()])
    file = topic.replace(" ", "_")
    file = f"{file}.jsonl"
    topic2 = "'"+topic+"'"
    !grep $topic2 /content/drive/Shareddrives/ontocord_llc/unified/unified_abstract_infill.jsonl > $file
    if topic.upper() == topic: continue
    if topic.lower() != topic:
      topic = topic.lower()
      print (topic)
      file = topic.replace(" ", "_")
      file = f"{file}.jsonl"
      topic2 = "'"+topic+"'"
      !grep $topic2  /content/drive/Shareddrives/ontocord_llc/unified/unified_abstract_infill.jsonl > $file
    if " and " in topic:
      topic3 = topic.split(" and ")
      for topic in topic3:
        topic = topic.replace("abstract", "").replace("formal", "").replace("high school", "").replace("college", "").replace("professional", "").replace("US", "").replace("human", "").replace("international", "").replace("business" , "").replace("elementary", "").replace("conceptual", "").strip()
        if len(topic) > 4:
          print (topic)
          file = topic.replace(" ", "_")
          file = f"{file}.jsonl"
          topic2 = "'"+topic+"'"
          !grep $topic2 /content/drive/Shareddrives/ontocord_llc/unified/unified_abstract_infill.jsonl  > $file
    elif " " in topic:
      topic = topic.replace("abstract", "").replace("formal", "").replace("high school", "").replace("college", "").replace("professional", "").replace("US", "").replace("human", "").replace("international", "").replace("business" , "").replace("elementary", "").replace("conceptual", "").strip()
      topic= topic.strip("- ")
      if len(topic) > 4:
        print (topic)
        file = topic.replace(" ", "_")
        file = f"{file}.jsonl"
        topic2 = "'"+topic+"'"
        !grep $topic2  /content/drive/Shareddrives/ontocord_llc/unified/unified_abstract_infill.jsonl > $file

import glob, json, random

names = ['Aimee',
 'Molly',
 'Denis',
 'Gerard',
 'Gillian',
 'Karen',
 'Rita',
 'Rachael',
 'Julia',
 'Elliott',
 'Kate',
 'Elaine',
 'Mark',
 'Dylan',
 'Jack',
 'Leanne',
 'Mitchell',
 'Andrea',
 'Leah',
 'Kathryn',
 'Michael',
 'Katy',
 'Joel',
 'Joe',
 'Stanley',
 'Joan',
 'Roger',
 'Mohammed',
 'Dennis',
 'Pauline',
 'Steven',
 'Lawrence',
 'Charles',
 'Marian',
 'Amber',
 'Jodie',
 'Fiona',
 'Dorothy',
 'Damien',
 'Gerald',
 'Jay',
 'Joyce',
 'Eileen',
 'Donald',
 'Keith',
 'Louis',
 'Callum',
 'Ruth',
 'Owen',
 'Carl',
 'Joanne',
 'Ian',
 'Rachel',
 'Edward',
 'Frances',
 'Denise',
 'Rosemary',
 'Dawn',
 'Anne',
 'Bethan',
 'Carly',
 'Christian',
 'Angela',
 'Teresa',
 'Beverley',
 'Ashley',
 'Jacob',
 'Paul',
 'Leigh',
 'Matthew',
 'Jenna',
 'Shannon',
 'Sian',
 'Iain',
 'Ann',
 'Rhys',
 'Victoria',
 'Amanda',
 'Douglas',
 'Stephen',
 'Philip',
 'Lesley',
 'Stephanie',
 'Amelia',
 'Jemma',
 'Melanie',
 'Daniel',
 'Damian',
 'Brian',
 'Connor',
 'Scott',
 'Maria',
 'Ryan',
 'Lindsey',
 'Terry',
 'Josephine',
 'Joseph',
 'George',
 'Howard',
 'Mohammad',
 'Nicola',
 'Grace',
 'Nathan',
 'Leslie',
 'Clive',
 'Valerie',
 'Jonathan',
 'Colin',
 'Luke',
 'Jean',
 'Guy',
 'Gordon',
 'Sarah',
 'Anna',
 'Bethany',
 'Cheryl',
 'Tom',
 'Dale',
 'Jasmine',
 'Amy',
 'Dean',
 'Justin',
 'Christopher',
 'Declan',
 'Yvonne',
 'Lydia',
 'Abdul',
 'Abigail',
 'Kyle',
 'Donna',
 'Toby',
 'Irene',
 'Shaun',
 'Abbie',
 'Benjamin',
 'Catherine',
 'Samantha',
 'Georgia',
 'Claire',
 'Alan',
 'Ben',
 'Kerry',
 'Jessica',
 'Henry',
 'Neil',
 'Emma',
 'Terence',
 'Zoe',
 'Lynn',
 'Jeffrey',
 'Wayne',
 'Kayleigh',
 'Suzanne',
 'David',
 'Joanna',
 'Hilary',
 'Megan',
 'Brett',
 'Robert',
 'Graham',
 'Ronald',
 'Deborah',
 'Lorraine',
 'Sheila',
 'Stewart',
 'Barbara',
 'Marilyn',
 'Eric',
 'Sean',
 'Ellie',
 'Rosie',
 'Jade',
 'Jacqueline',
 'Kenneth',
 'Victor',
 'Peter',
 'Max',
 'Patrick',
 'Francis',
 'Derek',
 'Michelle',
 'Kieran',
 'Nicole',
 'Katie',
 'Marie',
 'Kirsty',
 'Annette',
 'Reece',
 'Allan',
 'Gemma',
 'Wendy',
 'Lynne',
 'Andrew',
 'James',
 'Ashleigh',
 'Karl',
 'Bradley',
 'Maurice',
 'Kelly',
 'Frank',
 'Gregory',
 'Patricia',
 'Raymond',
 'Lisa',
 'Natalie',
 'Margaret',
 'Marion',
 'Jordan',
 'Gavin',
 'Olivia',
 'Garry',
 'Elizabeth',
 'Josh',
 'Richard',
 'Julian',
 'Phillip',
 'Craig',
 'Pamela',
 'Stacey',
 'Dominic',
 'Georgina',
 'Danielle',
 'Alison',
 'Liam',
 'Nigel',
 'Judith',
 'Natasha',
 'Leonard',
 'Adam',
 'Hayley',
 'Frederick',
 'Martin',
 'Gareth',
 'Mary',
 'Oliver',
 'Caroline',
 'Eleanor',
 'Sandra',
 'Bernard',
 'Ross',
 'Conor',
 'Roy',
 'Rebecca',
 'Beth',
 'Christine',
 'Aaron',
 'Norman',
 'Clare',
 'Hugh',
 'Barry',
 'Alexandra',
 'Melissa',
 'Naomi',
 'Timothy',
 'Harriet',
 'Hazel',
 'Bruce',
 'Shane',
 'Alex',
 'Paige',
 'Chelsea',
 'Sharon',
 'Jason',
 'Simon',
 'Trevor',
 'Charlotte',
 'Sam',
 'Janice',
 'Paula',
 'Tracey',
 'Louise',
 'Brenda',
 'Debra',
 'Lee',
 'Hannah',
 'Mandy',
 'Lewis',
 'Ricky',
 'Holly',
 'Adrian',
 'Jill',
 'Susan',
 'Jamie',
 'Marcus',
 'Tracy',
 'Tina',
 'Marc',
 'Malcolm',
 'Stuart',
 'Laura',
 'Elliot',
 'Helen',
 'Katherine',
 'Jayne',
 'Nicholas',
 'Vanessa',
 'Darren',
 'Graeme',
 'Albert',
 'Jake',
 'Janet',
 'Cameron',
 'Kevin',
 'Francesca',
 'Chloe',
 'Alice',
 'Sylvia',
 'Billy',
 'Joshua',
 'Sally',
 'Lynda',
 'Heather',
 'Leon',
 'Geraldine',
 'Kim',
 'Antony',
 'Samuel',
 'Sophie',
 'Lucy',
 'Charlie',
 'Glen',
 'Maureen',
 'Clifford',
 'Martyn',
 'Hollie',
 'Mohamed',
 'Anthony',
 'Jennifer',
 'Geoffrey',
 'Carol',
 'Harry',
 'Kathleen',
 'Brandon',
 'William',
 'Thomas',
 'Charlene',
 'Diane',
 'Lauren',
 'Vincent',
 'Glenn',
 'Emily',
 'Kimberley',
 'Arthur',
 'Alexander',
 'Gail',
 'Bryan',
 'Diana',
 'Jeremy',
 'Linda',
 'Carolyn',
 'Danny',
 'Shirley',
 'Jane',
 'Julie',
 'Gary',
 'Russell',
 'Robin',
 'Mathew',
 'Duncan',
 'Carole',
 'John',
 'Tony',
 'Sara',
 'June']

def gather_and_write_abstract_infl(file, outf, type_label, type_word_groups={}, seen=None, max_per_domain=10000):
  with open(file) as inf:
      curr_text = ""
      choice = random.randint(0,7)
      for idx, l in enumerate(inf):
        if idx > max_per_domain: break
        dat = json.loads(l)
        dat['meta'] = dat['metadata']
        del dat['metadata']
        dat['meta']['type_label'] = type_label
        text =  dat['text']
        if text[:100] in seen:
          continue
        seen[text[:100]] = 1
        if " murder" in dat['text'] or  " rape " in dat['text'] or " raped " in dat['text'] or "discography" in dat['text'] or "album" in dat['text'] or  "sports" in dat['text'] or "fictional character" in dat['text'] or "popular" in dat['text'] or "ball " in dat['text'] or "baller " in dat['text'] or "television show" in dat['text'] or "television series" in dat['text']: continue
        dat['text'] = dat['text'].replace("\n<human>: next\n<bot>: ", "").replace("\n<human>: more please\n<bot>: ", "").replace("<human>: continue\n<bot>: ", "")
        text =  dat['text']
        if "Background:" in text:
          if random.randint(0,1):
            a, b = text = text.split("<human>:",1)
            text = a.replace("Background: ", "").strip()+"<|endoftext|><human>: "+b.strip()
          else:
            if random.randint(0,1):
              text = text.replace("Background: ", "Context:\n")
            elif random.randint(0,1):
              text = text.replace("Background: ", "Article:\n")
            elif random.randint(0,1):
              text = text.replace("Background: ", "Discussion:\n")
        if "discussing in detail"  in text:
          if random.randint(0,1):
            text = text.replace("discussing in detail"  ,"with more about:")
          elif random.randint(0,1):
            text = text.replace("discussing in detail" ,"emphasizing")
          elif random.randint(0,1):
            text = text.replace("discussing in detail" ,"focusing on")
        if "Answer the question assuming the following"  in text:
          if random.randint(0,1):
            text = text.replace("Answer the question assuming the following" ,"Please assume:")
          elif random.randint(0,1):
            text = text.replace("Answer the question assuming the following" ,"Assuming:")
          elif random.randint(0,1):
            text = text.replace("Answer the question assuming the following" ,"Knowing:")
        if "I'm sorry I can't answer that question based on the information I have." in text:
          if random.randint(0,1):
            text = text.replace("I'm sorry I can't answer that question based on the information I have.", "Let's discuss this more. But first, I'll need some context.")
          elif random.randint(0,1):
            text = text.replace("I'm sorry I can't answer that question based on the information I have.", "Sorry about my earlier confusion. Can we start over?")
          elif random.randint(0,1):
            text = text.replace("I'm sorry I can't answer that question based on the information I have.", "Can you elaborate?")
          elif random.randint(0,1):
            text = text.replace("I'm sorry I can't answer that question based on the information I have.", "I apologize for my confusion. What is the context?", )
        if "Provide one sentence about" in text:
          if random.randint(0,1):
            text = text.replace("Provide one sentence about" ,"In this context, tell me about")
          elif random.randint(0,1):
            text = text.replace("Provide one sentence about" ,"Info about:")
          elif random.randint(0,1):
            text = text.replace("Provide one sentence about" ,"More about:")
        text = text.replace("in one paragraph", "").replace("in two paragraphs", "").replace("in three paragraphs", "").replace("in four paragraphs", "").replace("in five paragraphs", "")
        if random.randint(0,1):
          text = text.replace("Give me an article about", "Can we talk about")
        if random.randint(0,1):
          text = text.replace("me an article about", "information about")
        if random.randint(0,1):
          text = text.replace("me an article about", "me about")
        if "with a sentence about" in text:
          if random.randint(0,1):
            text = text.replace("with a sentence about" ,". Write about")
        if choice == 0:
          text = text.replace("<human>:", "Q:").replace("<bot>:", "A:")
        elif choice == 1:
          if random.randint(0,1):
            text = random.choice(["", "", "", "A "+random.choice(["quiz", "test", "exam"])+" given by a teacher to a student\n\n", "Student being asked questions\n\n", "The intelligent helpful teacher educates her student\n\n", "The intelligent helpful teacher educates his student\n\n"]) + text.replace("<human>: ", "### Teacher:\n").replace("<bot>: ", "### Student:\n")
          else:
            text =  random.choice(["", "", "", "A student "+random.choice(["asks", "inquires", "discusses"])+" her questions\n\n", "Student "+random.choice(["asks", "inquires", "discusses"])+" about his topics\n\n", "A student and teacher learning together\n\n", "A confused student asks a teacher for more "+random.choice(["information", "knowledge", "answers"])+"\n\n"]) + text.replace("<human>: ", "### Student:\n").replace("<bot>: ", "### Teacher:\n")
        elif choice == 2:
          text = text.replace("<human>: ","<|im_end|>\n<|im_start|>user\n").replace("<bot>: ", "<|im_end|>\n<|im_start|>assistant\n")
          if text.startswith("<|im_end|>\n"):
            text = text[len("<|im_end|>\n"):]
        elif choice == 3:
          text = text.replace("<human>:", "Quesiton:").replace("<bot>:", "Answer:")
        elif choice == 4:
          random.shuffle(names)
          text = text.replace("<human>:", names[0]+":").replace("<bot>:", names[1]+":")
          text = random.choice(["Here we have a", "This is a", "This is a story of a", "A"]) +" "+random.choice(["conversation", "discussion", "inquiry", "lesson"])+" between a " + random.choice(["teacher", "educator", "researcher", "very intelliget assistant", "helpful friend"]) + " named " + names[1] + " and a " + random.choice(["inquisitive", "polite", "thoughtful", "person named"]) + " " + names[0]+".\n\n" + text
        elif choice == 5:
          text = text.replace("<human>: ", "### Instruction:\n").replace("<bot>: ", "### Response:\n")
        elif choice == 6:
          text = text.replace("<human>: ", "[INST] ").replace("<bot>: ", " [\INST]")
          text = text.strip()
          if text.startswith("[INST]"):
            text = "<s>"+text
        elif choice == 7:
          text = text.replace("<bot>: ", "Answer: ")
          text_arr = text.split("<human>: ")
          if text_arr[0] == "":
            text = " ".join("\n"+str(idx)+". "+t.strip() for idx, t  in enumerate(text_arr) if t.strip())
          else:
            text = text_arr[0].strip() + "\n1." + text_arr[1].strip() + " ".join("\n"+str(idx+2)+". "+t.strip() for idx, t  in enumerate(text_arr[2:]) if t.strip())
          text = random.choice(["", "", "", "Lesson plan with excercises\n\n", "Helpful information\n\n", "Discussion\n\n"])+text.strip()
          if text.startswith("[INST]"):
            text = "<s>"+text
        text = text.replace("Lesson plan", random.choice(["Lesson plan", "Texbook", "Study book", "Notes"]))
        text = text.replace("student", random.choice(["student", "pupil", "scholar", "knowledge seeker", "novice", "mentee"]))
        if "mentee" in text:
           text = text.replace("teacher", "mentor")
        else:
          text = text.replace("teacher", random.choice(["teacher", "educator", "instructor", "tutor", "professor", "teaching assistant"]))
        text = text.replace("Provide", random.choice(["Provide", "Show", "Describe", "Write", "Tell"]))
        text = text.replace("Give", random.choice(["Discuss", "Give", "Describe", "Present", "Tell", "Prove"]))
        text = text.replace("about", random.choice(["relating to", "about", "of or about", "focused on", "focusing on"]))
        text = text.replace("article about", random.choice(["article on", "article about", "Wikipedia article with the subject of", "lesson about", "presentation pertaining to"]))
        text = text.replace("'CHEBI", "'CHEMICAL_OR_BIOLOGICAL").replace(" .", ".").replace(".,", ",").replace(": :", ":").replace("Give Tell", "Tell").replace("\n [", "\n[").replace("on The", "on the").replace("about The", "about the").replace(":.", ":")
        dat['text'] = text.strip()
        if '0) was ' in text or '1) was ' in text or '2) was ' in text or '3) was ' in text or '4) was ' in text  or '5) was ' in text  or '6) was ' in text  or '7) was ' in text  or '8) was ' in text  or '9) was ' in text:
          if random.randint(0,1): continue
        if 'is an American' in text or '0) is ' in text or '1) is ' in text or '2) is ' in text or '3) is ' in text or '4) is ' in text  or '5) is ' in text  or '6) is ' in text  or '7) is ' in text  or '8) is ' in text  or '9) is ' in text:
          if random.randint(0,3)> 0: continue
        text = text.strip()
        if len(curr_text+text.strip()) < 4000:
          curr_text = curr_text+"<|endoftext|>"+text.strip()
        else:
          curr_text = curr_text.strip()
          if curr_text.startswith("<|endoftext|>"):
            curr_text = curr_text[len("<|endoftext|>"):].strip()
          dat['text'] = curr_text.replace("'\\\"", "'").replace("\\\"", "\"").replace("\\\'", "\'")
          outf.write (json.dumps(dat)+"\n")
          curr_text = text
          choice = random.randint(0,7)
      if curr_text:
        if curr_text.startswith("<|endoftext|>"):
            curr_text = curr_text[len("<|endoftext|>"):].strip()
        dat['text'] = curr_text.replace("'\\\"", "'").replace("\\\"", "\"").replace("\\\'", "\'")
        outf.write (json.dumps(dat)+"\n")


seen = {}
with open("/content/drive/Shareddrives/ontocord_llc/mmlu_abstract_infill.jsonl", "w") as outf:
  for file in glob.glob("/content/drive/Shareddrives/ontocord_llc/unified_out/*.jsonl"):
    mmlu_type = file.split("/")[-1].replace(".jsonl", "")
    gather_and_write_abstract_infl(file, outf, type_label=mmlu_type, type_word_groups={}, seen=seen)
