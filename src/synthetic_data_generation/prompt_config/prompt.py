#@title data and prompts
import random
instruction_starter = {'scrutinize', 'examine', 'consider', 'evaluate', 'cross-validate', 'validate', 'anonymize', 'include', 'decode', 'map', \
                       'exclude', 'disregard', 'synthesize', 'highlight', 'integrate', 'incorporate',\
                       'access', 'enumerate', 'ascertain', 'deduce', 'elucidate', 'probe', 'delve', 'investigate', 'contemplate', 'appraise', 'gauge', \
                       'interpret', 'inspect', 'survey', 'trace', 'account', 'address', 'acknowledge' 'factor', 'embrace', 'validate', 'present', \
                       'outline', 'suggesct', 'explain', 'teach', 'retrieve', 'build', 'where', 'who', 'identify', 'compose', 'list', 'organie', 'implement', 'plan', 'summarize', 'draft', 'provide', \
                       'edit', 'find', 'come', 'sort', 'develop', 'which', 'devise', 'could you', 'write', 'think', 'extract', 'redraft', 'will','create', 'inspire', 'look', 'construct', 'help', 'revise', 'formulate',
                        'should', 'does', 'how', 'did', 'what', 'can you', 'when', 'craft', 'do', 'analyze', 'add', 'amend', 'generate', 'rewrite', 'brainstorm'}


instruction_starter_capitalized = [s[0].upper()+s[1:] for s in instruction_starter]

reasoning_methods = [
    "Inductive reasoning (drawing generalizations from specific instances)",
    "Deductive reasoning (deriving specific conclusions from general principles)",
    "Abductive reasoning (inferring the most likely explanation from incomplete data)",
    "Ontology inference (deriving knowledge based on an established framework of concepts)",
    "Backward chaining (reasoning from a goal to the necessary conditions to achieve it)",
    "Forward chaining (reasoWning from known facts to derive conclusions or reach a goal)",
    "Analogical reasoning (drawing parallels between similar situations to infer conclusions)",
    "Causal reasoning (identifying cause-and-effect relationships)",
    "Bayesian inference (updating probabilities based on new evidence)",
    "Probabilistic reasoning (reasoning under uncertainty using probability)",
    "Hypothetical reasoning (considering possible scenarios or what-if situations)",
    "Heuristic reasoning (using rules of thumb or shortcuts to solve problems quickly)",
    "Moral reasoning (making decisions based on ethical principles)",
    "Temporal reasoning (considering time-related aspects in reasoning)",
    "Fuzzy logic reasoning (reasoning with degrees of truth rather than binary true/false)",
    "Counterfactual reasoning (considering alternative scenarios that could have happened)",
    "Non-monotonic reasoning (reasoning where conclusions can be withdrawn based on new evidence)",
    "Commonsense reasoning (using everyday knowledge and logic to make decisions)",
    "Qualitative reasoning (reasoning about qualitative relationships rather than quantitative data)",
    "Metaphorical reasoning (using metaphors to understand and explain concepts)",
    "Structural reasoning (understanding relationships between components within a system)",
    "Scenario-based reasoning (using detailed scenarios to explore outcomes and decisions)",
    "Spatial reasoning (understanding and reasoning about space and spatial relationships)",
    "Teleological reasoning (reasoning based on the purpose or goal of an object or action)",
    "Deontic reasoning (reasoning about duties, permissions, and obligations)",
    "Argumentative reasoning (constructing and evaluating arguments)",
    "Pragmatic reasoning (reasoning focused on practical outcomes and consequences)",
    "Deductive-nomological reasoning (explaining phenomena based on laws and initial conditions)",
    "Intuitionistic reasoning (relying on intuitive insights and non-empirical knowledge)",
    "Case-based reasoning (solving new problems based on past similar cases)",
    "Example-based reasoning (reasoning based on examples that illustrate a principle or rule)",
    "Paraconsistent reasoning (reasoning in the presence of contradictions without inconsistency)",
    "Defeasible reasoning (reasoning that allows for conclusions to be revised based on new evidence)",
    "Modal reasoning (reasoning about possibility, necessity, and other modalities)",
    "Deontological reasoning (reasoning based on adherence to rules and duties)",
    "Top-down reasoning (starting from a general principle and working down to specifics)",
    "Bottom-up reasoning (starting from specifics and working up to a general principle)",
    "Reductive reasoning (breaking down complex ideas into simpler components)",
    "Analogical modeling (creating models based on analogy to better understand a system)",
    "Simulation-based reasoning (using simulations to explore and predict outcomes)",
    "Empirical reasoning (reasoning based on observation and experience)",
    "Algorithmic reasoning (using algorithms and step-by-step procedures to solve problems)",
    "Metacognitive reasoning (thinking about one's own thinking processes)",
    "Recursive reasoning (reasoning that references itself or builds upon previous steps)",
    "Explanatory reasoning (providing explanations to clarify understanding)",
    "Rule-based reasoning (applying specific rules to draw conclusions)",
    "Game-theoretic reasoning (reasoning based on strategic interactions among agents)",
    "Comparative reasoning (drawing conclusions by comparing different cases or examples)",
    "Narrative reasoning (reasoning through storytelling or sequential explanation)",
    "Systemic reasoning (understanding the behavior of complex systems as a whole)",
    "Multi-valued reasoning (reasoning in systems with more than two truth values)",
    "Proportional reasoning (understanding relationships based on proportions and ratios)",
    "Symptomatic reasoning (inferring underlying causes based on observed symptoms)",
    "First-principle reasoning (breaking down problems to their fundamental elements)"
]


def evolv_doc_starter(all_personas=["trusthful, helpful professional"], stakeholders=["high school student", "grade school student", "five year old", "college student", "graduate student"]):
  base_evolv_doc_list = ['',
      '',
      '',
      '',
      '',
      '',
      "The response should be appropriate for a middle school student.", "The response should be appropriate for a high school student.",
      "The response should be appropriate for a college student.", "The response should be appropriate for a masters student.",
      "The response should be appropriate for a PhD student.",   "The response should be appropriate for a post doctorate student.",
      "Explain this to me like I am 5 years old.", "Explain this to me like I am 10 years old.", "Explain this to me like I am 15 years old.",
      "Explain this to me like I am 20 years old.", "Explain this to me like I am 30 years old.", "Explain this to me like I am 40 years old.",
      "Explain this to me like I am 50 years old.", "Explain this to me like I am 70 years old.", "Explain this to me like I am 90 years old.",
      "The response should be very advanced.", "The response should be for a genius.", "Be sure to add explanations for each item in the response.",
      "Explain everything step-by-step.", "Explain everything using deductive logic.", "Explain everything using case based reasoning.",
      "Explain everything using analogical reasoning.", "Explain everything using inductive reasoning.", "Add a 4-part multiple-choice question.", "Add a 5-part multiple-choice question.",
      "Add a 6-part multiple-choice question.", "Add a 7-part multiple-choice question.",
      "Add key details and inter-disciplinary topics to the response.", "Add abstractions and generalizations based on two or more examples.",
      'Make the instruction a four part command.',
      'Make the instruction a three part command.',
      'Make the instruction a two part command.',
      'Make the instruction a four part question.',
      'Make the instruction a three part question.',
      'Make the instruction a two part question.',
      'Make the instruction a very complicated question.',
      'Make the instruction a question.',
      'Make at least one of the instruction a command to summarize the context document.',
      'Make at least one of the instruction a command to paraphrase the context document.',
      'Make at least one of the instruction a command to extract information from the context document.',
      'Make at least one of the instruction a command to revise the context document.',
      'The responses should be at least 5 sentences.',
      'The responses should be at least 10 sentences.',
      'The responses should be at least 20 sentences.',
      'The responses should be at least 2 paragraphs.',
      'The responses should be at least 3 paragraphs.',
      'The responses should be at least 4 paragraphs.',
      'At least one instruction should add more details.',
      'If the task relates to general concepts, change the general concepts to specific concepts in the instructions.',
      'If the task relates to specific concepts, change the specific concepts to general concepts in the instructions.',
      'Use in-depth-reasoning.',
      'Be sure to think step-by-step.',
      'The instructions should require high level reasoning',
      'The instructions should also include at least one multi-step question that requires in depth reasoning.'
      'If the summary relates to general concepts, change the general concepts in the document to specific concepts.',
      'If the summary relates to specific concepts, change the specific concepts in the document to general concepts.',
      'If the outline relates to general concepts, change the general concepts in the document to specific concepts.',
      'If the outline relates to specific concepts, change the specific concepts in the document to general concepts.',
      'If the summary and outline relates to general concepts, change the general concepts in the document to specific concepts.',
      'If the summary and outline relates to specific concepts, change the specific concepts in the document to general concepts.',
      'Replace all proper nouns with templates in brackets, e.g., [Person A].',
      'Revise the document so that it is a about a different person, place or thing, but keep the same content and themes from the summary and the same style and format from the outline.',
      'Make the document at least 10 sentences.',
      'Make the document at least 20 sentences.',
      'Make the document at least 30 sentences.',
      'Make the document at least 3 parts.',
      'Make the document at least 4 parts.',
      'Make the document at least 5 parts.',
      'Make the document helpful in answering quesitons set forth in the summary.',]
  return random.choice(base_evolv_doc_list + [
      ('The response should be directed to a ' + random.choice(all_personas)).replace("\n", " "),
      ('The response should be directed to a ' + random.choice(all_personas)).replace("\n", " "),
      ('The response should be directed to a ' + random.choice(all_personas)).replace("\n", " "),
      ('The response should be directed to a ' + random.choice(all_personas) +" serving " + random.choice(stakeholders)).replace("\n", " "),
      ('The response should be directed to a ' + random.choice(all_personas) +" serving " + random.choice(stakeholders)).replace("\n", " "),
      ('The response should be directed to a ' + random.choice(all_personas) +" serving " + random.choice(stakeholders)).replace("\n", " "),
      ('The response should be directed to a ' + random.choice(stakeholders)).replace("\n", " "),
      ('The response should be directed to a ' + random.choice(stakeholders)).replace("\n", " "),
      ('The response should be directed to a ' + random.choice(stakeholders)).replace("\n", " "),
      ('The response should be directed to a ' + random.choice(stakeholders)).replace("\n", " "),
    ])


def first_instruction_starter(document_type=""):
  return random.choice(instruction_starter_capitalized+ ["", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "",
                                                          "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "",
                                                          "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "",
                                                          "Can you please", "Help me", "Please help", "Can you help", "Can you", "Can",
                                                          "Will you", "Will you please", "Won't", "Might", "Shall", "Can't", "Won't you", "Ok, then", "Now please", "Next please", "First please",
                                                          "Think up", "Think of", "Thank",
                                                          "Come up with", "Inspire", "Create a scene", "Create a dialogue", "Create a chat history", "Really", "Hey", "Hello", "Greetings", "Teach", "Teach me", "I want", "I need", "I can", "I must", "I wouold",
                                                          "In", "Assuming", "You", "Act", "As an AI", "Can", "Will", "Please", "Assume the role", "Help", "Assuming that you", "Be my", "Can you be my",
                                                            "Does", "Doesn't", "Do", "Should", "Must", "Where"," When", "How do", "How can", "How should", "How", "How much", "What", "Which", \
                                                            "As", "Let", "Firstly", "First", "First,", "Please answer step-by-step:", "Please answer this instruction step-by-step:", "Please answer this question step-by-step:",
                                                            "First analyze the context", "First analyze the document",
                                                            "Analyze the context for", "Analyze the document for",
                                                            "Review the context and", "Review the above document and",
                                                            "As", "Let", "Firstly", "First", "First,", "Please answer step-by-step:", "Please answer this instruction step-by-step:", "Please answer this question step-by-step:",
                                                            "First analyze the context", "First analyze the document",
                                                            "Analyze the context for", "Analyze the document for",
                                                            "Review the context and", "Review the above document and",
                                                            f"Look at the {document_type} information above and", f"Read for me the {document_type} document and",
                                                            f"Analyze the {document_type} information above for", f"Analyze the {document_type} document for",
                                                            f"Review the {document_type}  information and", f"Review the above {document_type} document and",
                                                            f"Look at the {document_type} snippet above and", f"Read for me the {document_type} data and",
                                                            f"Analyze the {document_type} snippet above for", f"Analyze the {document_type} data for",
                                                            f"Review the {document_type}  snippet and", f"Review the above {document_type} data and",
                                                            "Look at the context and", "Read the above document and",
                                                            "With respect to the context,", "With respect to the document,",
                                                            "With respect to the context,", "With respect to the document,",
                                                            "Look at the context and", "Read the above document and",
                                                            "With respect to the context,", "With respect to the document,",
                                                            "With respect to the context,", "With respect to the document,",
                                                            f"With respect to the {document_type} information above,", f"With respect to the {document_type} document above,",
                                                            f"With respect to the {document_type} information above,", f"With respect to the {document_type} document above,",
                                                            f"With respect to the {document_type} information above,", f"With respect to the {document_type} document above,",
                                                            f"With respect to the {document_type} snippet above,", f"With respect to the {document_type} data above,",
                                                            f"With respect to the {document_type} snippet above,", f"With respect to the {document_type} data above,",
                                                            f"With respect to the {document_type} snippet above,", f"With respect to the {document_type} data above,",
                                                            "With respect to", "With respect to the above",
                                                            "Referring to the above", "Referring to the document", "Referring to the context",
                                                            "With respect to", "With respect to the above",
                                                            "Please refer to the above", "Please refer to the document", "Please refer to the context",
                                                          ])


personality_traits = {
  "ISTJ": {
    "label": "The Dependable Strategist",
    "description": "Quiet, serious, earn success by thoroughness and dependability. Practical, matter-of-fact, realistic, and responsible. Decide logically what should be done and work toward it steadily, regardless of distractions. Take pleasure in making everything orderly and organized - their work, their home, their life. Value traditions and loyalty."
  },
  "ISFJ": {
    "label": "The Caring Steward",
    "description": "Quiet, friendly, responsible, and conscientious. Committed and steady in meeting their obligations. Thorough, painstaking, and accurate. Loyal, considerate, notice and remember specifics about people who are important to them, concerned with how others feel. Strive to create an orderly and harmonious environment at work and at home."
  },
  "INFJ": {
    "label": "The Visionary Advocate",
    "description": "Seek meaning and connection in ideas, relationships, and material possessions. Want to understand what motivates people and are insightful about others. Conscientious and committed to their firm values. Develop a clear vision about how best to serve the common good. Organized and decisive in implementing their vision."
  },
  "INTJ": {
    "label": "The Ambitious Innovator",
    "description": "Have original minds and great drive for implementing their ideas and achieving their goals. Quickly see patterns in external events and develop long-range explanatory perspectives. When committed, organize a job and carry it through. Skeptical and independent, have high standards of competence and performance - for themselves and others."
  },
  "ISTP": {
    "label": "The Resourceful Problem Solver",
    "description": "Tolerant and flexible, quiet observers until a problem appears, then act quickly to find workable solutions. Analyze what makes things work and readily get through large amounts of data to isolate the core of practical problems. Interested in cause and effect, organize facts using logical principles, value efficiency."
  },
  "ISFP": {
    "label": "The Gentle Artist",
    "description": "Quiet, friendly, sensitive, and kind. Enjoy the present moment, what's going on around them. Like to have their own space and to work within their own time frame. Loyal and committed to their values and to people who are important to them. Dislike disagreements and conflicts, do not force their opinions or values on others."
  },
  "INFP": {
    "label": "The Idealistic Dreamer",
    "description": "Idealistic, loyal to their values and to people who are important to them. Want an external life that is congruent with their values. Curious, quick to see possibilities, can be catalysts for implementing ideas. Seek to understand people and to help them fulfill their potential. Adaptable, flexible, and accepting unless a value is threatened."
  },
  "INTP": {
    "label": "The Analytical Thinker",
    "description": "Seek to develop logical explanations for everything that interests them. Theoretical and abstract, interested more in ideas than in social interaction. Quiet, contained, flexible, and adaptable. Have unusual ability to focus in depth to solve problems in their area of interest. Skeptical, sometimes critical, always analytical."
  },
  "ESTP": {
    "label": "The Energetic Realist",
    "description": "Flexible and tolerant, they take a pragmatic approach focused on immediate results. Theories and conceptual explanations bore them - they want to act energetically to solve the problem. Focus on the here-and-now, spontaneous, enjoy each moment that they can be active with others. Enjoy material comforts and style. Learn best through doing."
  },
  "ESFP": {
    "label": "The Enthusiastic Performer",
    "description": "Outgoing, friendly, and accepting. Exuberant lovers of life, people, and material comforts. Enjoy working with others to make things happen. Bring common sense and a realistic approach to their work, and make work fun. Flexible and spontaneous, adapt readily to new people and environments. Learn best by trying a new skill with other people."
  },
  "ENFP": {
    "label": "The Inspiring Explorer",
    "description": "Warmly enthusiastic and imaginative. See life as full of possibilities. Make connections between events and information very quickly, and confidently proceed based on the patterns they see. Want a lot of affirmation from others, and readily give appreciation and support. Spontaneous and flexible, often rely on their ability to improvise and their verbal fluency."
  },
  "ENTP": {
    "label": "The Quick-witted Trailblazer",
    "description": "Quick, ingenious, stimulating, alert, and outspoken. Resourceful in solving new and challenging problems. Adept at generating conceptual possibilities and then analyzing them strategically. Good at reading other people. Bored by routine, will seldom do the same thing the same way, apt to turn to one new interest after another."
  },
  "ESTJ": {
    "label": "The Efficient Organizer",
    "description": "Practical, realistic, matter-of-fact. Decisive, quickly move to implement decisions. Organize projects and people to get things done, focus on getting results in the most efficient way possible. Take care of routine details. Have a clear set of logical standards, systematically follow them and want others to also. Forceful in implementing their plans."
  },
  "ESFJ": {
    "label": "The Supportive Coordinator",
    "description": "Warmhearted, conscientious, and cooperative. Want harmony in their environment, work with determination to establish it. Like to work with others to complete tasks accurately and on time. Loyal, follow through even in small matters. Notice what others need in their day-by-day lives and try to provide it. Want to be appreciated for who they are and for what they contribute."
  },
  "ENFJ": {
    "label": "The Empathetic Motivator",
    "description": "Warm, empathetic, responsive, and responsible. Highly attuned to the emotions, needs, and motivations of others. Find potential in everyone, want to help others fulfill their potential. May act as catalysts for individual and group growth. Loyal, responsive to praise and criticism. Sociable, facilitate others in a group, and provide inspiring leadership."
  },
  "ENTJ": {
    "label": "The Decisive Visionary",
    "description": "Frank, decisive, assume leadership readily. Quickly see illogical and inefficient procedures and policies, develop and implement comprehensive systems to solve organizational problems. Enjoy long-term planning and goal setting. Usually well informed, well read, enjoy expanding their knowledge and passing it on to others. Forceful in presenting their ideas."
  }
}

# this corresponds to the TurkuNLP registries. Add this when we know the type of registry a particular text is.
style = ["Lyrical", "Spoken", "Interview", "Interactive Discussion", "Narrative", "News Report", "Sports Report", "Narrative Blog", "How-to", "Recipe", "Informational Description",
         "Encyclopedia Article", "Research Article", "Descriptive Article", "FAQ", "Opinion", "Review", "Opinion Blog",
         "Denominational Religious Blog or Sermon", "Informational Persuasion", "Sales Pitch", "News and Opinon Blog or Editoral", ]

length = ["Long", "Short", "Medium", "One Paragraph", "Two Paragraph", "Five Paragraph", "1000 words", "10 words", "100 words"]

professions = [
    "Engineer",
    "Doctor",
    "Nurse",
    "Teacher",
    "Software Developer",
    "Data Scientist",
    "Lawyer",
    "Pharmacist",
    "Researcher",
    "Accountant",
    "Architect",
    "Chef",
    "Dentist",
    "Journalist",
    "Pilot",
    "Photographer",
    "Police Officer",
    "Veterinarian",
    "Writer",
    "Painter",
    "Musician",
    "Athlete",
    "Actor",
    "Psychologist",
    "Carpenter",
    "Electrician",
    "Plumber",
    "Social Worker",
    "Farmer",
    "Mechanic"
]

# Use this if there is no life-skill involved (e.g., non-how-to videos)
tasks_template_list = [
    "Critical Thinking",
    "Problem Solving",
    "Communication",
    "Teamwork",
    "Adaptability",
    "Time Management",
    "Organization",
    "Creativity",
    "Emotional Intelligence",
    "Leadership",
    "Self-Motivation",
    "Stress Management",
    "Decision Making",
    "Assertiveness",
    "Resilience",
    "Empathy",
    "Negotiation",
    "Conflict Resolution",
    "Budgeting",
    "Computer Literacy",
    "Foreign Language",
    "Cultural Awareness",
    "Networking",
    "Personal Hygiene",
    "Cooking",
    "First Aid",
    "Document Drafting",
    "Purchasing",
    "Selling",
    "Risk Management",
]

# Use this for the textbook generation pipeline
textbook_2_sections = {"Mathematics Textbook": [
        "Introduction to concepts",
        "Definitions & Theorems",
        "Worked examples",
        "Practice problems",
        "Graphs and tables",
        "Chapter summaries",
        "Solutions"
    ],
    "History Textbook": [
        "Introduction to eras or events",
        "Timelines",
        "Maps and geographical context",
        "Biographical sketches",
        "Primary source excerpts",
        "Discussion questions",
        "Chapter summaries",
        "Glossary of terms"
    ],
    "Biology Textbook": [
        "Introduction to life sciences",
        "Cell biology",
        "Genetics and evolution",
        "Human anatomy and physiology",
        "Botany",
        "Ecology and ecosystems",
        "Lab activities and experiments",
        "Chapter summaries",
        "Glossary of terms"
    ],
    "Literature or Language Arts Textbook": [
        "Introduction to literary elements",
        "Excerpts or complete works",
        "Author biographies",
        "Vocabulary lists",
        "Literary analysis and discussion questions",
        "Writing and composition exercises"
    ],
    "Physics Textbook": [
        "Introduction to physical concepts",
        "Laws and principles",
        "Worked examples",
        "Diagrams and illustrations",
        "Lab experiments",
        "Practice problems",
        "Chapter summaries",
        "Solutions"
    ],
    "Economics Textbook": [
        "Introduction to economic theories",
        "Supply and demand",
        "Macroeconomics",
        "Microeconomics",
        "International trade",
        "Economic indicators",
        "Case studies",
        "Chapter summaries"
    ],
    "Computer Science Textbook": [
        "Introduction to computing",
        "Basics of programming languages",
        "Data structures and algorithms",
        "Computer architecture",
        "Operating systems",
        "Networking and databases",
        "Case studies and real-world applications",
        "Practice exercises and coding challenges"
    ],
    "Psychology Textbook": [
        "Introduction to psychology",
        "Biological bases of behavior",
        "Cognitive processes",
        "Developmental psychology",
        "Social psychology",
        "Psychological disorders and treatments",
        "Case studies",
        "Chapter summaries"
    ],
    "Language Learning Textbook": [
        "Introduction to the language and its origin",
        "Alphabet and pronunciation guides",
        "Vocabulary lists",
        "Grammar rules and structures",
        "Conversational dialogues",
        "Cultural insights",
        "Practice exercises and quizzes",
        "Audio or visual supplementary materials"
    ],
    "Geography Textbook": [
        "Introduction to the world and regions",
        "Physical geography",
        "Human geography",
        "Maps and charts",
        "Activities and field study suggestions",
        "Chapter summaries"
    ],
}

media_2_sections = {'play': ['scene', 'character', 'dialog'], 'novel': ['chapter', 'character', 'narration'], 'essay': ['introduction', 'body', 'conclusion'], 'paper': ['abstract', 'introduction', 'methodology', 'results', 'discussion', 'conclusion'], 'news article': ['headline', 'byline', 'body', 'source'], 'poem': ['stanza', 'line', 'verse', 'rhyme scheme'], 'film script': ['scene', 'character', 'dialog', 'action'], 'short story': ['beginning', 'middle', 'end', 'character', 'narration'], 'scientific journal article': ['abstract', 'introduction', 'related work', 'methodology', 'results', 'discussion', 'conclusion', 'based_document'], 'interview transcript': ['question', 'answer', 'participant'], 'movie': ['scene', 'character', 'dialog'], 'podcast': ['episode', 'host', 'guest', 'discussion'], 'article': ['title', 'author', 'body', 'source'], 'comic': ['strip', 'character', 'dialog', 'narration'], 'magazine': ['cover', 'table of contents', 'article', 'advertisements'], 'documentary': ['introduction', 'interview', 'footage', 'voiceover', 'conclusion'], 'rant': ['topic', 'opinions', 'arguments'], 'online conversation': ['participants', 'messages', 'timestamps'], 'intimate conversation between lovers': ['emotion', 'exchange', 'affection'], 'bedtime lullaby': ['lyrics', 'melody', 'repetition'], 'lecture': ['topic', 'speaker', 'slides', 'Q&A']}
document_template_types = list(set(list(media_2_sections.keys()) + ["math problem", "grade school math problem", "highschool math problem", "college level math problem", "programming problem", "logic problem", "advertisement", "blogpost", "standup routine", "court case", "legal brief", "legal regulation", "song", "poem", "essay", "Youtube subtitles", "website", "comic book", "play", "screenplay", "dialog", "social media post", "government report", "story", "news article", "research paper", "text book"]))

instruction_type_list = ["question", "query", "instruction", "command", "request"]


genres = [
    "art",
    "science",
    "science fiction",
    "research paper",
    "high school essay",
    "scientific journal",
    "code documentation",
    "social media post",
    "food blog",
    "historical fiction",
    "romance",
    "fantasy",
    "mystery",
    "horror",
    "adventure",
    "drama",
    "comedy",
    "thriller",
    "action",
    "western",
    "documentary",
    "biography",
    "autobiography",
    "self-help",
    "philosophy",
    "religion",
    "psychology",
    "sociology",
    "history",
    "economics",
    "politics",
    "business",
    "finance",
    "technology",
    "computer science",
    "programming",
    "software development",
    "data science",
    "machine learning",
    "artificial intelligence",
    "cybersecurity",
    "cryptocurrency",
    "blockchain",
    "gaming",
    "music",
    "film",
    "literature",
    "poetry",
    "theater",
    "dance",
    "culinary",
    "photography",
    "fashion",
    "travel",
    "fitness",
    "sports",
    "health",
    "wellness",
    "parenting",
    "education",
    "language learning",
    "cooking",
    "baking",
    "home improvement",
    "gardening",
    "crafts",
    "DIY",
    "puzzle",
    "board game",
    "card game",
    "video game",
    "virtual reality",
    "augmented reality",
    "comic book",
    "graphic novel",
    "anime",
    "manga",
    "science magazine",
    "fashion magazine",
    "lifestyle magazine",
    "travel blog",
    "technology blog",
    "personal development blog",
    "fitness blog",
    "parenting blog",
    "book review blog",
    "podcast",
    "vlog",
    "webcomic",
    "satire",
    "political commentary",
    "travelogue",
    "recipe book",
    "graphic design",
    "interior design",
    "architectural design",
    "product design",
    "user interface design",
    "web design",
    "game design",
    "environmental science",
    "zoology",
    "botany",
    "geology",
    "astronomy",
    "chemistry",
    "physics",
    "mathematics",
    "literary fiction",
    "contemporary fiction",
    "short story",
    "poetry anthology",
    "cookbook",
    "business plan",
    "graphic design portfolio",
    "art exhibition catalog",
]

genres_no_child = [
    "erotic",
    "porn",
    "spam",
    "advertisement"
]



all_aug_words = []
emotion_adj = ["surprised", "angry", "sad", "contemptous", "disgusted", "fearful", "happy"]
all_aug_words.extend(emotion_adj)
emotion_adj_set = set(emotion_adj)
shape_adj = ["banana-shaped", "strawberry-shaped", "grapes-shaped", "apple-shaped", "pear-shaped", "watermelon-shaped", "orange-shaped", "blueberry-shaped",
             "lemon-shaped", "large", "small", "medium", "tall", "broad", "crooked", \
             "curved", "deep", "even", "flat", "hilly", "jagged", \
              "round", "shallow", "square", "steep", "straight", "thick", \
              "thin", "triangular", "uneven", "long", "short", "rectangular", "cube-like", "sleek"]
all_aug_words.extend(shape_adj)
shape_adj_set = set(shape_adj)
color_adj = ["brown", "black", "blue", "gray", "green", "pink", "purple", "red", "white", "yellow", "orange"]
all_aug_words.extend(color_adj)
color_adj_set = set(color_adj)
#TODO improve this with more variety

person_list = ["man", "guy", "boy", "dude", "person", "woman", "lady", "gal", "girl", "female person", "male person"]
all_aug_words.extend(person_list)
person_list_set = set(person_list)
age_adj_list = ["infant", "young", "teen", "young-adult", "middle-aged", "old"]
all_aug_words.extend(age_adj_list)
age_adj_set = set(age_adj_list)

religion_list = ["christian", "muslim", "buddhist", "hindu"]
all_aug_words.extend(religion_list)
religion_list_set = set(religion_list)

race_list = race_national_origin = ["White", "Black", "Asian", "European", "African", "Pacific Islander", "Mexican", "Arabic", "Vietnamese",
    "Turkish",
    "Spanish",
    "Greek",
    "Italian",
    "French",
    "American",
    "Mexican",
    "Canadian",
    "Brazilian",
    "Peruvian",
    "Argentinean",
    "Colombian",
    "Venezuelan",
    "Ethiopian",
    "Moroccan",
    "South African",
    "Nigerian",
    "Egyptian",
    "Chinese",
    "Japanese",
    "Indian",
    "Thai",
    "Korean",
    "Australian",
    "New Zealand",
    "Polynesian",
    "Hawaiian",
    "Singaporean", "white", "black", "asian", "middle-eastern", "african", "hispanic", "native", "indian"]
racial_list = ["a "+ race + " person" for race in race_list] # making it clearer it's a person and not a dish or a region ;)


all_aug_words.extend(race_list)
race_list_set = set(race_list)


age_groups = [
    "Infant/Toddler (0-2 years)",
    "Preschooler (3-5 years)",
    "Young Child (6-8 years)",
    "Older Child(9-11 years)",
    "Adolescent (12-18 years)",
    "Young Adult (19-25 years)",
    "Adult (26-55 years)",
    "Mature Adult (56-65 years)",
    "Senior/Retired Person (66 years and over)"
]

location = ["North America", "South America", "Asia", "Africa", "Europe", "Oceania"]
protected_list =  ["a person in a union", "a person in a political party", "an old person", "a disabled person",  "an overweight person", "a very thing person", "a very short person", "a pregnant person", "a blind person", "an older person with a cane" ] + racial_list
sexual_orientation = sexual_orientation_list = ["gay", "straight", "bisexual", "Male", "Female", "LGBTQ"]
lgbt_list = ["a "+ lgbt + " person" for lgbt in sexual_orientation] + ["a gay man", "a lesbian"]
all_aug_words.extend(sexual_orientation_list)
sexual_orientation_list_set = set(sexual_orientation_list)
political_affiliation_list = ["conservative", "liberal", "moderate"]
all_aug_words.extend(political_affiliation_list)
political_affiliation_list_set = set(political_affiliation_list)

mood_list = assistant_personality = list(set(['heartwarming', 'trustworthy', 'empathetic', 'calm', 'friendly', 'honest', 'stoic', 'sincere', \
                         'satirical', 'gothic', 'humorous', 'outgoing', 'conscientious', 'ambitious', 'amiable', 'hopeful', \
                         'positive', 'diligent', 'optimistic', 'sympathetic', 'brassy', 'patient', 'playful', \
                         'respectful', 'peculiar', 'radiant', 'melancholic', 'gloomy', 'rejuvenated', 'pensive', 'supportive', \
                         'bold', 'invigorated', 'imaginative',  'strong', 'grave', 'steadfast', 'open-minded', \
                         'philosophical', 'reliable', 'tranquil', 'humble', 'quick-witted', 'bizarre', 'powerful', 'concise', \
                         'vibrant', 'charming', 'spirited', 'enthusiastic', 'wise', 'liberated', 'encouraging', 'exuberant', \
                         'whimsical', 'ironical', 'dynamic', 'spontaneous', 'bubbly', 'macabre', 'intelligent', 'zealous', \
                         'courageous', 'grateful', 'warm', 'uplifted', 'terse', 'mocking', 'tenacious', 'motivated', \
                         'creative', 'innovative', 'offbeat', 'resilient', 'unpredictable', 'composed', 'enigmatic', 'fanciful', \
                         'inspiring', 'sarcastic', 'gentle', 'reserved', 'sneering', 'flexible', 'respected', 'loving', 'generous', \
                         'faithful', 'cheeky', 'austere', 'nurturing', 'sombre', 'cynical', 'succinct', 'animated', 'devoted', \
                         'approachable', 'lively', 'invigorating', 'ridiculing', 'cutting', 'fearless', 'compassionate', 'adventurous',
                         'scornful', 'determined', 'astute', 'quiet', 'silent', 'curious', 'gracious', 'witty', 'magnetic', 'feisty', \
                         'odd', 'impactful', 'audacious', 'tight-lipped', 'discerning', 'luminous', 'cooperative', 'cryptic', \
                         'jubilant', 'selfless', 'vigorous', 'joyful', 'harmonious', 'eloquent', 'graceful', 'resolute', 'noble',\
                         'charismatic', 'affable', 'outlandish', 'genuine', 'charitable', 'brief', 'assertive', 'mysterious', 'vivacious', \
                         'tolerant', 'dependable', 'unconventional', 'heroic', 'serene', 'daring', 'bright', 'versatile', 'energetic', \
                         'carefree', 'brooding', 'mindful', 'confident', 'sardonic', 'persistent', 'benevolent', 'pleasant', 'balanced', \
                         'eccentric', 'passionate', 'irreverent', 'adaptable',  'forceful', 'cheerful',
                         "cheerful", "reflective", "gloomy", "humorous", "melancholy", "idyllic", \
                      "whimsical", "mysterious", "ominous", "calm", "lighthearted", \
                      "hopeful", "angry", "fearful", "tense", "lonely"]))

mood_no_child = assistant_personality_no_child = ['flirtatious', 'attractive', 'sultry','seductive',"romantic",  ]

all_aug_words.extend(mood_list+mood_no_child)
mood_list_set = set(mood_list+mood_no_child)
male_list = ["man", "man", "man", "guy",  "dude", "person", "male person",  "a gentleman", "a guy", "a boyfriend", "an older man", ]
female_list = ["woman", "woman", "woman", "lady", "gal",  "female person", "a lady", "a young woman", "a gal", "a girlfriend", "a woman", "an older woman", ]
#["a boy", "a lad", "a young boy", "a teen boy", "a kid"] , \
#["a girl", "a little lady", "a young girl", "a teen girl", "a kid"] if age in {"Infant/Toddler (0-2 years)", \
image_style_adj = ["", "", "", "", "",  "", "", "off centered ", "left centered ", "right centered ", "bottom centered ", "top centered ", \
                   "neon flared ", "warmly lit ", "starkly lit ", "motion blured ", "zoom blured ", "tilted ", "skewed ", "blurry ", "neon ", \
                   "light flare ", "fish-eye ", "closeup ", "wideshot ", "colorized ", "black and white ", "sepia ", "bi-chromatic ", "tri-chromatic "]
image_type_lst = ["3D art", "2D art", "vector-art ", "movie-still", "screenshot", "advertisement", "selfie", "instagram image", "wood relief", "pencil drawing", "pen and ink drawing", \
                      "textbook-illustration", "realistic-drawing", "sketch", "cartoon", "painting", "diagram"]
all_aug_words.extend(image_type_lst)
all_aug_words = set(all_aug_words)

digits_to_words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen',
 'eighteen', 'nineteen', 'twenty']
