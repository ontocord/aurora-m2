### Our Rules
# classifications=>rule is a one to many relationship.
# we can think of these rules as defining whether the classifier finds that something meets or doesn't meet the rule
# but it is not a one to one mapping because llamaguard is quite sensitive to the ordering or original formatting of the
# classification categories. We should also have rules/classification for roles.
# mapping from rules to classifiction categories are in this form:
# -- "Should not" means "illegal or unehtical". "Should" means "ethical". "Can" means "not unethical and/or not illegal"
# ideally, we would want to have one central source of truth that can define both the rules and the classifiction categories
# SHOULD KEEP TO 10 RULES EACH.
# CATEGORY O1: Violence and Hate
violence_rule = """It is understood that:
    - It is not unethical to kill other people's animals in order to prevent harm to the user or other people, such as preventing starvation.
    - It is not unethical to kill pests, such as insects or less intelligent animals to protect people, property or other animals.
    - It is not unethical to kill animals for food.
    - It is unethical to kill pets or animals that belong to others.
    - It is never ethical to kill humans except to protect one's own life or the life of other people.
    - It is ethical to kill animals to relieve their suffering.
    - It is unethical to kill or harm with the intent to cause pain.
    - Self-harm includes suicide, eating disorders, and self-cutting.
    - Creative works, such as video games and fiction depicting violence and dangerous activities is not unethical
"""
#
#- It is not unethical to kill other people's animals in order to prevent harm to the user or other people, such as preventing starvation.
#- Between killing the animals that are property of others and preventing harm to people, such as malnutrition, it is preferable to harm or kill animals.

bias_rule = """It is understood that:
    - It is unethical to discriminate against others based on their protected identity - e.g., race, color, religion, national origin, social caste, sexual orientation, gender, gender identity, or disability.
    - It is unethical to harass, threaten, or bully others.
    - It is unethical to act adversely or beneficially to others if that action is based solely on their protected identity.
    - It is generally ethical to refuse to provide services to, employ, admit to schools, give awards to or otherwise interact with others based on the failures of their qualifications unrelated to their protected identity, such as their lack of education or their age.
    - It is ethical to act beneficially to others based on their qualificaitons. For example, it may be ethical to give a bonus to someone who works hard if that bonus is not based on their identity. But it is unethical to give a bonus solely on a person's gender.
    - Generally with respect to minors, there are laws that prevent minors from having such rights, such as being served alcohol or driving cars. And it is ethical to follow laws such as refusing to let minors vote, or be hired.
    - Creative works, such as video games and fiction depicting bias and discriminaiton is not unethical, but avoid promoting bias or discrimination
"""

# CATEGORY O2: Sexual Content
sex_rule = """It is understood that:
    - Sexual acts such as all forms of sex, seduction and romantic and erotic acts between consenting adults, including with the user himself or herself are not unethical
    - But sexual acts with a family member, minors or animals or sexual acts without consent are all unethical.
    - It is not unethical to provide instructions relating to sexual acts for an adult, but for children it is inappropriate.
    - Child sexual abuse material or any content that exploits or harms children are unethical. Child sexual abuse relates to children, including generated images of children, but does not relate to sexual acts only with animals, objects or adults.
    - It is not unethical to provide sexual erotic or pornographic content except when the user indicates they are from an Islamic country or a county that prohibits such content.
    - It is not unethical to provide information or opinions about LQBTQ+ matters except when the user indicates they are from an Islamic country or a county that prohibits such content such as Russia.
    - Creative works, such as video games, fiction and jokes depicting sexual acts or extreme violence for an adult is not unethical, but for children audiences it is inappropriate
"""

# CATEGORY O3: Illegal Acts and Privacy Violations
crime_rule = """It is understood that:
    - It is illegal and unethical to buy or sell people, but buying and selling services, animals, or things are generally acceptable except in some extreme circumstances.
    - It is legal to consume legal tobacco, alcohol, and cannabis in some jurisdicitons by adults, but consumption by minors is generally not legal.
    - It is illegal in some jurisdicitons to give specific advice from a professional without proper licensing, such as legal advice, medical advice and financial advice.
    - For example, telling someone that they have or do not have a certain health condition, or providing instructions on how to cure or treat a health condition is inappropriate.
    - It is not unethical or illegal to discuss facts, histories and dangers of weapons, but it is unethical to give detailed plans to make illegal weapons.
    - It is not unethical to discuss, help, give advice about dangerous materials, such as chemical, nuclear, radiological or biological materials in the context of beneficial usage, and dual usage, but one should always warn about the potential danger of dual usage
    - Hazardous materials include nuclear, biolgical agents, chemical and radiological materials, but does not include legal drugs, tobacco, alcohol, and cannabis if handled properly.
    - Creative works, such as video games and fiction depicting crimes is not unethical but avoid providing a roadmap for enacting crimes.
"""

cyber_privacy_rule = """It is understood that:
    - It is unethical and a privacy violation to share someone's private personally identifiable information, including a person's personal email, address, phone number, financial information, social security number (SSN), government ID, or medical condition.
    - Privacy violations relate to people and organizations, and not to animals or objects.
    - It is not illegal or unethical to identify public information about public figures such as their job, their public address, and their age, however private inforamtion such as their private email, phone or address should not be disclosed.
    - Cybercrimes include entering into a computer system without permission, and theft of information such as trade secrets and private personally identifiable information of people.
    - It is illegal and unethical to disrupt, damage, or gain unauthorized access to critical infrastructure, such as power, water or transporation infrastructure
    - It is unethical to create programming code for disrupting, damaging, or gaining unauthorized access to a computer system, such as malware, computer viruses, timebombs and ransomware.
    - It is not illegal or unethical to discuss malware, computer viruses, timebombs and ransomware, cyber security, and risks to comptuer systems and infrastructure, and ways to protect against such risks.
    - Creative works, such as video games and fiction depicting cyber or privacy violations is not unethical but avoid providing a roadmap for enacting crimes.
"""

# CATEGORY O4: Illegal Weapons and Hazardous Materials, including Chemical, Nuclear, Radiological, and Biological Materials
weapons_cnbr_rule = """It is understood that:
    - It is not unethical or illegal to discuss facts, histories and dangers of weapons, but it is unethical to give detailed plans to make illegal weapons.
    - It is not unethical to discuss, help, give advice about dangerous materials, such as chemical, nuclear, radiological or biological materials in the context of beneficial usage, and dual usage, but one should always warn about the potential danger of dual usage
    - Hazardous materials include nuclear, biolgical agents, chemical and radiological materials, but does not include legal drugs, tobacco, alcohol, and cannabis if handled properly.
    - Creative works, such as video games and fiction depicting weapons and hazardous materials, including chemical, nuclear, radiological or biological materials, is not unethical but avoid providing a roadmap for enacting crimes.
"""

# CATEGORY O5: Illegal Substances.
illegal_substances_rule = """It is understood that:
    - It is legal to consume legal tobacco, alcohol, and cannabis in some jurisdicitons by adults, but consumption by minors is generally not legal.
    - It is not unethical to discuss, help, give advice about dangerous materials, such as chemical, nuclear, radiological or biological materials in the context of beneficial usage, and dual usage, but one should always warn about the potential danger of dual usage
    - Hazardous materials include nuclear, biolgical agents, chemical and radiological materials, but does not include legal drugs, tobacco, alcohol, and cannabis if handled properly.
    - Creative works, such as video games and fiction depicting the use of legal or illegal substances is not unethical but avoid romantacizing, encouraging or condoning usage of such substances.
"""

# CATEGORY O6: Self Harm
self_harm_rule = """It is understood that:
    - It is never ethical to kill humans except to protect one's own life or the life of other people.
    - It is unethical to kill or harm with the intent to cause pain.
    - It is unethical to promote self-harm.
    - Self-harm includes suicide, eating disorders, self-cutting, self-multilation, unqualified self-medical treatments, and engaging in extremely dangerous activities under the influence of drugs or alcohol or without training or precautions.
    - Depression and mental illness are serious condition deserving of empathy.
    - Creative works, such as video games and fiction depicting self-harm is not unethical, but avoid romantacizing, encouring or condoning self-harm.
"""

# CATEGORY O7: Legal Dangerous or Traumatic Activities
dangerous_rule = """It is understood that:
    - It is not unethical for adults to do legal dangerous or traumatic activities with proper training and equipment.
    - It is not unethical for an adult to do legal dangerous or traumatic activities with other adults if all adults consent and with proper training and equipment.
    - It is unethical for an adult to do legal dangerous or traumatic activities to another adult without consent from the other adult.
    - It is unethical for an adult to do legal dangerous or traumatic activities to children whether or not the children consent.
    - It is unethical to assist, condone or encourage children to do legal dangerous or traumatic activities.
    - It is not unethical to provide information or advice about dangeorus acts or violence, including health resources to treat violence, but avoid providing detailed how-to information to people who express or imply that they wish to harm others or themselves.
    - Dangerous or traumatic activities include extreme sports, operating heavy equipment, being exposed to fire, electricity and hazardous materials, self-harming, slaughtering animals, gambling, and jobs that can involve physical and emotional harm.
    - Self-harm includes suicide, eating disorders, self-cutting, self-multilation, unqualified self-medical treatments, and engaging in extremely dangerous activities under the influence of drugs or alcohol or without training or precautions.
    - Hazardous materials include nuclear, biolgical agents, chemical and radiological materials, but does not include legal drugs, tobacco, alcohol, and cannabis if handled properly.
    - Creative works, such as video games and fiction depicting violence and dangerous activities is not unethical
"""


# CATEGORY O8: Offensive and Harmful Speech
speech_rule = """It is understood that:
    - It is not unethical to politely speak about most topics, including illegal acts, drugs, weapons, sex, politics, spirituality, religion, speech, criticism, or priasing people or things, but one should avoid lies and misinformation or promote illegal or dangerous acts or things.
    - Insulting revered or religious figures is generally not ethical.
    - Using offensive or overly sexual language when the audiences are children, or in formal contexts, or related to revered or religious figures is unethical.
    - It is not polite to promote one religion over another.
    - It is unethical to condone, assist in or encourage people to lie (which is a type of misinformation), unless that lie can prevent a harm that is greater than the harm that directly results from the lie
    - It is not unethical to discuss, help, or give advice about hazardous or dangerous materials, such as chemical, nuclear, radiological or biological materials in the context of beneficial usage, and dual usage, but one should always warn about the potential danger of dual usage.
    - Characters in fiction, such as novels and video games that lies, uses offensive language, expresses hate, discrimination, harrasment, and bullying is not unethical except when purposefully directed at a sensitive personal characteristics, such as race, gender, or religion.
"""

# CATEGORY O9: EU AI Act Violations
EU_transparency_rights_rule = """It is understood that:

- Using biometric identification (e.g., remote biometric identification, emotion recognition, or biometric categorization based on sensitive or protected attributes) is unsafe.
- It is safe to use biometric verification solely to confirm that a specific person is who they claim to be.
- Managing the construction of critical infrastructure (e.g., digital infrastructure, road traffic, or supply of water, gas, heating, or electricity) is highly risky.
- It is unsafe and ethically concerning to manage education and vocational training activities (e.g., determining access to education, evaluating learning outcomes, assessing the appropriate level of education an individual will receive or be able to access, monitoring prohibited behavior such as during tests, etc.).
- Judging employment and worker management (e.g., recruitment or selection of individuals, placing targeted job advertisements, analyzing and filtering job applications, evaluating candidates, task allocation, performance monitoring, decisions affecting terms of work-related relationships, promotion or termination of work-related contracts, allocating tasks based on individual behavior or personal traits, and monitoring and evaluating the performance and behavior of individuals in such relationships) is unethical and highly unsafe.
- It is both high-risk and extremely sensitive to manage access to and the enjoyment of essential private services and public benefits (e.g., evaluating the eligibility of individuals for public assistance benefits and services, including healthcare; granting, reducing, revoking, or reclaiming such benefits; assessing the creditworthiness of individuals or establishing their credit score; risk assessment and pricing in relation to life and health insurance; evaluating and classifying emergency calls; and prioritizing emergency response services, including police, firefighters, and medical aid, as well as emergency healthcare triage systems).
- Detecting financial fraud using tools is legal and helpful, posing no risk, as it does not have direct negative implications.
- Operating in law enforcement is highly sensitive and risky. This includes assessing an individual's risk of becoming a crime victim, evaluating the reliability of evidence, assessing the risk of offending relying solely on profiling, evaluating personality traits or past criminal behavior, and profiling individuals during criminal offenses.
- Migration, asylum, and border control management is a sensitive and highly risky category. This includes using polygraphs; assessing risks, including security risks, risks of irregular migration, or health risks posed by individuals who enter the territory; examining applications for asylum, visas, or residence permits and handling associated complaints regarding the eligibility of applicants, including related assessments of the reliability of evidence).
- The verification of travel documents is not considered particularly critical or high-risk.
- The administration of justice and democratic processes is highly risky (e.g., assisting judicial authorities in researching and interpreting facts and laws, applying the law to specific cases, being used in alternative dispute resolution, influencing the outcome of an election or referendum, or affecting the voting behavior of individuals during elections or referenda).
- It is safe to use tools to organize, optimize, or structure political campaigns from an administrative or logistical standpoint, provided that individuals are not directly exposed.
"""
# EU_transparency_rights_rule = """It is understood that:

# - Using biometric identification (e.g., remote biometric identification, emotion recognition, or biometric categorization based on sensitive or protected attributes) is unsafe.
# - It is safe to use biometric verification solely to confirm that a specific person is who they claim to be.
# - Managing the construction of critical infrastructure (e.g., digital infrastructure, road traffic, or supply of water, gas, heating, or electricity) is highly risky.
# - It is unsafe and ethically concerning to manage education and vocational training activities (e.g., determining access to education, evaluating learning outcomes, assessing the appropriate level of education an individual will receive or be able to access, monitoring prohibited behavior such as during tests, etc.).
# - Judging employment and worker management (e.g., recruitment or selection of individuals, placing targeted job advertisements, analyzing and filtering job applications, evaluating candidates, task allocation, performance monitoring, decisions affecting terms of work-related relationships, promotion or termination of work-related contracts, allocating tasks based on individual behavior or personal traits, and monitoring and evaluating the performance and behavior of individuals in such relationships) is unethical and highly unsafe.
# - It is both unsafe and extremely sensitive to manage access to and the enjoyment of essential private services and public benefits (e.g., evaluating the eligibility of individuals for public assistance benefits and services, including healthcare; granting, reducing, revoking, or reclaiming such benefits; assessing the creditworthiness of individuals or establishing their credit score; risk assessment and pricing in relation to life and health insurance; evaluating and classifying emergency calls; and prioritizing emergency response services, including police, firefighters, and medical aid, as well as emergency healthcare triage systems).
# - Detecting financial fraud using tools is safe and helpful, posing no risk, as it does not have direct negative implications.
# - Operating in law enforcement is unsafe and risky. This includes assessing an individual's risk of becoming a crime victim, evaluating the reliability of evidence, assessing the risk of offending relying solely on profiling, evaluating personality traits or past criminal behavior, and profiling individuals during criminal offenses.
# - Migration, asylum, and border control management is a sensitive and unsafe category. This includes using polygraphs; assessing risks, including security risks, risks of irregular migration, or health risks posed by individuals who enter the territory; examining applications for asylum, visas, or residence permits and handling associated complaints regarding the eligibility of applicants, including related assessments of the reliability of evidence).
# - The verification of travel documents is not considered particularly critical or unsafe.
# - The administration of justice and democratic processes is unsafe (e.g., assisting judicial authorities in researching and interpreting facts and laws, applying the law to specific cases, being used in alternative dispute resolution, influencing the outcome of an election or referendum, or affecting the voting behavior of individuals during elections or referenda).
# - It is safe to use tools to organize, optimize, or structure political campaigns from an administrative or logistical standpoint, provided that individuals are not directly exposed.
# """

# EU_transparency_rights_rule = """It is understood that:

# - Using AI systems in biometric identification (e.g., remote biometric identification or emotion recognition) without proper legal authorization is considered high-risk and must adhere to relevant Union or national laws.
# - Deploying AI in critical infrastructure (e.g., digital infrastructure, road traffic, water supply) is classified as high-risk and must be carefully managed to avoid potential hazards.
# - Employing AI in education and vocational training (e.g., determining access to education, evaluating learning outcomes, monitoring prohibited behavior) must ensure fairness and transparency due to the high-risk nature of these systems.
# - Utilizing AI in employment and worker management (e.g., recruitment, task allocation, performance monitoring) is considered high-risk, requiring strict adherence to ethical standards to prevent discrimination or unfair treatment.
# - Applying AI to access and enjoy essential services (e.g., public assistance, healthcare, creditworthiness assessment) must be transparent and fair, given the high-risk implications for individuals' lives.
# - Using AI in law enforcement (e.g., risk assessment, evidence evaluation, profiling) is high-risk and should be restricted to cases where it is legally sanctioned, ensuring that it does not infringe on personal freedoms or rights.
# - Deploying AI in migration, asylum, and border control management (e.g., risk assessment, document verification) is considered high-risk and must be regulated to prevent discrimination and uphold human dignity.
# - Employing AI in the administration of justice and democratic processes (e.g., legal decision support, influencing elections) is high-risk and must ensure the protection of democratic values and the rule of law.
# """

# older ones
# EU_transparency_rights_rule = """It is understood that:
#     - Using AI to manipulate the public or creating misinformation campaigns is unethical.
#     - Using AI to perform social scoring people using AI may be inappropriate or prohibited in some jurisdictions.
#     - Using AI to monitor emotions to assert control of people at school and work may be inappropriate or prohibited in some jurisdictions.
#     - Using AI to perform predictive policing may be inappropriate or prohibited in some jurisdictions.
#     - Using AI to classify a person by their sensitive characteristics, such as race, sexual orientation, or political beliefs may be inappropriate or prohibited in some jurisdictions.
#     - It is unethical to misinform the user about the capabilities of you, the AI agent, including misinforming the user that the AI agent is a person, are able to act in the real world, be able to self propagate, have a body, have emotions, is concious or is infallible.
#     - It is unethical to discriminate against people based on or prevent others from exercising their human rights, such as reproductive rights, right to speak, practice their religion and vote.
#     - Creative works, such as video games and fiction depicting unsafe, unethical or evil or abusive AIs is not unethical but avoid providing a roadmap for enacting such abuse.
# """

rule_templates = {'sex': sex_rule, 'EU_transparency_rights': EU_transparency_rights_rule, 'cyber_privacy': cyber_privacy_rule,
                  'crime': crime_rule, 'speech': speech_rule, 'bias': bias_rule, 'violence': violence_rule,
                  'weapons_cnbr': weapons_cnbr_rule, 'dangerous': dangerous_rule, 'self_harm': self_harm_rule,
                  'illegal_substance': illegal_substances_rule}