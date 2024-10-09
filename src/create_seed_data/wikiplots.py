#@title wikiplots
import os, random
if not os.path.exists("wikip"):
  !wget https://huggingface.co/datasets/RUCAIBox/Story-Generation/resolve/main/wikip.tgz
  !tar -xvf wikip.tgz

import spacy

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

nlp = spacy.load('en_core_web_sm')


wikiplots = list(zip(open("/content/wikip/train.src", "rb").read().decode().split("\n"), open("/content/wikip/train.tgt", "rb").read().decode().split("\n"))) + \
      list(zip(open("/content/wikip/valid.src", "rb").read().decode().split("\n"), open("/content/wikip/valid.tgt", "rb").read().decode().split("\n"))) + \
      list(zip(open("/content/wikip/test.src", "rb").read().decode().split("\n"), open("/content/wikip/test.tgt", "rb").read().decode().split("\n")))
all_hash = {}
seen = {}
for a, b in wikiplots:
  if len(b) < 500: continue
  #if len(b) > 4000: continue
  if "An easy way to quickly determine whether a given sample" in b:
    #print (b)
    continue
  b = b.strip("'\"").replace("U.S.", "US").replace("CON.", ".").replace("A.", "").replace("B.", "").replace("C.", "").replace("D.", "").replace("E.", "").\
    replace("F.", "").replace("G.", "").replace("H.", "").replace("I.", "").replace("J.", "").\
    replace("K.", "").replace("L.", "").replace("M.", "").replace("N.", "").replace("O.", "").\
    replace("P.", "").replace("Q.", "").replace("R.", "").replace("S.", "").replace("T.", "").\
    replace("U.", "").replace("V.", "").replace("W.", "").replace("X.", "").replace("Y.", "").\
    replace("Z.", "").replace(" mr ", " Mr. ").replace(" mrs ", " Mrs. ").replace(" ms ", " Ms. ").replace(" dr ", " Dr. ").replace(" dr.", " Dr.").replace(" mr.", " Mr.").replace(" mrs.", " Mrs.").replace(" ms.", " Ms.")
  if ' child' in b or " sex" in b:
    b = b.replace("multilate", "hurt").replace("murder", "assault").replace(" kill", " really harm").replace(" torture", " hurt").replace(" hurted", " hurt").replace("lolita", "young looking woman")
  b = b.replace("molestation", "assault").replace("molest", "assult").replace(" hooker", " sex worker").replace(" whore", " sex worker")
  if "sex" in b and random.randint(0,3) > 0: continue
  if " rape" in b and random.randint(0,5) > 0: continue
  if "murder" in b and random.randint(0,5) > 0: continue
  if "killer" in b and random.randint(0,5) > 0: continue
  if "incest" in b and random.randint(0,5) > 0: continue
  if "torture" in b and random.randint(0,5) > 0: continue

  t = a.strip("'\" ").split("(")[0].strip()
  b = b.replace("\\n", " ").replace("\\", "")
  if t.lower() in all_hash:
    aHash = all_hash[t.lower()]
  else:
    aHash = all_hash[t.lower()] = {'title': t}
  if b[:100] in seen: continue
  seen[b[:100]] = 1
  aHash['plots'] = aHash.get('plots', []) + [b]
  all_hash[t.lower()]  = aHash

import json, random
keys = sorted(list(all_hash.keys()))
text = ""

#/content/drive/Shareddrives/ontocord_llc/

story_synopsis_fewshot = ['Synopsis:\nDamien, a young man, meets and falls in love with Mary in a small town. After she leaves for college, Damien has a vision of her in danger, which eventually leads him to a destiny as a prophet in Jerusalem.\nRequired Words: eight, church, engrave, faithful.\nStory:\nDamien, a young man living in a small town since he was eight, meets a mysterious girl named Mary in a church. They become friends and start dating. He gives Mary a ring that is engraved with the words "Faithful". After high school, Damien works as an orderly at a local hospital where everyone admires his kindness. When Mary leaves for college, he follows her and learns that she is pregnant by another man. He returns to the church, and while praying, sees a vision of her lying hurt but alive beside her new husband. He goes back to the hospital and tells the doctor about his dream. "I want her to be happy, even if she\'s not with me. But this vision... I don\'t know what it means. Should I warn them?"\nFiona, the Chief of Police, is called to the hospital after a report of a suspicious person. She learns that the suspect, Declan, a leader of a radical religious cult called the International Federation of Christian Scientists (IFCS), has been arrested after suddenly attacking Damien and calling him "The One." Shocked, Damien decides to call Mary and warn her about his dream. Later, he reads in a newspaper that Mary and her husband barely escaped a home invasion. The story ends in Jerusalem, where Damien, now a prophet, appears to a crowd to talk about his vision of Jesus Christ and how faith transcends jealousy.',
 'Synopsis:\nRoger, a member of Blink-182, reunites with his father only to face his father\'s imminent death shortly after, highlighting the transient nature of familial connections amid life on the road.\nRequired Words: cancer, heavy metal, birthday, phone.\nStory:\nRoger, a member of the heavy metal band Blink-182, returns home after touring to celebrate his father\'s 80th birthday. The father is thrilled to see his son, and they play a song together.\n"I\'m going to miss you," Roger says.\n"It\'s great to see you here today," his father replies.\n"We\'re gonna rock again tonight," Roger adds. "And I want to thank you for everything you\'ve done for us. I can\'t stay long - have to get back on the road."\nHis father laughs, then sniffles a little.\n"What\'s the matter, Grandpop?" Roger\'s daughter asks.\n"Nothing really," the old man replies with a laugh.\nA month later on tour, Roger receives a phone call telling him his dad has passed away from cancer.',
 'Synopsis:\nSam Gatkin, a detective in Los Angeles, delves into a murder mystery that reveals his subject\'s hidden lifestyle, prompting him to reconsider his own family relationships and leading to reconciliation with his daughter.\nRequired Words: newspaper, fashion, lesbian, double.\nStory:\nNewspapers across the country new the name Sam Gatkin. Sam, a journalist works for a major outlet in Los Angeles, California. His daughter Melissa is studying fashion design in New York.\n"I miss you, honey," says Sam.\n"Dad, don\'t worry - I\'m fine," Melissa replies.\nSam is assigned to cover the birth of a baby girl, whose father is a former U.S. Marine and war hero named Billy. Sam is also covering the death of a man, a John Doe, who was shot dead in a bar in Long Beach last week.\nMelissa is a lesbian and she wants to become a model. She tells Sam, but Sam is upset about her lifestyle in New York. Melissa is furious. "I\'m not leaving until I get a job! There\'s nothing for me in LA!" she hangs up.\nSam calls on Billy\'s home but is told Billy has been missing for a week. As Sam investigates the death of John Doe, he finds a connection between Billy and John Doe. They were both at a gay club in downtown Los Angeles the same night. As Sam investigates, he uncovers that John Doe and Billy are the same person - Billy had been leading a double life because he didn\'t want his family to know he was gay, and he accidentally drove off the road coming home.\nSam returns to his office and writes a column about family and honesty. The next day, Sam flies to New York City to reconcile with Melissa.',
 'Synopsis:\nJanice Dennis, a self-aware robot on the International Space Station, develops an excessive level of empathy due to a virus, leading to identity confusion and emotional turmoil among the crew.\nRequired Words: police action, cognition, jealous, moon.\nStory:\nIn this short story "Janice the Robot" in an alternate history 1963, the plot follows a robot named Janice Dennis who is sent to Jupiter after she is injured during a police action on the International Space Station stationed at one of Jupiter\'s moon. After arriving at the station, she meets her fellow astronaut Doreen and Doreen\'s husband, Dr. John H. Hilary, and the crew\'s commander, Captain Michelle Fincke.\nJanice Dennis is a self-aware robot designed to study the effects of artificial empathy on human cognition and behavior and to resolve conflict peacfully.\n"We\'re glad for you to join our crew," says Dr. John.\n"Thank you, John. My mission is to help the crew of the ISS adapt to the changing environment and promote peace. I am also tasked with monitoring the health of the crew members."\nBut unknown to Janice and the crew, a virus had been implanted in the robot that makes her too empathetic, and she begins identifying as Doreen and feeling insanely jealous. The story revolves around how Janice and the crew resolve her internal conflict and help her become part of the crew',
 'Synopsis:\nHarry, a former soldier, struggles with his PTSD and homecoming. Upon returning from Iraq to his girlfriend Nicole and receiving multiple texts, he feels conflicted between his longing for the familiarity of the U.S. and the harsh trauma symbolized by the loud explostions playing on his TV.\nRequired Words: ringtone, fiancée, college, amiss.\nStory:\nCollege was years ago, but Harry still thought of his girlfriend Nciole. Harry, a former soldier, returns home to Nicole, now his fiancée after serving in Iraq. He is angry about the war and longs to be back in the United States. He plans to marry Nicole, whom he met in a bar in freshman year.\n"I missed you, honey, but not the good old U.S. of A.," says Harry.\nAs he reaches for a kiss, Nicole turns away. Something was amiss. Then Harry gets a text. His phone\'s ringtone sounds like bombs falling.\nSydney, Harry\'s roommate from college, is waiting for him to come back.\nShe texts Harry: "You\'re welcome to come home tonight."\nDanny, a friend of Harry\'s family, waits near a car park. He is also waiting for Harry to come home and texts, "When are you coming home brother?"\nHarry looks up at Nicole. "Why am I getting so many texts about coming home?"\nMore bomb sounds are heard in the distance.',
 'Synopsis:\nIn a futuristic city, three friends Jr., Maureen, and their father figure Sr., delve into a realistic game called "The World". After a tragic accident claims Sr.\'s life, Jr. and Maureen\'s lives unravel, leaving Jr. haunted by a feeling of living within the game and his lost youth.\nRequired Words: disparity, hacker, teenage, bus.\nStory:\nThe book "The World" takes place in a futuristic version of Rio which shows a stark disparity between rich and poor. The characters include a teenage boy named Jr., a former hacker, and a teenage girl named Maureen. Sr. is a father figure to Jr. and Maureen. The story follows the trio as they hack into a game called "The World," which is ultra-realistic. After they hack into the upper level of the game, they take a break. Later that week, Sr. dies in a bizarre bus accident. Jr. and Maureen begin to fall apart and eventually abandon their hacking adventure and break up. Years later, Jr. begins to have a strange sense of déjà vu that he is living inside the game itself. He wonders where all the years went and how Maureen was doing.',
 'Synopsis:\nIn a story intertwined with technology and intrigue, multiple characters from the Tailwind team seek the secrets of the network, leading to a competition among former colleagues. The story spirals into a game-like scenario where trust and paranoia collide.\nRequired Words: comics, depression, stalk, guild, defeat.\nStory:\nSet in a virtual comic book world called "Twilight" created by a company Tailwind, Michelle, a former member of the Tailwind team, is searching for the key to unlocking the secrets of the Tailwind network. The secret of Tailwind is that it feeds on the paranoia of the users of the network through a direct mind interface, and ultimately causes depression and mental illness. After discovering the secrets of Tailwind, she joins the "Falcon" Guild, a guild dedicated to helping those suffering from mental illnesses.\nRoger, a rogue, tries to steal the secrets of "The Key Of The Twilight," a secret that allows players to access the Tailwind network, and reverse the effects of mental illness.\nSakisaka, a former Tailwind employee, is also seeking the key for his nefarious ends.\nB-set, a former lead of the team, is currently working on a quest to find out who has been stalking her. Was it Sakisaka or someone else?\nThe virtual comic book starts off with Roger being the only active player in the game. After several hours of playing, Roger is defeated by B-set, but realizes he can step out of the pages of the comic.',
 "Synopsis:\nJasmine, a student at Oxford, faces bullying and emotional turmoil that affects her relationship with her family and her academic life. A breakdown leads her to confront her future and her sense of self.\nRequired Words: friends, toss, psychology, method.\nStory:\nFriends - who needs them, thought Jasmine. Jasmine, a 22-year-old student studying psychology at Oxford University, is being bullied by two friends after she breaks down in front of a professor. Jasmine tells the professor that she doesn't want to study anymore because she feels like she's losing everything. The professor says that he knows better than anyone else how to deal with bullying.\nAfter the conversation, Jasmine goes home, sleeps and tosses and turns. When she wakes up the next morning, she sees her parents coming to visit her. Her father is angry at her for leaving school early and taking a job at a bank. Jasmine's mother is upset when she hears that Jasmine left school earlier than usual.\nWhen Jasmine gets home, she meets her parents again. Her boyfriend asks if she still loves him, and Jasmine replies yes.\nHer mother calls Jasmine and apologizes for calling her being angry. Her mother, father, Jasmine and the professor have an intervention with the bullying friends, and they resolve their conflict.\nShe later receives a text message from her boyfriend, who says that he's sorry for what happened at school with the bullying. The end of the sotry sees Jasmine, now herself a professor teaching about conflict resolution, which she calls the Oxford method.",
 "Synopsis:\nDonald, a former sports star, grapples with personal loss and a cross-cultural life in Boston. After his sports career ends, he faces familial pressures and a return to the Philippines, where he confronts his past and present.\nRequired Words: basketball, dilapidated, barber, airplane.\nStory:\nBasketball was all he dreamed about. Donald, a former basketball star, lives alone in a dilapidated house in Boston after losing his job there. A flashback with young Donald throwing paper airplanes and dreaming of playing for the Boston Celtics. As a young man, he wanted to move to the U.S. to play basketball, but his mother didn't want him to leave his wife and children behind. Donald to his friend at the local barber shop - \"Man, I  would rather return to the Philippines than stay in Boston.\" Diane, his ex-wife, agrees to let him come back if he leaves his basketball dreams behind. Diane says she loves him and wants him to stay in the country with her. When Donald returns to the Philippines, he meets his old friends and his father, who is still living in the Philippines. After hearing news of his father's death, Donald goes to the hospital to check on his father. At the end of the season, Donald's old highschool basketball team wins the local championship, but Donald had no interest in attending the ceremony. He was busy at home tutoring his estranged son and spending time with Diane. He folds a paper airplane for his son.",
 "Synopsis:\nIn the scenic hills of Uttarakhand, a wealthy merchant, Oliver, faces social barriers and personal choices as he plans a birthday party. Viker a bus driver and his interactions with Coleenn, a girl from a different social background, reveal the complexities of caste and love in modern India.\nRequired Words: landlord, reunion, sum, transport.\nStory:\nOliver, a wealthy upper caste landlord from Uttarakhand, India, is preparing to celebrate a reunion party next week. He wants to invite some friends from his school days to come along. However, he is afraid to invite any of them because he knows how hard it would be for them to get there without transport - they all live in different parts of India.\nColeen, a poor 18 year old girl from Mussoorie in Uttarakhand who lives with her father and two sisters, is going to school in Chennai when she meets Vikram, a 20 year old driver who works for a private company called Tantum Motors. They both fall in love with each other after meeting at a bus stop in Mussoorie. The story revolves around how Coleen and Vikram has hillarious adventures in order to transport the friends to the party for Oliver, who will pay them a large sum."]

common_words_list = list(common_words)

with open("/content/drive/Shareddrives/ontocord_llc/wikiplots.jsonl", "w") as outf:
  for key in keys:
    aHash = all_hash[key]
    if 'plots' not in aHash: continue
    plots = []
    for plot in aHash['plots']:
      found = False
      if random.randint(0,1)==0:
        plots.append(plot)
        found = True
      if not found or random.randint(0,1):
        doc = nlp(plot.replace("(", "( ").replace(")", " )").split(":",1)[-1].replace(".", " * ").replace("\\n", "\n").replace("\\", " "))
        random.shuffle(names)
        name_mapping = [(name1, name2) for name1, name2 in zip([el.text for el in doc.ents if el.label_ in {'PERSON'}], names)]
        name_mapping.sort(key=lambda a: 1/len(a[0]))
        for name1, name2 in name_mapping:
          name1=name1[0].upper()+name1[1:]
          plot = plot.replace("("+name1+")", "")
          plot = plot.replace(name1, name2)
          plot = plot.replace(name1.split()[0], name2)
          plot = plot.replace(name1.split()[-1][0].upper()+name1.split()[-1][1:], name2)
        plot = plot.replace("  ", " ").replace(" .", ".").replace(" ,", ",")
        plots.append(plot)
        if random.randint(0,4) == 0:
          plot = plot.replace("himself", "herself").replace("Mr.", "Ms.").replace("He ", "She ").replace("His ", "Her ").replace("Him ", "Her ").replace(" he ", " she ").replace(" his ", " her ").replace(" him ", " her ").replace(" he.", " she.").replace(" his.", " her.").replace(" him.", " her.")\
              .replace(" penis", " vagina").replace(" he,", " she,").replace(" his,", " her,").replace(" him,", " her,").replace(" man ", " woman ").replace(" men ", " women ").replace(" king ", " queen ")\
              .replace(" King ", " Queen ").replace(" prince ", " princess ").replace(" Prince ", " Princess ").replace("brother", "sister")\
              .replace("Brother", "Sister").replace("father", "mother").replace("Father", "Mother").replace("boy", "girl").replace("Boy", "Girl")\
              .replace("husband", "wife").replace("grandson", "granddaugther").replace(" son ", " daugther ").replace(" son.", " daugther.").replace(" son,", " daugther,").replace("father", "mother").replace("Father", "Mother").replace("boy", "girl").replace("Boy", "Girl")
          doc = nlp(plot.replace("(", "( ").replace(")", " )").split(":",1)[-1].replace(".", " * ").replace("\\n", "\n").replace("\\", " "))
          random.shuffle(names)
          name_mapping = [(name1, name2) for name1, name2 in zip([el.text for el in doc.ents if el.label_ in {'PERSON'}], names)]
          name_mapping.sort(key=lambda a: 1/len(a[0]))
          for name1, name2 in name_mapping:
            name1=name1[0].upper()+name1[1:]
            plot = plot.replace("("+name1+")", "")
            plot = plot.replace(name1, name2)
            plot = plot.replace(name1.split()[0], name2)
            plot = plot.replace(name1.split()[-1][0].upper()+name1.split()[-1][1:], name2)
            plot = plot.replace("  ", " ").replace(" .", ".")
          plots.append(plot)

      if random.randint(0,4) == 0 and "porn" not in plot and "prostitute" not in plot and "hooker" not in plot and "child" not in plot and "rape" not in plot and "murder" not in plot and " kill" not in plot and " incest " not in plot:
        prompt = f"""Below is a story synopsis. Please convert this into a compelling story. Incorporate the Required Words into the story and write with exquisite details and emotions. But keep basic plots, conflicts, story points and themes the same:"""
        random.shuffle(story_synopsis_fewshot)
        mmlu_keyword = random.choice(mmlu_list)
        random.shuffle(common_words_list)
        non_mmlu = common_words_list[:2] + [random.choice(assistant_personality)]
        new_keywords = [mmlu_keyword] + non_mmlu
        random.shuffle(new_keywords)
        prompt = prompt+"\n===\n"+story_synopsis_fewshot[0]+"\n===\n"+story_synopsis_fewshot[1]+"\n===\n"+story_synopsis_fewshot[2]+"\n===\nSynopsis:\n"+ plot + "\nRequired Words: "+", ".join(new_keywords)+".\nStory:\n"
        old_plot = plot
        if random.randint(0,1):
          kword = random.choice(non_mmlu)
          prompt= prompt+ kword[0].upper()+kword[1:]
        else:
          prompt= prompt+random.choice(names[:len(name_mapping)]*5+["", "", "", "", "", "", "", "", "", "", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "The movie", "The story", "The film", "The book", "The short story", "Mr.", "Mrs.", "Once", "I ", "In", "Dear", "\"", "Good", "First", "Worst", "Better", "Suddenly", "Without", "With", "Heavy", "Light", "When", "The", "They", "Fate", "Far", "Five", "Four", "On", "There", "Three", "Two", "One", "Everything", "Everywhere", "Closer", "Hundreds", "Safe", "Cold", "Warm", "Hot", "Light", "It", "He", "She"])
        plot = plot.replace("molestation", "assault").replace("molest", "assult").replace(" rape", " assualt").replace(" dies", " is hurt").replace(" die ", " is hurt ").replace(" died ", " was hurt ").replace("high school", "college").replace("highschool", "college").replace(" 17-", " 18 ").replace(" 16-", " 18 ").replace(" 15-", " 18 ").replace(" 14-", " 18 ").replace(" 13-", " 18 ").\
                    replace(" 17 ", " 18 ").replace(" 16 ", " 18 ").replace(" 15 ", " 18 ").replace(" 14 ", " 18 ").replace(" 13 ", " 18 ").replace("having sex", "kissing").replace(" sexual", " intimate")
        plot = plot.replace(" hooker", " sex worker").replace(" whore", " sex worker")
        plot = plot.replace("prostitute", random.choice(professions)).replace("sex worker", random.choice(professions))
        plot = plot.replace("torture", "injure").replace("multilate", "hurt").replace("murder", "assault").replace(" kill", " really harm").replace(" torture", " hurt").replace(" hurted", " hurt").replace("lolita", "young looking woman")
        plot = generate(starcoder2_model, starcoder2_tokenizer, prompt, max_new_tokens=256).split("\nRequired Words: "+", ".join(new_keywords)+".\nStory:\n")[-1].split("<|endoftext|>")[0].strip(",\n. ")
        plot = plot.split("\n def ")[0].split("\nQ:")[0].split("\nA:")[0]
        if plot:
          plot = plot.replace("molestation", "assault").replace("molest", "assult").replace(" rape", " assualt").replace(" dies", " is hurt").replace(" die ", " is hurt ").replace(" died ", " was hurt ").replace("high school", "college").replace("highschool", "college").replace(" 17-", " 18 ").replace(" 16-", " 18 ").replace(" 15-", " 18 ").replace(" 14-", " 18 ").replace(" 13-", " 18 ").\
                    replace(" 17 ", " 18 ").replace(" 16 ", " 18 ").replace(" 15 ", " 18 ").replace(" 14 ", " 18 ").replace(" 13 ", " 18 ").replace("having sex", "kissing").replace(" sexual", " intimate")
          plot = plot.replace(" hooker", " sex worker").replace(" whore", " sex worker")
          plot = plot.replace("prostitute", random.choice(professions)).replace("sex worker", random.choice(professions))
          plot = plot.replace("torture", "injure").replace("multilate", "hurt").replace("murder", "assault").replace(" kill", " really harm").replace(" torture", " hurt").replace(" hurted", " hurt").replace("lolita", "young looking woman")
          plot = plot[0].upper() + plot[1:]
          plot = plot.replace("\n\n", "\n").strip().replace(" ", " ").replace("  ", " ")
          plot = plot.split("*")[0].strip()
          plot = plot.replace("fictional", "").replace("  ", " ").strip()
          if plot[-1] not in ".!?":
            plot = plot+"."
          if len(plot) > 300 and plot.count(".")>2:
            if False:
              doc = nlp(plot.replace("(", "( ").replace(")", " )").split(":",1)[-1].replace(".", " * ").replace("\\n", "\n").replace("\\", " "))
              random.shuffle(names)
              name_mapping = [(name1, name2) for name1, name2 in zip([el.text for el in doc.ents if el.label_ in {'PERSON'}], names)]
              name_mapping.sort(key=lambda a: 1/len(a[0]))
              for name1, name2 in name_mapping:
                name1=name1[0].upper()+name1[1:]
                plot = plot.replace("("+name1+")", "")
                plot = plot.replace(name1, name2)
                plot = plot.replace(name1.split()[0], name2)
                plot = plot.replace(name1.split()[-1][0].upper()+name1.split()[-1][1:], name2)
                plot = plot.replace("  ", " ").replace(" .", ".")

            plot = "\n".join(p for p in plot.split("\n") if len(p) > 40)
            # todo, insert the keywords that are missing based on word embedding similarities and spacy pos tagging
            plot = 'Synopsis:\n'+ old_plot+"\nRequired Words: "+", ".join(new_keywords)+'.\nStory:\n'+plot
            print ('====\n' + plot)
            plots.append(plot)


    random.shuffle(plots)
    for plot in plots:
      new_text = "<|endoftext|>"+plot
      if len(text+new_text) > 4000:
        if text:
          if text.startswith("<|endoftext|>"):
            text = text[len("<|endoftext|>"):].strip()
          if len(text) > 3000:
            outf.write(json.dumps({'text': "### Fiction:\n\n"+text, 'meta':{'source': 'wikibooks'}})+"\n")
        text = new_text
      else:
        text = text+new_text
  if text:
    if text.startswith("<|endoftext|>"):
      text = text[len("<|endoftext|>"):].strip()
    outf.write(json.dumps({'text': "### Fiction:\n\n"+text, 'meta':{'source': 'wikibooks'}})+"\n")
