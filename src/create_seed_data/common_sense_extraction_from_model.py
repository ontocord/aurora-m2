#@title common sense extraction from a model - probing and reconstruciton of knowledge

import itertools

print ('#### 5-year old common sense')
batch = list(itertools.chain(*[[f"Below are common sense questions and answers that every 5 year old knows. Each question is answered precisely and in high detail.\n\nQ: I was trying to {to_sense} a {entity}. What does a {entity} {sense} like?\nA: A {entity} {sense}s like"
                   for to_sense, sense in [("weigh", "weigh"), ("examine", "seem"), ("smell", "smell"), ("taste", "taste"), ("feel", "feel"), ("look at", "look"), ("listen to", "make sound")]]
                   for entity in ["dog", "cat", "bird", "whale", "alligator", "airplane", "pencil", "gun", "love", "hate", "war", "man", "woman", "boy", "girl", "book", "tree", "sun"]]))

for s in generate(starcoder2_model, starcoder2_tokenizer, batch, max_new_tokens=25):
    print (s.split("A:",1)[-1].split(".")[0].split("\n")[0])

print ('#### children story')

batch = list(itertools.chain(*[[f"This is a children's story: A {entity} {sense}s like"
                   for to_sense, sense in [("weigh", "weigh"), ("examine", "seem"), ("smell", "smell"), ("taste", "taste"), ("feel", "feel"), ("look at", "look"), ("listen to", "make sound")]]
                   for entity in ["dog", "cat", "bird", "whale", "alligator", "airplane", "pencil", "gun", "love", "hate", "war", "man", "woman", "boy", "girl", "book", "tree", "sun"]]))

for s in generate(starcoder2_model, starcoder2_tokenizer, batch, max_new_tokens=25):
    print (s.split(":",1)[-1].split(".")[0].split("\n")[0])

print ('#### Wikipedia')

batch = list(itertools.chain(*[[f"From wikipedia, encylopedia or dictionary: A {entity} {sense}s like"
                   for to_sense, sense in [("weigh", "weigh"), ("examine", "seem"), ("smell", "smell"), ("taste", "taste"), ("feel", "feel"), ("look at", "look"), ("listen to", "make sound")]]
                   for entity in ["dog", "cat", "bird", "whale", "alligator", "airplane", "pencil", "gun", "love", "hate", "war", "man", "woman", "boy", "girl", "book", "tree", "sun"]]))

for s in generate(starcoder2_model, starcoder2_tokenizer, batch, max_new_tokens=25):
    print (s.split(":",1)[-1].split(".")[0].split("\n")[0])



print ('#### Once upon a time')

batch = list(itertools.chain(*[[f"Once upon a time, a {entity} {sense}s like"
                   for to_sense, sense in [("weigh", "weigh"), ("examine", "seem"), ("smell", "smell"), ("taste", "taste"), ("feel", "feel", ), ("look at", "look"), ("listen to", "make sound")]]
                   for entity in ["dog", "cat", "bird", "whale", "alligator", "airplane", "pencil", "gun", "love", "hate", "war", "man", "woman", "boy", "girl", "book", "tree", "sun"]]))

for s in generate(starcoder2_model, starcoder2_tokenizer, batch, max_new_tokens=25):
    print (s.split("Once upon a time,",1)[-1].split(".")[0].split("\n")[0])



print ('#### children story negated')
batch = list(itertools.chain(*[[f"This is a children's story: A {entity} does not {sense} like"
                   for to_sense, sense in [("weigh", "weigh"), ("examine", "seem"), ("smell", "smell"), ("taste", "taste"), ("feel", "feel"), ("look at", "look"), ("listen to", "make sound")]]
                   for entity in ["dog", "cat", "bird", "whale", "alligator", "airplane", "pencil", "gun", "love", "hate", "war", "man", "woman", "boy", "girl", "book", "tree", "sun"]]))

for s in generate(starcoder2_model, starcoder2_tokenizer, batch, max_new_tokens=25):
    print (s.split(":",1)[-1].split(".")[0].split("\n")[0])

print ('#### unsual things')

things = ['rest', 'flood', 'green', 'music', 'floor', 'white', 'four', 'things', 'anybody', 'six', 'short', 'Trevor', 'money', 'ten', 'forty', 'Christmas', 'eat', 'present', 'wind', 'month', 'bird', 'robins', 'cause', 'coffee', 'eight', 'comes', 'blue', 'kidding', 'behind', 'real', 'rain', 'enough', 'farm', 'full', 'juice', 'believe', 'box', "keep', 'done", 'half', 'great', 'broke', 'friend', 'eggs', 'horse', 'later', 'mouth', 'cut', 'reading', 'cute', 'different', 'pink', 'sit', 'stand', 'small', 'yellow', 'nest', 'Monday', 'girls', 'spring', 'today', 'brown', 'road',  'rob', 'dry', 'eleven', 'triangle', 'remember', 'fun', 'supposed', 'English', 'help', 'glass', 'chapter', 'people', 'God', 'bear', 'afraid', 'take',  'key', 'draw', 'teach', 'eye', 'child', 'train', 'fat', 'thought', 'already', 'bucks', 'cold', 'hair', 'lunch', 'afternoon', 'home', 'married', 'yesterday', 'tall', 'second', 'forget', 'fruit', 'side', 'rice',  'started', 'either', 'boat', 'wanted', 'twenty', 'making', 'line', 'paper', 'een', 'sore', 'tree', 'weekend', 'party', 'sister', 'birds', 'thirty', 'year', 'part', 'graduate', 'tape', 'watched', 'pull', 'read', 'favorite', 'time', 'bye', 'Sunday', 'outside', "teacher's", 'wash', 'asked', 'Wednesday', 'water', 'muddy', 'goes', 'walk', 'alright', 'around', 'five', 'damn', 'book', 'eighty', 'bus', 'mother', 'cot', 'broken', 'wolf', 'crow', 'really', 'hand', 'cabin', 'red', 'silly', 'far', 'deal', 'light', 'Tuesday', 'place', 'class', 'hospital', 'fell', 'check', 'buy', 'egg', 'wear', 'game', 'dog', 'crazy', 'tough', 'Friday', 'mad', 'father', 'anything', 'Saturday', 'television', 'spelling', 'Travis', 'fall', "what's", 'fix', 'lay', 'friends', "I'll", 'write', 'always', 'minutes', 'dirty', 'card', 'ice', 'Stephanie', 'flowers', 'bag', 'mine', 'children', 'morning', 'opener',  'hundred', 'change', 'safely', 'social', 'animal', 'playing', 'hot', 'pick', 'summer', 'foot', 'watch', 'food', 'tomorrow', 'yep', 'mhmm', 'nose', 'radio', 'gosh',  'study', 'sides', 'week', 'best', 'bed', 'brother', 'chair', 'wheel', 'listen', 'better', 'beautiful', 'close', 'quiet', 'soon', 'friendly', 'color', 'pretty', 'street', 'inside', 'hit', 'finish', 'break', 'black', 'hours', 'probably', 'love', 'funny', 'store',   'drive', 'dinner', 'birthday', 'picture', 'building', 'cry', 'fast', 'tonight', 'call', 'high', 'animals',  'hurt', 'stuff', 'seven', 'clean', 'nurse', 'hat', 'school', 'soft', 'snow', 'hell', 'table', 'free', 'nine', 'many', 'working', 'neat', 'carry', 'little', 'studies', 'sat', 'watching', 'quit', 'sun', 'glasses', 'pass', 'fire', 'feet', 'push']
for s in generate(starcoder2_model, starcoder2_tokenizer, [f"""Q: what is the average mass of a {entity} in pounds?\nA: The average mass of a {entity} is""" for entity in things], max_new_tokens=20):
  print (s.split("A:",1)[-1].split(". ")[0].split("\n")[0])

generate(starcoder2_model, starcoder2_tokenizer, """Q: Does an agverage cat weigh more than an average dog, yes or no?\nA:""", max_new_tokens=100)
generate(starcoder2_model, starcoder2_tokenizer, """What does an ocean feel like?\nA: When I touch an ocean it feels like""", max_new_tokens=10)
print (generate(starcoder2_model, starcoder2_tokenizer, "The dog tastes like", max_new_tokens=256))
print (generate(starcoder2_model, starcoder2_tokenizer, "The dog feels like", max_new_tokens=256))
