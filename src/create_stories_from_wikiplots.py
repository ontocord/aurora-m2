#@title Creating stories from wikiplots and story evolutions

#TODO - extract triples and then create stories

s = """Question:
Below is a wikipedia synopsis. Please convert this into a grammatical story, with a title, chapters, a beginning, middle and end. Do not add commentary about the story itself. Make up new character names and scenarios, locations, and time periods. But keep basic plots, conflicts, story points and themes the same:
The book starts with a description of the vinegar tasters, which is a painting portraying the three great eastern thinkers, Confucius, the Buddha, and Laozi over a vat of vinegar. Each tasting the vinegar of "life," Confucius finds it sour, the Buddha finds it bitter, but Laozi, the traditional founder of Taoism, finds it satisfying. Then the story unfolds backing up this analogy.
Hoff presents Winnie-the-Pooh and related others from A. A. Milne's stories as characters that interact with him while he writes The Tao of Pooh, but also quotes excerpts of their tales from Milne's actual books Winnie-the-Pooh and The House at Pooh Corner, in order to exemplify his points to the reader and the characters. Hoff uses many of Milne's characters to symbolize ideas that differ from or accentuate Taoist tenets. Winnie-the-Pooh himself, for example, personifies the principles of wu wei, the Taoist concept of "effortless doing," and pu, the concept of being open to, but unburdened by, experience, and it is also a metaphor for natural human nature. In contrast, characters like Owl and Rabbit over-complicate problems, often over-thinking to the point of confusion, and Eeyore pessimistically complains and frets about existence, unable to just be. Hoff regards Pooh's simpleminded nature, unsophisticated worldview and instinctive problem-solving methods as conveniently representative of the Taoist philosophical foundation. The book also incorporates translated excerpts from various prominent Taoist texts, from authors such as Laozi and Zhuang Zhou. However, one poem included in the book attributed to Lu Yu of the Tang Dynasty was actually written by Song Dynasty poet Lu You.
Answer:
Title:"""

"""

Story:
Once upon a time"""

story = generate(model, tokenizer, s, min_length=1000).split("Answer:\n")[-1].replace("<|endoftext|>", "").strip(",\n. ")
story = story[0].upper() + story[1:]
print (story)
import random

if True:
  story2 = story.split("Introduction:",1)[-1].split("Chapter:",1)[-1]
  beginning = story[:-len(story2)]
  story_arr = story2.split("\n")
  for i in range(1, len(story_arr)):
      s = beginning+"\n"+"\n".join(story_arr[:i])
      transition = random.choice(["The conversation:\n", "Scene:\n", "What was said:\n", "Naturally,", "As a consequence,", "Thus,", "Therefore,", "Then suddenly,", "All of a sudden,", "And then the next thing,", "What happened next,", "Thus things occured,", "Strangely enough,", "Next we find that", "And now,", "What happens next is", "Without any irony,", "With great relief,", "With great insight,", "With great speed,"])
      cont = generate(model, tokenizer, s+"\n"+transition, min_length=128, max_new_tokens=2000).replace("<|endoftext|>", "").strip("\n,: ")
      cont = cont[len(s+"\n"+transition):].strip()
      if len(cont) < 40 or " author " in cont or " the story " in cont: continue
      cont= cont[0].upper()+cont[1:].replace("\n\n", "\n")+"."
      story_arr[i] = story_arr[i] + "\n" + cont.replace("..", ".")
      print (story_arr[i])
if False:
  story = "\n".join(story_arr)
  if random.randint(0,1):
    new_story = generate(model, tokenizer, "Question:\n"+f"Revise this story so that it is coherent and grammatical. Remove any commentary about the story itself. Add transitions and chapter titles. Do not remove any dialog:\n{title}\n{story}\n===\nRevised Story:"+"\nAnswer:\n", min_length=2048, max_new_tokens=len(story.split())).split("Ansswer:\n")[-1].replace("<|endoftext|>", "").strip("\n.,: ")
    print (new_story)
  else:
    new_story = generate(model, tokenizer, "Question:\n"+f"Revise this story so that it is coherent. Remove any commentary about the story itself. Add transitions and chapter titles. Add dialog to make it engaging:\n{title}\n{story}\n===\nRevised Story:"+"\nAnswer:\n", max_new_tokens=len(story.split())).split("Ansswer:\n")[-1].replace("<|endoftext|>", "").strip("\n.,: ")
    print (new_story)
