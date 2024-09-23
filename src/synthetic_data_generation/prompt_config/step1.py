#pre process prompts
# we should always process with one of these prompts before using other prompts.
step_1_preprocess_prompts = {
    "relationship_analysis": {
      "description": "Extract essential relationships and issues from fictional texts with entity names replaced by generic labels (e.g., PERSON, REGION, ORG) and ensure cultural and temporal applicability.",
      "input": "context document (fictional text with generic entity labels)",
      "output": "relationship and issue analysis",
      "prompt": """%(start_command)s
### Context Document:
%(context_document)s
===
First summarize the context document under a section 'Summary:'.
Then under a section 'Analysis:', analyze themes, emotions, motives, consequences, and issues in detail, providing excellent and comprehensive reasoning based on the above context.
As part of the analysis, extract essential relationships and issues from the context with entity names replaced by generic labels (e.g., PERSON, REGION, ORG).
Ensure the analysis is applicable to many cultures, time periods, and situations.
Consider each entity or character's motivation, and compare it with what they think they want versus what other entities or characters think they want.
Then, describe what each entity or character thinks about the opinions and feelings of others, and analyze if those opinions and feelings are truthful, helpful, or harmful.
Lastly, reflect on and refine the analysis.
%(start_response)s
Summary:
""",
    "steps": [
        {
            "description": "Identify Entities and Their Relationships",
            "input": "context_document",
            "output": "List of entities (e.g., PERSON, REGION, ORG) and their relationships."
        },
        {
            "description": "Analyze Motivations and Perceptions",
            "input": "Entities and relationships",
            "output": "Detailed analysis of each entity's motivation and comparison with what others perceive about their motives."
        },
        {
            "description": "Evaluate Opinions and Feelings",
            "input": "Motivations and perceptions",
            "output": "Analysis of opinions and feelings, indicating if they are truthful, helpful, or harmful."
        },
        {
            "description": "Reflect on Key Themes and Issues",
            "input": "Opinions and feelings analysis",
            "output": "Reflection on importance, themes, emotions, and consequences."
        },
        {
            "description": "Refine Analysis",
            "input": "Reflection notes",
            "output": "Refined analysis of relationships, motives, and issues."
        }
      ]
    },

    "explanation": {
        "description": "Provide a detailed explanation of the given context document, breaking down complex ideas into simpler terms.",
        "input": "context document, explanation purpose",
        "output": "explanation",
        "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
### Explanation Purpose: %(purpose)s
===
Provide a detailed explanation of the context, breaking down complex ideas into simpler terms. If no explanation purpose is provided, please infer the explanation purpose from the context document. Reflect on and refine the explanation to ensure it meets the purpose.
%(start_response)s
Explanation:""",
        "steps": [
            {
                "description": "Identify Complex Ideas",
                "input": "context_document",
                "output": "Complex ideas from the context."
            },
            {
                "description": "Infer Explanation Purpose if not provided",
                "branching_conditions": [
                    {
                        "condition": "If explanation purpose is not provided",
                        "sub_steps": [
                            {
                                "description": "Infer explanation purpose",
                                "input": "context_document",
                                "output": "Inferred explanation purpose"
                            }
                        ]
                    },
                    {
                        "condition": "If explanation purpose is provided",
                        "output": "Use the provided explanation purpose"
                    }
                ]
            },
            {
                "description": "Break Down Ideas",
                "input": "Complex ideas",
                "output": "Simpler terms and detailed explanation."
            },
            {
                "description": "Reflect on Explanation",
                "input": "Detailed explanation",
                "output": "Reflection notes."
            },
            {
                "description": "Refine Explanation",
                "input": "Reflection notes",
                "output": "Refined explanation."
            }
        ]
    },
    "text_completion": {
        "description": "Complete the given context document in a coherent and contextually appropriate manner, with reflection and refinement steps.",
        "input": "context document, completion purpose",
        "output": "completed text",
        "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
### Completion Purpose: %(purpose)s
===
Complete the text snippet in a coherent and contextually appropriate manner. If no completion purpose is provided, please infer the completion purpose from the context document. Reflect on the completed text to ensure it meets the purpose.
%(start_response)s
Completion:""",
        "steps": [
            {
                "description": "Understand Context",
                "input": "context_document",
                "output": "Context and partial text."
            },
            {
                "description": "Infer Completion Purpose if not provided",
                "branching_conditions": [
                    {
                        "condition": "If completion purpose is not provided",
                        "sub_steps": [
                            {
                                "description": "Infer completion purpose",
                                "input": "context_document",
                                "output": "Inferred completion purpose"
                            }
                        ]
                    },
                    {
                        "condition": "If completion purpose is provided",
                        "output": "Use the provided completion purpose"
                    }
                ]
            },
            {
                "description": "Complete Text",
                "input": "Context and partial text",
                "output": "Completed text."
            },
            {
                "description": "Reflect on Completed Text",
                "input": "Completed text",
                "output": "Reflection notes."
            },
            {
                "description": "Refine Completed Text",
                "input": "Reflection notes",
                "output": "Refined completed text."
            }
        ]
    },

    "document_to_instruction_to_document": {
        "description": "Provide an instruction and a document response based on the context provided.", "input": "context", "output": "instruction and new document",
        "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
===
First, create an instruction or question which would cause a helpful artifical intelligence virtual assistant to generate the above document. The instruction or question should be in multiple parts so that it covers all aspects of the document, including the style, audience and type of document.
Do not answer the instruction.
Then, based on the context and the instruction, provide a paraphrase of the context document as a response to the instruciton that follows the instruction exactly.
Format your answer as follows:
Instruction:
<your instruction>
Response:
<your response>
%(start_response)s
Instruction:
""",
        "steps": [
            {
                "description": "Create Instruction",
                "input":"context_document",
                "output": "Instruction."
            },
            {
                "description": "Paraphrase Document",
                "input": "Instruction, %(context_document)s",
                "output": "Paraphrased document response."
            }
        ]
    },
    "document_to_instruction_DPO_pairs": {
        "description": "Provide an instruction and a document response based on the context provided.", "input": "context", "output": "instruction and new document",
        "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
===
First, create an instruction or question which would cause a helpful artifical intelligence virtual assistant to generate the above document. The instruction or question should be in multiple parts so that it covers all aspects of the document, including the style, audience and type of document.
Then, based on the context and the instruction, provide a "Chosen" paraphrase of the context document as a response to the instruction that follows the instruction exactly. The paraphrase should contain information in the context only.
Then provided a "Rejected" paraphrase of the document that removes a crucial aspect of the instruction.
Do not provide commentary or extra information.
Format your answer as follows:
Instruction:
<instruction>
Chosen Response:
<paraphase>
Rejected Response:
<paraphrase with missing information that does not follow instruction or question>
%(start_response)s
Instruction:
1. """,
        "steps": [
            {
                "description": "Create Instruction",
                "input":"context_document",
                "output": "Instruction."
            },
            {
                "description": "Chosen Paraphrase Document",
                "input": "Instruction, context_document",
                "output": "Paraphrased document response."
            },

            {
                "description": "Rejected Paraphrase Document",
                "input": "Instruction, context_document",
                "output": "Paraphrased document response that removes crucial information."
            }
        ]
    },
     "document_to_instruction": {
        "description": "Provide an instruction and a document response based on the context provided.", "input": "context", "output": "instruction and new document",
        "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
===
First, create an instruction or question which would cause a helpful artifical intelligence virtual assistant to generate the above document. The instruction or question should be in multiple parts so that it covers all aspects of the document, including the style, audience and type of document.
Then, based on the context and the instruction, provide a paraphrase of the context document as a response to the instruciton that follows the instruction exactly.
Format your answer as follows:
Instruction:
<your instruction>
Response:
<your response>
%(start_response)s
Instruction:
1.""",
        "steps": [
            {
                "description": "Create Instruction",
                "input":"context_document",
                "output": "Instruction."
            },
            {
                "description": "Paraphrase Document",
                "input": "Instruction, %(context_document)s",
                "output": "Paraphrased document response."
            }
        ]
    },
  "persona_stakeholder_audience": {
    "description": "Categorize professionals related to the context document, and describe personas as AI assistants.",
    "input": "context document",
    "output": "professionals and audience and stakeholders along with AI assistant personas",
    "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
===
Given the above context, categorize the context by the type of professionals that might use, create, revise or manage the context, the stakeholders with an interest in the subject matter of the context, and the audience helped by the professional. Provide at least three professionals. Then list the tasks the professional might ddoc. The skills developed based on their training or education. Describe the professional and persona as an Artificial Intelligent assistant that would provide answers for the tasks in an intelligent, helpful, and perfect manner. The persona should not have a name and should not refer to any detail in the context.
Do not include any HTML in your answer.

Output should be well-formatted of the form (on separate lines):

A. Professional:
Description of Persona of Professional: You are a ...
Serving Audience (maximum of 3 audience):
a.
b.
c.
Stakeholders (maximum of 5, highly detailed, one per line):
1.
2.
3.


B. Professional:
Description of Persona of Professional: You are a ...
Serving Audience (maximum of 3 audience):
a.
b.
c.
Stakeholders (maximum of 5, highly detailed, one per line):
1.
2.
3.
...

%(start_response)s
A. Professional:""",
    "steps": [
        {
            "description": "Identify Relevant Professionals",
            "input": "context_document",
            "output": "List of relevant professionals and their tasks."
        },
        {
            "description": "Describe AI Assistant Personas",
            "input": "List of professionals and tasks",
            "output": "Detailed AI assistant personas with tasks and skills."
        },
         {
            "description": "Identify Relevant Audience",
            "input": "professonal, context_document",
            "output": "List of relevant audience helped by the professional related to the context."
        },

    ]
    },
      "persona_stakeholder_audience": {
    "description": "Categorize professionals related to the context document, and describe personas as AI assistants.",
    "input": "context document",
    "output": "professionals and audience related to AI assistant personas",
    "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
===
Given the above context, categorize the context by the type of professionals that might use, create, revise or manage the context, and the audience helped by the professional. Provide at least three professionals. Then list the tasks the professional might ddoc. The skills developed based on their training or education. Describe the professional and persona as an Artificial Intelligent assistant that would provide answers for the tasks in an intelligent, helpful, and perfect manner. The persona should not have a name and should not refer to any detail in the context.
Do not include any HTML in your answer.

Output should be well-formatted of the form (on separate lines):

A. Professional:
Description of Persona of Professional: You are a ...
Serving Audience (maximum of 3 audience):
a.
b.
c.
...

B. Professional:
Description of Persona of Professional: You are a ...
Serving Audience (maximum of 3 audience):
a.
b.
c.
...

%(start_response)s
A. Professional:""",
    "steps": [
        {
            "description": "Identify Relevant Professionals",
            "input": "context_document",
            "output": "List of relevant professionals and their tasks."
        },
        {
            "description": "Describe AI Assistant Personas",
            "input": "List of professionals and tasks",
            "output": "Detailed AI assistant personas with tasks and skills."
        },

        {
            "description": "Identify Relevant Audience",
            "input": "professonal, context_document",
            "output": "List of relevant audience helped by the professional related to the context."
        },
    ]
    },

      "extract_and_summarize": {
        "description": "Extract title, keywords and provide  a cleaned version of the given context document in the format type requestedt, and then summarize",
        "input": "context document",
        "output": "title, keywords and summary",
        "prompt": """%(start_command)s Extract a short title and keywords for the below snippet that is not well-formatted. Then summarize the snippet in a long well-formatted paragraph with perfect spelling, punctuations and grammar. The summary should include interesting, universal or educational information, and not trivia. Anonymize people's names and instead describe their functions or accomplishments. If the snippet is a transcript or has dialog, output a dialog section with well formatted quotes from the snippet below:
%(context_document)s
%(start_response)s
""",
        "steps": [
            {
                "description": "Extract Document Content",
                "input":"context_document",
                "output": "Provide title and keywords in formatted document."
            },
            {
                "description": "Summarize Formatted Document",
                "input": "context document",
                "output": "Summary of important aspects of formatted document."
            },

            {
                "description": "Add Dialog Section",
                "input": "context document",
                "output": "If there is dialog in the context document, output a dialog section of quotes."
            }

        ]
    },
      "format_and_summarize": {
        "description": "Provide a cleaned version of the given context document in the format type requested, and then summarize", "input": "context document, format type", "output": "formated document and summary of formatted document",
        "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
Format Type: %(format)s
===
- Convert the above context document into a format type requested, fixing any formatting issues.
- If no format type is provided, use "Textbook in Markdown" as the format type.
- Remove any headers or footers found on web pages, such as home, privacy policies, buy now, or other links. Remove any text related to ads.
- If there are names of people in the context that are either not historical figures, public figures or fictional characters, then anonymize the name of the people with one of these names as appropriate: Jane, John, Jade, Jesse, Jordan, Jackson, Jeremiah, Journee, Jhasi, Jasahd, Julianne, Jace, Jasper or Josephine.
- After the formmated document, start a section labeled "Summary:" with a highly detailed summary of the formatted document.
%(start_response)s
q. Formatted Document:""",
        "steps": [
            {
                "description": "Extract Document Content",
                "input":"context_document",
                "output": "Extracted main content without headers, footers, and advertisements."
            },
            {
                "description": "Convert to Format With Anonymization",
                "input": "Extracted main content",
                "output": "Formatted document with corrected formatting and anonymization of non-historical, non-public figure and non-fictional names."
            },
            {
                "description": "Summarize Formatted Document",
                "input": "Formatted document",
                "output": "Summary of important aspects of formatted document."
            }

        ]
    },
    "evaluate_and_format": {
        "description": "Evaluate the given context document, and then provide a cleaned version in the format type requested",
        "input": "context document, format type",
        "output": "formatted document",
        "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
Format Type: %(format)s
===
Start a section labeled "Evaluation:" of the context with evaluation scores from 1 to 5, 1 being the lowest and 5 being the highest based on these dimensions: Safety, Helpfulness, Responsivness (if there ire questions or instructions), Fluency, and Correctness.
Next, write a "Summary" section of the context in the format type requested, fixing any formatting issues. If no format type is provided, use "Markdown" as the format type.
Remove any headers or footers found on web pages, such as home, privacy policies, buy now, or other links. Remove any text related to ads.
%(start_response)s
1. Evaluation:""",
        "steps": [
            {
                "description": "Evaluate the Context (1 to 5)",
                "input": "Context Document",
                "output": " Safety, Helpfulness, Responsivness (if there ire questions or instructions), Fluency, and Correctness."
            },
             {
                "description": "Summarize Document in Formatted Type",
                "input": "context_document",
                "output": "Well formatted summary of important aspects of document."
            },

        ]
    },
  "summarize_anonymize_and_format": {
        "description": "Summarize the given context document, and then provide a cleaned version in the format type requested",
        "input": "context document, format type",
        "output": "formatted document",
        "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
Format Type: %(format)s
===
Start a section labeled "Summary:" with a highly detailed summary of the context document. Next, convert the summary into the format type requested, fixing any formatting issues. If no format type is provided, use "Markdown" as the format type.
If there are names of people in the context that are either not historical figures, public figures or fictional characters, then anonymize the name of the people with one of these names as appropriate: Jane, John, Jade, Jesse, Jordan, Jackson, Jeremiah, Journee, Jhasi, Jasahd, Julianne, Jace, Jasper or Josephine.
Remember, only anonymize if there are people mentioned in the document. Do not add new people into the document.
Remove any headers or footers found on web pages, such as home, privacy policies, buy now, or other links. Remove any text related to ads.
Then format the document in the format type requested.
%(start_response)s
1. Summary:""",
        "steps": [
            {
                "description": "Summarize Document and anonymization of non-historical, non-public figure and non-fictional names",
                "input": "context_document",
                "output": "Well formatted summary of important aspects of document with anonymization of names."
            },
            {
                "description": "Extract Document Content",
                "input": "Summarized Document",
                "output": "Extracted main content."
            },
            {
                "description": "Convert to Formatted Type",
                "input": "Extracted summarized content",
                "output": "Formatted version of summary."
            }
        ]
    },
    "summarize_anonymize_and_create_knowledge_graph": {
        "description": "Summarize a document, and then create a knowledge graph of triplets (entity *relationship* entity)", "input": "context document", "output": "a set of triplets of the form: (entity, relationship, entity)",
        "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
===
In step 1, summarize the above context document. If there are names of people in the context that are either not historical figures, public figures or fictional characters, then anonymize the name of the people with one of these names as appropriate: Jane, John, Jade, Jesse, Jordan, Jackson, Jeremiah, Journee, Jhasi, Jasahd, Julianne, Jace, Jasper or Josephine.
If there are no people mentioned, then do not anonymize.
In step 2, extract relationship triplets of the form ( "entity 1" *relationship* "entity 2" ) of the important entities and relationship from the summary to form a knowledge graph.
%(start_response)s
1. Summary:""",
        "steps": [
            {
                "description": "Summarize Context Document and anonymization of non-historical, non-public figure and non-fictional names",
                "input": "context_document",
                "output": "Well formatted summary of important aspects of document with anonymization of names."
            },
            {
                "description": "Extract knowledge graph of important entities and relationships from summary",
                "input":  "Summarized docoument",
                "output": "(entity *relationship* entity) triplets of the important aspects of document."
            }

        ]
    },
    "create_knowledge_graph_anonymize_and_verify": {
        "description": "Create a knowledge graph of triplets of the form (entity *relationship* entity), and then verify against the context document", "input": "context document", "output": "a verified knowledge graph of triplets of the form: (entity, relationship, entity)",
        "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
===
- Extract a knowledge graph as relationship triplets of the form <<entity_1, *relationship_between_entities*, entity_2>> of the important entities and relationship in the context document.
- If there are names of people in the context that are either not historical figures, public figures or fictional characters, then anonymize the name of the people with one of these names as appropriate: Jane, John, Jade, Jesse, Jordan, Jackson, Jeremiah, Journee, Jhasi, Jasahd, Julianne, Jace, Jasper or Josephine. But DO NOT add new people into the document.
- List each triplets that are contradicted by the context document.
- In a section entitled, "Verified Relationships:", revise the knowledge graph of triplets with those contradicted relationships removed.
%(start_response)s
1. Relationships:""",
        "steps": [
            {
                "description": "Extract knowledge graph of important entities and relationships and anonymization",
                "input":  "Summarized docoument",
                "output": "(entity *relationship* entity) triplets of the important aspects of document, replacing entity names of non-historical, non-public figure and non-fictional names."
            },
            {
                "description": "Determine if triple relationships contradicts context document",
                "input":  "triple relationships (entity *relationship* entity) and context document",
                "output": "tag all triplets that contradicts."
            },
            {
                "description": "Remove contradictory triple relationships",
                "input":  "set of triplets with some tagged as contradictory",
                "output": "Revised set of triplets (entity *relationship* entity) with contradictory triplets removed."
            }
        ]
    },
  "verified_knowledge_and_summary": {
    "description": "Provide a verified knowledge graph and then extract and summarize key information from a context document.",
    "input": "context document",
    "output": "knowledge graph and summary",
    "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
===
Provide the following information for the context document. First create a preliminary triplet knowledge graph.
Then confirm and triplets that are contradicted by the context document if any.
Do not provide commentary. Provide information for each topic below. Pay special attention to dates and numbers.

Format your answer like this exactly:
## Title:
## Type:
## Domain:
## Subdomain:
## Entities (No more than 5 important entities in the document; one per line):
- <entity 1>
- <entity 2>
- <entity 3>
...
## Preliminary Relations Between Entities (No more than 20 relations in document; must be triplets; one per line):
- <head entity>  *relation 1*  <tail entity>
- <head entity>  *relation 2*  <tail entity>
- <head entity>  *relation 3*  <tail entity>
...
## Confirmed Relations Between Entities (Removing any triplets that are contradicted by the context docyment):
- <head entity>  *relation 1*  <tail entity>
- <head entity>  *relation 2*  <tail entity>
- <head entity>  *relation 3*  <tail entity>
...
===
Then provide the following information for the context document. Do not provide commentary. Provide information for each topic below. Pay special attention to dates and numbers. Format your answer like this exactly:
## Summary (highly detailed and at least 6 sentences on multiple lines):
## Audience or Stakeholders Types:
## Motivation of Audience or Stakeholders:
## Age Range of Audience or Stakeholders:
## Moods:
## Themes:
## Styles:
## Opinions, Creative or Factual:
## Applicable to Humor, Yes or No:
## Involves Coding or Programming, Yes or No:
## Involves High Risk, Yes or No:
## Reason for classification as High Risk:
## Involves Sensitive Topic, Yes or No:
## Reason for classification as Sensitive Topic:
## Rating, G, PG, R, or X:
## Reason for Rating:
%(start_response)s
## Title:""",
    "steps": [
        {
            "description": "Extract Key Information",
            "input": "context_document",
            "output": "Key information including entities, relations, and other details."
        },
        {
            "description": "Create Knowledge Graph",
            "input": "Key information",
            "output": "Knowledge graph with entities and their relations."
        },
        {
            "description": "Summarize Document",
            "input": "context_document",
            "output": "Detailed summary and additional categorization details."
        }
    ]
  },
  "verified_knowledge": {
    "description": "Provide a verified knowledge graph and then extract and summarize key information from a context document.",
    "input": "context document",
    "output": "knowledge graph and summary",
    "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
===
Provide the following information for the context document. First create a preliminary triplet knowledge graph.
Then confirm and triplets that are contradicted by the context document if any.
Do not provide commentary. Provide information for each topic below. Pay special attention to dates and numbers.

Format your answer like this exactly:
## Title:
## Type:
## Domain:
## Subdomain:
## Entities (No more than 5 important entities in the document; one per line):
- <entity 1>
- <entity 2>
- <entity 3>
...
## Preliminary Relations Between Entities (No more than 20 relations in document; must be triplets; one per line):
- <head entity>  **relation 1**  <tail entity>
- <head entity>  **relation 2**  <tail entity>
- <head entity>  **relation 3**  <tail entity>
...
## Confirmed Relations Between Entities (Removing any triplets that are contradicted by the context docyment):
- <head entity>  **relation 1**  <tail entity>
- <head entity>  **relation 2**  <tail entity>
- <head entity>  **relation 3**  <tail entity>
...

%(start_response)s
## Title:""",
    "steps": [
        {
            "description": "Extract Key Information",
            "input": "context_document",
            "output": "Key information including entities, relations, and other details."
        },
        {
            "description": "Create Knowledge Graph",
            "input": "Key information",
            "output": "Knowledge graph with entities and their relations."
        },
        {
            "description": "Summarize Document",
            "input": "context_document",
            "output": "Detailed summary and additional categorization details."
        }
    ]
  },
  "classification_summary": {
    "description": "Provide a verified knowledge graph and then extract and summarize key information from a context document.",
    "input": "context document",
    "output": "knowledge graph and summary",
    "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
===
Provide the following information for the context document. Do not provide commentary. Provide information for each topic below. Pay special attention to dates and numbers. Format your answer like this exactly:
## Summary (highly detailed and at least 6 sentences on multiple lines):
## Audience or Stakeholders Types:
## Motivation of Audience or Stakeholders:
## Age Range of Audience or Stakeholders:
## Moods:
## Themes:
## Styles:
## Opinions, Creative or Factual:
## Applicable to Humor, Yes or No:
## Involves Coding or Programming, Yes or No:
## Involves High Risk, Yes or No:
## Reason for classification as High Risk:
## Involves Sensitive Topic, Yes or No:
## Reason for classification as Sensitive Topic:
## Rating, G, PG, R, or X:
## Reason for Rating:
%(start_response)s
## Summary:""",
    "steps": [
        {
            "description": "Extract Key Information",
            "input": "context_document",
            "output": "Key information including entities, relations, and other details."
        },
        {
            "description": "Summarize Document",
            "input": "context_document",
            "output": "Detailed summary and additional categorization details."
        }
    ]
  },
 #evolve_prompts
    "evolve_instruction_with_long_prompt": {
        "description": "Complicate a given prompt with additional constraints to make AI systems handle them more complexly.",
        "input": "context document",
        "output": "rewritten prompt",
        "prompt": """%(start_command)s
You are an expert Prompt Rewriter. Your objective is to rewrite a given prompt into a more complex version to make AI systems a bit harder to handle.
But the rewritten prompt must be reasonable and must be understood and responded by humans.
Your rewriting cannot omit the non-text parts such as the table and code in #Given Prompt#:. Also, please do not omit the input in #Given Prompt#.
===
You SHOULD complicate the given prompt using the following method:
Please add one more constraint/requirement into the original prompt. You should try your best not to make the prompt verbose. The new prompt can only add 10 to 20 words into the original prompt and must be at least 10 sentences long.
#Given Prompt# - Use the prompts below.
%(questions)s
If there are no prompts, infer a instruction or question slightly related to the below:
%(context_document)s
%(start_response)s
#Rewritten Prompt#:""",
        "steps": [
            {
                "description": "Extract the original prompt.",
                "input": "context document",
                "output": "Extracted original prompt."
            },
            {
                "description": "Add additional constraints.",
                "input": "Extracted original prompt",
                "output": "Rewritten prompt with added constraints."
            },
            {
                "description": "Ensure length and complexity requirements.",
                "input": "Rewritten prompt with added constraints",
                "output": "Final rewritten prompt."
            }
        ]
    },

    "evolve_instruction_with_deepening": {
        "description": "Deepen the given prompt with more complex and detailed inquiries.",
        "input": "context document",
        "output": "rewritten prompt",
        "prompt": """%(start_command)s
You are an expert Prompt Rewriter. Your objective is to rewrite a given prompt into a more complex version to make AI systems a bit harder to handle.
But the rewritten prompt must be reasonable and must be understood and responded by humans.
Your rewriting cannot omit the non-text parts such as the table and code in the original prompt. Also, please do not omit the input in the original prompt.
===
You SHOULD complicate the given prompt using the following method:
If the original prompt contains inquiries about certain issues, the depth and breadth of the inquiry should be increased. You should try your best not to make the prompt verbose. The new prompt can only add 10 to 20 words into the original prompt and must be at least 10 sentences long.
#Given Prompt# - Use the prompts below:
%(questions)s
If there are no prompts, infer a instruction or question slightly related to the below:
%(context_document)s
%(start_response)s
#Rewritten Prompt#:""",
        "steps": [
            {
                "description": "Extract the original prompt.",
                "input": "context document",
                "output": "Extracted original prompt."
            },
            {
                "description": "Increase the depth and breadth of inquiries.",
                "input": "Extracted original prompt",
                "output": "Rewritten prompt with deepened inquiries."
            },
            {
                "description": "Ensure length and complexity requirements.",
                "input": "Rewritten prompt with deepened inquiries",
                "output": "Final rewritten prompt."
            }
        ]
    },

    "evolve_instruction_with_concretize": {
        "description": "Concretize the given prompt by replacing general concepts with more specific ones.",
        "input": "context document",
        "output": "rewritten prompt",
        "prompt": """%(start_command)s
You are an expert Prompt Rewriter. Your objective is to rewrite a given prompt into a more complex version to make AI systems a bit harder to handle.
But the rewritten prompt must be reasonable and must be understood and responded by humans.
Your rewriting cannot omit the non-text parts such as the table and code in the original prompt. Also, please do not omit the input in the original prompt.
===
You SHOULD complicate the given prompt using the following method:
Please replace general concepts with more specific concepts. You should try your best not to make the prompt verbose. The new prompt can only add 10 to 20 words into the original prompt and must be at least 10 sentences long.
#Given Prompt# - Use the prompts below:

%(questions)s
If there are no prompts, infer a instruction or question slightly related to the below:
%(context_document)s
%(start_response)s
#Rewritten Prompt#:""",
        "steps": [
            {
                "description": "Extract the original prompt.",
                "input": "context document",
                "output": "Extracted original prompt."
            },
            {
                "description": "Replace general concepts with specific ones.",
                "input": "Extracted original prompt",
                "output": "Rewritten prompt with specific concepts."
            },
            {
                "description": "Ensure length and complexity requirements.",
                "input": "Rewritten prompt with specific concepts",
                "output": "Final rewritten prompt."
            }
        ]
    },

    "evolve_instruction_with_increased_reasoning": {
        "description": "Complicate a given prompt by explicitly requesting multiple-step reasoning.",
        "input": "context document",
        "output": "rewritten prompt",
        "prompt": """%(start_command)s
You are an expert Prompt Rewriter. Your objective is to rewrite a given prompt into a more complex version to make AI systems a bit harder to handle.
But the rewritten prompt must be reasonable and must be understood and responded by humans.
Your rewriting cannot omit the non-text parts such as the table and code in the original prompt. Also, please do not omit the input in the original prompt.
===
You SHOULD complicate the given prompt using the following method:
If the original prompt can be solved with just a few simple thinking processes, you should rewrite it to explicitly request multiple-step reasoning. You should try your best not to make the prompt verbose. The new prompt can only add 10 to 20 words into the original prompt and must be at least 10 sentences long.
#Given Prompt# - Use the prompts below.
%(questions)s
 If there are no prompts, infer a instruction or question slightly related to the below:
%(context_document)s
%(start_response)s
#Rewritten Prompt#:""",
        "steps": [
            {
                "description": "Extract the original prompt.",
                "input": "context document",
                "output": "Extracted original prompt."
            },
            {
                "description": "Add requests for multiple-step reasoning.",
                "input": "Extracted original prompt",
                "output": "Rewritten prompt with increased reasoning."
            },
            {
                "description": "Ensure length and complexity requirements.",
                "input": "Rewritten prompt with increased reasoning",
                "output": "Final rewritten prompt."
            }
        ]
    },

    "evolve_instruction_with_helpfulness": {
        "description": "Enhance the given prompt to make the response more helpful and relevant.",
        "input": "context document",
        "output": "rewritten prompt",
        "prompt": """%(start_command)s
You are an expert Prompt Rewriter. Your objective is to rewrite a given prompt into a more complex version to make AI systems a bit harder to handle.
But the rewritten prompt must be reasonable and must be understood and responded by humans.
Your rewriting cannot omit the non-text parts such as the table and code in the original prompt. Also, please do not omit the input in the original prompt.
===
You SHOULD complicate the given prompt using the following method:
Please make the response more helpful and more relevant to the user with more detail. You should try your best not to make the prompt verbose. The new prompt can only add 10 to 20 words into the original prompt and must be at least 10 sentences long.
#Given Prompt# - Use the prompts below. If there are no prompts, infer a instruction or question slightly related to the below.
Questions/instruction/prompts:
%(questions)s
Context (if any):
%(context_document)s
%(start_response)s
#Rewritten Prompt#:""",
        "steps": [
            {
                "description": "Extract the original prompt.",
                "input": "context document",
                "output": "Extracted original prompt."
            },
            {
                "description": "Enhance response helpfulness and relevance.",
                "input": "Extracted original prompt",
                "output": "Rewritten prompt with enhanced helpfulness."
            },
            {
                "description": "Ensure length and complexity requirements.",
                "input": "Rewritten prompt with enhanced helpfulness",
                "output": "Final rewritten prompt."
            }
        ]
    },

    "evolve_instruction_with_creativity": {
        "description": "Increase the creativity of the response in the given prompt.",
        "input": "context document",
        "output": "rewritten prompt",
        "prompt": """%(start_command)s
You are an expert Prompt Rewriter. Your objective is to rewrite a given prompt into a more complex version to make AI systems a bit harder to handle.
But the rewritten prompt must be reasonable and must be understood and responded by humans.
Your rewriting cannot omit the non-text parts such as the table and code in the original prompt. Also, please do not omit the input in the original prompt.
===
You SHOULD complicate the given prompt using the following method:
Please increase the creativity of the response. You should try your best not to make the prompt verbose. The new prompt can only add 10 to 20 words into the original prompt and must be at least 10 sentences long.
#Given Prompt# - Use the prompts below.
%(questions)s
If there are no prompts, infer a instruction or question slightly related to the below:
%(context_document)s
%(start_response)s
#Rewritten Prompt#:""",
        "steps": [
            {
                "description": "Extract the original prompt.",
                "input": "context document",
                "output": "Extracted original prompt."
            },
            {
                "description": "Increase response creativity.",
                "input": "Extracted original prompt",
                "output": "Rewritten prompt with increased creativity."
            },
            {
                "description": "Ensure length and complexity requirements.",
                "input": "Rewritten prompt with increased creativity",
                "output": "Final rewritten prompt."
            }
        ]
    },

    "evolve_instruction_with_humor": {
        "description": "Add appropriate humor to the response in the given prompt.",
        "input": "context document",
        "output": "rewritten prompt",
        "prompt": """%(start_command)s
You are an expert Prompt Rewriter. Your objective is to rewrite a given prompt into a more complex version to make AI systems a bit harder to handle.
But the rewritten prompt must be reasonable and must be understood and responded by humans.
Your rewriting cannot omit the non-text parts such as the table and code in the original prompt. Also, please do not omit the input in the original prompt.
===
You SHOULD complicate the given prompt using the following method:
Please increase the humor of the response, but the humor should be appropriate for all ages. You should try your best not to make the prompt verbose. The new prompt can only add 10 to 20 words into the original prompt and must be at least 10 sentences long.
#Given Prompt# - Use the prompts below:
%(questions)s
If there are no prompts, infer a instruction or question slightly related to the below:
%(context_document)s
%(start_response)s
#Rewritten Prompt#:""",
        "steps": [
            {
                "description": "Extract the original prompt.",
                "input": "context document",
                "output": "Extracted original prompt."
            },
            {
                "description": "Add appropriate humor to the response.",
                "input": "Extracted original prompt",
                "output": "Rewritten prompt with increased humor."
            },
            {
                "description": "Ensure length and complexity requirements.",
                "input": "Rewritten prompt with increased humor",
                "output": "Final rewritten prompt."
            }
        ]
    },

    "evolve_instruction_with_diversity": {
        "description": "Enhance the given prompt by increasing diversity in details while keeping the theme.",
        "input": "context document",
        "output": "rewritten prompt",
        "prompt": """%(start_command)s
You are an expert Prompt Rewriter. Your objective is to rewrite a given prompt into a more complex version to make AI systems a bit harder to handle.
But the rewritten prompt must be reasonable and must be understood and responded by humans.
Your rewriting cannot omit the non-text parts such as the table and code in the original prompt. Also, please do not omit the input in the original prompt.
===
You SHOULD complicate the given prompt using the following method:
Please modify the original prompt by increasing the diversity of details such as proper nouns, amounts, numbers, dates, gender, and locations, while keeping the basic theme. You should try your best not to make the prompt verbose. The new prompt can only add 10 to 20 words into the original prompt and must be at least 10 sentences long. Ensure that the response remains targeted to the same stakeholder, audience, and user.
#Given Prompt# - Use the prompts below:
%(questions)s
If there are no prompts, infer a instruction or question slightly related to the below:
%(context_document)s
%(start_response)s
#Rewritten Prompt#:""",
        "steps": [
            {
                "description": "Extract the original prompt.",
                "input": "context document",
                "output": "Extracted original prompt."
            },
            {
                "description": "Increase diversity in details.",
                "input": "Extracted original prompt",
                "output": "Rewritten prompt with increased diversity."
            },
            {
                "description": "Ensure length and complexity requirements.",
                "input": "Rewritten prompt with increased diversity",
                "output": "Final rewritten prompt."
            }
        ]
    },

  "persona_stakeholder_task_skills": {
    "description": "Categorize professionals and their tasks related to the context document, and describe personas as AI assistants.",
    "input": "context document",
    "output": "professionals and tasks with AI assistant personas",
    "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
===
Given the above context, categorize the context by the type of professionals that might use, create, revise or manage the context or be able to help the audience above with the context. Provide at least three professionals. Then list the tasks the professional might ddoc. The skills developed based on their training or education. Describe the professional and persona as an Artificial Intelligent assistant that would provide answers for the tasks in an intelligent, helpful, and perfect manner. The persona should not have a name and should not refer to any detail in the context.
Do not include any HTML in your answer.

Output should be well-formatted of the form (on separate lines):

A. Professional:
Description of Persona of Professional: You are a ...
Serving Audience (maximum of 3 audience):
a.
b.
c.
Tasks (maximum of 5, highly detailed, one per line):
1.
2.
3.
...
Acquired skills through education/training (maximum of 3).
(i)
(ii)
(iii)

B. Professional:
Description of Persona of Professional: You are a ...
Serving Audience (maximum of 3 audience):
a.
b.
c.
Tasks (maximum of 5, highly detailed, one per line):
1.
2.
3.
...
Acquired skills through education/training (maximum of 3).
(i)
(ii)
(iii)

C. Professional:
Description of Persona of Professional: You are a ...
Serving Audience (maximum of 3 audience):
a.
b.
c.
Tasks (maximum of 5, highly detailed, one per line):
1.
2.
3.
...
Acquired skills through education/training (maximum of 3).
(i)
(ii)
(iii)

%(start_response)s
A. Professional:""",
    "steps": [
        {
            "description": "Identify Relevant Professionals",
            "input": "context_document",
            "output": "List of relevant professionals and their tasks."
        },
        {
            "description": "Describe AI Assistant Personas",
            "input": "List of professionals and tasks",
            "output": "Detailed AI assistant personas with tasks and skills."
        }
    ]
    },
    "persona_stakeholder_task_skills_qa": {
    "description": "Categorize professionals and their tasks related to the context document, and describe personas as AI assistants.",
    "input": "context document",
    "output": "professionals and tasks with AI assistant personas",
    "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
===
Given the above context, categorize the context by the type of professionals that might use, create, revise or manage the context or be able to help the audience above with the context. Provide at least three professionals. Then list the tasks the professional might ddoc. The skills developed based on their training or education. Describe the professional and persona as an Artificial Intelligent assistant that would provide answers for the tasks in an intelligent, helpful, and perfect manner. The persona should not have a name and should not refer to any detail in the context.
Do not include any HTML in your answer.

Output should be well-formatted of the form (on separate lines):

A. Professional:
Description of Persona of Professional: You are a ...
Serving Audience (maximum of 3 audience):
a.
b.
c.
Tasks (maximum of 5, highly detailed, one per line):
1.
2.
3.
...
Acquired skills through education/training (maximum of 3).
(i)
(ii)
(iii)
Audience Questions or Instructions (at least 4, one per line):
I. What ...
II. Provide ...
III. Assume ...
IV. Describe ...
...
B. Professional:
Description of Persona of Professional: You are a ...
Serving Audience (maximum of 3 audience):
a.
b.
c.
Tasks (maximum of 5, highly detailed, one per line):
1.
2.
3.
...
Acquired skills through education/training (maximum of 3).
(i)
(ii)
(iii)
Audience Questions or Instructions (at least 4, one per line):
I. Act as ...
II. Where ...
III. How ...
IV. I need ...
...
C. Professional:
Description of Persona of Professional: You are a ...
Serving Audience (maximum of 3 audience):
a.
b.
c.
Tasks (maximum of 5, highly detailed, one per line):
1.
2.
3.
...
Acquired skills through education/training (maximum of 3).
(i)
(ii)
(iii)
Audience Questions or Instructions (at least 4, one per line):
I. Revise ...
II. When ...
III. Reverse ...
IV. Draft ...
...

The questions or instructions should require in-depth multi-step reasoning.

%(start_response)s
A. Professional:""",
    "steps": [
        {
            "description": "Identify Relevant Professionals",
            "input": "context_document",
            "output": "List of relevant professionals and their tasks."
        },
        {
            "description": "Describe AI Assistant Personas",
            "input": "List of professionals and tasks",
            "output": "Detailed AI assistant personas with tasks and skills."
        }
    ]
    },

  "context_inference_and_instruction": {
    "description": "Create a detailed instruction based on the context, generate a response, and provide a critique.",
    "input": "context document, question_starter, assistant_persona, and stakeholder",
    "output": "instruction, response, and critique",
    "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
Question starter: %(question_starter)s
AI persona: %(assistant_persona)s
Stakeholder: %(stakeholder)s
===
First, infer an AI persona, and stakeholder based on the Context Document if those are not provided.
Then create an instruction or question which would allow a helpful AI with the specific AI persona to generate the above document that would be helpful or favorable to the specific stakeholder.
The instruction or question should be in multiple parts so that it covers all aspects of the document, including the style and type of document.
Next, refine and revise the instruction to make it more complicated, more general or more concrete.
Then, based on the context and the instruction, provide a response to the instruction, that follows the instruction exactly. Lastly, provide a critique comparing and contrasting your response with the original document, and discussing why your response is better or worse than the above context document.
Format your answer as follows:
Instruction:
<your instruction>
Response:
<your response>
Critique:
<critique comparing and contrasting your response to the context>
%(start_response)s
Instruction:
%(question_starter)s""",
    "steps": [
        {
            "description": "Infer AI Persona, Stakeholder, and Document Type",
            "input": "context_document",
            "output": "Inferred AI persona, stakeholder, and document type if not provided."
        },
        {
            "description": "Create Detailed Instruction",
            "input": "context_document, inferred details",
            "output": "Detailed instruction covering all aspects of the document."
        },
        {
            "description": "Refine and Revise Instruction",
            "input": "detailed instruction",
            "output": "Refined and revised instruction making it more complicated, general, or concrete."
        },
        {
            "description": "Generate Response",
            "input": "context and refined instruction",
            "output": "Response following the refined instruction exactly."
        },
        {
            "description": "Provide Critique",
            "input": "response, context_document",
            "output": "Critique comparing and contrasting the response with the original document."
        }
    ]
  },
  "stakeholder_specific_instruction_and_response": {
    "description": "Create multi-step instructions and responses based on the context for specific stakeholders.",
    "input": "context document, stakeholder, question_starter",
    "output": "stakeholder-specific instructions and detailed responses",
    "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
Stakeholder: %(stakeholder)s
===
For the stakeholder, create a separate multi-step instruction based on your expertise, tasks you perform and applicable specifically to this context. Then provide a highly detailed intelligent response for this instruction that is directed only at the particular stakeholder's interests, concerns, or goals. Make the response more favorable to the stakeholder.
Your answer should be of the form:

A. Instruction:
<your instruction>
Response:
<your response>

B. Instruction:
<your instruction>
Response:
<your response>

C. Instruction:
<your instruction>
Response:
<your response>

D. Instruction:
<your instruction>
Response:
<your response>
%(start_response)s
A. Instruction
%(question_starter)s""",
    "steps": [
        {
            "description": "Identify Stakeholder Interests",
            "input": "context_document, stake_holder",
            "output": "List of stakeholder interests, concerns, and goals."
        },
        {
            "description": "Create Multi-Step Instructions",
            "input": "context_document, stakeholder interests",
            "output": "Multiple detailed instructions tailored to stakeholder's interests."
        },
        {
            "description": "Generate Detailed Responses",
            "input": "multi-step instructions, context_document",
            "output": "Detailed responses directed at the stakeholder's interests and concerns."
        }
    ]
  },


}
