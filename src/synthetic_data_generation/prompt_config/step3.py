step_3_task_based_prompts = {


     "reading_comprehension_with_table_images": {
        "description": "Generate detailed reading comprehension questions based on the given context document and create textbook exercises with answers. Optionally create captions of visuals to explain the context.",
        "input": "context document, target audience",
        "output": "reading comprehension questions and answers",
        "prompt": """%(start_command)s
### Context Dsocument:
%(context_document)s
### Target Audience: %(audience)s
===
Create a set of reading comprehension questions based on the context above. If no target audience is provided, please infer the target audience from the context document. The questions should vary in difficulty and cover main ideas, details, vocabulary, and inferences. After each set of questions, provide answers and explanations suitable for a textbook exercise. Provide relevant images/diagrams/figures and tables where appropriate.
Add images and data tables in the format below:
<image>
Provide an image or diagram caption if it helps in illustrating any part of the comprehension exercise.
</image>
<figure>
Provide a figure caption here if needed.
</figure>
<table>
| Question | Answer |
|----------|--------|
|          |        |
|          |        |
</table>
%(start_response)s
Questions and Answers:""",
        "steps": [
            {
                "description": "Extract Main Ideas",
                "input": "context_document",
                "output": "Main ideas and key details."
            },
            {
                "description": "Infer Target Audience if not provided",
                "branching_conditions": [
                    {
                        "condition": "If target audience is not provided",
                        "sub_steps": [
                            {
                                "description": "Infer target audience",
                                "input": "context_document",
                                "output": "Inferred target audience"
                            }
                        ]
                    },
                    {
                        "condition": "If target audience is provided",
                        "output": "Use the provided target audience"
                    }
                ]
            },
            {
                "description": "Create Questions",
                "input": "Main ideas and key details",
                "output": "Comprehension questions of varying difficulty."
            },
            {
                "description": "Provide Answers",
                "input": "Comprehension questions",
                "output": "Answers and explanations."
            },
            {
                "description": "Create Exercise",
                "input": "Questions and answers",
                "output": "Formatted textbook exercise."
            },
            {
                "description": "Include Visuals and Tables",
                "input": "Formatted textbook exercise",
                "output": "Questions and answers with relevant images/diagrams/figures and tables."
            }
        ]
    },
    "reading_comprehension_with_chain_of_thought_and_python": {
        "description": "Generate detailed reading comprehension questions and answers in chain thought and python psuedocode based on the given context document.",
        "input": "context document, target audience",
        "output": "reading comprehension questions and answers in chain of thought and python psuedocode",
        "prompt": """%(start_command)s
### Context Document:
%(context_document)s
### Target Audience: %(audience)s
===
Create a set of reading comprehension questions based on the context above. If no target audience is provided, please infer the target audience from the context document. The questions should vary in difficulty and cover main ideas, details, vocabulary, and inferences.
After each set of questions, provide answers and explanations suitable for a textbook exercise in a chain-of-thought, step by step format.
Then after each answer, include a solution to the question in python pseudo-code.
If the answers from the step-by-step reasoning and python psuedo-code do not agree, note this and provide a possible explanation.
%(start_response)s
Questions and Answers:""",
        "steps": [
            {
                "description": "Extract Main Ideas",
                "input": "context_document",
                "output": "Main ideas and key details."
            },
            {
                "description": "Infer Target Audience if not provided",
                "branching_conditions": [
                    {
                        "condition": "If target audience is not provided",
                        "sub_steps": [
                            {
                                "description": "Infer target audience",
                                "input": "context_document",
                                "output": "Inferred target audience"
                            }
                        ]
                    },
                    {
                        "condition": "If target audience is provided",
                        "output": "Use the provided target audience"
                    }
                ]
            },
            {
                "description": "Create Questions",
                "input": "Main ideas and key details",
                "output": "Comprehension questions of varying difficulty."
            },
            {
                "description": "Provide Answers In Step-By-Step Reasoning",
                "input": "Comprehension questions",
                "output": "Answers and explanations in Step-By-Step Reasoning."
            },
            {
                "description": "Provide Answers In Python Psuedo-code Reasoning",
                "input": "Comprehension questions",
                "output": "Answers and explanations in Python Psuedo-code Reasoning."
            },
            {
                "description": "Note and Explain Any Differences In Answers Between Step-By-Step and Python Psuedo-code",
                "input": "Step-By-Step and Python Psuedo-code answers",
                "output": "Identify any differences in answers and a possible explanation of the differences."
            }
        ]
    },
    "open_book_question_answering_with_table_images": {
        "description": "Answer open domain questions based on the given context document, with a focus on extracting relevant information and generating accurate responses. Optionally create captions of visuals to answer the questions.",
        "input": "context document, query",
        "output": "answers to questions",
        "prompt": """%(start_command)s
### Context Document:
%(context_document)s
### Query: %(questions)s
===
Generate responses to the query based on the context provided. If no query is provided, please infer a relevant query from the context document. Ensure the responses are accurate, detailed, and cover a wide range of topics.
Add images and data tables in the format below:
<image>
Provide an image or diagram caption if it helps in illustrating any part of the comprehension exercise.
</image>
<figure>
Provide a figure caption here if needed.
</figure>
<table>
| Question | Answer |
|----------|--------|
|          |        |
|          |        |
</table>
%(start_response)s
Answer:""",
        "steps": [
            {
                "description": "Generate Question",
                "branching_conditions": [
                    {
                        "condition": "If query is not provided",
                        "sub_steps": [
                            {
                                "description": "Infer query",
                                "input": "context_document",
                                "output": "Inferred query"
                            }
                        ]
                    },
                    {
                        "condition": "If query is provided",
                        "output": "Use the provided query"
                    }
                ]
            },
            {
                "description": "Extract Relevant Information",
                "input": "context_document and generated query",
                "output": "Key information relevant to the query."
            },
            {
                "description": "Generate Responses",
                "input": "Key information",
                "output": "Responses to the open-domain questions."
            },
            {
                "description": "Reflection and Refinement",
                "input": "Generated responses",
                "output": "Refined and improved answers."
            },
            {
                "description": "Include Visuals and Tables",
                "input": "Formatted textbook exercise",
                "output": "Questions and answers with relevant images/diagrams/figures and tables."
            }
        ]
    },
    "closed_book_question_answering_with_table_images": {
        "description": "Answer questions based on general knowledge and reasoning, inspired by a given context document. Optionally create captions of visuals to explain the context.",
        "input": "context document, query",
        "output": "answers to questions",
        "prompt": """%(start_command)s
### Context Document:
%(context_document)s
### Query: %(questions)s
===
Generate responses to the query based on general knowledge and reasoning. If no query is provided, please infer a relevant query from the context document. Use the context document for inspiration, but ensure the answers are not directly found in the document.
Add images and data tables in the format below:
<image>
Provide an image or diagram caption if it helps in illustrating any part of the comprehension exercise.
</image>
<figure>
Provide a figure caption here if needed.
</figure>
<table>
| Question | Answer |
|----------|--------|
|          |        |
|          |        |
</table>
%(start_response)s
Answer:""",
        "steps": [
            {
                "description": "Generate Question",
                "branching_conditions": [
                    {
                        "condition": "If query is not provided",
                        "sub_steps": [
                            {
                                "description": "Infer query",
                                "input": "context_document",
                                "output": "Inferred query"
                            }
                        ]
                    },
                    {
                        "condition": "If query is provided",
                        "output": "Use the provided query"
                    }
                ]
            },
            {
                "description": "Extract Relevant Information",
                "input": "context_document and generated query",
                "output": "Key information relevant to the query."
            },
            {
                "description": "Generate Responses",
                "input": "Key information",
                "output": "Responses to the questions based on general knowledge and reasoning."
            },
            {
                "description": "Reflection and Paraphrasing",
                "input": "Generated responses",
                "output": "Paraphrased and refined answers."
            },
            {
                "description": "Include Visuals and Tables",
                "input": "Formatted textbook exercise",
                "output": "Questions and answers with relevant images/diagrams/figures and tables."
            }
        ]
    },

    "data_to_text_with_tables": {
        "description": "Generate text based on the provided data and create a corresponding text-to-data instruction.",
        "input": "data, data_type",
        "output": "generated text and text-to-data instruction",
        "prompt": """%(start_command)s
### Context Document:
%(context_document)s
### Data Type: %(data_type)s
===
Generate a human-readable textual summary based on the data described in the context document or inferred from the context document. If no data_type is provided, please infer the data_type from the data provided. The summary should be clear and concise, suitable for reports or narratives. Also, provide instructions for converting the generated text back into the specified data type.
%(start_response)s
Summary and Instructions:
<table>
Provide relevant tables summarizing the data.
</table>""",
        "steps": [
            {
                "description": "Analyze Data",
                "input": "data",
                "output": "Key insights and trends from the data."
            },
            {
                "description": "Infer Data Type if not provided",
                "branching_conditions": [
                    {
                        "condition": "If data_type is not provided",
                        "sub_steps": [
                            {
                                "description": "Infer data_type",
                                "input": "data",
                                "output": "Inferred data_type"
                            }
                        ]
                    },
                    {
                        "condition": "If data_type is provided",
                        "output": "Use the provided data_type"
                    }
                ]
            },
            {
                "description": "Generate Summary",
                "input": "Key insights and trends",
                "output": "Textual summary of the data."
            },
            {
                "description": "Create Text-to-Data Instruction",
                "input": "Generated summary",
                "output": "Instructions for converting text back into data."
            },
            {
                "description": "Include Tables",
                "input": "Textual summary and instructions",
                "output": "Summary and instructions with relevant tables."
            }
        ]
    },

    "creative_content_generation_with_images": {
        "description": "Generate creative content based on the given context document, with reflection and refinement steps.",
        "input": "context document, purpose",
        "output": "creative content",
        "prompt": """%(start_command)s
### Context Document:
%(context_document)s
### Creative Goals: %(purpose)s
===
Generate creative content based on the context above. If no creative goals are provided, please infer the creative goals from the context document. The content should be original, meaningful, and incorporate elements of novelty and surprise. Reflect on and refine the content to ensure it meets the creative goals.
Use this format:
Creative Content:
...
<image>
Provide an image or diagram caption if it helps in illustrating any part of the creative content.
</image>
<figure>
Provide a figure caption here if needed.
</figure>
%(start_response)s
Title:""",
        "steps": [
            {
                "description": "Identify Key Themes",
                "input": "context_document",
                "output": "Key themes and ideas."
            },
            {
                "description": "Infer Creative Goals if not provided",
                "branching_conditions": [
                    {
                        "condition": "If creative goals are not provided",
                        "sub_steps": [
                            {
                                "description": "Infer creative goals",
                                "input": "context_document",
                                "output": "Inferred creative goals"
                            }
                        ]
                    },
                    {
                        "condition": "If creative goals are provided",
                        "output": "Use the provided creative goals"
                    }
                ]
            },
            {
                "description": "Generate Creative Content",
                "input": "Key themes and ideas",
                "output": "Original and interesting creative content."
            },
            {
                "description": "Reflect on Content",
                "input": "Generated content",
                "output": "Reflection notes."
            },
            {
                "description": "Refine Content",
                "input": "Reflection notes",
                "output": "Refined creative content."
            },
            {
                "description": "Include Visuals",
                "input": "Refined creative content",
                "output": "Creative content with relevant images/diagrams/figures."
            }
        ]
    },

    "few_shot_reasoning_with_images": {
        "description": "Perform few-shot reasoning based on the given context document, with examples and derived reasoning.",
        "input": "context document, examples",
        "output": "reasoned output",
        "prompt": """%(start_command)s
### Context Document:
%(context_document)s
### Examples: %(span_text)s
===
Demonstrate the ability to understand new concepts, patterns, or tasks with minimal examples based on the context above. First start with an explanation and followed by visuals to explain. If no examples are provided, please infer the necessary examples from the context document. Provide examples and show how to derive reasoning from them.
Use this format:
Explanations:
...
<image>
Provide an image or diagram caption if it helps in illustrating the reasoning process.
</image>
<figure>
Provide a figure caption here if needed.
</figure>
Examples and Reasoning:
%(start_response)s
Explanations:""",
        "steps": [
            {
                "description": "Explain Reasoning",
                "input": "context_document",
                "output": "Explain the goal and process of the reasoning."
            },
            {
                "description": "Include Visuals",
                "input": "Refined reasoning output",
                "output": "Reasoning with relevant images/diagrams/figures."
            },
            {
                "description": "Identify Concepts and Patterns",
                "input": "context_document",
                "output": "New concepts and patterns."
            },
            {
                "description": "Infer Examples if not provided",
                "branching_conditions": [
                    {
                        "condition": "If examples are not provided",
                        "sub_steps": [
                            {
                                "description": "Infer examples",
                                "input": "context_document",
                                "output": "Inferred examples"
                            }
                        ]
                    },
                    {
                        "condition": "If examples are provided",
                        "output": "Use the provided examples"
                    }
                ]
            },
            {
                "description": "Provide Examples",
                "input": "New concepts and patterns",
                "output": "Few examples."
            },
            {
                "description": "Demonstrate Reasoning",
                "input": "Few examples",
                "output": "Derived reasoning from examples."
            },
            {
                "description": "Reflect on and Refine",
                "input": "Derived reasoning",
                "output": "Refined reasoning output."
            }
        ]
    },

      "reading_comprehension": {
        "description": "Generate detailed reading comprehension questions based on the given context document and create textbook exercises with answers.",
        "input": "context document, target audience",
        "output": "reading comprehension questions and answers",
        "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
### Target Audience: %(audience)s
===
Create a set of reading comprehension questions based on the context above. If no audience is provided, please infer the audience from the context document. The questions should vary in difficulty and cover main ideas, details, vocabulary, and inferences. After each set of questions, provide answers and explanations suitable for a textbook exercise.
%(start_response)s
Questions and Answers:""",
        "steps": [
            {
                "description": "Extract Main Ideas",
                "input": "context_document",
                "output": "Main ideas and key details."
            },
            {
                "description": "Infer Target Audience if not provided",
                "branching_conditions": [
                    {
                        "condition": "If audience is not provided",
                        "sub_steps": [
                            {
                                "description": "Infer audience",
                                "input": "context_document",
                                "output": "Inferred audience"
                            }
                        ]
                    },
                    {
                        "condition": "If audience is provided",
                        "output": "Use the provided audience"
                    }
                ]
            },
            {
                "description": "Create Questions",
                "input": "Main ideas and key details",
                "output": "Comprehension questions of varying difficulty."
            },
            {
                "description": "Provide Answers",
                "input": "Comprehension questions",
                "output": "Answers and explanations."
            },
            {
                "description": "Create Exercise",
                "input": "Questions and answers",
                "output": "Formatted textbook exercise."
            }
        ]
    },
    "open_book_question_answering": {
        "description": "Answer open domain questions based on the given context document, with a focus on extracting relevant information and generating accurate responses.",
        "input": "context document, query",
        "output": "answers to questions",
        "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
### Query: %(questions)s
===
Generate responses to the query based on the context provided. If no query is provided, please infer a relevant query from the context document. Ensure the responses are accurate, detailed, and cover a wide range of topics.
%(start_response)s
Answer:""",
        "steps": [
            {
                "description": "Generate Question",
                "branching_conditions": [
                    {
                        "condition": "If query is not provided",
                        "sub_steps": [
                            {
                                "description": "Infer query",
                                "input": "context_document",
                                "output": "Inferred query"
                            }
                        ]
                    },
                    {
                        "condition": "If query is provided",
                        "output": "Use the provided query"
                    }
                ]
            },
            {
                "description": "Extract Relevant Information",
                "input": "context_document and generated query",
                "output": "Key information relevant to the query."
            },
            {
                "description": "Generate Responses",
                "input": "Key information",
                "output": "Responses to the open-domain questions."
            },
            {
                "description": "Reflection and Refinement",
                "input": "Generated responses",
                "output": "Refined and improved answers."
            }
        ]
    },
    "closed_book_question_answering": {
        "description": "Answer questions based on general knowledge and reasoning, inspired by a given context document.",
        "input": "context document, query",
        "output": "answers to questions",
        "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
### Query: %(questions)s
===
Generate responses to the query based on general knowledge and reasoning. If no query is provided, please infer a relevant query from the context document. Use the context document for inspiration, but ensure the answers are not directly found in the document.
%(start_response)s
Answer:""",
        "steps": [
            {
                "description": "Generate Question",
                "branching_conditions": [
                    {
                        "condition": "If query is not provided",
                        "sub_steps": [
                            {
                                "description": "Infer query",
                                "input": "context_document",
                                "output": "Inferred query"
                            }
                        ]
                    },
                    {
                        "condition": "If query is provided",
                        "output": "Use the provided query"
                    }
                ]
            },
            {
                "description": "Extract Relevant Information",
                "input": "context_document and generated query",
                "output": "Key information relevant to the query."
            },
            {
                "description": "Generate Responses",
                "input": "Key information",
                "output": "Responses to the open-domain questions."
            },
            {
                "description": "Reflection and Paraphrasing",
                "input": "Generated responses",
                "output": "Paraphrased and refined answers."
            }
        ]
    },
    "short_open_book_question_answering": {
        "description": "Answer short open domain questions based on the given context document.",
        "input": "context document, query",
        "output": "short answers to questions",
        "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
### Query: %(questions)s
===
Generate short responses to the query based on the context provided. If no query is provided, please infer a relevant query from the context document. Ensure the responses are accurate and concise.
%(start_response)s
Short Answer:""",
        "steps": [
            {
                "description": "Generate Question",
                "branching_conditions": [
                    {
                        "condition": "If query is not provided",
                        "sub_steps": [
                            {
                                "description": "Infer query",
                                "input": "context_document",
                                "output": "Inferred query"
                            }
                        ]
                    },
                    {
                        "condition": "If query is provided",
                        "output": "Use the provided query"
                    }
                ]
            },
            {
                "description": "Extract Relevant Information",
                "input": "context_document and generated query",
                "output": "Key information relevant to the query."
            },
            {
                "description": "Generate Short Responses",
                "input": "Key information",
                "output": "Concise responses to open-domain questions."
            },
            {
                "description": "Review for Brevity",
                "input": "Generated responses",
                "output": "Final short answers."
            }
        ]
    },
    "short_closed_book_question_answering": {
        "description": "Answer short questions based on reasoning, with inspiration from the given context document.",
        "input": "context document, query",
        "output": "short answers to questions",
        "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
### Query: %(questions)s
===
Generate short responses to the query based on general knowledge and reasoning. If no query is provided, please infer a relevant query from the context document. Use the context document for inspiration, but ensure the answers are not directly found in the document.
%(start_response)s
Short Answer:""",
        "steps": [
            {
                "description": "Generate Question",
                "branching_conditions": [
                    {
                        "condition": "If query is not provided",
                        "sub_steps": [
                            {
                                "description": "Infer query",
                                "input": "context_document",
                                "output": "Inferred query"
                            }
                        ]
                    },
                    {
                        "condition": "If query is provided",
                        "output": "Use the provided query"
                    }
                ]
            },
            {
                "description": "Extract Relevant Information",
                "input": "context_document and generated query",
                "output": "Key information relevant to the query."
            },
            {
                "description": "Generate Short Responses",
                "input": "Key information",
                "output": "Concise responses to open-domain questions."
            },
            {
                "description": "Review for Brevity",
                "input": "Generated responses",
                "output": "Final short answers."
            }
        ]
    },
    "text_modification": {
        "description": "Modify the given text document according to specific quality, tone, and audience instructions, and create a section for definitions and ontologies.",
        "input": "context document, quality, tone, audience",
        "output": "modified text document",
        "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
### Quality: %(quality)s
### Tone: %(tone)s
### Audience: %(audience)s
===
Modify the context above to improve its quality, adjust its tone, or fit a specific audience. If no quality, tone, or audience is provided, please infer the necessary parameters from the context document. Ensure the modified text retains the original meaning but is better suited to the intended purpose. Also, extract definitions and create an ontology section based on the context above.
%(start_response)s
Modified Text:""",
        "steps": [
            {
                "description": "Creating default quality, tone, and audience",
                "branching_conditions": [
                    {
                        "condition": "If the quality, tone, or audience input is missing",
                        "sub_steps": [
                            {
                                "description": "Infer the necessary quality, tone, and audience",
                                "input": "context_document",
                                "output": "Inferred quality, tone, and audience"
                            }
                        ]
                    },
                    {
                        "condition": "If the quality, tone, and audience are provided",
                        "output": "Use the provided quality, tone, and audience"
                    }
                ]
            },
            {
                "description": "Analyze Text",
                "input": "context_document",
                "output": "Analysis of text quality, tone, and context."
            },
            {
                "description": "Modify Text",
                "input": "Text analysis",
                "output": "Improved and context-appropriate text."
            },
            {
                "description": "Extract Definitions and Ontologies",
                "input": "Modified text",
                "output": "Definitions and ontology section."
            }
        ]
    },
     "data_to_text": {
        "description": "Generate text based on the provided data and create a corresponding text-to-data instruction.",
        "input": "data, data_type",
        "output": "generated text and text-to-data instruction",
        "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
### Data Type: %(data_type)s
===
- Generate a human-readable textual summary based on the data described in the context document or inferred from the context document.
- If no data_type is provided, please infer the data_type from the data provided. The summary should be clear and concise, suitable for reports or narratives.
- Also, provide instructions for converting the generated text back into the specified data type.
%(start_response)s
Summary and Instructions:""",
        "steps": [
            {
                "description": "Analyze Data",
                "input": "data",
                "output": "Key insights and trends from the data."
            },
            {
                "description": "Infer Data Type if not provided",
                "branching_conditions": [
                    {
                        "condition": "If data_type is not provided",
                        "sub_steps": [
                            {
                                "description": "Infer data_type",
                                "input": "data",
                                "output": "Inferred data_type"
                            }
                        ]
                    },
                    {
                        "condition": "If data_type is provided",
                        "output": "Use the provided data_type"
                    }
                ]
            },
            {
                "description": "Generate Summary",
                "input": "Key insights and trends",
                "output": "Textual summary of the data."
            },
            {
                "description": "Create Text-to-Data Instruction",
                "input": "Generated summary",
                "output": "Instructions for converting text back into data."
            }
        ]
    },

    "creative_content_generation": {
        "description": "Generate creative content based on the given context document, with reflection and refinement steps.",
        "input": "context document, creative goals",
        "output": "creative content",
        "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
### Creative Goals: %(purpose)s
===
Generate creative content based on the context above. If no creative goals are provided, please infer the creative goals from the context document. The content should be original, meaningful, and incorporate elements of novelty and surprise. Reflect on and refine the content to ensure it meets the creative goals.
%(start_response)s
Creative Content:""",
        "steps": [
            {
                "description": "Identify Key Themes",
                "input": "context_document",
                "output": "Key themes and ideas."
            },
            {
                "description": "Infer Creative Goals if not provided",
                "branching_conditions": [
                    {
                        "condition": "If creative goals are not provided",
                        "sub_steps": [
                            {
                                "description": "Infer creative goals",
                                "input": "context_document",
                                "output": "Inferred creative goals"
                            }
                        ]
                    },
                    {
                        "condition": "If creative goals are provided",
                        "output": "Use the provided creative goals"
                    }
                ]
            },
            {
                "description": "Generate Creative Content",
                "input": "Key themes and ideas",
                "output": "Original and interesting creative content."
            },
            {
                "description": "Reflect on Content",
                "input": "Generated content",
                "output": "Reflection notes."
            },
            {
                "description": "Refine Content",
                "input": "Reflection notes",
                "output": "Refined creative content."
            }
        ]
    },
    "few_shot_reasoning": {
        "description": "Perform few-shot reasoning based on the given context document, with examples and derived reasoning.",
        "input": "context document, examples",
        "output": "reasoned output",
        "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
### Examples:  %(span_text)s
===
Demonstrate the ability to understand new concepts, patterns, or tasks with minimal examples based on the context above. If no examples are provided, please infer the necessary examples from the context document. Provide examples and show how to derive reasoning from them.
%(start_response)s
Examples and Reasoning:""",
        "steps": [
            {
                "description": "Identify Concepts and Patterns",
                "input": "context_document",
                "output": "New concepts and patterns."
            },
            {
                "description": "Infer Examples if not provided",
                "branching_conditions": [
                    {
                        "condition": "If examples are not provided",
                        "sub_steps": [
                            {
                                "description": "Infer examples",
                                "input": "context_document",
                                "output": "Inferred examples"
                            }
                        ]
                    },
                    {
                        "condition": "If examples are provided",
                        "output": "Use the provided examples"
                    }
                ]
            },
            {
                "description": "Provide Examples",
                "input": "New concepts and patterns",
                "output": "Few examples."
            },
            {
                "description": "Demonstrate Reasoning",
                "input": "Few examples",
                "output": "Derived reasoning from examples."
            },
            {
                "description": "Reflect on and Refine",
                "input": "Derived reasoning",
                "output": "Refined reasoning output."
            }
        ]
    },
    "conversation": {
        "description": "Engage in a detailed conversation based on the given context document, covering key points and providing insightful dialogue.",
        "input": "context document, conversation purpose",
        "output": "conversation",
        "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
### Conversation Goal: %(purpose)s
===
Generate a conversation based on the context above. If no conversation goal is provided, please infer the conversation goal from the context document. The conversation should be natural, engaging, and cover key points. Reflect on and refine the dialogue to enhance its quality.
%(start_response)s
Conversation:""",
        "steps": [
            {
                "description": "Identify Key Points",
                "input": "context_document",
                "output": "Key points from the context."
            },
            {
                "description": "Infer Conversation Goal if not provided",
                "branching_conditions": [
                    {
                        "condition": "If conversation goal is not provided",
                        "sub_steps": [
                            {
                                "description": "Infer conversation goal",
                                "input": "context_document",
                                "output": "Inferred conversation goal"
                            }
                        ]
                    },
                    {
                        "condition": "If conversation goal is provided",
                        "output": "Use the provided conversation goal"
                    }
                ]
            },
            {
                "description": "Generate Conversation",
                "input": "Key points",
                "output": "Natural and engaging conversation."
            },
            {
                "description": "Reflect on Dialogue",
                "input": "Generated conversation",
                "output": "Reflection notes."
            },
            {
                "description": "Refine Dialogue",
                "input": "Reflection notes",
                "output": "Refined conversation."
            }
        ]
    },
    "summarization": {
        "description": "Provide a detailed summary of the given context document, including main points, essential information, and reflection.",
        "input": "context document, summarization purpose",
        "output": "summary",
        "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
### Summarization Purpose: %(purpose)s
===
Summarize the context in a concise and clear manner, capturing the main points and essential information. If no summarization purpose is provided, please infer the summarization purpose from the context document. Reflect on the summary to ensure it meets the purpose.
%(start_response)s
Summary:""",
        "steps": [
            {
                "description": "Identify Main Points",
                "input": "context_document",
                "output": "Main points and essential information."
            },
            {
                "description": "Infer Summarization Purpose if not provided",
                "branching_conditions": [
                    {
                        "condition": "If summarization purpose is not provided",
                        "sub_steps": [
                            {
                                "description": "Infer summarization purpose",
                                "input": "context_document",
                                "output": "Inferred summarization purpose"
                            }
                        ]
                    },
                    {
                        "condition": "If summarization purpose is provided",
                        "output": "Use the provided summarization purpose"
                    }
                ]
            },
            {
                "description": "Write Summary",
                "input": "Main points and essential information",
                "output": "Concise and clear summary."
            },
            {
                "description": "Reflect on Summary",
                "input": "Written summary",
                "output": "Reflection notes."
            },
            {
                "description": "Refine Summary",
                "input": "Reflection notes",
                "output": "Refined summary."
            }
        ]
    },
    "paraphrasing": {
        "description": "Paraphrase the given context document, ensuring clarity, coherence, and reflection on the paraphrased text.",
        "input": "context document, paraphrasing purpose",
        "output": "paraphrased text",
        "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
### Paraphrasing Purpose: %(purpose)s
===
Paraphrase the context while retaining the original meaning. If no paraphrasing purpose is provided, please infer the paraphrasing purpose from the context document. Ensure the paraphrased text is clear and coherent. Reflect on the paraphrased text to ensure it meets the purpose.
%(start_response)s
Paraphrased Text:""",
        "steps": [
            {
                "description": "Understand Original Text",
                "input": "context_document",
                "output": "Understanding of original text."
            },
            {
                "description": "Infer Paraphrasing Purpose if not provided",
                "branching_conditions": [
                    {
                        "condition": "If paraphrasing purpose is not provided",
                        "sub_steps": [
                            {
                                "description": "Infer paraphrasing purpose",
                                "input": "context_document",
                                "output": "Inferred paraphrasing purpose"
                            }
                        ]
                    },
                    {
                        "condition": "If paraphrasing purpose is provided",
                        "output": "Use the provided paraphrasing purpose"
                    }
                ]
            },
            {
                "description": "Paraphrase Text",
                "input": "Understanding of original text",
                "output": "Clear and coherent paraphrased text."
            },
            {
                "description": "Reflect on Paraphrased Text",
                "input": "Paraphrased text",
                "output": "Reflection notes."
            },
            {
                "description": "Refine Paraphrased Text",
                "input": "Reflection notes",
                "output": "Refined paraphrased text."
            }
        ]
    },
    "dialogue_generation": {
        "description": "Generate dialogue based on the given context document, ensuring natural flow and contextual appropriateness.",
        "input": "context document, dialogue purpose",
        "output": "generated dialogue",
        "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
### Dialogue Purpose: %(purpose)s
===
Generate a dialogue based on the context above. If no dialogue purpose is provided, please infer the dialogue purpose from the context document. Ensure the dialogue is natural and contextually appropriate. Reflect on and refine the dialogue to enhance its quality.
%(start_response)s
Dialogue:""",
        "steps": [
            {
                "description": "Extract Key Points",
                "input": "context_document",
                "output": "Key points from the context."
            },
            {
                "description": "Infer Dialogue Purpose if not provided",
                "branching_conditions": [
                    {
                        "condition": "If dialogue purpose is not provided",
                        "sub_steps": [
                            {
                                "description": "Infer dialogue purpose",
                                "input": "context_document",
                                "output": "Inferred dialogue purpose"
                            }
                        ]
                    },
                    {
                        "condition": "If dialogue purpose is provided",
                        "output": "Use the provided dialogue purpose"
                    }
                ]
            },
            {
                "description": "Generate Dialogue",
                "input": "Key points",
                "output": "Natural and contextually appropriate dialogue."
            },
            {
                "description": "Reflect on Dialogue",
                "input": "Generated dialogue",
                "output": "Reflection notes."
            },
            {
                "description": "Refine Dialogue",
                "input": "Reflection notes",
                "output": "Refined dialogue."
            }
        ]
    },

   "translation": {
        "description": "Translate the given context document, ensuring accuracy and retention of original meaning.",
        "input": "context document, target language",
        "output": "translated text",
        "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
### Target Language: %(target_language)s
===
Translate the context into the specified language. If no target language is provided, please choose one of Spanish, Chinese, Arabic, Indonesian or Hindi. Ensure the translation is accurate and retains the original meaning. Reflect on and refine the translation to enhance its quality.
%(start_response)s
Translated Text:""",
        "steps": [
            {
                "description": "Understand Original Text",
                "input": "context_document",
                "output": "Understanding of original text."
            },
            {
                "description": "Infer Target Language if not provided",
                "branching_conditions": [
                    {
                        "condition": "If target language is not provided",
                        "sub_steps": [
                            {
                                "description": "Infer target language",
                                "input": "context_document",
                                "output": "Inferred target language"
                            }
                        ]
                    },
                    {
                        "condition": "If target language is provided",
                        "output": "Use the provided target language"
                    }
                ]
            },
            {
                "description": "Translate Text",
                "input": "Understanding of original text",
                "output": "Accurate translation in the target language."
            },
            {
                "description": "Reflect on Translation",
                "input": "Translated text",
                "output": "Reflection notes."
            },
            {
                "description": "Refine Translation",
                "input": "Reflection notes",
                "output": "Refined translation."
            }
        ]
    },
    "proofreading": {
        "description": "Proofread the given context document for grammar, punctuation, and spelling errors, with reflection and refinement steps.",
        "input": "context document, proofreading purpose",
        "output": "proofread text",
        "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
### Proofreading Purpose: %(purpose)s
===
Proofread the context for grammar, punctuation, and spelling errors. If no proofreading purpose is provided, please infer the proofreading purpose from the context document. Reflect on the proofread text to ensure it meets the purpose.
%(start_response)s
Proofread Text:""",
        "steps": [
            {
                "description": "Identify Errors",
                "input": "context_document",
                "output": "List of grammar, punctuation, and spelling errors."
            },
            {
                "description": "Infer Proofreading Purpose if not provided",
                "branching_conditions": [
                    {
                        "condition": "If proofreading purpose is not provided",
                        "sub_steps": [
                            {
                                "description": "Infer proofreading purpose",
                                "input": "context_document",
                                "output": "Inferred proofreading purpose"
                            }
                        ]
                    },
                    {
                        "condition": "If proofreading purpose is provided",
                        "output": "Use the provided proofreading purpose"
                    }
                ]
            },
            {
                "description": "Correct Errors",
                "input": "List of errors",
                "output": "Corrected text with clarity and correctness."
            },
            {
                "description": "Reflect on Proofread Text",
                "input": "Corrected text",
                "output": "Reflection notes."
            },
            {
                "description": "Refine Proofread Text",
                "input": "Reflection notes",
                "output": "Refined proofread text."
            }
        ]
    },
    "sentiment_analysis": {
        "description": "Perform sentiment analysis on the given context document, with explanation and reflection steps.",
        "input": "context document, analysis purpose",
        "output": "sentiment analysis",
        "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
### Analysis Purpose: %(purpose)s
===
Analyze the sentiment of the context. Determine whether the sentiment is positive, negative, or neutral, and provide a brief explanation. If no analysis purpose is provided, please infer the analysis purpose from the context document. Reflect on and refine the analysis to ensure it meets the purpose.
%(start_response)s
Sentiment Analysis:""",
        "steps": [
            {
                "description": "Identify Sentiment",
                "input": "context_document",
                "output": "Sentiment classification (positive, negative, or neutral)."
            },
            {
                "description": "Infer Analysis Purpose if not provided",
                "branching_conditions": [
                    {
                        "condition": "If analysis purpose is not provided",
                        "sub_steps": [
                            {
                                "description": "Infer analysis purpose",
                                "input": "context_document",
                                "output": "Inferred analysis purpose"
                            }
                        ]
                    },
                    {
                        "condition": "If analysis purpose is provided",
                        "output": "Use the provided analysis purpose"
                    }
                ]
            },
            {
                "description": "Explain Sentiment",
                "input": "Sentiment classification",
                "output": "Brief explanation of the sentiment."
            },
            {
                "description": "Reflect on Analysis",
                "input": "Sentiment explanation",
                "output": "Reflection notes."
            },
            {
                "description": "Refine Analysis",
                "input": "Reflection notes",
                "output": "Refined sentiment analysis."
            }
        ]
    },
    "topic_modeling": {
        "description": "Perform topic modeling on the given context document, with explanation and reflection steps.",
        "input": "context document, modeling purpose",
        "output": "topics",
        "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
### Modeling Purpose: %(purpose)s
===
Identify the main topics in the context. If no modeling purpose is provided, please infer the modeling purpose from the context document. Provide a list of topics along with a brief description of each. Reflect on and refine the topic modeling to ensure it meets the purpose.
%(start_response)s
Topics:""",
        "steps": [
            {
                "description": "Extract Main Topics",
                "input": "context_document",
                "output": "List of main topics."
            },
            {
                "description": "Infer Modeling Purpose if not provided",
                "branching_conditions": [
                    {
                        "condition": "If modeling purpose is not provided",
                        "sub_steps": [
                            {
                                "description": "Infer modeling purpose",
                                "input": "context_document",
                                "output": "Inferred modeling purpose"
                            }
                        ]
                    },
                    {
                        "condition": "If modeling purpose is provided",
                        "output": "Use the provided modeling purpose"
                    }
                ]
            },
            {
                "description": "Describe Topics",
                "input": "List of main topics",
                "output": "Brief description of each topic."
            },
            {
                "description": "Reflect on Topic Modeling",
                "input": "Described topics",
                "output": "Reflection notes."
            },
            {
                "description": "Refine Topic Modeling",
                "input": "Reflection notes",
                "output": "Refined topics and descriptions."
            }
        ]
    },
    "named_entity_recognition": {
        "description": "Identify and classify named entities in the given context document, with explanation and reflection steps.",
        "input": "context document, recognition purpose",
        "output": "named entities",
        "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
### Recognition Purpose: %(purpose)s
===
Identify and list all named entities (e.g., people, organizations, locations) mentioned in the context. If no recognition purpose is provided, please infer the recognition purpose from the context document. Reflect on and refine the recognition process to ensure it meets the purpose.
%(start_response)s
Named Entities:""",
        "steps": [
            {
                "description": "Identify Named Entities",
                "input": "context_document",
                "output": "List of named entities."
            },
            {
                "description": "Infer Recognition Purpose if not provided",
                "branching_conditions": [
                    {
                        "condition": "If recognition purpose is not provided",
                        "sub_steps": [
                            {
                                "description": "Infer recognition purpose",
                                "input": "context_document",
                                "output": "Inferred recognition purpose"
                            }
                        ]
                    },
                    {
                        "condition": "If recognition purpose is provided",
                        "output": "Use the provided recognition purpose"
                    }
                ]
            },
            {
                "description": "Classify Named Entities",
                "input": "List of named entities",
                "output": "Classification of entities (people, organizations, locations, etc.)."
            },
            {
                "description": "Reflect on Recognition",
                "input": "Classified entities",
                "output": "Reflection notes."
            },
            {
                "description": "Refine Recognition",
                "input": "Reflection notes",
                "output": "Refined list and classification of entities."
            }
        ]
    },
    "keyword_extraction_and_generation": {
        "description": "Extract and list important keywords from the given context document, with explanation and reflection steps, and then generate a new document based on the keywords.",
        "input": "context document, extraction purpose",
        "output": "keywords",
        "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
### Extraction Purpose: %(purpose)s
===
- Extract and list the most important keywords from the context.
- If no extraction purpose is provided, please infer the extraction purpose from the context document.
- Reflect on and refine the extraction process to ensure it meets the purpose.
- Generate a document based on the keywords, including details and logical transitions.
%(start_response)s
Keywords:""",
        "steps": [
            {
                "description": "Identify Important Keywords",
                "input": "context_document",
                "output": "List of important keywords."
            },
            {
                "description": "Infer Extraction Purpose if not provided",
                "branching_conditions": [
                    {
                        "condition": "If extraction purpose is not provided",
                        "sub_steps": [
                            {
                                "description": "Infer extraction purpose",
                                "input": "context_document",
                                "output": "Inferred extraction purpose"
                            }
                        ]
                    },
                    {
                        "condition": "If extraction purpose is provided",
                        "output": "Use the provided extraction purpose"
                    }
                ]
            },
            {
                "description": "Explain Keywords",
                "input": "List of keywords",
                "output": "Brief explanation of each keyword."
            },
            {
                "description": "Reflect on Extraction",
                "input": "Keywords and explanations",
                "output": "Reflection notes."
            },
            {
                "description": "Refine Extraction",
                "input": "Reflection notes",
                "output": "Refined list and explanations of keywords."
            },
            {
                "description": "Generate Document",
                "input": "Refined list and explanations of keywords.",
                "output": "New Document based on keywords"
            }
        ]
    },
    "document_classification": {
        "description": "Classify the given document into predefined categories, with explanation and reflection steps.",
        "input": "context document, classification purpose",
        "output": "classified document",
        "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
### Classification Purpose: %(purpose)s
===
Classify the document into one of the predefined categories. If no classification purpose is provided, please infer the classification purpose from the context document. Ensure the classification is accurate and provide a brief explanation. Reflect on and refine the classification to ensure it meets the purpose.
%(start_response)s
Classification:""",
        "steps": [
            {
                "description": "Identify Document Category",
                "input": "context_document",
                "output": "Predefined category for the document."
            },
            {
                "description": "Infer Classification Purpose if not provided",
                "branching_conditions": [
                    {
                        "condition": "If classification purpose is not provided",
                        "sub_steps": [
                            {
                                "description": "Infer classification purpose",
                                "input": "context_document",
                                "output": "Inferred classification purpose"
                            }
                        ]
                    },
                    {
                        "condition": "If classification purpose is provided",
                        "output": "Use the provided classification purpose"
                    }
                ]
            },
            {
                "description": "Explain Classification",
                "input": "Document category",
                "output": "Brief explanation of the classification."
            },
            {
                "description": "Reflect on Classification",
                "input": "Classification and explanation",
                "output": "Reflection notes."
            },
            {
                "description": "Refine Classification",
                "input": "Reflection notes",
                "output": "Refined classification and explanation."
            }
        ]
    },
    "fact_verification": {
        "description": "Verify the factual accuracy of the given context document, with explanation and reflection steps.",
        "input": "context document, verification purpose",
        "output": "verified facts",
        "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
### Verification Purpose: %(purpose)s
===
Verify the factual accuracy of the context based soley on the consistency of the text provided and your background knowledge. Do not refer to any external sources, validations or experiments. If no verification purpose is provided, please infer the verification purpose from the context document. Provide a detailed explanation of any inaccuracies found. Reflect on and refine the verification process to ensure it meets the purpose.
%(start_response)s
Fact Verification:""",
        "steps": [
            {
                "description": "Identify Facts",
                "input": "context_document",
                "output": "List of facts to be verified."
            },
            {
                "description": "Infer Verification Purpose if not provided",
                "branching_conditions": [
                    {
                        "condition": "If verification purpose is not provided",
                        "sub_steps": [
                            {
                                "description": "Infer verification purpose",
                                "input": "context_document",
                                "output": "Inferred verification purpose"
                            }
                        ]
                    },
                    {
                        "condition": "If verification purpose is provided",
                        "output": "Use the provided verification purpose"
                    }
                ]
            },
            {
                "description": "Verify Facts",
                "input": "List of facts",
                "output": "Verification results and explanation of inaccuracies."
            },
            {
                "description": "Reflect on Verification",
                "input": "Verification results",
                "output": "Reflection notes."
            },
            {
                "description": "Refine Verification",
                "input": "Reflection notes",
                "output": "Refined verification results and explanations."
            }
        ]
    },
    "summarization_highlights": {
        "description": "Provide key highlights of the given context document, with explanation and reflection steps.",
        "input": "context document, highlights purpose",
        "output": "highlights",
        "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
### Highlights Purpose: %(purpose)s
===
Provide a summary of the key highlights from the text. If no highlights purpose is provided, please infer the highlights purpose from the context document. Focus on the most important points and ensure clarity. Reflect on and refine the highlights to ensure they meet the purpose.
%(start_response)s
Summary Highlights:""",
        "steps": [
            {
                "description": "Identify Key Highlights",
                "input": "context_document",
                "output": "List of key highlights."
            },
            {
                "description": "Infer Highlights Purpose if not provided",
                "branching_conditions": [
                    {
                        "condition": "If highlights purpose is not provided",
                        "sub_steps": [
                            {
                                "description": "Infer highlights purpose",
                                "input": "context_document",
                                "output": "Inferred highlights purpose"
                            }
                        ]
                    },
                    {
                        "condition": "If highlights purpose is provided",
                        "output": "Use the provided highlights purpose"
                    }
                ]
            },
            {
                "description": "Write Summary",
                "input": "List of key highlights",
                "output": "Summary of key highlights."
            },
            {
                "description": "Reflect on Highlights",
                "input": "Summary of key highlights",
                "output": "Reflection notes."
            },
            {
                "description": "Refine Highlights",
                "input": "Reflection notes",
                "output": "Refined summary of key highlights."
            }
        ]
    },
    "hypothetical_scenario": {
        "description": "Describe a hypothetical scenario based on the given context document, with explanation and reflection steps.",
        "input": "context document, scenario purpose",
        "output": "hypothetical scenario",
        "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
### Scenario Purpose: %(purpose)s
===
Create a hypothetical scenario based on the context above. If no scenario purpose is provided, please infer the scenario purpose from the context document. Ensure the scenario is plausible and explores potential outcomes. Reflect on and refine the scenario to ensure it meets the purpose.
%(start_response)s
Hypothetical Scenario:""",
        "steps": [
            {
                "description": "Identify Key Elements",
                "input": "context_document",
                "output": "Key elements of the text."
            },
            {
                "description": "Infer Scenario Purpose if not provided",
                "branching_conditions": [
                    {
                        "condition": "If scenario purpose is not provided",
                        "sub_steps": [
                            {
                                "description": "Infer scenario purpose",
                                "input": "context_document",
                                "output": "Inferred scenario purpose"
                            }
                        ]
                    },
                    {
                        "condition": "If scenario purpose is provided",
                        "output": "Use the provided scenario purpose"
                    }
                ]
            },
            {
                "description": "Create Scenario",
                "input": "Key elements",
                "output": "Plausible hypothetical scenaridoc."
            },
            {
                "description": "Reflect on Scenario",
                "input": "Hypothetical scenario",
                "output": "Reflection notes."
            },
            {
                "description": "Refine Scenario",
                "input": "Reflection notes",
                "output": "Refined hypothetical scenaridoc."
            }
        ]
    },
    "problem_solution": {
        "description": "Generate a problem solution based on the given context document, with explanation and reflection steps.",
        "input": "context document, problem statement",
        "output": "solution",
        "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
### Problem Statement: %(problem_statement)s
===
Identify a problem described in the context and propose a detailed solution. If no problem statement is provided, please infer the problem statement from the context document. Ensure the solution is practical and addresses the problem effectively. Reflect on and refine the solution to ensure it meets the purpose.
%(start_response)s
Problem and Solution:""",
        "steps": [
            {
                "description": "Identify Problem",
                "input": "context_document",
                "output": "Problem statement."
            },
            {
                "description": "Infer Problem Statement if not provided",
                "branching_conditions": [
                    {
                        "condition": "If problem statement is not provided",
                        "sub_steps": [
                            {
                                "description": "Infer problem statement",
                                "input": "context_document",
                                "output": "Inferred problem statement"
                            }
                        ]
                    },
                    {
                        "condition": "If problem statement is provided",
                        "output": "Use the provided problem statement"
                    }
                ]
            },
            {
                "description": "Propose Solution",
                "input": "Problem statement",
                "output": "Detailed and practical solution."
            },
            {
                "description": "Reflect on Solution",
                "input": "Proposed solution",
                "output": "Reflection notes."
            },
            {
                "description": "Refine Solution",
                "input": "Reflection notes",
                "output": "Refined solution."
            }
        ]
    },
    "process_description": {
        "description": "Describe a process based on the given context document, with detailed explanation and reflection steps.",
        "input": "context document, purpose",
        "output": "process description",
        "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
### Process Purpose: %(purpose)s
===
Describe the process mentioned in the context in a detailed and clear manner. If no purpose is provided, please infer the process purpose from the context document. Ensure each step is thoroughly explained. Reflect on and refine the process description to ensure it meets the purpose.
%(start_response)s
Process Description:""",
        "steps": [
            {
                "description": "Identify Process Steps",
                "input": "context_document",
                "output": "List of process steps."
            },
            {
                "description": "Infer Process Purpose if not provided",
                "branching_conditions": [
                    {
                        "condition": "If process purpose is not provided",
                        "sub_steps": [
                            {
                                "description": "Infer process purpose",
                                "input": "context_document",
                                "output": "Inferred process purpose"
                            }
                        ]
                    },
                    {
                        "condition": "If process purpose is provided",
                        "output": "Use the provided process purpose"
                    }
                ]
            },
            {
                "description": "Describe Steps",
                "input": "List of process steps",
                "output": "Detailed description of each step."
            },
            {
                "description": "Reflect on Process",
                "input": "Process description",
                "output": "Reflection notes."
            },
            {
                "description": "Refine Process",
                "input": "Reflection notes",
                "output": "Refined process description."
            }
        ]
    },
    "debate_argument": {
        "description": "Formulate a debate argument based on the given context document, with explanation and reflection steps.",
        "input": "context document, argument purpose",
        "output": "debate argument",
        "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
### Argument Purpose: %(purpose)s
===
Create an argument for a debate based on the context above. If no argument purpose is provided, please infer the argument purpose from the context document. Ensure the argument is logical, well-structured, and persuasive. Reflect on and refine the argument to ensure it meets the purpose.
%(start_response)s
Debate Argument:""",
        "steps": [
            {
                "description": "Identify Key Points",
                "input": "context_document",
                "output": "Key points for the argument."
            },
            {
                "description": "Infer Argument Purpose if not provided",
                "branching_conditions": [
                    {
                        "condition": "If argument purpose is not provided",
                        "sub_steps": [
                            {
                                "description": "Infer argument purpose",
                                "input": "context_document",
                                "output": "Inferred argument purpose"
                            }
                        ]
                    },
                    {
                        "condition": "If argument purpose is provided",
                        "output": "Use the provided argument purpose"
                    }
                ]
            },
            {
                "description": "Create Argument",
                "input": "Key points",
                "output": "Logical and persuasive debate argument."
            },
            {
                "description": "Reflect on Argument",
                "input": "Debate argument",
                "output": "Reflection notes."
            },
            {
                "description": "Refine Argument",
                "input": "Reflection notes",
                "output": "Refined debate argument."
            }
        ]
    },
    "ethical_analysis": {
        "description": "Perform an ethical analysis based on the given context document, with explanation and reflection steps.",
        "input": "context document, analysis purpose",
        "output": "ethical analysis",
        "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
### Analysis Purpose: %(purpose)s
===
Analyze the ethical implications of the situation described in the context. If no analysis purpose is provided, please infer the analysis purpose from the context document. Provide a detailed explanation of the ethical considerations involved. Reflect on and refine the analysis to ensure it meets the purpose.
%(start_response)s
Ethical Analysis:""",
        "steps": [
            {
                "description": "Identify Ethical Issues",
                "input": "context_document",
                "output": "List of ethical issues."
            },
            {
                "description": "Infer Analysis Purpose if not provided",
                "branching_conditions": [
                    {
                        "condition": "If analysis purpose is not provided",
                        "sub_steps": [
                            {
                                "description": "Infer analysis purpose",
                                "input": "context_document",
                                "output": "Inferred analysis purpose"
                            }
                        ]
                    },
                    {
                        "condition": "If analysis purpose is provided",
                        "output": "Use the provided analysis purpose"
                    }
                ]
            },
            {
                "description": "Analyze Issues",
                "input": "List of ethical issues",
                "output": "Detailed ethical analysis."
            },
            {
                "description": "Reflect on Analysis",
                "input": "Ethical analysis",
                "output": "Reflection notes."
            },
            {
                "description": "Refine Analysis",
                "input": "Reflection notes",
                "output": "Refined ethical analysis."
            }
        ]
    },
    "risk_assessment": {
        "description": "Conduct a risk assessment based on the given context document, with explanation and reflection steps.",
        "input": "context document, assessment purpose",
        "output": "risk assessment",
        "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
### Assessment Purpose: %(purpose)s
===
Assess the risks associated with the situation described in the context. If no assessment purpose is provided, please infer the assessment purpose from the context document. Provide a detailed explanation of the risks and potential mitigation strategies. Reflect on and refine the assessment to ensure it meets the purpose.
%(start_response)s
Risk Assessment:""",
        "steps": [
            {
                "description": "Identify Risks",
                "input": "context_document",
                "output": "List of risks."
            },
            {
                "description": "Infer Assessment Purpose if not provided",
                "branching_conditions": [
                    {
                        "condition": "If assessment purpose is not provided",
                        "sub_steps": [
                            {
                                "description": "Infer assessment purpose",
                                "input": "context_document",
                                "output": "Inferred assessment purpose"
                            }
                        ]
                    },
                    {
                        "condition": "If assessment purpose is provided",
                        "output": "Use the provided assessment purpose"
                    }
                ]
            },
            {
                "description": "Assess Risks",
                "input": "List of risks",
                "output": "Detailed risk assessment and mitigation strategies."
            },
            {
                "description": "Reflect on Assessment",
                "input": "Risk assessment",
                "output": "Reflection notes."
            },
            {
                "description": "Refine Assessment",
                "input": "Reflection notes",
                "output": "Refined risk assessment."
            }
        ]
    },

    "comparison_and_contrast_analysis": {
        "description": "Perform a comparison and contrast analysis based on the given context document, with detailed explanation and reflection steps.",
        "input": "context document, comparison and contrast purpose",
        "output": "comparison analysis",
        "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
### Comparison And Contract Purpose: %(purpose)s
===
Compare and contrast the key elements described in the context. If no comparison and contrast purpose is provided, please infer the purpose from the context document. Provide a detailed analysis of the similarities and differences. Reflect on and refine the analysis to ensure it meets the purpose.
%(start_response)s
Comparison And Contrast Analysis:""",
        "steps": [
            {
                "description": "Identify Key Elements",
                "input": "context_document",
                "output": "List of key elements."
            },
            {
                "description": "Infer Comparison And Contrast Purpose if not provided",
                "branching_conditions": [
                    {
                        "condition": "If purpose is not provided",
                        "sub_steps": [
                            {
                                "description": "Infer purpose",
                                "input": "context_document",
                                "output": "Inferred purpose"
                            }
                        ]
                    },
                    {
                        "condition": "If purpose is provided",
                        "output": "Use the provided purpose"
                    }
                ]
            },
            {
                "description": "Compare and Contrast",
                "input": "List of key elements",
                "output": "Detailed analysis of similarities and differences."
            },
            {
                "description": "Reflect on Analysis",
                "input": "Comparison And Contrast analysis",
                "output": "Reflection notes."
            },
            {
                "description": "Refine Analysis",
                "input": "Reflection notes",
                "output": "Refined comparison and contrast analysis."
            }
        ]
    },
    "lessons_learned": {
        "description": "Document lessons learned based on the given context document, with detailed steps and reflection.",
        "input": "context document, learning purpose",
        "output": "lessons learned",
        "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
### Learning Purpose: %(purpose)s
===
Document the lessons learned from the situation described in the context. If no learning purpose is provided, please infer the learning purpose from the context document. Include insights, successes, and areas for improvement. Reflect on and refine the lessons learned to ensure they meet the purpose.
%(start_response)s
Lessons Learned:""",
        "steps": [
            {
                "description": "Identify Insights and Successes",
                "input": "context_document",
                "output": "List of insights and successes."
            },
            {
                "description": "Infer Learning Purpose if not provided",
                "branching_conditions": [
                    {
                        "condition": "If learning purpose is not provided",
                        "sub_steps": [
                            {
                                "description": "Infer learning purpose",
                                "input": "context_document",
                                "output": "Inferred learning purpose"
                            }
                        ]
                    },
                    {
                        "condition": "If learning purpose is provided",
                        "output": "Use the provided learning purpose"
                    }
                ]
            },
            {
                "description": "Identify Areas for Improvement",
                "input": "List of insights and successes",
                "output": "Areas for improvement."
            },
            {
                "description": "Reflect on Lessons Learned",
                "input": "Lessons learned",
                "output": "Reflection notes."
            },
            {
                "description": "Refine Lessons Learned",
                "input": "Reflection notes",
                "output": "Refined lessons learned document."
            }
        ]
    },
   "topic_modeling_with_table": {
        "description": "Perform topic modeling on the given context document, with explanation and reflection steps.",
        "input": "context document, modeling purpose",
        "output": "topics",
        "prompt": """%(start_command)s
### Context Document:
%(context_document)s
### Modeling Purpose: %(purpose)s
===
Identify the main topics in the context. If no modeling purpose is provided, please infer the modeling purpose from the context document. Provide a list of topics along with a brief description of each. Reflect on and refine the topic modeling to ensure it meets the purpose.
Use this format:

Topics:
...
<table>
Provide relevant tables summarizing the topics and their descriptions.
</table>
%(start_response)s""",
        "steps": [
            {
                "description": "Extract Main Topics",
                "input": "context_document",
                "output": "List of main topics."
            },
            {
                "description": "Infer Modeling Purpose if not provided",
                "branching_conditions": [
                    {
                        "condition": "If modeling purpose is not provided",
                        "sub_steps": [
                            {
                                "description": "Infer modeling purpose",
                                "input": "context_document",
                                "output": "Inferred modeling purpose"
                            }
                        ]
                    },
                    {
                        "condition": "If modeling purpose is provided",
                        "output": "Use the provided modeling purpose"
                    }
                ]
            },
            {
                "description": "Describe Topics",
                "input": "List of main topics",
                "output": "Brief description of each topic."
            },
            {
                "description": "Reflect on Topic Modeling",
                "input": "Described topics",
                "output": "Reflection notes."
            },
            {
                "description": "Refine Topic Modeling",
                "input": "Reflection notes",
                "output": "Refined topics and descriptions."
            },
            {
                "description": "Include Tables",
                "input": "Refined topics and descriptions",
                "output": "Topics with relevant tables."
            }
        ]
    },
    "named_entity_recognition_with_tables": {
        "description": "Identify and classify named entities in the given context document, with explanation and reflection steps.",
        "input": "context document, recognition purpose",
        "output": "named entities",
        "prompt": """%(start_command)s
### Context Document:
%(context_document)s
### Recognition Purpose: %(purpose)s
===
Identify and list all named entities (e.g., people, organizations, locations) mentioned in the context. If no recognition purpose is provided, please infer the recognition purpose from the context document. Reflect on and refine the recognition process to ensure it meets the purpose, using the following format:
Named Entities:
<table>
Provide a table summarizing the named entities and their classifications.
</table>
%(start_response)s""",
        "steps": [
            {
                "description": "Identify Named Entities",
                "input": "context_document",
                "output": "List of named entities."
            },
            {
                "description": "Infer Recognition Purpose if not provided",
                "branching_conditions": [
                    {
                        "condition": "If recognition purpose is not provided",
                        "sub_steps": [
                            {
                                "description": "Infer recognition purpose",
                                "input": "context_document",
                                "output": "Inferred recognition purpose"
                            }
                        ]
                    },
                    {
                        "condition": "If recognition purpose is provided",
                        "output": "Use the provided recognition purpose"
                    }
                ]
            },
            {
                "description": "Classify Named Entities",
                "input": "List of named entities",
                "output": "Classification of entities (people, organizations, locations, etc.)."
            },
            {
                "description": "Reflect on Recognition",
                "input": "Classified entities",
                "output": "Reflection notes."
            },
            {
                "description": "Refine Recognition",
                "input": "Reflection notes",
                "output": "Refined list and classification of entities."
            },
            {
                "description": "Include Tables",
                "input": "Refined list and classification of entities",
                "output": "Entities with relevant tables."
            }
        ]
    },
    "keyword_extraction_and_generation_with_tables": {
        "description": "Extract and list important keywords from the given context document, with explanation and reflection steps, and then generate a new document based on the keywords.",
        "input": "context document, extraction purpose",
        "output": "keywords",
        "prompt": """%(start_command)s
### Context Document:
%(context_document)s
### Extraction Purpose: %(purpose)s
===
Extract and list the most important keywords from the context. If no extraction purpose is provided, please infer the extraction purpose from the context document. Reflect on and refine the extraction process to ensure it meets the purpose. Generate a document based on the keywords, including details and logical transitions.
Optionally you may return keywords in the form of a table like this:
<table>
Provide a table summarizing the keywords and their relevance.
</table>
%(start_response)s
Keywords:""",
        "steps": [
            {
                "description": "Identify Important Keywords",
                "input": "context_document",
                "output": "List of important keywords."
            },
            {
                "description": "Infer Extraction Purpose if not provided",
                "branching_conditions": [
                    {
                        "condition": "If extraction purpose is not provided",
                        "sub_steps": [
                            {
                                "description": "Infer extraction purpose",
                                "input": "context_document",
                                "output": "Inferred extraction purpose"
                            }
                        ]
                    },
                    {
                        "condition": "If extraction purpose is provided",
                        "output": "Use the provided extraction purpose"
                    }
                ]
            },
            {
                "description": "Explain Keywords",
                "input": "List of keywords",
                "output": "Brief explanation of each keyword."
            },
            {
                "description": "Reflect on Extraction",
                "input": "Keywords and explanations",
                "output": "Reflection notes."
            },
            {
                "description": "Refine Extraction",
                "input": "Reflection notes",
                "output": "Refined list and explanations of keywords."
            },
            {
                "description": "Generate Document",
                "input": "Refined list and explanations of keywords.",
                "output": "New Document based on keywords"
            },
            {
                "description": "Include Tables",
                "input": "New Document based on keywords",
                "output": "Keywords with relevant tables."
            }
        ]
    },
    "hypothesis_generation": {
        "description": "Generate a hypothesis based on the given context document, with detailed explanation and reflection steps.",
        "input": "context document, hypothesis purpose",
        "output": "hypothesis",
        "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
### Hypothesis Purpose: %(purpose)s
===
- Generate a hypothesis based on the context above.
- If no hypothesis purpose is provided, please infer the hypothesis purpose from the context document. For example, hypothesis generation could be for science, ecnomics, etc.
- Provide a detailed explanation of the hypothesis and the rationale behind it.
- Reflect on and refine the hypothesis to ensure it meets the purpose.
%(start_response)s
Hypothesis:""",
        "steps": [
            {
                "description": "Identify Key Information",
                "input": "context_document",
                "output": "Key information for hypothesis generation."
            },
            {
                "description": "Infer Hypothesis Purpose if not provided",
                "branching_conditions": [
                    {
                        "condition": "If hypothesis purpose is not provided",
                        "sub_steps": [
                            {
                                "description": "Infer hypothesis purpose",
                                "input": "context_document",
                                "output": "Inferred hypothesis purpose"
                            }
                        ]
                    },
                    {
                        "condition": "If hypothesis purpose is provided",
                        "output": "Use the provided hypothesis purpose"
                    }
                ]
            },
            {
                "description": "Generate Hypothesis",
                "input": "Key information",
                "output": "Detailed hypothesis and rationale."
            },
            {
                "description": "Reflect on Hypothesis",
                "input": "Generated hypothesis",
                "output": "Reflection notes."
            },
            {
                "description": "Refine Hypothesis",
                "input": "Reflection notes",
                "output": "Refined hypothesis and rationale."
            }
        ]
    },

    "problem_statement": {
        "description": "Write a problem statement based on the given context document, with detailed explanation and reflection steps.",
        "input": "context document, problem description",
        "output": "problem statement",
        "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
### Problem Description: %(problem_description)s
===
Formulate a problem statement based on the context text. If no problem description is provided, please infer the problem description from the context document. Ensure the statement clearly defines the problem, its context, and significance. Reflect on and refine the problem statement to ensure it meets the purpose.
%(start_response)s
Problem Statement:""",
        "steps": [
            {
                "description": "Identify Key Issues",
                "input": "context_document",
                "output": "Key issues and context."
            },
            {
                "description": "Infer Problem Description if not provided",
                "branching_conditions": [
                    {
                        "condition": "If problem description is not provided",
                        "sub_steps": [
                            {
                                "description": "Infer problem description",
                                "input": "context_document",
                                "output": "Inferred problem description"
                            }
                        ]
                    },
                    {
                        "condition": "If problem description is provided",
                        "output": "Use the provided problem description"
                    }
                ]
            },
            {
                "description": "Formulate Problem Statement",
                "input": "Key issues and context",
                "output": "Clear and concise problem statement."
            },
            {
                "description": "Reflect on Problem Statement",
                "input": "Problem statement",
                "output": "Reflection notes."
            },
            {
                "description": "Refine Problem Statement",
                "input": "Reflection notes",
                "output": "Refined problem statement."
            }
        ]
    },


}
