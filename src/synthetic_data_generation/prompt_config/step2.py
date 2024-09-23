step_2_enhance_prompts = {

  "analogical_reasoning": {
    "description": "Given a context document, create analogous versions of the context document and explain the analogical relationship.",
    "input": "context document, target words",
    "output": "analogous documents, analogies and explanations",
    "prompt": """%(start_command)s
### Original Context Document:
%(context_document)s
===
First extract a set of triplets from the document (entity *relationship* entity).
Generate three analogical versions of the document that change the triplets into analogous triplets. For example:
  (Teacher instructs students) -> (Coach trains athletes)
  (Doctor treats patient) -> (Mechanic repairs car)
  (Sun rises in the east) -> (Moon appears at night)
  (Bird builds nest) -> (Spider weaves web)
  (Farmer plants crops) -> (Gardener grows flowers)
  (Chef cooks meal) -> (Artist paints picture)
  (Author writes book) -> (Composer creates music)
  (Engineer designs planes) -> (Architect builds building)
These examples maintain the underlying relational structure while substituting analogous entities and actions.

For each version of the document, list the new triplets and explanation why they are analogical to the original set of triplets.
Format your answer as below:

Triplet Relationships:
- <head entity> *relation 1* <tail entity>
- <head entity> *relation 2* <tail entity>
- <head entity> *relation 3* <tail entity>

1. Versions Aligning with the Target Triplet Relationships:
New Context Document:
...
New Triplets:
...
Explanations:
...

2. Versions Aligning with the Target Triplet Relationships:
New Context Document:
...
New Triplets:
...
Explanations:
...

3. Versions Aligning with the Target Triplet Relationships:
New Context Document:
...
New Triplets:
...
Explanations:
...
%(start_response)s
Triplet Relationships:
""",
    "steps": [
      {
        "description": "Extract Triplets",
        "input": "context document",
        "output": "Set of triplets (entity *relationship* entity)"
      },
      {
        "description": "Create Analogical Versions",
        "input": "context document, triplets",
        "output": "Three analogous versions of the document with new triplets"
      },
      {
        "description": "Generate Explanations",
        "input": "analogous documents, original triplets",
        "output": "Explanations for why new triplets are analogous to the original triplets"
      }
    ]
  },
    "rule_abduction_from_examples": {
        "description": "Given a context document, create versions of the context document with slight variations that covers a label and that do not. Create rules and explanations to cover variations, and choose the best rule/examples.",
        "input": "context document, label(s)",
        "output": "varied documents, rules and explanations",
        "prompt": """%(start_command)s
### Original Context Document With Missing Fill In Blanks:
%(context_document)s
### Labels: %(label)s
===
Generate three slight variations of the document that satisfy the label and three versions that do not.
If no label is provided, then infer the label that is interesting and educational from the context.
Then create two sets of rules and explanations: one set that covers only the versions that satisfy the label, and one that do not.
The rules must be based on the words, meaning, and patterns of the generated versions of the context documents only.
Finally, decide which explanation is the best (e.g., simplest, most coherent, and with broadest scope).
Format your answers as follows:
### Labels: <the label>

A. Versions Satisfying the Label:
1. Fill-In-Blank Document 1 (similar to original context if original context satisfies the label)
   Set of Spans 1 (e.g., different phrases, nouns, etc. but still covering the labels)
2.
3.

===
B. Versions Not Satisfying the Label:
1. Fill-In-Blank Document 1 (similar to original context if original context does NOT satisfy the label)
   Set of Spans 1 (e.g., different phrases, nouns, etc. but still covering the labels)
2.
3.

===
X. Rules Covering Satisfied Versions A But Not Versions B:
1. Rule 1.
   - Explanation 1:
2.
3.

===
Y. Rules Covering Unsatisfied Versions B But Not Versions A:
1. Rule 1.
   - Explanation 1:
2.
3.
%(start_response)s
### Labels:""",
        "steps": [
            {
                "description": "Infer the label if none is provided",
                "input": "context document, labels",
                "output": "A label where text in the document does or does not satisfy the label."
            },
             {
                "description": "Create Versions Satisfying the Label",
                "input": "context document, labels",
                "output": "Three slight variations of the document that satisfy the label."
            },
            {
                "description": "Create Versions Not Satisfying the Label",
                "input": "context document, spans, classification dimension, labels",
                "output": "Three slight variations of the document that do not satisfy the label."
            },
            {
                "description": "Generate Rules and Explanations",
                "input": "varied documents",
                "output": "Rules and explanations covering both satisfied and unsatisfied versions."
            }
        ]
    },

    # tasks that have an element of time, decisions, conditional branching
    "planning_timeline_creation": {
        "description": "Create a timeline based on the given context document, with detailed explanation and reflection steps.",
        "input": "context document, timeline purpose",
        "output": "timeline",
        "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
### Timeline Purpose: %(purpose)s
===
Create a timeline of events based on the context above. If no timeline purpose is provided, please infer the timeline purpose from the context document. Ensure the timeline is clear and includes all significant events. Reflect on each step and if other actions or events could have occured, and their consequences.  Refine the timeline to ensure it meets the purpose.
%(start_response)s
Timeline:""",
        "steps": [
            {
                "description": "Identify Events",
                "input": "context_document",
                "output": "List of significant events."
            },
            {
                "description": "Infer Timeline Purpose if not provided",
                "branching_conditions": [
                    {
                        "condition": "If timeline purpose is not provided",
                        "sub_steps": [
                            {
                                "description": "Infer timeline purpose",
                                "input": "context_document",
                                "output": "Inferred timeline purpose"
                            }
                        ]
                    },
                    {
                        "condition": "If timeline purpose is provided",
                        "output": "Use the provided timeline purpose"
                    }
                ]
            },
            {
                "description": "Create Timeline",
                "input": "List of significant events",
                "output": "Clear and detailed timeline."
            },
            {
                "description": "Reflect on timeline, and what alternative actions or events could have occured, and their consequences",
                "input": "Timeline",
                "output": "Reflection notes."
            },
            {
                "description": "Refine Timeline",
                "input": "Reflection notes",
                "output": "Refined timeline."
            }
        ]
    },
    "planning_process_description_with_images": {
        "description": "Describe a process based on the given context document, with detailed explanation and reflection steps.",
        "input": "context document",
        "output": "process description",
        "prompt": """%(start_command)s
### Context Document:
%(context_document)s
===
Describe the process mentioned in the context in a detailed and clear manner. If no process description is provided, please infer the process description from the context document. Ensure each step is thoroughly explained. Reflect on and refine the process description to ensure it meets the purpose. As part of reflection, analyze  each step and if other actions or events could have occured, and their consequences.
In every case, add images and figures to enhance the response:
<image>
Provide an image or diagram caption if it helps in illustrating the process.
</image>
<figure>
Provide a figure caption here if needed.
</figure>
%(start_response)s
Process Description:""",
        "steps": [
            {
                "description": "Identify Process Steps",
                "input": "context_document",
                "output": "List of process steps."
            },
            {
                "description": "Infer Process Description if not provided",
                "branching_conditions": [
                    {
                        "condition": "If process description is not provided",
                        "sub_steps": [
                            {
                                "description": "Infer process description",
                                "input": "context_document",
                                "output": "Inferred process description"
                            }
                        ]
                    },
                    {
                        "condition": "If process description is provided",
                        "output": "Use the provided process description"
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
            },
            {
                "description": "Include Visuals",
                "input": "Refined process description",
                "output": "Process with relevant images/diagrams/figures."
            }
        ]
    },
    "planning_timeline_creation_with_images": {
        "description": "Create a timeline based on the given context document, with detailed explanation and reflection steps.",
        "input": "context document, timeline purpose",
        "output": "timeline",
        "prompt": """%(start_command)s
### Context Document:
%(context_document)s
### Timeline Purpose: %(purpose)s
===
Create a timeline of events based on the context above. If no timeline purpose is provided, please infer the timeline purpose from the context document. Ensure the timeline is clear and includes all significant events. Reflect on and refine the timeline to ensure it meets the purpose. Reflect on what alternative actions or events could have occured, and their consequences.
Use a graph, diagram and/or image to explain better:
<image>
Provide an image or diagram caption if it helps in illustrating the timeline.
</image>
<figure>
Provide a figure caption here if needed.
</figure>
%(start_response)s
Timeline:""",
        "steps": [
            {
                "description": "Identify Events",
                "input": "context_document",
                "output": "List of significant events."
            },
            {
                "description": "Infer Timeline Purpose if not provided",
                "branching_conditions": [
                    {
                        "condition": "If timeline purpose is not provided",
                        "sub_steps": [
                            {
                                "description": "Infer timeline purpose",
                                "input": "context_document",
                                "output": "Inferred timeline purpose"
                            }
                        ]
                    },
                    {
                        "condition": "If timeline purpose is provided",
                        "output": "Use the provided timeline purpose"
                    }
                ]
            },
            {
                "description": "Create Timeline",
                "input": "List of significant events",
                "output": "Clear and detailed timeline."
            },
            {
                "description": "Reflect on Timeline, and what alternative actions or events could have occured, and their consequences",
                "input": "Timeline",
                "output": "Reflection notes."
            },
            {
                "description": "Refine Timeline",
                "input": "Reflection notes",
                "output": "Refined timeline."
            },
            {
                "description": "Include Visuals",
                "input": "Refined timeline",
                "output": "Timeline with relevant images/diagrams/figures."
            }
        ]
    },
      "planning_goal_setting": {
        "description": "Set goals based on the given context document, with detailed steps and reflection.",
        "input": "context document, goal setting purpose",
        "output": "goals",
        "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
### Goal Setting Purpose: %(purpose)s
===
- based on the context above, set clear and achievable goals. If no goal setting purpose is provided, please infer the goal setting purpose from the context document.
- Provide a detailed explanation of each goal and the steps necessary to achieve them.
- Provide a forward path from starting state to end goal, and a backwards path from goal state to starting state.
- Reflect on and refine the goals to ensure they meet the intended purpose.
%(start_response)s
Goals:""",
        "steps": [
            {
                "description": "Identify Objectives",
                "input": "context_document",
                "output": "List of objectives."
            },
            {
                "description": "Infer Goal Setting Purpose if not provided",
                "branching_conditions": [
                    {
                        "condition": "If goal setting purpose is not provided",
                        "sub_steps": [
                            {
                                "description": "Infer goal setting purpose",
                                "input": "context_document",
                                "output": "Inferred goal setting purpose"
                            }
                        ]
                    },
                    {
                        "condition": "If goal setting purpose is provided",
                        "output": "Use the provided goal setting purpose"
                    }
                ]
            },
            {
                "description": "Set Goals",
                "input": "List of objectives",
                "output": "Clear and achievable goals with detailed steps, both forward and backwards path."
            },
            {
                "description": "Reflect on Goals",
                "input": "Set goals",
                "output": "Reflection notes."
            },
            {
                "description": "Refine Goals",
                "input": "Reflection notes",
                "output": "Refined goals and steps."
            }
        ]
    },
    "planning_trend_analysis": {
        "description": "Perform a trend analysis based on the given context document, with detailed explanation and reflection steps.",
        "input": "context document, trend analysis purpose",
        "output": "trend analysis",
        "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
### Trend Analysis Purpose: %(purpose)s
===
- Analyze the trends described in the context.
- If no trend analysis purpose is provided, please infer the trend analysis purpose from the context document. For example, time-series analysis, buying trends, political trends, etc.
- Provide a detailed explanation of the trends and their potential implications.
- Reflect on and refine the trend analysis to ensure it meets the purpose.
%(start_response)s
Trend Analysis:""",
        "steps": [
            {
                "description": "Identify Trends",
                "input": "context_document",
                "output": "List of trends."
            },
            {
                "description": "Infer Trend Analysis Purpose if not provided",
                "branching_conditions": [
                    {
                        "condition": "If trend analysis purpose is not provided",
                        "sub_steps": [
                            {
                                "description": "Infer trend analysis purpose",
                                "input": "context_document",
                                "output": "Inferred trend analysis purpose"
                            }
                        ]
                    },
                    {
                        "condition": "If trend analysis purpose is provided",
                        "output": "Use the provided trend analysis purpose"
                    }
                ]
            },
            {
                "description": "Analyze Trends",
                "input": "List of trends",
                "output": "Detailed analysis of trends and implications."
            },
            {
                "description": "Reflect on Analysis",
                "input": "Trend analysis",
                "output": "Reflection notes."
            },
            {
                "description": "Refine Analysis",
                "input": "Reflection notes",
                "output": "Refined trend analysis."
            }
        ]
    },

    # Agent type tasks

    "retrieval_augmented_generation_with_images": {
        "description": "Generate text using retrieval-augmented techniques based on the given context document and include an evaluation step.",
        "input": "context document, query",
        "output": "generated text",
        "prompt": """%(start_command)s
### Context Document:
%(context_document)s
### Query: %(question))s
===
Generate a response to the query treating the context document as one or more relevant documents that is retrieved. Determine which documents are most relevant and pay attention only to those documents. Using the most relevant documents, generate a detailed answer. If no query is provided, please infer a relevant query from the context document. Evaluate the generated response for accuracy and completeness.
%(start_response)sm
Query Response:
<image>
Provide an image or diagram caption if it helps in illustrating any part of the response.
</image>
<figure>
Provide a figure caption here if needed.
</figure>""",
        "steps": [
            {
                "description": "Treat Context Document as Retrieved Relevant Documents",
                "input": "context_document",
                "output": "Relevant documents retrieved."
            },
            {
                "description": "Determine Most Relevant Documents",
                "input": "Relevant documents retrieved.",
                "output": "Most relevant documents."
            },
            {
                "description": "Generate Response",
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
                "description": "Evaluate Response",
                "input": "Generated response",
                "output": "Evaluation report of the response."
            },
            {
                "description": "Include Visuals",
                "input": "Evaluation report",
                "output": "Response with relevant images/diagrams/figures."
            }
        ]
    },
    "tool_use_with_images": {
        "description": "Use specific tools to process the given context document and demonstrate a round trip process.",
        "input": "context document, tools",
        "output": "processed document and tool application instructions",
        "prompt": """%(start_command)s
### Context Document:
%(context_document)s
### Tools: %(tools)s
===
Describe how to use the provided tools to achieve the goals mentioned in the context. If no tools are provided, please infer the necessary tools from the context document. Demonstrate a round trip process where the output of the tools is converted back to the original form.
%(start_response)s
Tools and Process:
<image>
Provide an image or diagram caption if it helps in illustrating the tool use process.
</image>
<figure>
Provide a figure caption here if needed.
</figure>""",
        "steps": [
            {
                "description": "Identify Tools",
                "branching_conditions": [
                    {
                        "condition": "If tools are not provided",
                        "sub_steps": [
                            {
                                "description": "Infer tools",
                                "input": "context_document",
                                "output": "Inferred tools"
                            }
                        ]
                    },
                    {
                        "condition": "If tools are provided",
                        "output": "Use the provided tools"
                    }
                ]
            },
            {
                "description": "Describe Tool Use",
                "input": "List of tools",
                "output": "Detailed instructions and examples for each tool."
            },
            {
                "description": "Apply Tools",
                "input": "List of tools and context_document",
                "output": "Step-by-step application of the tools and output."
            },
            {
                "description": "Demonstrate Round Trip",
                "input": "Processed output",
                "output": "Instructions for converting output back to original form."
            },
            {
                "description": "Include Visuals",
                "input": "Instructions for converting output back to original form",
                "output": "Tool use process with relevant images/diagrams/figures."
            }
        ]
    },

      "retrieval_augmented_generation": {
        "description": "Generate text using retrieval-augmented techniques based on the given context document and include an evaluation step.",
        "input": "context document, query",
        "output": "generated text",
        "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
### Query: %(questions)s
===
You will simulate a retrieval of snippets based on a query, and then answer the query based on the retrieved documents:
- If no query is provided, please infer a relevant query from the context document.
- Determine the most relevant part of the context document that can be used to reply to the query. This part will be treated as the "relevant"  simulated  retrieval snippets.
- Treat 3 other parts of the context document as distractor snippets. For each part, rewrite the snippets, each in a different style. Use different one of these styles: blog post, advertisement, government report, social media discussion.
- List the retriel document in a random order.
- Generate a response to the query based on the relevant retrieval document.
- Evaluate the generated response for accuracy and completeness.
%(start_response)s
Query Response:""",
        "steps": [

            {
                "description": "Determine Query",
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
                "description": "Simulate Retrieval Of Documents",
                "input": "context_document",
                "output": "Relevant documents retrieved.",
                "sub_steps": [
                            {
                                "description": "Convert most relevant part of the context document into the relevant snippet",
                                "input": "context_document",
                                "output": "Relevant Snippet"
                            },
                            {
                                "description": "Treat 3 other part of the context document as distractor snippets",
                                "input": "context_document",
                                "output": "3 Distractor snippets in different styles"
                            }
                        ]
            },
            {
                "description": "Determine Most Relevant Documents",
                "input": "Relevant snippets retrieved.",
                "output": "Most relevant snippet."
            },

            {
                "description": "Generate Response",
                "input": "Query and snippets.",
                "output": "Response based on query and snippets."
            },
            {
                "description": "Evaluate Response",
                "input": "Generated response",
                "output": "Evaluation report of the response."
            }
        ]
    },
    "tool_use": {
        "description": "Use specific tools to process the given context document and demonstrate a round trip process.",
        "input": "context document, tools",
        "output": "processed document and tool application instructions",
        "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
### Tools: %(tools)s
===
Describe how to use the provided tools to achieve the goals mentioned in the context. If no tools are provided, please infer the necessary tools from the context document. Demonstrate a round trip process where the output of the tools is converted back to the original form.
%(start_response)s
Tools and Process:""",
        "steps": [
            {
                "description": "Identify Tools",
                "branching_conditions": [
                    {
                        "condition": "If tools are not provided",
                        "sub_steps": [
                            {
                                "description": "Infer tools",
                                "input": "context_document",
                                "output": "Inferred tools"
                            }
                        ]
                    },
                    {
                        "condition": "If tools are provided",
                        "output": "Use the provided tools"
                    }
                ]
            },
            {
                "description": "Describe Tool Use",
                "input": "List of tools",
                "output": "Detailed instructions and examples for each tool."
            },
            {
                "description": "Apply Tools",
                "input": "List of tools and context_document",
                "output": "Step-by-step application of the tools and output."
            },
            {
                "description": "Demonstrate Round Trip",
                "input": "Processed output",
                "output": "Instructions for converting output back to original form."
            }
        ]
    },

   # teaching reasoning. UL2 type, FIM, etc.
   "infer_missing_spans_with_explanations_first": {
        "description": "Generate snippets for a fill-in-the-blank problem and provide explanations.",
        "input": "context document with missing fill-in-the-blank spans, span type, span labels",
        "output": "fill-in-the-blank snippets and explanations",
        "prompt": """%(start_command)s  Generate explanations of why a set of snippets can fill the missing spans in the given document, then provide the snippets:
### Context Document With Fill-in-the-blank Span(s):
%(context_document)s
===
Infer the span type if none is provided. Start with the explanations.  The span should be for whole words, sentences or paragraphs and should not be partial words.
%(start_response)s
Explanations:""",
        "steps": [
            {
                "description": "Extract Document Content",
                "input": "context_document",
                "output": "Extracted main content with missing spans."
            },
            {
                "description": "Generate Explanations",
                "input": "Extracted main content",
                "output": "Explanations for missing spans."
            },
            {
                "description": "Provide Snippets",
                "input": "Explanations",
                "output": "Fill-in-the-blank snippets."
            }
        ]
    },
    "infer_missing_spans_with_explanations": {
        "description": "Generate snippets for a fill-in-the-blank problem and provide explanations.",
        "input": "context document with missing fill-in-the-blank spans, span type, span labels",
        "output": "fill-in-the-blank snippets and explanations",
        "prompt": """%(start_command)s  Generate snippets for a fill-in-the-blank problem and provide explanations.
### Context Document With Fill-in-the-blank Span(s):
%(context_document)s
===
Infer the span type if none is provided. The snippet should be for whole words, sentences or paragraphs and should not be partial words.
%(start_response)s
Snippets:""",
        "steps": [
            {
                "description": "Extract Document Content",
                "input": "context_document",
                "output": "Extracted main content with missing spans."
            },
            {
                "description": "Provide Snippets",
                "input": "Extracted main content",
                "output": "Fill-in-the-blank snippets."
            },
            {
                "description": "Generate Explanations",
                "input": "Extracted main content",
                "output": "Explanations for missing spans."
            }
        ]
    },
  "infer_missing_spans_with_explanations_first_reversed": {
    "description": "Generate snippets for a fill-in-the-blank problem in reverse order and provide explanations.",
    "input": "context document with missing fill-in-the-blank spans",
    "output": "fill-in-the-blank snippets and explanations",
    "prompt": """%(start_command)s  Generate explanations of why a set of snippets can fill the missing spans in the given document in reverse order, then provide the snippets also in reverse order:
### Context Document With Fill-in-the-blank Span(s):
%(context_document)s
===
Infer the snippets if none is provided.
Start with the explanations from the last snippet. The snippet should be for whole words, sentences or paragraphs and should not be partial words.
%(start_response)s
Explanations Starting With Last Span:""",
    "steps": [
        {
            "description": "Extract Document Content",
            "input": "context_document",
            "output": "Extracted main content with missing spans."
        },
        {
            "description": "Generate Explanations",
            "input": "Extracted main content",
            "output": "Explanations for missing spans, starting with the last snippet."
        },
        {
            "description": "Provide Snippets",
            "input": "Explanations",
            "output": "Fill-in-the-blank snippets in reverse order."
        }
    ]
  },
  "infer_missing_spans_with_explanations_reversed": {
    "description": "Generate snippets for a fill-in-the-blank problem in reverse order and provide explanations.",
    "input": "context document with missing fill-in-the-blank spans",
    "output": "fill-in-the-blank snippets and explanations",
    "prompt": """%(start_command)s  Generate snippets for a fill-in-the-blank problem in reverse order and then provide explanations.
### Context Document With Fill-in-the-blank Span(s):
%(context_document)s
===
Infer the span type if none is provided. The span should be for whole words, sentences or paragraphs and should not be partial words.
%(start_response)s
Snippets, Starting With Last Span:""",
    "steps": [
        {
            "description": "Extract Document Content",
            "input": "context_document",
            "output": "Extracted main content with missing spans."
        },
        {
            "description": "Provide Snippets",
            "input": "Extracted main content",
            "output": "Fill-in-the-blank snippets in reverse order."
        },
        {
            "description": "Generate Explanations",
            "input": "Extracted main content",
            "output": "Explanations for missing spans, starting with the last snippet."
        }
    ]
  },
  "infer_correct_ordering_with_explanations": {
        "description": "Reorder a set of sentences and provide explanations.",
        "input": "set of sentences in random order",
        "output": "correct ordering with explanations",
        "prompt": """%(start_command)s  Given a set of sentences in random order, reorder the sentences logically and explain why that order is best.
### Sentences in random order:
%(context_document)s
===
%(start_response)s
""",
        "steps": [
            {
                "description": "Reorder Sentences",
                "input": "random_sentences",
                "output": "Correctly ordered sentences."
            },
            {
                "description": "Provide Explanation",
                "input": "Correctly ordered sentences",
                "output": "Explanation for the ordering."
            }
        ]
    },
    "explain_correct_ordering_of_sentences": {
        "description": "Explain why a given ordering of sentences is appropriate.",
        "input": "set of sentences in random order, and correct ordering",
        "output": "correct ordering with explanations",
        "prompt": """%(start_command)s  Given a set of sentences in random order and the proposed correct order (if no ordering is provided, then propose one), explain why that ordering is best and then format the whole document.
### Sentences in random order:
%(context_document)s
### Correct ordering:
%(span_text)s
===
%(start_response)s
Explanation:""",
        "steps": [
            {
                "description": "Reorder Sentences",
                "input": "random_sentences",
                "output": "Correctly ordered sentences."
            },
            {
                "description": "Provide Explanation",
                "input": "Correctly ordered sentences",
                "output": "Explanation for the ordering."
            },
            {
                "description": "Format Document",
                "input": "Correctly ordered sentences with explanation",
                "output": "Formatted document."
            }
        ]
    },
    "infer_context_document_with_explanations": {
        "description": "Generate a fill-in-the-blank context from text snippets.",
        "input": "A set of text snippets in a particular order",
        "output": "Context document with fill-in-the-blank spans",
        "prompt": """%(start_command)s  Given a set of text snippets, generate a fill-in-the-blank context document and explain why this context is appropriate.
### Ordered Text Snippets for Spans:
%(context_document)s
===
The span should be for whole words, sentences or paragraphs and should not be partial words.
%(start_response)s
### Context Document With Fill-in-the-blank Span(s):
""",
        "steps": [
            {
                "description": "Extract Snippets",
                "input": "snippets",
                "output": "Ordered text snippets."
            },
            {
                "description": "Generate Context Document",
                "input": "Ordered text snippets",
                "output": "Proposed fill-in-the-blank context document."
            },
            {
                "description": "Provide Explanation",
                "input": "Proposed context document",
                "output": "Explanation for the context."
            }
        ]
    },
    "infer_context_document_with_explanations_first": {
        "description": "Generate a fill-in-the-blank context with explanations from text snippets.",
        "input": "A set of text snippets in a particular order and context",
        "output": "Document with fill-in-the-blank spans and explanations",
        "prompt": """%(start_command)s  Given a set of text snippets, generate a fill-in-the-blank document and explain why this context is appropriate. The fill-in-the-blank should be labeled <<SPAN 1>>, etc.
If no text snippets are provided infer them from the context. The span should be for whole words, sentences or paragraphs and should not be partial words. In your response. begin with the explanations, then provide the context document.
### Context:
%(context_document)s
### Ordered Text Snippets for Spans:
%(span_text)s
===
%(start_response)s
Explanations:
""",
        "steps": [
            {
                "description": "Extract Snippets",
                "input": "snippets",
                "output": "Ordered text snippets."
            },
            {
                "description": "Provide Explanation",
                "input": "Ordered text snippets",
                "output": "Explanation for the context."
            },
            {
                "description": "Generate Document",
                "input": "Explanation",
                "output": "Proposed fill-in-the-blank document."
            }
        ]
    },
    "provide_explanations_for_given_context_document": {
        "description": "Explain why a proposed fill-in-the-blank context document is appropriate for a set of snippets.",
        "input": "A set of text snippets in order, and a proposed fill-in-the-blank context document",
        "output": "Explanation why the context document is appropriate",
        "prompt": """%(start_command)s  Given a set of text snippets and a proposed fill-in-the-blank context document using those snippets, explain why the context document is appropriate. The span should be for whole words, sentences or paragraphs and should not be partial words.
### Ordered Text Snippets for Spans:
%(span_text)s
### Context Document With Fill-in-the-blank Span(s):
%(context_document)s
===
%(start_response)s
Explanations:
""",
        "steps": [
            {
                "description": "Extract Snippets",
                "input": "snippets",
                "output": "Ordered text snippets."
            },
            {
                "description": "Generate Context Document",
                "input": "Ordered text snippets",
                "output": "Proposed fill-in-the-blank context document."
            },
            {
                "description": "Provide Explanation",
                "input": "Proposed context document",
                "output": "Explanation for the context."
            }
        ]
    },
    "provide_explanations_for_given_span": {
        "description": "Explain why a proposed text is appropriate for a missing span in a context document.",
        "input": "context document with a single fill-in-the-blank, proposed snippet",
        "output": "Explanation why proposed snippet is appropriate",
        "prompt": """%(start_command)s  For a context document with a single missing span and a proposed text for the span, explain why the proposed text is appropriate. The span should be for whole words, sentences or paragraphs and should not be partial words.
### Context Document Missing Span:
%(context_document)s
### Proposed Span: %(span_text)s
===
If no span_text is provided, choose on a short span of no less than 3 words and no more than one paragraph from the context document. Your explanation should treat the span as if it was missing form the context document.
%(start_response)s
Explanation:""",
        "steps": [
            {
                "description": "Extract Document Content",
                "input": "context_document",
                "output": "Extracted main content with missing span."
            },
            {
                "description": "Generate Explanation",
                "input": "Extracted main content",
                "output": "Explanation for proposed span."
            },
            {
                "description": "Provide Proposed Span",
                "input": "Explanation",
                "output": "Proposed text to fill the missing span."
            }
        ]
    },
  "three_multiple_choice_ordering_with_explanations": {
    "description": "Reorder a set of sentences, provide three possible orderings, choose the best one, and explain why.",
    "input": "set of sentences in random order",
    "output": "best ordering with explanations",
    "prompt": """%(start_command)s  Given a set of sentences in random order, provide three possible logical orderings, choose the best one, and explain why that order is best.
### Sentences in random order:
%(context_document)s
===
Your answer should be in this format:
Possible Orderings:
1. Ordering 1:
2. Ordering 2:
3. Ordering 3:
Best Ordering and Explanation:
%(start_response)s
Possible Orderings:
""",
    "steps": [
        {
            "description": "Generate Three Orderings",
            "input": "random_sentences",
            "output": "Three logical orderings of sentences."
        },
        {
            "description": "Choose Best Ordering",
            "input": "Three logical orderings",
            "output": "Best ordering with explanation."
        }
    ]
  },
  "four_multiple_choice_ordering_with_explanations": {
    "description": "Reorder a set of sentences, provide four possible orderings, choose the best one, and explain why.",
    "input": "set of sentences in random order",
    "output": "best ordering with explanations",
    "prompt": """%(start_command)s  Given a set of sentences in random order, provide four possible logical orderings, choose the best one, and explain why that order is best.
### Sentences in random order:
%(context_document)s
===
Your answer should be in this format:
Possible Orderings:
1. Ordering 1:
2. Ordering 2:
3. Ordering 3:
4. Ordering 4:
Best Ordering and Explanation:
%(start_response)s
Possible Orderings:
""",
    "steps": [
        {
            "description": "Generate Three Orderings",
            "input": "random_sentences",
            "output": "Four logical orderings of sentences."
        },
        {
            "description": "Choose Best Ordering",
            "input": "Four logical orderings",
            "output": "Best ordering with explanation."
        }
    ]
  },
}
