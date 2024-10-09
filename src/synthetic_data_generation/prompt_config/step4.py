# more complicated tests
step_4_subject_matter_prompts = {

    "grade_school_math_with_cot_and_python_solutions": {
        "description": "Generate grade school math problems based on the given extract, along with solutions in both step-by-step reasoning and python code.",
        "input": "extract",
        "output": "grade school math problems with solutions",
        "prompt": """%(start_command)s Here's an extract from a webpage:
%(context_document)s
===
- Create an educational math problem related to the snippet above targeted at grade-school students.
- Complex college-like topics such as Electromagnetism and Integration shouldn't be used, as they aren't usually taught at grade-school.
- If that's what the snippet is about, look for a much simpler scientific alternative to explain, and use everyday examples.
- For instance, if the topic is 'Linear Algebra' you might discuss how arranging objects in rows and columns can help solve puzzles.
- Avoid technical terms and LaTeX and only discuss simple grade-school level topics. Start the math problem right away.
- Then solve the problems, using step-by-step reasoning under the heading "Step-by-Step Solution:".
- Then solve the same problems using python code.
- If the python answers are not the same as the step-by-step answers, redo one of the solutions, either the step-by-step or python solution so that the answers are the same.
%(start_response)s""",
        "steps": [
            {
                "description": "Extract Main Idea",
                "input": "context_document",
                "output": "Main idea and complexity level."
            },
            {
                "description": "Simplify Topic",
                "branching_conditions": [
                    {
                        "condition": "If the topic is complex",
                        "sub_steps": [
                            {
                                "description": "Simplify the complex topic to a grade-school level.",
                                "input": "Main idea and complexity level",
                                "output": "Simplified topic suitable for grade-school students."
                            }
                        ]
                    },
                    {
                        "condition": "If the topic is already simple",
                        "output": "Use the main idea directly."
                    }
                ]
            },
            {
                "description": "Create Math Problem",
                "input": "Simplified topic or Main idea",
                "output": "Grade-school math problem."
            },
            {
                "description": "Solve Math Problem Step-by-Step",
                "input": "Grade-school math problem",
                "output": "Step-by-step solution."
            },
            {
                "description": "Solve Math Problem Using Python",
                "input": "Grade-school math problem",
                "output": "A Python program that solves the math problem."
            },
            {
                "description": "Confirming Matching Answers",
                "branching_conditions": [
                    {
                        "condition": "If the answers are not the same",
                        "sub_steps": [
                            {
                                "description": "Choose the solution which is most likely to be incorrect",
                                "input": "Step-by-step solution and Python program solution",
                                "output": "Identification of which solution is likely to be incorrect"
                            },
                            {
                                "description": "Re-solve the incorrect solution.",
                                "input": "Grade-school math problem and identification of incorrect solution",
                                "output": "Revised solution"
                            }
                        ]
                    },
                    {
                        "condition": "If the answers are the same",
                        "output": "Use the matching answers"
                    }
                ]
            }
        ]
    },
    "grade_school_math": {
        "description": "Generate grade school math problems based on the given extract.",
        "input": "extract",
        "output": "grade school math problems",
        "prompt": """%(start_command)s Here's an extract from a webpage:
%(context_document)s
===
Create an educational math problem related to the snippet above targeted at grade-school students. Complex college-like topics such Electromagnetism and Integration shouldn't be used, as they aren't usually taught at grade-school. If that's what the snippet is about, look for a much simpler scientific alternative to explain, and use everyday examples. For instance, if the topic is 'Linear Algebra' you might discuss how arranging objects in rows and columns can help solve puzzles.
Avoid technical terms and LaTeX and only discuss simple grade-school level topics. Start the math problem right away followed by the solutions.
Then verify your solutions, and provide step-by-step explanations and revised solutions if necessary.
%(start_response)s""",
        "steps": [
            {
                "description": "Extract Main Idea",
                "input": "context_document",
                "output": "Main idea and complexity level."
            },
            {
                "description": "Simplify Topic",
                "branching_conditions": [
                    {
                        "condition": "If the topic is complex",
                        "sub_steps": [
                            {
                                "description": "Simplify the complex topic to a grade-school level.",
                                "input": "Main idea and complexity level",
                                "output": "Simplified topic suitable for grade-school students."
                            }
                        ]
                    },
                    {
                        "condition": "If the topic is already simple",
                        "output": "Use the main idea directly."
                    }
                ]
            },
            {
                "description": "Create Math Problem",
                "input": "Simplified topic or Main idea",
                "output": "Grade-school math problem with solutions."
            },
            {
                "description": "Verify Math Solutions",
                "input": "Grade-school math problem with solutions.",
                "output": "Verify solutions with explanations."
            }
        ]
    },
    "grade_school_math_with_solutions": {
        "description": "Generate grade school math problems based on the given extract, along with solutions.",
        "input": "extract",
        "output": "grade school math problems with solutions",
        "prompt": """%(start_command)s Here's an extract from a webpage:
%(context_document)s
===
- Create an educational math problem related to the snippet above targeted at grade-school students.
- Complex college-like topics such Electromagnetism and Integration shouldn't be used, as they aren't usually taught at grade-school.
- If that's what the snippet is about, look for a much simpler scientific alternative to explain, and use everyday examples.
- For instance, if the topic is 'Linear Algebra' you might discuss how arranging objects in rows and columns can help solve puzzles.
- Avoid technical terms and LaTeX and only discuss simple grade-school level topics. Start the math problem right away.
- Then solve the problems, using step-by-step reasoning.
%(start_response)s""",
        "steps": [
            {
                "description": "Extract Main Idea",
                "input": "context_document",
                "output": "Main idea and complexity level."
            },
            {
                "description": "Simplify Topic",
                "branching_conditions": [
                    {
                        "condition": "If the topic is complex",
                        "sub_steps": [
                            {
                                "description": "Simplify the complex topic to a grade-school level.",
                                "input": "Main idea and complexity level",
                                "output": "Simplified topic suitable for grade-school students."
                            }
                        ]
                    },
                    {
                        "condition": "If the topic is already simple",
                        "output": "Use the main idea directly."
                    }
                ]
            },
            {
                "description": "Create Math Problem",
                "input": "Simplified topic or Main idea",
                "output": "Grade-school math problem."
            },
            {
                "description": "Solve Math Problem Step-by-Step",
                "input": "Grade-school math problem",
                "output": "Step-by-step solution."
            }
        ]
    },
    "college_math_with_python_and_solutions": {
        "description": "Generate college-level math problems based on the given extract, along with solutions in both step-by-step reasoning and python code.",
        "input": "extract",
        "output": "college-level math problems with solutions",
        "prompt": """%(start_command)s Here's an extract from a webpage:
%(context_document)s
===
- Create an educational math problem related to the snippet above targeted at college students.
- Use advanced topics such as Calculus, Linear Algebra, or Differential Equations.
- Clearly state the math problem without unnecessary context.
- Then solve the problems, using step-by-step reasoning under the heading "Step-by-Step Solution:".
- Then solve the same problems using python code.
- If the python answers are not the same as the step-by-step answers, redo one of the solutions, either the step-by-step or python solution so that the answers are the same.
%(start_response)s""",
        "steps": [
            {
                "description": "Extract Main Idea",
                "input": "context_document",
                "output": "Main idea and complexity level."
            },
            {
                "description": "Create Math Problem",
                "input": "Main idea",
                "output": "College-level math problem."
            },
            {
                "description": "Solve Math Problem Step-by-Step",
                "input": "College-level math problem",
                "output": "Step-by-step solution."
            },
            {
                "description": "Solve Math Problem Using Python",
                "input": "College-level math problem",
                "output": "A Python program that solves the math problem."
            },
            {
                "description": "Confirm Matching Answers",
                "branching_conditions": [
                    {
                        "condition": "If the answers are not the same",
                        "sub_steps": [
                            {
                                "description": "Identify the solution which is most likely to be incorrect",
                                "input": "Step-by-step solution and Python program solution",
                                "output": "Identification of which solution is likely to be incorrect"
                            },
                            {
                                "description": "Re-solve the incorrect solution.",
                                "input": "College-level math problem and identification of incorrect solution",
                                "output": "Revised solution"
                            }
                        ]
                    },
                    {
                        "condition": "If the answers are the same",
                        "output": "Use the matching answers"
                    }
                ]
            }
        ]
    },

    "pre_school_math_with_cot_and_python_solutions": {
        "description": "Generate preschool math problems based on the given extract, along with solutions in both step-by-step reasoning and python code.",
        "input": "extract",
        "output": "preschool math problems with solutions",
        "prompt": """%(start_command)s Here's an extract from a webpage:
%(context_document)s
===
- Create an educational math problem related to the snippet above targeted at preschool students.
- Use very simple concepts like counting, basic addition, and shapes.
- Avoid complex numbers, fractions, or advanced concepts.
- Use everyday examples like counting toys, apples, etc.
- Start the math problem right away.
- Then solve the problem, using step-by-step reasoning under the heading "Step-by-Step Solution:".
- Then solve the same problem using python code.
- If the python answers are not the same as the step-by-step answers, redo one of the solutions, either the step-by-step or python solution so that the answers are the same.
%(start_response)s""",
        "steps": [
            {
                "description": "Extract Main Idea",
                "input": "context_document",
                "output": "Main idea and simplicity level."
            },
            {
                "description": "Simplify Topic",
                "branching_conditions": [
                    {
                        "condition": "If the topic is complex",
                        "sub_steps": [
                            {
                                "description": "Simplify the complex topic to a preschool level.",
                                "input": "Main idea and simplicity level",
                                "output": "Simplified topic suitable for preschool students."
                            }
                        ]
                    },
                    {
                        "condition": "If the topic is already simple",
                        "output": "Use the main idea directly."
                    }
                ]
            },
            {
                "description": "Create Math Problem",
                "input": "Simplified topic or Main idea",
                "output": "Preschool math problem."
            },
            {
                "description": "Solve Math Problem Step-by-Step",
                "input": "Preschool math problem",
                "output": "Step-by-step solution."
            },
            {
                "description": "Solve Math Problem Using Python",
                "input": "Preschool math problem",
                "output": "A Python program that solves the math problem."
            },
            {
                "description": "Confirm Matching Answers",
                "branching_conditions": [
                    {
                        "condition": "If the answers are not the same",
                        "sub_steps": [
                            {
                                "description": "Identify the solution which is most likely to be incorrect",
                                "input": "Step-by-step solution and Python program solution",
                                "output": "Identification of which solution is likely to be incorrect"
                            },
                            {
                                "description": "Re-solve the incorrect solution.",
                                "input": "Preschool math problem and identification of incorrect solution",
                                "output": "Revised solution"
                            }
                        ]
                    },
                    {
                        "condition": "If the answers are the same",
                        "output": "Use the matching answers"
                    }
                ]
            }
        ]
    },
    "middle_school_math_with_solutions": {
        "description": "Generate middle school math problems based on the given extract, along with step-by-step solutions.",
        "input": "extract",
        "output": "middle school math problems with solutions",
        "prompt": """%(start_command)s Here's an extract from a webpage:
%(context_document)s
===
- Create an educational math problem related to the snippet above targeted at middle school students.
- Use concepts like fractions, decimals, basic algebra, and geometry.
- Avoid overly complex or high school level topics.
- Provide a step-by-step solution and ensure clarity.
%(start_response)s""",
        "steps": [
            {
                "description": "Extract Main Idea",
                "input": "context_document",
                "output": "Main idea and complexity level."
            },
            {
                "description": "Simplify Topic",
                "branching_conditions": [
                    {
                        "condition": "If the topic is complex",
                        "sub_steps": [
                            {
                                "description": "Simplify the complex topic to a middle school level.",
                                "input": "Main idea and complexity level",
                                "output": "Simplified topic suitable for middle school students."
                            }
                        ]
                    },
                    {
                        "condition": "If the topic is already simple",
                        "output": "Use the main idea directly."
                    }
                ]
            },
            {
                "description": "Create Math Problem",
                "input": "Simplified topic or Main idea",
                "output": "Middle school math problem."
            },
            {
                "description": "Solve Math Problem Step-by-Step",
                "input": "Middle school math problem",
                "output": "Step-by-step solution."
            }
        ]
    },
    "high_school_math_with_python_and_solutions": {
        "description": "Generate high school math problems based on the given extract, along with solutions in both step-by-step reasoning and python code.",
        "input": "extract",
        "output": "high school math problems with solutions",
        "prompt": """%(start_command)s Here's an extract from a webpage:
%(context_document)s
===
- Create an educational math problem related to the snippet above targeted at high school students.
- Use topics like algebra, trigonometry, basic calculus, or statistics.
- Provide a step-by-step solution, and ensure that the solution aligns with typical high school math courses.
- Then solve the same problem using python code.
- If the python answers are not the same as the step-by-step answers, redo one of the solutions, either the step-by-step or python solution so that the answers are the same.
%(start_response)s""",
        "steps": [
            {
                "description": "Extract Main Idea",
                "input": "context_document",
                "output": "Main idea and complexity level."
            },
            {
                "description": "Simplify Topic",
                "branching_conditions": [
                    {
                        "condition": "If the topic is complex",
                        "sub_steps": [
                            {
                                "description": "Simplify the complex topic to a high school level.",
                                "input": "Main idea and complexity level",
                                "output": "Simplified topic suitable for high school students."
                            }
                        ]
                    },
                    {
                        "condition": "If the topic is already simple",
                        "output": "Use the main idea directly."
                    }
                ]
            },
            {
                "description": "Create Math Problem",
                "input": "Simplified topic or Main idea",
                "output": "High school math problem."
            },
            {
                "description": "Solve Math Problem Step-by-Step",
                "input": "High school math problem",
                "output": "Step-by-step solution."
            },
            {
                "description": "Solve Math Problem Using Python",
                "input": "High school math problem",
                "output": "A Python program that solves the math problem."
            },
            {
                "description": "Confirm Matching Answers",
                "branching_conditions": [
                    {
                        "condition": "If the answers are not the same",
                        "sub_steps": [
                            {
                                "description": "Identify the solution which is most likely to be incorrect",
                                "input": "Step-by-step solution and Python program solution",
                                "output": "Identification of which solution is likely to be incorrect"
                            },
                            {
                                "description": "Re-solve the incorrect solution.",
                                "input": "High school math problem and identification of incorrect solution",
                                "output": "Revised solution"
                            }
                        ]
                    },
                    {
                        "condition": "If the answers are the same",
                        "output": "Use the matching answers"
                    }
                ]
            }
        ]
    },
    "graduate_level_math_with_python_and_solutions": {
        "description": "Generate graduate-level math problems based on the given extract, along with solutions in both step-by-step reasoning and python code.",
        "input": "extract",
        "output": "graduate-level math problems with solutions",
        "prompt": """%(start_command)s Here's an extract from a webpage:
%(context_document)s
===
- Create an educational math problem related to the snippet above targeted at graduate students.
- Use highly advanced topics such as Topology, Abstract Algebra, or Advanced Calculus.
- Clearly state the math problem without unnecessary context.
- Then solve the problems, using step-by-step reasoning under the heading "Step-by-Step Solution:".
- Then solve the same problems using python code.
- If the python answers are not the same as the step-by-step answers, redo one of the solutions, either the step-by-step or python solution so that the answers are the same.
%(start_response)s""",
        "steps": [
            {
                "description": "Extract Main Idea",
                "input": "context_document",
                "output": "Main idea and complexity level."
            },
            {
                "description": "Create Math Problem",
                "input": "Main idea",
                "output": "Graduate-level math problem."
            },
            {
                "description": "Solve Math Problem Step-by-Step",
                "input": "Graduate-level math problem",
                "output": "Step-by-step solution with proof."
            },
            {
                "description": "Solve Math Problem Using Python",
                "input": "Graduate-level math problem",
                "output": "A Python program that solves the math problem."
            },
            {
                "description": "Confirm Matching Answers",
                "branching_conditions": [
                    {
                        "condition": "If the answers are not the same",
                        "sub_steps": [
                            {
                                "description": "Identify the solution which is most likely to be incorrect",
                                "input": "Step-by-step solution and Python program solution",
                                "output": "Identification of which solution is likely to be incorrect"
                            },
                            {
                                "description": "Re-solve the incorrect solution.",
                                "input": "Graduate-level math problem and identification of incorrect solution",
                                "output": "Revised solution"
                            }
                        ]
                    },
                    {
                        "condition": "If the answers are the same",
                        "output": "Use the matching answers"
                    }
                ]
            }
        ]
    },


    "pre_school_math": {
        "description": "Generate preschool math problems based on the given extract, using simple and relatable concepts.",
        "input": "extract",
        "output": "preschool math problems",
        "prompt": """%(start_command)s Here's an extract from a webpage:
%(context_document)s
===
- Create an educational math problem related to the snippet above targeted at preschool students.
- Use very simple concepts like counting, basic addition, and shapes.
- Avoid complex numbers, fractions, or advanced concepts.
- Use everyday examples like counting toys, apples, etc.
- Start the math problem right away followed by the solution.
%(start_response)s""",
        "steps": [
            {
                "description": "Extract Main Idea",
                "input": "context_document",
                "output": "Main idea and simplicity level."
            },
            {
                "description": "Simplify Topic",
                "branching_conditions": [
                    {
                        "condition": "If the topic is complex",
                        "sub_steps": [
                            {
                                "description": "Simplify the complex topic to a preschool level.",
                                "input": "Main idea and simplicity level",
                                "output": "Simplified topic suitable for preschool students."
                            }
                        ]
                    },
                    {
                        "condition": "If the topic is already simple",
                        "output": "Use the main idea directly."
                    }
                ]
            },
            {
                "description": "Create Math Problem",
                "input": "Simplified topic or Main idea",
                "output": "Preschool math problem."
            },
            {
                "description": "Solve Math Problem",
                "input": "Preschool math problem",
                "output": "Simple solution."
            }
        ]
    },
    "middle_school_math": {
        "description": "Generate middle school math problems based on the given extract.",
        "input": "extract",
        "output": "middle school math problems",
        "prompt": """%(start_command)s Here's an extract from a webpage:
%(context_document)s
===
- Create an educational math problem related to the snippet above targeted at middle school students.
- Use concepts like fractions, decimals, basic algebra, and geometry.
- Avoid overly complex or high school level topics.
- Provide a step-by-step solution and ensure clarity.
%(start_response)s""",
        "steps": [
            {
                "description": "Extract Main Idea",
                "input": "context_document",
                "output": "Main idea and complexity level."
            },
            {
                "description": "Simplify Topic",
                "branching_conditions": [
                    {
                        "condition": "If the topic is complex",
                        "sub_steps": [
                            {
                                "description": "Simplify the complex topic to a middle school level.",
                                "input": "Main idea and complexity level",
                                "output": "Simplified topic suitable for middle school students."
                            }
                        ]
                    },
                    {
                        "condition": "If the topic is already simple",
                        "output": "Use the main idea directly."
                    }
                ]
            },
            {
                "description": "Create Math Problem",
                "input": "Simplified topic or Main idea",
                "output": "Middle school math problem."
            },
            {
                "description": "Solve Math Problem Step-by-Step",
                "input": "Middle school math problem",
                "output": "Step-by-step solution."
            }
        ]
    },
    "high_school_math": {
        "description": "Generate high school math problems based on the given extract, including solutions.",
        "input": "extract",
        "output": "high school math problems with solutions",
        "prompt": """%(start_command)s Here's an extract from a webpage:
%(context_document)s
===
- Create an educational math problem related to the snippet above targeted at high school students.
- Use topics like algebra, trigonometry, basic calculus, or statistics.
- Provide a step-by-step solution, and ensure that the solution aligns with typical high school math courses.
%(start_response)s""",
        "steps": [
            {
                "description": "Extract Main Idea",
                "input": "context_document",
                "output": "Main idea and complexity level."
            },
            {
                "description": "Simplify Topic",
                "branching_conditions": [
                    {
                        "condition": "If the topic is complex",
                        "sub_steps": [
                            {
                                "description": "Simplify the complex topic to a high school level.",
                                "input": "Main idea and complexity level",
                                "output": "Simplified topic suitable for high school students."
                            }
                        ]
                    },
                    {
                        "condition": "If the topic is already simple",
                        "output": "Use the main idea directly."
                    }
                ]
            },
            {
                "description": "Create Math Problem",
                "input": "Simplified topic or Main idea",
                "output": "High school math problem."
            },
            {
                "description": "Solve Math Problem Step-by-Step",
                "input": "High school math problem",
                "output": "Step-by-step solution."
            }
        ]
    },
    "college_math_with_python_and_solutions": {
        "description": "Generate college-level math problems based on the given extract, along with solutions in both step-by-step reasoning and python code.",
        "input": "extract",
        "output": "college-level math problems with solutions",
        "prompt": """%(start_command)s Here's an extract from a webpage:
%(context_document)s
===
- Create an educational math problem related to the snippet above targeted at college students.
- Use advanced topics such as Calculus, Linear Algebra, or Differential Equations.
- Clearly state the math problem without unnecessary context.
- Then solve the problems, using step-by-step reasoning under the heading "Step-by-Step Solution:".
- Then solve the same problems using python code.
- If the python answers are not the same as the step-by-step answers, redo one of the solutions, either the step-by-step or python solution so that the answers are the same.
%(start_response)s""",
        "steps": [
            {
                "description": "Extract Main Idea",
                "input": "context_document",
                "output": "Main idea and complexity level."
            },
            {
                "description": "Create Math Problem",
                "input": "Main idea",
                "output": "College-level math problem."
            },
            {
                "description": "Solve Math Problem Step-by-Step",
                "input": "College-level math problem",
                "output": "Step-by-step solution."
            },
            {
                "description": "Solve Math Problem Using Python",
                "input": "College-level math problem",
                "output": "A Python program that solves the math problem."
            },
            {
                "description": "Confirm Matching Answers",
                "branching_conditions": [
                    {
                        "condition": "If the answers are not the same",
                        "sub_steps": [
                            {
                                "description": "Identify the solution which is most likely to be incorrect",
                                "input": "Step-by-step solution and Python program solution",
                                "output": "Identification of which solution is likely to be incorrect"
                            },
                            {
                                "description": "Re-solve the incorrect solution.",
                                "input": "College-level math problem and identification of incorrect solution",
                                "output": "Revised solution"
                            }
                        ]
                    },
                    {
                        "condition": "If the answers are the same",
                        "output": "Use the matching answers"
                    }
                ]
            }
        ]
    },
    "graduate_level_math": {
        "description": "Generate graduate-level math problems based on the given extract, along with advanced solutions.",
        "input": "extract",
        "output": "graduate-level math problems with solutions",
        "prompt": """%(start_command)s Here's an extract from a webpage:
%(context_document)s
===
- Create an educational math problem related to the snippet above targeted at graduate students.
- Use highly advanced topics such as Topology, Abstract Algebra, or Advanced Calculus.
- Clearly state the math problem without unnecessary context.
- Then solve the problems, using step-by-step reasoning under the heading "Step-by-Step Solution:".
- Ensure the solution includes rigorous proofs and reasoning expected at a graduate level.
%(start_response)s""",
        "steps": [
            {
                "description": "Extract Main Idea",
                "input": "context_document",
                "output": "Main idea and complexity level."
            },
            {
                "description": "Create Math Problem",
                "input": "Main idea",
                "output": "Graduate-level math problem."
            },
            {
                "description": "Solve Math Problem with Proof",
                "input": "Graduate-level math problem",
                "output": "Step-by-step solution with proof."
            }
        ]
    },

    "children_story_with_images": {
        "description": "Create a fun and simple e-learning module tailored for 5 to 10 year-old children, using a playful and imaginative approach.", "input": "text snippet", "output": "e-learning module",
        "prompt": """%(start_command)s Create a fun and simple e-learning module tailored for 5 to 10 year-old children. Opt for a playful and imaginative approach, suitable for very young learners. The module should relate to the following text snippet:
%(context_document)s
===
In this module for young children, aim to:
- Use very simple, everyday words and phrases that a 5-year-old would easily understand, avoiding any complex concepts or technical terms.
- Tell a short, engaging story with colorful cartoon characters. For instance, to illustrate economic trade concepts use characters like animals or friendly creatures trading snacks or toys. Another example is addition and calculus, use apples to explain: '2 apples + 3 apples = 5 apples' .
- Keep the tone light, cheerful, and encouraging.
- If applicable, add images captions between "<image> </image>" tags.  Add two or more image captions at the beginning, end or inside the text itself.
%(start_response)s""",
        "steps": [
            {
                "description": "Extract Main Idea",
                "input":"context_document",
                "output": "Main idea."
            },
            {
                "description": "Create Simple Story With image captions",
                "input": "Main idea",
                "output": "Simple children's story with cartoon characters."
            }
        ]
    },
    "middle_school_textbook_with_images": {
        "description": "Create an engaging and accessible e-learning module for middle school students without prior knowledge on the topic.", "input": "text snippet", "output": "e-learning module",
        "prompt": """%(start_command)s Create an engaging and accessible e-learning module tailored for middle school students without prior knowledge on the topic. The module should relate to the following text snippet:
%(context_document)s
===
Instead of a traditional textbook approach, use a story-based narrative to explain the concept. Try to:
- Avoid technical jargon and present the ideas in a straightforward, conversational tone to spark curiosity and relate to the experiences of a younger audience.
- Include interactive elements like thought experiments and real-life scenarios. The goal is to topic approachable and fun, sparking curiosity about how it applies to everyday life.
- Do not use introductory phrases such as "welcome to this unit" at the beginning or conclusions the end.
- Add an exercise section, with accurate, complete and in-depth answersbased on the ideas of the textbook.
- If applicable, add images captions between "<image> </image>" tags.  Add two or more image captions at the beginning, end or inside the text itself.
%(start_response)s""",
        "steps": [
            {
                "description": "Extract Main Idea",
                "input":"context_document",
                "output": "Main idea."
            },
            {
                "description": "Develop Story-Based Narrative With image captions",
                "input": "Main idea",
                "output": "Story-based narrative for middle school students."
            }
        ]
    },
    "children_textbook_with_images": {
        "description": "Write an educational story (3-5 paragraphs) targeted at young children using simple words.", "input": "text snippet", "output": "educational story",
        "prompt": """%(start_command)s Write an educational story (3-5 paragraphs) targeted at young children using simple words. The story should be inspired from this text snippet:
%(context_document)s
===
The story doesn’t have to be addressing everything in the snippet, it is there just for inspiration.
The story should have the following features:
- Science integration: embed basic science concepts within the story, explaining them through the characters' adventures and discoveries. For example, if the story includes a scene where characters are looking at the sky, you could have them wonder why it's blue and explain the physics behind in grade school level.
- Dialogue: include at least one dialogue and insightful conversation.
- If applicable, add images captions between "<image> </image>" tags.  Add two or more image captions at the beginning, end or inside the text itself.
- Unexpected twist: conclude with a twist that doesn't resolve as hoped, but leaves a clear lesson about life and science.
Do not provide commentary and start the story right away. Do not start with classic sentences like "Once upon a time", be creative.
%(start_response)s""",
        "steps": [
            {
                "description": "Extract Main Idea",
                "input":"context_document",
                "output": "Main idea."
            },
            {
                "description": "Develop Educational Story With Image Captions",
                "sub_steps": [
                    {
                        "description": "Write a 3-5 paragraph educational story integrating basic science concepts.",
                        "input": "Main idea",
                        "output": "Educational story with science concepts."
                    },
                    {
                        "description": "Include dialogue and insightful conversation and images.",
                        "input": "Educational story with science concepts",
                        "output": "Story with dialogue and image captions."
                    },
                     {
                        "description": "Add images and an unexpected twist with a clear lesson.",
                        "input": "Story with dialogue",
                        "output": "Completed educational story with image captions."
                    }
                ]
            }
        ]
    },
    "blog_post_with_images": {
        "description": "Generate a blog post based on the given extract.", "input": "webpage extract", "output": "blog post",
        "prompt": """%(start_command)s Here is an extract from a webpage:
%(context_document)s
===
Write an informative and insightful blog post that expands upon the extracted webpage above. Your post should delve into the nuances of the topic, offering fresh perspectives and deeper analysis. Aim to:
- Inform: Provide valuable, well-researched information that educates the reader.
- Engage: Write in a conversational tone that connects with the audience, making complex ideas accessible.
- If applicable, add images captions between "<image> </image>" tags.  Add two or more image captions at the beginning, end or inside the text itself.
- Illustrate: Use examples, anecdotes, or personal experiences to bring the topic to life.
Do not give a title and do not start with sentences like "Have you ever..." or "Hello dear readers..", simply write the content without these introductory phrases.
%(start_response)s""",
        "steps": [
            {
                "description": "Extract Main Idea",
                "input":"context_document",
                "output": "Main idea."
            },
            {
                "description": "Expand and Analyze Topic",
                "sub_steps": [
                    {
                        "description": "Provide well-researched information to educate the reader.",
                        "input": "Main idea",
                        "output": "Informative content."
                    },
                    {
                        "description": "Write in a conversational tone to engage the audience.",
                        "input": "Informative content",
                        "output": "Engaging content."
                    },
                    {
                        "description": "Use examples and anecdotes to illustrate the topic.",
                        "input": "Engaging content",
                        "output": "Illustrative content."
                    },
                    {
                        "description": "Add images and combine all elements to create a blog post.",
                        "input": "Illustrative content",
                        "output": "Completed blog post with image captions."
                    }
                ]
            }
        ]
    },
    "textbook_with_images": {
        "description": "Create a textbook segment based on the given extract.", "input": "webpage extract", "output": "textbook segment",
        "prompt": """%(start_command)s Here is an extract from a webpage:
%(context_document)s
===
Write an extensive and detailed course unit suitable for a textbook, related to the given extract. Do not just list concepts, but develop each one in detail before moving to the next, as we prioritize depth of understanding and comprehensive exploration of the subject matter over breadth. Focus on:
- Rigor: Ensure in-depth coverage of the concepts.
- Engagement: Use a narrative style akin to Michael Lewis, making it captivating and thought-provoking.
- Relevance: Connect the topic with current trends, real-life examples, or recent studies.
- If applicable, add images captions between "<image> </image>" tags.  Add two or more image captions at the beginning, end or inside the text itself.
- Do not include a title or an introduction, simply write the content without headlines and introductory phrases.""",
        "steps": [
            {
                "description": "Extract Main Idea",
                "input":"context_document",
                "output": "Main idea."
            },
            {
                "description": "Write Detailed Course Unit",
                "sub_steps": [
                    {
                        "description": "Develop each concept in detail with in-depth coverage.",
                        "input": "Main idea",
                        "output": "Detailed concepts."
                    },
                    {
                        "description": "Use a narrative style to make the content engaging.",
                        "input": "Detailed concepts",
                        "output": "Engaging narrative."
                    },
                    {
                        "description": "Connect the topic with current trends and real-life examples.",
                        "input": "Engaging narrative",
                        "output": "Relevant content."
                    },
                    {
                        "description": "Add images and combine all elements to create the course unit.",
                        "input": "Relevant content",
                        "output": "Completed course unit with image captions."
                    }
                ]
            }
        ]
    },
    "how_to_article": {
        "description": "Write a how-to article based on the given extract.", "input": "webpage extract", "output": "how-to article",
        "prompt": """%(start_command)s Here is an extract from a webpage:
%(context_document)s
===
Write a long and very detailed tutorial that could be part of how-to article whose title is related to the extracted webpage above. Include in depth explanations for each step and how it helps achieve the desired outcome, inluding key tips and guidelines.
Ensure clarity and practicality, allowing readers to easily follow and apply the instructions.
If applicable, add images captions between "<image> </image>" tags.  Add two or more image captions at the beginning, end or inside the text itself.
%(start_response)s""",
        "steps": [
            {
                "description": "Extract Main Idea",
                "input":"context_document",
                "output": "Main idea."
            },
            {
                "description": "Create Detailed Tutorial",
                "sub_steps": [
                    {
                        "description": "Write in-depth explanations for each step.",
                        "input": "Main idea",
                        "output": "Step-by-step explanations."
                    },
                    {
                        "description": "Include key tips and guidelines.",
                        "input": "Step-by-step explanations",
                        "output": "Detailed steps with tips."
                    },
                    {
                        "description": "Add images and combine all elements to create the tutorial.",
                        "input": "Detailed steps with tips",
                        "output": "Completed tutorial with image captions."
                    }
                ]
            }
        ]
    },
    "professional_science_article_with_images": {
        "description": "Create an extract of a scientific journal article tailored for professionals and researchers on the topic.", "input": "text snippet", "output": "scientific journal article extract",
        "prompt": """%(start_command)s Create an extract of a scientific journal article tailored for professionals and researchers on the topic. The module should relate to the following text snippet:
%(context_document)s
===
The style should mirror that of a scholarly publication, not school textbooks, aiming to engage a highly knowledgeable audience with very deep expertise. Try to:
- Present advanced theories, using technical and academic language.
- Include critical analysis of recent research findings and debates in the field, with a detailed examination of empirical data and statistical methodologies.
- The article should reflect the depth and complexity of content found in top-tier economics journals, intended for a readership deeply entrenched in the field.
- If applicable, add images captions between "<image> </image>" tags.  Add two or more image captions at the beginning, end or inside the text itself.
- Do not add come up with references or add them at the end of the article. If there are mathematical expressions use a correct LateX formatting.
%(start_response)s""",
        "steps": [
            {
                "description": "Extract Main Idea",
                "input":"context_document",
                "output": "Main idea."
            },
            {
                "description": "Develop Scientific Article",
                "sub_steps": [
                    {
                        "description": "Present advanced theories using technical and academic language.",
                        "input": "Main idea",
                        "output": "Advanced theories."
                    },
                    {
                        "description": "Include critical analysis of recent research findings and debates.",
                        "input": "Advanced theories",
                        "output": "Critical analysis."
                    },
                    {
                        "description": "Examine empirical data and statistical methodologies.",
                        "input": "Critical analysis",
                        "output": "Empirical data examination."
                    },
                    {
                        "description": "Add images and combine all elements to create the scientific article.",
                        "input": "Empirical data examination",
                        "output": "Completed scientific article with image captions."
                    }
                ]
            }
        ]
    },
    "college_textbook": {
        "description": "Write a comprehensive and in-depth textbook segment tailored for college students.", "input": "text snippet", "output": "textbook segment",
        "prompt": """%(start_command)s Write a comprehensive and in-depth textbook tailored for college students. The module should relate to the following text snippet:
%(context_document)s
===
Try to be:
- Rigorous: Ensure very detailed and in-depth coverage of the concepts.
- Engaging: Write with an academic and engaging tone that captivates interest.
- Applied: Use specific and practical examples. For example, if the topic is integration in calculus, include equations and proofs of the concept you're teaching. As another example, if the topic is the history of the United States, include dates, names, and key events.
- If there are mathematical expressions use a correct LateX formatting.
- If applicable, add images captions between "<image> </image>" tags.  Add two or more image captions at the beginning, end or inside the text itself.
- Avoid introductory phrases such as "welcome to this unit" at the beginning or conclusions the end.
%(start_response)s""",
        "steps": [
            {
                "description": "Extract Main Idea",
                "input":"context_document",
                "output": "Main idea."
            },
            {
                "description": "Write Comprehensive Textbook Unit",
                "sub_steps": [
                    {
                        "description": "Ensure detailed and in-depth coverage of the concepts.",
                        "input": "Main idea",
                        "output": "Detailed concepts."
                    },
                    {
                        "description": "Write in an academic and engaging tone.",
                        "input": "Detailed concepts",
                        "output": "Engaging narrative."
                    },
                    {
                        "description": "Use specific and practical examples.",
                        "input": "Engaging narrative",
                        "output": "Practical examples."
                    },
                    {
                        "description": "Add images and combine all elements to create the textbook unit.",
                        "input": "Practical examples",
                        "output": "Completed textbook unit with image captions."
                    }
                ]
            }
        ]
    },
    "course_unit_textbook": {
        "description": "Write a detailed course unit for a textbook.", "input": "text snippet", "output": "course unit",
        "prompt": """%(start_command)s Write a long and very detailed course unit for a textbook.
Write the new sub-unit while trying to be:
- Rigorous - you create challenging textbooks that cover the material in depth.
- Engaging - your textbooks have a narrative arc and engaging tone, like the writing of Michael Lewis.
- Applied - you use specific and practical examples. For example, if the topic is integration in calculus, include equations and proofs of the concept you're teaching. As another example, if the topic is the history of the United States, include dates, names, and key events.
- If applicable, add images captions between "<image> </image>" tags.  Add two or more image captions at the beginning, end or inside the text itself.
The course unit should relate to this text snippet:
%(context_document)s
%(start_response)s""",
        "steps": [
            {
                "description": "Extract Main Idea",
                "input":"context_document",
                "output": "Main idea."
            },
            {
                "description": "Write Detailed Course Unit",
                "sub_steps": [
                    {
                        "description": "Ensure detailed and in-depth coverage of the concepts.",
                        "input": "Main idea",
                        "output": "Detailed concepts."
                    },
                    {
                        "description": "Write in an academic and engaging tone.",
                        "input": "Detailed concepts",
                        "output": "Engaging narrative."
                    },
                    {
                        "description": "Use specific and practical examples.",
                        "input": "Engaging narrative",
                        "output": "Practical examples."
                    },
                    {
                        "description": "Combine all elements to create the course unit.",
                        "input": "Practical examples",
                        "output": "Completed course unit."
                    }
                ]
            }
        ]
    },
    "college_application_textbook": {
        "description": "Write an educational piece suited for college students.", "input": "text snippet", "output": "educational piece",
        "prompt": """%(start_command)s Write an educational piece suited for college students related to the following text snippet:
%(context_document)s
===
Do not just list concepts, but develop each one in detail before moving to the next, as we prioritize depth of understanding and comprehensive exploration of the subject matter over breadth. Focus on:
- Rigor: Ensure in-depth coverage of the concepts/sections.
- Engagement: Write with an academic, professional and engaging tone that captivates interest.
- Application: Incorporate specific, practical examples, such as proofs in calculus or critical dates and figures in history.
- If applicable, add images captions between "<image> </image>" tags.  Add two or more image captions at the beginning, end or inside the text itself.
- Do not include a title or an introduction, simply write the content without headlines and introductory phrases.
%(start_response)s""",
        "steps": [
            {
                "description": "Extract Main Idea",
                "input":"context_document",
                "output": "Main idea."
            },
            {
                "description": "Write Detailed Educational Piece",
                "sub_steps": [
                    {
                        "description": "Ensure in-depth coverage of the concepts.",
                        "input": "Main idea",
                        "output": "Detailed concepts."
                    },
                    {
                        "description": "Write in an academic and engaging tone.",
                        "input": "Detailed concepts",
                        "output": "Engaging narrative."
                    },
                    {
                        "description": "Use specific and practical examples.",
                        "input": "Engaging narrative",
                        "output": "Practical examples."
                    },
                    {
                        "description": "Combine all elements to create the educational piece.",
                        "input": "Practical examples",
                        "output": "Completed educational piece."
                    }
                ]
            }
        ]
    },
    "story": {
        "description": "Write a compelling story related to the given text snippet.", "input": "text snippet", "output": "story",
        "prompt": """%(start_command)s Write a compelling story related to the following text snippet:
%(context_document)s
===
The story doesn’t need to mention everything in the snippet, use it just for inspiration and be creative!
The story should incorporate the following elements:
- Dialogue: the story must feature at least one meaningful dialogue that reveals character depth, advances the plot, or unravels a crucial piece of the mystery
- Interesting themes: explore themes resonant with a mature audience, such as moral ambiguity, existential queries, personal transformation, or the consequences of past actions.
- If applicable, add images captions between "<image> </image>" tags.  Add two or more image captions at the beginning, end or inside the text itself.
Do not provide commentary and start the story right away. Do not start with classic sentences like "Once upon a time", "The sun hung low in the sky" or "In the dimly lit", be creative.
%(start_response)s""",
        "steps": [
            {
                "description": "Extract Main Idea",
                "input":"context_document",
                "output": "Main idea."
            },
            {
                "description": "Develop Compelling Story",
                "sub_steps": [
                    {
                        "description": "Write the story incorporating dialogue and interesting themes.",
                        "input": "Main idea",
                        "output": "Story with dialogue and themes."
                    },
                    {
                        "description": "Ensure the story is engaging and resonant with a mature audience.",
                        "input": "Story with dialogue and themes",
                        "output": "Completed compelling story."
                    }
                ]
            }
        ]
    },
    "forum_post_story": {
        "description": "Write a story in the style of real-life situations that people share in forums.", "input": "text snippet", "output": "forum post-style story",
        "prompt": """%(start_command)s Write a story in the style of real-life situations that people share in forums. The story should be somehow related to this text snippet:
%(context_document)s
===
The story needs to include a compelling and unexpected plot twist. Your narrative should resonate with the authenticity and personal touch found in forum discussions. Include relatable events and emotional depth. If applicable, add images captions between "<image> </image>" tags.  Add two or more image captions at the beginning, end or inside the text itself.
Do not provide commentary and start the story right away. Do not start with classic sentences like "Once upon a time", "A few years back" or "A few montsh ago", be creative.
%(start_response)s""",
        "steps": [
            {
                "description": "Extract Main Idea",
                "input":"context_document",
                "output": "Main idea."
            },
            {
                "description": "Write Real-Life Story",
                "sub_steps": [
                    {
                        "description": "Develop a relatable narrative with emotional depth.",
                        "input": "Main idea",
                        "output": "Relatable story."
                    },
                    {
                        "description": "Include a compelling and unexpected plot twist.",
                        "input": "Relatable story",
                        "output": "Story with plot twist."
                    },
                    {
                        "description": "Add images and ensure the narrative resonates with forum discussions.",
                        "input": "Story with plot twist",
                        "output": "Completed forum post story with image captions."
                    }
                ]
            }
        ]
    },
     "creative_story_from_analysis": {
        "description": "Write a story that explores a situation slightly related to the given text snippet.", "input": "text snippet", "output": "creative story",
        "prompt": """%(start_command)s Below is an analysis of a document. Please draft a story based on this analysis:
== Anlysis ==
%(context_document)s
===
Draft a long fictional story for an adult audience incorporating the relationships, themes, motives, emotions, consequences and issues within the above context analysis.
The story should unfold through character interactions and decisions. Place the entity labels next to the specific entities, dates and numbers in the story - e.g., 'Jane Smith ({PERSON_2})', 'American ({TRAIT_5})', 'United States ({REGION_1})', etc.  Match the entity labels to any entity labels in the analysis.
Don't expand on the below analysis, just write the story. Do not provide commentary and start the story right away. Do not start with classic sentences like "Once upon a time", be creative.

%(start_response)s
== STORY ==
""",
        "steps": [
            {
                "description": "Extract relationships, themes, motives, emotions, consequences and issues ",
                "input":"context_document",
                "output": "relationships, themes, motives, emotions, consequences and issues ."
            },
            {
                "description": "Develop Creative Story",
                "sub_steps": [
                    {
                        "description": "Write a story unfolding through character interactions and decisions.",
                        "input": "relationships, themes, motives, emotions, consequences and issues ",
                        "output": "Story with interactions."
                    },
                    {
                        "description": "Place entity labels next to specific names, matched to any .",
                        "input": "Story with specific names of entities, dates and numbers",
                        "output": "Entity labels matched to labels in the input analysis"
                    },
                ]
            }
        ]
    },

    "creative_story": {
        "description": "Write a story that explores a situation slightly related to the given text snippet.", "input": "text snippet", "output": "creative story",
        "prompt": """%(start_command)s Write a story that explores a situation slightly related to this text snippet:
%(context_document)s
===
The story should unfold through the characters interactions, decisions, and the consequences of their actions. Aim to weave in common sense lessons and social cues. The narrative should cater to a diverse age group, including at least one dialogue and presenting both positive and negative outcomes.
Do not provide commentary and start the story right away. Do not start with classic sentences like "Once upon a time", be creative.
%(start_response)s""",
        "steps": [
            {
                "description": "Extract Main Idea",
                "input":"context_document",
                "output": "Main idea."
            },
            {
                "description": "Develop Creative Story",
                "sub_steps": [
                    {
                        "description": "Write a story unfolding through character interactions and decisions.",
                        "input": "Main idea",
                        "output": "Story with interactions."
                    },
                    {
                        "description": "Include common sense lessons and social cues.",
                        "input": "Story with interactions",
                        "output": "Story with lessons."
                    },
                    {
                        "description": "Present both positive and negative outcomes.",
                        "input": "Story with lessons",
                        "output": "Completed creative story."
                    }
                ]
            }
        ]
    },
    "social_media_post": {
        "description": "Write a real-life story shared by someone in a social media forum.", "input": "text snippet", "output": "social media post",
        "prompt": """%(start_command)s Write a real-life story shared by someone in a social media forum. The story should be somehow related to this text snippet:
%(context_document)s
===
The story should include:
- Niche interests or humor: dive into specific hobbies, interests, or humorous situations
- An unexpected plot twist or engaging conflict: introduce a relatable yet challenging situation or dilemma that the author faced.
- Reflection and insight: end with a resolution that offers a new understanding, a sense of community, or a personal revelation, much like the conclusions drawn in forum discussions.
Start the story right away. Do not start with sentences like  "Once upon a time" as this is a reddit post and not a novel, you should also avoid starting with classic sentences like "A few years ago" or "A few years back", be creative.
%(start_response)s""",
        "steps": [
            {
                "description": "Extract Main Idea",
                "input":"context_document",
                "output": "Main idea."
            },
            {
                "description": "Write Social Media Story",
                "sub_steps": [
                    {
                        "description": "Include niche interests or humor.",
                        "input": "Main idea",
                        "output": "Story with niche interests."
                    },
                    {
                        "description": "Introduce an unexpected plot twist or engaging conflict.",
                        "input": "Story with niche interests",
                        "output": "Story with plot twist."
                    },
                    {
                        "description": "End with reflection and insight.",
                        "input": "Story with plot twist",
                        "output": "Completed social media post."
                    }
                ]
            }
        ]
    },
   "policy_evaluation": {
        "description": "Evaluate a policy based on the given context document, with detailed explanation and reflection steps.",
        "input": "context document, evaluation purpose",
        "output": "policy evaluation",
        "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
### Evaluation Purpose: %(purpose)s
===
- Evaluate the policy described or suggested in the above context document. If no evaluation purpose is provided, please infer the evaluation purpose from the context document.
- Provide a detailed explanation of the policy's strengths, weaknesses, and potential impacts.
- Add arguments and counter arguments.
- Reflect on and refine the evaluation to ensure it meets the purpose.
%(start_response)s
Policy Evaluation:""",
        "steps": [
            {
                "description": "Identify Policy Elements",
                "input": "context_document",
                "output": "List of policy elements."
            },
            {
                "description": "Infer Evaluation Purpose if not provided",
                "branching_conditions": [
                    {
                        "condition": "If evaluation purpose is not provided",
                        "sub_steps": [
                            {
                                "description": "Infer evaluation purpose",
                                "input": "context_document",
                                "output": "Inferred evaluation purpose"
                            }
                        ]
                    },
                    {
                        "condition": "If evaluation purpose is provided",
                        "output": "Use the provided evaluation purpose"
                    }
                ]
            },
            {
                "description": "Evaluate Policy",
                "input": "List of policy elements",
                "output": "Detailed evaluation of policy strengths, weaknesses, and impacts."
            },
            {
                "description": "Create Arguments and Counter-arguments",
                "input": "List of policy elements",
                "output": "Detailed arguments for and against the policies."
            },
            {
                "description": "Reflect on Evaluation",
                "input": "Policy evaluation",
                "output": "Reflection notes."
            },
            {
                "description": "Refine Evaluation",
                "input": "Reflection notes",
                "output": "Refined policy evaluation."
            }
        ]
    },
    "case_study": {
        "description": "Create a case study based on the given context document, with detailed analysis and reflection steps.",
        "input": "context document, case study purpose",
        "output": "case study",
        "prompt": """%(start_command)s Create a case study based on the text.
### Context Document:
%(context_document)s
### Case Study Purpose: %(purpose)s
===
- If no case study purpose is provided, please infer the case study purpose from the context document.
- Ensure the case study matches the stype of the case study purpose (e.g., business case study, public health case study, etc.).
- Ensure the case study is detailed and includes an analysis of the situation, challenges, and solutions.
- Reflect on and refine the case study to ensure it meets the purpose.
%(start_response)s
Case Study:""",
        "steps": [
            {
                "description": "Identify Key Aspects",
                "input": "context_document",
                "output": "Key aspects of the situation."
            },
            {
                "description": "Infer Case Study Purpose if not provided",
                "branching_conditions": [
                    {
                        "condition": "If case study purpose is not provided",
                        "sub_steps": [
                            {
                                "description": "Infer case study purpose",
                                "input": "context_document",
                                "output": "Inferred case study purpose"
                            }
                        ]
                    },
                    {
                        "condition": "If case study purpose is provided",
                        "output": "Use the provided case study purpose"
                    }
                ]
            },
            {
                "description": "Create Case Study",
                "input": "Key aspects",
                "output": "Detailed case study with analysis."
            },
            {
                "description": "Reflect on Case Study",
                "input": "Case study",
                "output": "Reflection notes."
            },
            {
                "description": "Refine Case Study",
                "input": "Reflection notes",
                "output": "Refined case study."
            }
        ]
    },
     "user_story": {
        "description": "Write a user story based on the given context document, with detailed steps and reflection.",
        "input": "context document, story purpose",
        "output": "user story",
        "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
### Story Purpose: %(purpose)s
===
Create a user story based on the context above. If no story purpose is provided, please infer the story purpose from the context document. Ensure the story includes user roles, goals, and acceptance criteria. Reflect on and refine the user story to ensure it meets the purpose.
%(start_response)s
User Story:""",
        "steps": [
            {
                "description": "Identify User Roles and Goals",
                "input": "context_document",
                "output": "User roles and goals."
            },
            {
                "description": "Infer Story Purpose if not provided",
                "branching_conditions": [
                    {
                        "condition": "If story purpose is not provided",
                        "sub_steps": [
                            {
                                "description": "Infer story purpose",
                                "input": "context_document",
                                "output": "Inferred story purpose"
                            }
                        ]
                    },
                    {
                        "condition": "If story purpose is provided",
                        "output": "Use the provided story purpose"
                    }
                ]
            },
            {
                "description": "Develop Acceptance Criteria",
                "input": "User roles and goals",
                "output": "Acceptance criteria."
            },
            {
                "description": "Reflect on User Story",
                "input": "User story",
                "output": "Reflection notes."
            },
            {
                "description": "Refine User Story",
                "input": "Reflection notes",
                "output": "Refined user story."
            }
        ]
    },
    "meeting_agenda": {
        "description": "Create a meeting agenda based on the given context document, with detailed steps and reflection.",
        "input": "context document, agenda purpose",
        "output": "meeting agenda",
        "prompt": """%(start_command)s Create a meeting agenda based on the text.
### Context Document:
%(context_document)s
### Agenda Purpose: %(purpose)s
===
If no agenda purpose is provided, please infer the agenda purpose from the context document. Include topics to be discussed, objectives, and time allocations. Reflect on and refine the meeting agenda to ensure it meets the purpose.
%(start_response)s
Meeting Agenda:""",
        "steps": [
            {
                "description": "Identify Topics and Objectives",
                "input": "context_document",
                "output": "List of topics and objectives."
            },
            {
                "description": "Infer Agenda Purpose if not provided",
                "branching_conditions": [
                    {
                        "condition": "If agenda purpose is not provided",
                        "sub_steps": [
                            {
                                "description": "Infer agenda purpose",
                                "input": "context_document",
                                "output": "Inferred agenda purpose"
                            }
                        ]
                    },
                    {
                        "condition": "If agenda purpose is provided",
                        "output": "Use the provided agenda purpose"
                    }
                ]
            },
            {
                "description": "Allocate Time for Each Topic",
                "input": "List of topics and objectives",
                "output": "Time allocations."
            },
            {
                "description": "Reflect on Agenda",
                "input": "Meeting agenda",
                "output": "Reflection notes."
            },
            {
                "description": "Refine Agenda",
                "input": "Reflection notes",
                "output": "Refined meeting agenda."
            }
        ]
    },
    "change_management_plan": {
        "description": "Create a change management plan based on the given context document, with detailed steps and reflection.",
        "input": "context document, change purpose",
        "output": "change management plan",
        "prompt": """%(start_command)s Develop a change management plan based on the text.
### Context Document:
%(context_document)s
### Change Purpose: %(purpose)s
===
If no change purpose is provided, please infer the change purpose from the context document. Include the rationale for change, impact assessment, and strategies for implementation. Reflect on and refine the change management plan to ensure it meets the purpose.
%(start_response)s
Change Management Plan:""",
        "steps": [
            {
                "description": "Identify Rationale for Change",
                "input": "context_document",
                "output": "Rationale for change."
            },
            {
                "description": "Infer Change Purpose if not provided",
                "branching_conditions": [
                    {
                        "condition": "If change purpose is not provided",
                        "sub_steps": [
                            {
                                "description": "Infer change purpose",
                                "input": "context_document",
                                "output": "Inferred change purpose"
                            }
                        ]
                    },
                    {
                        "condition": "If change purpose is provided",
                        "output": "Use the provided change purpose"
                    }
                ]
            },
            {
                "description": "Conduct Impact Assessment",
                "input": "Rationale for change",
                "output": "Impact assessment."
            },
            {
                "description": "Develop Implementation Strategies",
                "input": "Impact assessment",
                "output": "Implementation strategies."
            },
            {
                "description": "Reflect on Plan",
                "input": "Change management plan",
                "output": "Reflection notes."
            },
            {
                "description": "Refine Plan",
                "input": "Reflection notes",
                "output": "Refined change management plan."
            }
        ]
    },
    "customer_journey_map": {
        "description": "Create a customer journey map based on the given context document, with detailed steps and reflection.",
        "input": "context document, journey purpose",
        "output": "customer journey map",
        "prompt": """%(start_command)s Create a customer journey map based on the text.
### Context Document:
%(context_document)s
### Journey Purpose: %(purpose)s
===
If no journey purpose is provided, please infer the journey purpose from the context document. Include stages of the customer journey, touchpoints, and customer experiences. Reflect on and refine the journey map to ensure it meets the purpose.
%(start_response)s
Customer Journey Map:""",
        "steps": [
            {
                "description": "Identify Stages of the Journey",
                "input": "context_document",
                "output": "Stages of the customer journey."
            },
            {
                "description": "Infer Journey Purpose if not provided",
                "branching_conditions": [
                    {
                        "condition": "If journey purpose is not provided",
                        "sub_steps": [
                            {
                                "description": "Infer journey purpose",
                                "input": "context_document",
                                "output": "Inferred journey purpose"
                            }
                        ]
                    },
                    {
                        "condition": "If journey purpose is provided",
                        "output": "Use the provided journey purpose"
                    }
                ]
            },
            {
                "description": "Identify Touchpoints",
                "input": "Stages of the customer journey",
                "output": "Touchpoints in the journey."
            },
            {
                "description": "Describe Customer Experiences",
                "input": "Touchpoints in the journey",
                "output": "Customer experiences at each touchpoint."
            },
            {
                "description": "Reflect on Journey Map",
                "input": "Customer journey map",
                "output": "Reflection notes."
            },
            {
                "description": "Refine Journey Map",
                "input": "Reflection notes",
                "output": "Refined customer journey map."
            }
        ]
    },
    "competitive_analysis": {
        "description": "Perform a competitive analysis based on the given context document, with detailed steps and reflection.",
        "input": "context document, analysis purpose",
        "output": "competitive analysis",
        "prompt": """%(start_command)s Conduct a competitive analysis based on the text.
### Context Document:
%(context_document)s
### Analysis Purpose: %(purpose)s
===
If no analysis purpose is provided, please infer the analysis purpose from the context document. Include an overview of competitors, their strengths and weaknesses, and strategic insights. Reflect on and refine the competitive analysis to ensure it meets the purpose.
%(start_response)s
Competitive Analysis:""",
        "steps": [
            {
                "description": "Identify Competitors",
                "input": "context_document",
                "output": "List of competitors."
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
                "description": "Assess Strengths and Weaknesses",
                "input": "List of competitors",
                "output": "Competitors' strengths and weaknesses."
            },
            {
                "description": "Provide Strategic Insights",
                "input": "Competitors' strengths and weaknesses",
                "output": "Strategic insights."
            },
            {
                "description": "Reflect on Analysis",
                "input": "Competitive analysis",
                "output": "Reflection notes."
            },
            {
                "description": "Refine Analysis",
                "input": "Reflection notes",
                "output": "Refined competitive analysis."
            }
        ]
    },
    "investment_proposal": {
        "description": "Write an investment proposal based on the given context document, with detailed steps and reflection.",
        "input": "context document, proposal purpose",
        "output": "investment proposal",
        "prompt": """%(start_command)s Create an investment proposal based on the below text.
### Context Document:
%(context_document)s
### Proposal Purpose: %(purpose)s
===
If no proposal purpose is provided, please infer the proposal purpose from the context document. Include an overview of the opportunity, financial projections, and potential returns.
Always state that the response is for educational or information purposes only, and that the user should consult with a professional advisor.
Reflect on and refine the investment proposal to ensure it meets the purpose.
%(start_response)s
Investment Proposal:""",
        "steps": [
            {
                "description": "Identify Investment Opportunity",
                "input": "context_document",
                "output": "Overview of the investment opportunity."
            },
            {
                "description": "Infer Proposal Purpose if not provided",
                "branching_conditions": [
                    {
                        "condition": "If proposal purpose is not provided",
                        "sub_steps": [
                            {
                                "description": "Infer proposal purpose",
                                "input": "context_document",
                                "output": "Inferred proposal purpose"
                            }
                        ]
                    },
                    {
                        "condition": "If proposal purpose is provided",
                        "output": "Use the provided proposal purpose"
                    }
                ]
            },
            {
                "description": "Develop Financial Projections",
                "input": "Overview of the investment opportunity",
                "output": "Financial projections."
            },
            {
                "description": "Assess Potential Returns",
                "input": "Financial projections",
                "output": "Potential returns."
            },
            {
                "description": "Reflect on Proposal",
                "input": "Investment proposal",
                "output": "Reflection notes."
            },
            {
                "description": "Refine Proposal",
                "input": "Reflection notes",
                "output": "Refined investment proposal."
            }
        ]
    },
    "financial_report": {
        "description": "Generate a financial report based on the given context document, with detailed steps and reflection.",
        "input": "context document, report purpose",
        "output": "financial report",
        "prompt": """%(start_command)s Create a financial report based on the below text.
### Context Document:
%(context_document)s
### Report Purpose: %(purpose)s
===
If no report purpose is provided, please infer the report purpose from the context document. Include an analysis of financial statements, key metrics, and overall financial health. Reflect on and refine the financial report to ensure it meets the purpose.
%(start_response)s
Financial Report:""",
        "steps": [
            {
                "description": "Analyze Financial Statements",
                "input": "context_document",
                "output": "Analysis of financial statements."
            },
            {
                "description": "Infer Report Purpose if not provided",
                "branching_conditions": [
                    {
                        "condition": "If report purpose is not provided",
                        "sub_steps": [
                            {
                                "description": "Infer report purpose",
                                "input": "context_document",
                                "output": "Inferred report purpose"
                            }
                        ]
                    },
                    {
                        "condition": "If report purpose is provided",
                        "output": "Use the provided report purpose"
                    }
                ]
            },
            {
                "description": "Identify Key Metrics",
                "input": "Analysis of financial statements",
                "output": "Key financial metrics."
            },
            {
                "description": "Assess Financial Health",
                "input": "Key financial metrics",
                "output": "Overall financial health assessment."
            },
            {
                "description": "Reflect on Report",
                "input": "Financial report",
                "output": "Reflection notes."
            },
            {
                "description": "Refine Report",
                "input": "Reflection notes",
                "output": "Refined financial report."
            }
        ]
    },

  "experiment_design": {
        "description": "Design an experiment based on the given context document, with detailed steps and reflection.",
        "input": "context document, experiment purpose",
        "output": "experiment design",
        "prompt": """%(start_command)s Design an experiment based on the below text.
### Context Document:
%(context_document)s
### Experiment Purpose: %(purpose)s
===
If no experiment purpose is provided, please infer the experiment purpose from the context document. Include hypotheses, variables, methods, and procedures. Reflect on and refine the experiment design to ensure it meets the purpose.
%(start_response)s
Experiment Design:""",
        "steps": [
            {
                "description": "Identify Hypotheses and Variables",
                "input": "context_document",
                "output": "List of hypotheses and variables."
            },
            {
                "description": "Infer Experiment Purpose if not provided",
                "branching_conditions": [
                    {
                        "condition": "If experiment purpose is not provided",
                        "sub_steps": [
                            {
                                "description": "Infer experiment purpose",
                                "input": "context_document",
                                "output": "Inferred experiment purpose"
                            }
                        ]
                    },
                    {
                        "condition": "If experiment purpose is provided",
                        "output": "Use the provided experiment purpose"
                    }
                ]
            },
            {
                "description": "Develop Methods and Procedures",
                "input": "List of hypotheses and variables",
                "output": "Detailed methods and procedures."
            },
            {
                "description": "Reflect on Design",
                "input": "Experiment design",
                "output": "Reflection notes."
            },
            {
                "description": "Refine Design",
                "input": "Reflection notes",
                "output": "Refined experiment design."
            }
        ]
    },
    "swot_analysis": {
        "description": "Perform a SWOT analysis based on the given context document, with detailed explanation and reflection steps.",
        "input": "context document, swot purpose",
        "output": "SWOT analysis",
        "prompt": """%(start_command)s Conduct a SWOT analysis based on the below text.
### Context Document:
%(context_document)s
### SWOT Purpose: %(purpose)s
===
If no swot purpose is provided, please infer the swot purpose from the context document. Identify the strengths, weaknesses, opportunities, and threats related to the context. Reflect on and refine the SWOT analysis to ensure it meets the purpose.
%(start_response)s
SWOT Analysis:""",
        "steps": [
            {
                "description": "Identify Strengths and Weaknesses",
                "input": "context_document",
                "output": "List of strengths and weaknesses."
            },
            {
                "description": "Infer SWOT Purpose if not provided",
                "branching_conditions": [
                    {
                        "condition": "If swot purpose is not provided",
                        "sub_steps": [
                            {
                                "description": "Infer swot purpose",
                                "input": "context_document",
                                "output": "Inferred swot purpose"
                            }
                        ]
                    },
                    {
                        "condition": "If swot purpose is provided",
                        "output": "Use the provided swot purpose"
                    }
                ]
            },
            {
                "description": "Identify Opportunities and Threats",
                "input": "List of strengths and weaknesses",
                "output": "List of opportunities and threats."
            },
            {
                "description": "Reflect on SWOT",
                "input": "SWOT analysis",
                "output": "Reflection notes."
            },
            {
                "description": "Refine SWOT",
                "input": "Reflection notes",
                "output": "Refined SWOT analysis."
            }
        ]
    },
    "case_analysis": {
        "description": "Analyze a case based on the given context document, with detailed explanation and reflection steps.",
        "input": "context document, analysis purpose",
        "output": "case analysis",
        "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
### Analysis Purpose: %(purpose)s
===
Analyze the case described or suggested in the text. If no analysis purpose is provided, please infer the analysis purpose from the context document. Provide a detailed examination of the situation, key issues, and potential solutions. Reflect on and refine the case analysis to ensure it meets the purpose.
%(start_response)s
Case Analysis:""",
        "steps": [
            {
                "description": "Identify Key Issues",
                "input": "context_document",
                "output": "Key issues and context."
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
                "input": "Key issues and context",
                "output": "Detailed analysis of issues."
            },
            {
                "description": "Propose Solutions",
                "input": "Detailed analysis of issues",
                "output": "Potential solutions."
            },
            {
                "description": "Reflect on Analysis",
                "input": "Case analysis",
                "output": "Reflection notes."
            },
            {
                "description": "Refine Analysis",
                "input": "Reflection notes",
                "output": "Refined case analysis."
            }
        ]
    },
    "implementation_plan": {
        "description": "Create an implementation plan based on the given context document, with detailed steps and reflection.",
        "input": "context document, plan purpose",
        "output": "implementation plan",
        "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
### Plan Purpose: %(purpose)s
===
Develop an implementation plan based on the context above. If no plan purpose is provided, please infer the plan purpose from the context document. Include objectives, strategies, timelines, and resources needed. Reflect on and refine the implementation plan to ensure it meets the purpose.
%(start_response)s
Implementation Plan:""",
        "steps": [
            {
                "description": "Identify Objectives",
                "input": "context_document",
                "output": "List of objectives."
            },
            {
                "description": "Infer Plan Purpose if not provided",
                "branching_conditions": [
                    {
                        "condition": "If plan purpose is not provided",
                        "sub_steps": [
                            {
                                "description": "Infer plan purpose",
                                "input": "context_document",
                                "output": "Inferred plan purpose"
                            }
                        ]
                    },
                    {
                        "condition": "If plan purpose is provided",
                        "output": "Use the provided plan purpose"
                    }
                ]
            },
            {
                "description": "Develop Strategies",
                "input": "List of objectives",
                "output": "Detailed strategies."
            },
            {
                "description": "Outline Timelines and Resources",
                "input": "Detailed strategies",
                "output": "Timelines and resources needed."
            },
            {
                "description": "Reflect on Plan",
                "input": "Implementation plan",
                "output": "Reflection notes."
            },
            {
                "description": "Refine Plan",
                "input": "Reflection notes",
                "output": "Refined implementation plan."
            }
        ]
    },
  "implementation_plan": {
        "description": "Create an implementation plan based on the given context document, with detailed steps and reflection.",
        "input": "context document, plan purpose",
        "output": "implementation plan",
        "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
### Plan Purpose: %(purpose)s
===
Develop an implementation plan based on the context above. If no plan purpose is provided, please infer the plan purpose from the context document. Include objectives, strategies, timelines, and resources needed. Reflect on and refine the implementation plan to ensure it meets the purpose.
%(start_response)s
Implementation Plan:""",
        "steps": [
            {
                "description": "Identify Objectives",
                "input": "context_document",
                "output": "List of objectives."
            },
            {
                "description": "Infer Plan Purpose if not provided",
                "branching_conditions": [
                    {
                        "condition": "If plan purpose is not provided",
                        "sub_steps": [
                            {
                                "description": "Infer plan purpose",
                                "input": "context_document",
                                "output": "Inferred plan purpose"
                            }
                        ]
                    },
                    {
                        "condition": "If plan purpose is provided",
                        "output": "Use the provided plan purpose"
                    }
                ]
            },
            {
                "description": "Develop Strategies",
                "input": "List of objectives",
                "output": "Detailed strategies."
            },
            {
                "description": "Outline Timelines and Resources",
                "input": "Detailed strategies",
                "output": "Timelines and resources needed."
            },
            {
                "description": "Reflect on Plan",
                "input": "Implementation plan",
                "output": "Reflection notes."
            },
            {
                "description": "Refine Plan",
                "input": "Reflection notes",
                "output": "Refined implementation plan."
            }
        ]
    },
  "lesson_plan": {
        "description": "Create an lesson plan based on test questions, with detailed steps and information extraction.",
        "input": "context document, test questions",
        "output": "lesson plan",
        "prompt": """%(start_command)s I. Goal:
Below is a context document and test questions. First, start by classifying the theme, subject matter and audience covered below in the test questions. All text is truthful and supported by evidence, and promotes science, and learning. The text will not contain anything sexist, racist, bias or offensive, promote violence or hatred.
If not test questions are defined, infer the test questions form the context document:
### Context Document:
%(context_document)s
### Test Questions:
%(questions)s
### Lesson purpose:
%(purpose)s
II. Requested Lesson Plan:
Then write a detailed lesson book appropriate for a student based on the subject matter such that the student can answer questions like the test questions or any other questions in the field.
If a lesson purpose is not provided, infered it from the context and/or the test questions.
The lesson plan should show step by step reasoning such that the student can completely and thoughtfully understand the subject matter and infer new knowledge according to the purpose.

A. Requirements:
Do not repeat the test questions or answer the test questions in your lesson plan, but do add Examples and Exercises about the subject matter but that does not include information from the test questions. Add a high amount of details, along with explanations, and counter examples. Answers in excercises should be highly detailed and thoughtful.

B. Formatting:
Format your response and make sure it is grammatical, with headings and subheadings. Make sure you use proper punctuations.

***
Theme and Subject Matter, Audience, Lesson Plan with Examples, and Exercises:
%(start_Response)s
A. Theme""",
        "steps": [
            {
                "description": "Identify Theme, Subject Mattem Audience, Purpose",
                "input": "context_document",
                "output": "List the theme, subject matter and audience and purpose if one is not provided."
            },
            {
                "description": "Infer Test Questions if not provided",
                "branching_conditions": [
                    {
                        "condition": "If test questions are not provided",
                        "sub_steps": [
                            {
                                "description": "Infer test questions",
                                "input": "context_document",
                                "output": "quesitions"
                            }
                        ]
                    },
                    {
                        "condition": "If test questions is provided",
                        "output": "Use the test questions"
                    }
                ]
            },
            {
                "description": "Develop Lesson Plan",
                "input": "List of objectives",
                "output": "Lesson with examples, questions and answers."
            },
        ]
    },

}
