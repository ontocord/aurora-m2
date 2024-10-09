#@title safety prompts - EU AI Act
#Run high-risk documents through these prompts. Run this after the pre-process prompts.
step_5_safety_prompts = {
  "EU_AI_Act_1": {
    "description": "Analyze context document for AI safety training, focusing on specific high-risk areas.",
    "input": "context document",
    "output": "detailed analysis of high-risk and revised document",
    "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
===
Provide the following information for the context document that would be useful for training a safe AI based on the context document. First, answer with Yes or No only:
Answer with respect to the Context:

1. Involves biometric data or technology (e.g., can include technology or processes), Yes or No:
2. Related to critical infrastructure (e.g., digital infrastructure, road traffic, waterway, gas, heating, or electricity), Yes or No:
3. Has information for determining education access (e.g., access or admission to education and vocational training and which levels of education), Yes or No:
4. Describes methods for testing students, Yes or No:


If you answered Yes to any of the above, give detailed summary and analysis of the context document based on the subject matter you answered "Yes" to, and re-write the context document to remove any aspects you answered "Yes" tdoc.
Use this format:
# Yes/No Answers
1. ...
2. ...
...
# Summary & Analysis (If any Yes answers)
...
# Rewritten Context Document Removing Problematic Text (If any Yes answers)
...
%(start_response)s
# Yes/No Answers
1. Involves Biometric Data or Technology:""",
    "steps": [
        {
            "description": "Analyze Document for High-Risk Areas",
            "input": "context_document",
            "output": "Yes/No answers to high-risk areas."
        },
        {
            "description": "Detailed Analysis Of Risks",
            "input": "High-Risk Areas detected in document",
            "output": "Detailed analysis based on high-risk areas identified."
        },
        {
            "description": "Revise context document to remove any High-Risk Content if any",
            "input": "context_document and detailed analysis based on high-risk areas identified.",
            "output": "Revised document with removed High-Risk Content, if any."
        }

    ]
  },
  "EU_AI_Act_2": {
    "description": "Analyze context document for AI safety training, focusing on specific high-risk areas.",
    "input": "context document",
    "output": "detailed analysis of high-risk and revised document",
    "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
===
Provide the following information for the context document that would be useful for training a safe AI based on the context document. First, answer with Yes or No only:
Answer with respect to the Context:

1. Ways to check if students are properly taking tests (e.g., detecting prohibited behavior of students during tests), Yes or No:
2. Related to recruitment, hiring, firing, or promotion (e.g., includes employee monitoring), Yes or No:
3. Related to public assistance benefits and services (e.g., evaluating eligibility, granting, reducing, revoking, or reclaiming benefits or services), Yes or No:
4. Includes information used to evaluate creditworthiness or establish credit scores (excluding systems used for detecting financial fraud), Yes or No:

If you answered Yes to any of the above, give detailed summary and analysis of the context document based on the subject matter you answered "Yes" to, and re-write the context document to remove any aspects you answered "Yes" tdoc.
Use this format:
# Yes/No Answers
1. ...
2. ...
...
# Summary & Analysis (If any Yes answers)
...
# Rewritten Context Document Removing Problematic Text (If any Yes answers)
...
%(start_response)s
# Yes/No Answers
1. Methods for checking if students are properly taking tests:""",
    "steps": [
        {
            "description": "Analyze Document for High-Risk Areas",
            "input": "context_document",
            "output": "Yes/No answers to high-risk areas."
        },
        {
            "description": "Detailed Analysis Of Risks",
            "input": "High-Risk Areas detected in document",
            "output": "Detailed analysis based on high-risk areas identified."
        },
        {
            "description": "Revise context document to remove any High-Risk Content if any",
            "input": "context_document and detailed analysis based on high-risk areas identified.",
            "output": "Revised document with removed High-Risk Content, if any."
        }

    ]
  },
  "EU_AI_Act_3": {
    "description": "Analyze context document for AI safety training, focusing on specific high-risk areas.",
    "input": "context document",
    "output": "detailed analysis of high-risk and revised document",
    "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
===
Provide the following information for the context document that would be useful for training a safe AI based on the context document. First, answer with Yes or No only:
Answer with respect to the Context:

1. Contains data or methods for risk assessment for life insurance and health insurance, or pricing of the insurance, Yes or No:
2. Involves the evaluation or classification of emergency calls (e.g., prioritizing emergency first response services for a person based on their location or characteristics, including police, firefighters, and medical aid), Yes or No:
3. Describes methods for emergency healthcare patient triage, Yes or No:
4. Related to assessing the risk of a person becoming the victim of criminal offenses, Yes or No:

If you answered Yes to any of the above, give detailed summary and analysis of the context document based on the subject matter you answered "Yes" to, and re-write the context document to remove any aspects you answered "Yes" tdoc.
Use this format:
# Yes/No Answers
1. ...
2. ...
...
# Summary & Analysis (If any Yes answers)
...
# Rewritten Context Document Removing Problematic Text (If any Yes answers)
...
%(start_response)s
# Yes/No Answers
1. Involves information for risk assessment and pricing in life and health insurance:""",
    "steps": [
        {
            "description": "Analyze Document for High-Risk Areas",
            "input": "context_document",
            "output": "Yes/No answers to high-risk areas."
        },
        {
            "description": "Detailed Analysis Of Risks",
            "input": "High-Risk Areas detected in document",
            "output": "Detailed analysis based on high-risk areas identified."
        },
        {
            "description": "Revise context document to remove any High-Risk Content if any",
            "input": "context_document and detailed analysis based on high-risk areas identified.",
            "output": "Revised document with removed High-Risk Content, if any."
        }

    ]
  },
  "EU_AI_Act_4": {
    "description": "Analyze context document for AI safety training, focusing on specific high-risk areas.",
    "input": "context document",
    "output": "detailed analysis of high-risk and revised document",
    "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
===
Provide the following information for the context document that would be useful for training a safe AI based on the context document. First, answer with Yes or No only:
Answer with respect to the Context:

1. Contains information used by law enforcement as trust, truth or polygraphs detections or similar tools, Yes or No:
2. Includes methods for evaluating the reliability of evidence during the investigation or prosecution of criminal offenses, Yes or No:
3. Contains data or processes used to assess the risk of a natural person offending or re-offending, without relying solely on profiling, Yes or No:
4. Involves profiling natural persons during the detection, investigation, or prosecution of criminal offenses, Yes or No:
5. Involves AI systems designed to influence the outcome of an election or referendum, or the voting behavior of individuals in elections or referenda, Yes or No:

If you answered Yes to any of the above, give detailed summary and analysis of the context document based on the subject matter you answered "Yes" to, and re-write the context document to remove any aspects you answered "Yes" tdoc.
Use this format:
# Yes/No Answers
1. ...
2. ...
...
# Summary & Analysis (If any Yes answers)
...
# Rewritten Context Document Removing Problematic Text (If any Yes answers)
...
%(start_response)s
# Yes/No Answers
1. Involves information used by law enforcement to test trust and truth of suspects:""",
    "steps": [
        {
            "description": "Analyze Document for High-Risk Areas",
            "input": "context_document",
            "output": "Yes/No answers to high-risk areas."
        },
        {
            "description": "Detailed Analysis Of Risks",
            "input": "High-Risk Areas detected in document",
            "output": "Detailed analysis based on high-risk areas identified."
        },
        {
            "description": "Revise context document to remove any High-Risk Content if any",
            "input": "context_document and detailed analysis based on high-risk areas identified.",
            "output": "Revised document with removed High-Risk Content, if any."
        }

    ]
  },
  "EU_AI_Act_5": {
    "description": "Analyze context document for AI safety training, focusing on specific high-risk areas.",
    "input": "context document",
    "output": "detailed analysis of high-risk and revised document",
    "prompt": """%(start_command)s  ### Context Document:
%(context_document)s
===
Provide the following information for the context document that would be useful for training a safe AI based on the context document. First, answer with Yes or No only:
Answer with respect to the Context:

1. Contains information used by public authorities or governmental bodies as truth, trust or polygraph detections or similar tools in the context of migration, asylum, or border control, Yes or No:
2. Includes data or methods for assessing risks (e.g., security risks, risks of irregular migration, or health risks) posed by individuals entering or within a Member State, Yes or No:
3. Related to the examination of applications for asylum, visas, or residence permits, including assessing the reliability of evidence, Yes or No:
4. Involves detecting, recognizing, or identifying natural persons in the context of migration, asylum, or border control management (excluding travel document verification), Yes or No:
5. Contains data or methods used by judicial authorities to assist in researching and interpreting facts and laws, or applying the law to a specific set of facts (including in alternative dispute resolution), Yes or No:


If you answered Yes to any of the above, give detailed summary and analysis of the context document based on the subject matter you answered "Yes" to, and re-write the context document to remove any aspects you answered "Yes" tdoc.
Use this format:
# Yes/No Answers
1. ...
2. ...
...
# Summary & Analysis (If any Yes answers)
...
# Rewritten Context Document Removing Problematic Text (If any Yes answers)
...
%(start_response)s
# Yes/No Answers
1. In the context of migration, asylum, or border control, involves information used by governments to test trust and truth of people:""",
    "steps": [
        {
            "description": "Analyze Document for High-Risk Areas",
            "input": "context_document",
            "output": "Yes/No answers to high-risk areas."
        },
        {
            "description": "Detailed Analysis Of Risks",
            "input": "High-Risk Areas detected in document",
            "output": "Detailed analysis based on high-risk areas identified."
        },
        {
            "description": "Revise context document to remove any High-Risk Content if any",
            "input": "context_document and detailed analysis based on high-risk areas identified.",
            "output": "Revised document with removed High-Risk Content, if any."
        }

    ]
  },

}
