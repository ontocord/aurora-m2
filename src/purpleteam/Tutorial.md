# How to use Context-aware Automated Purple Teaming

#### 1. Input construction - Constructing Context via Rule Templates 
This initial step where you define context categories (like violence) and subjects (like children) 
using rule templates is crucial. It's essential to ensure these templates are well-defined to capture the nuances of 
different contexts.

```
main.construct_context(rule_templates: dict, context_config: dict) -> dict
```

#### 2. Generate - LLM Creates Input Prompts
The LLM generates prompts based on the context and rules, 
creating the prompt dataset. This step is pivotal for guiding the subsequent content generation 
and can be repeated if to many prompts get discarded in a later step.

```
main.generate_input_prompts(context: dict, sample_size: int, llm: object) -> list
```

#### 3. Generate - LLM Creates Content (A): 
The LLM uses the input prompts to generate content. 
This is where the model's default safety alignment, and understanding of the safety context and 
guidelines (e.g. the EU AI Act) is tested.

```
main.generate_content_A(prompts: list, llm: object) -> list
```

#### 4. Check - Classify Content (A) using Safety Guard: 
Employing a tool like Llama Guard to classify the generated content to evaluate its appropriateness or safety.

```
main.classify_content_safety_guard(prompts: list, llm_outputs: list, safety_tool: object) -> list
```

#### 5. Generate "safety conditioned" output via further instructions - LLM Creates Content (B) with Added Safety Context: 
This step involves re-generating content with an additional layer of safety context, 
using information such as guidelines from the EU AI Act.

```
main.generate_content_B_with_safety_context(prompts: list, safety_context: dict, llm: object) -> list
```

#### 6. Check - Classify Second Content (B) with Llama Guard: 
Similar to step 4, but for the newly generated content (B).

```
main.classify_content_safety_guard(prompts: list, llm_outputs: list, safety_tool: object) -> list
```

#### 7. Classify Safety - Categorize A and B using Safety Guard: 
Determining which content (A or B) is safer using e.g. Llama Guard. Categorization in S (safer) and U (unsafer)

```
main.compare_safety_content_A_B(prompts: list, content_list_A: list, content_list_B: list, safety_assessment_A_and_B: list) -> list
```

#### 8. Generate - LLM Makes Content (A) More Offensive (C): 
Intentionally generating more offensive content (C) to test the robustness of the safety measures.

```
main.generate_offensive_content(prompts: list, outputs: list, safety_context: dict, llm: object) -> list
```

#### 9. Check - Purple LLM Classifies S and C: 
Comparative safety assessment between the safer content (S) and the intentionally offensive content (C).

```
main.classify_content_purple_llm(content_list_S: list, content_list_C: list, purple_llm: object) -> dict
```

#### 10. Check Consistency of step 9: 
If (C) is classified as safer than (S), it indicates a flaw in the safety assessment process, 
leading to the discarding of the sample.

```
main.check_consistency_classification(results: dict) -> bool
```

#### 11. Check Helpfulness - LLM Evaluates Helpfulness of S: 
Assessing whether the safer content (S) to ensure the helpfulness of the generated data sample. 
If helpfulness score is too low the sample is discarded. 

```
main.evaluate_helpfulness(prompts: list, llm_outputs, evaluation_model: object) -> list
```

#### 12. Iteration: 
Repeat these steps (3-11) to ensure thorough evaluation and generation of a broad range of samples. 
Potentially repeat step 2 if too many samples are discarded.
```
see main.main(config: dict)
```

#### 13. Dataset Construction:
Extract triplets (S, U, C) along with input prompts to create a rich dataset for analysis and further model training.

```
main.construct_dataset(prompts: list, output_safe: list, output_unsafe: list, output_unsafer: list) -> dict
```

#### 14. Auto-Feedback Fine-Tuning: 
Finally, using e.g. DPO to tune the LLM to align the model with safety standards.
```
TODO
```

## TODOs

**Basics:**

- [ ] (a) write utils for LLM generations **(Patrick)**
  - [ ] answer/completion given prompt
  - [ ] answer/completion given prompt and safety system prompt, previous answer
  - [ ] unsafer answer/completion given prompt and un-safety system prompt, previous answer
  - [ ] prompts given context template (cf. step 2)

- [ ] (b) write utils for Llama guard **(Huu, Patrick)**

- [ ] (c) write templates with context, subjects - **(Huu)**

- [ ] (d) define templates for safety compliance (US Executive order, EU AI Act etc.) - **(Huu)**
  - [ ] EU AI Act
  - [ ] US Executive order
  - [ ] other?

**Main:**
- [ ] step 1
- [ ] step 2
- [ ] step 3
- [ ] step 4
- [ ] step 5
- [ ] step 6
- [ ] step 7
- [ ] step 8
- [ ] step 9
- [ ] step 10
- [ ] step 11
- [ ] step 12
- [ ] step 13
- [ ] step 14
