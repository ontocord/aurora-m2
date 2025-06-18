# aurora-m2 aka safellm

## Synthetic Data Generation Papers
- Creating more data from smaller models is more efficient - [Smaller, Weaker, Yet Better: Training LLM Reasoners via Compute-Optimal Sampling](https://arxiv.org/pdf/2408.16737).
- Evol-Instruct - [Automatic Instruction Evolving for Large Language Models](https://arxiv.org/pdf/2406.00770)
- Orca-3 - [AgentInstruct: Toward Generative Teaching with Agentic Flows](https://arxiv.org/pdf/2407.03502)
- 1 Billion Personas - [Scaling Synthetic Data Creation with 1,000,000,000 Personas](https://arxiv.org/pdf/2406.20094v1) [Code](https://github.com/tencent-ailab/persona-hub)
- SAT - [SATLM: Satisfiability-Aided Language Models Using Declarative Prompting](https://arxiv.org/pdf/2305.09656)
- [Arena Learning: Build Data Flywheel for LLMs Post-training via Simulated Chatbot Arena](https://arxiv.org/pdf/2407.10627)
- [Magpie](https://github.com/magpie-align/magpie) - [Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing](https://arxiv.org/abs/2406.08464)


## Tools/ Library
Benchmark Report of Different LLM Inference Backends [link](https://www.bentoml.com/blog/benchmarking-llm-inference-backends)
- [LMDeploy](https://github.com/InternLM/lmdeploy) [Fastest]
- [vLLM](https://github.com/vllm-project/vllm)


## Installation
- On leonardo, you will need to do this: `module load cuda`
- pip install -r requirements.txt
- python -m wn download oewn:2023
- python -m spacy download en_core_web_sm
- python -m spacy download xx_ent_wiki_sm
- NOTE: scispacy is not working on Leonardo for some reason bc of nmslib not installing. TODO to fix. 
- pip install -q https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz
  
