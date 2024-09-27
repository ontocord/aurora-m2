# safellm

## Synthetic Data Generation Papers
- Evol-Instruct - [Automatic Instruction Evolving for Large Language Models](https://arxiv.org/pdf/2406.00770)
- Orca-3 - [AgentInstruct: Toward Generative Teaching with Agentic Flows](https://arxiv.org/pdf/2407.03502)
- 1 Billion Personas - [Scaling Synthetic Data Creation with 1,000,000,000 Personas](https://arxiv.org/pdf/2406.20094v1) [Code](https://github.com/tencent-ailab/persona-hub)
- SAT - [SATLM: Satisfiability-Aided Language Models Using Declarative Prompting](https://arxiv.org/pdf/2305.09656)
- [Arena Learning: Build Data Flywheel for LLMs Post-training via Simulated Chatbot Arena](https://arxiv.org/pdf/2407.10627)


## Tools/ Library
Benchmark Report of Different LLM Inference Backends [link](https://www.bentoml.com/blog/benchmarking-llm-inference-backends)
- [LMDeploy](https://github.com/InternLM/lmdeploy) [Fastest]
- [vLLM](https://github.com/vllm-project/vllm)


## Installation
- In addition to pip install -r requirements.txt also do this:
- python -m wn download oewn:2023
- python -m spacy download en_core_web_sm
- pip install -q https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz
