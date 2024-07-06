from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, Pipeline


def get_llm(model_name: str, tokenizer_name: str) -> Pipeline:
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).half().cuda()
    # Set up text generation pipeline
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
    return pipe