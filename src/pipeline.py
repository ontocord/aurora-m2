from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, Pipeline


def get_llm(model_name: str, tokenizer_name: str, batch_size: int) -> Pipeline:
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).half().cuda()
    # Set up text generation pipeline
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto", batch_size=batch_size)
    return pipe


if __name__ == '__main__':
    llm = get_llm("facebook/opt-125m", "facebook/opt-125m")
    messages = ['tell me a pirate joke', 'tell me a joke prate']
    messages = [[{"role": "user", "content": f"tell me a pirate joke"}]]*2
    output_text = llm(messages, max_length=64, min_length=32, use_cache=True)
    print(len(output_text))
    print(output_text)
    from src.utils import postprocess_results
    pp_out = postprocess_results(output_text)
    print(pp_out)