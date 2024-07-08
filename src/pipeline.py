import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, Pipeline

def get_llm(model_name: str, tokenizer_name: str, batch_size: int, huggingface_or_vllm: str="huggingface"):
    if huggingface_or_vllm == "huggingface":
        get_hf_llm(model_name, tokenizer_name, batch_size)
    elif huggingface_or_vllm == "vllm":
        get_vllm_llm(model_name, tokenizer_name, batch_size)
    raise ValueError("huggingface_or_vllm must be either 'huggingface' or 'vllm'")


def get_hf_llm(model_name: str, tokenizer_name: str, batch_size: int) -> Pipeline:
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).half().cuda()
    # Set up text generation pipeline
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto", batch_size=batch_size)
    return lambda x: pipe(x, max_length=2048, min_length=512, use_cache=True)

def get_vllm_llm(model_name: str, tokenizer_name: str, batch_size: int) -> Pipeline:
    try:
        from vllm import LLM, SamplingParams
    except:
        raise ImportError("vllm must be installed to use vllm models")
    
    params = SamplingParams(
        max_tokens=2048,
        min_tokens=512,
        # POSSIBLE PARAMS
        # temperature=args.temperature if not args.use_beam_search else 0,
        # top_p=args.top_p if not args.use_beam_search else 1,
        # top_k=args.top_k if not args.use_beam_search else -1,        
        # repetition_penalty=1.05,
        # use_beam_search=args.use_beam_search,
        # best_of=3 if args.use_beam_search else 1,
        # skip_special_tokens=True,
    )

    # this is not the right way
    tensor_parallel_size = 4 if "70" in model_name else 1
    llm = LLM(
        model=model_name,
        tokenizer_mode="slow",
        tensor_parallel_size=tensor_parallel_size)
    
    return lambda x: llm.generate(x, sampling_params=params)

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