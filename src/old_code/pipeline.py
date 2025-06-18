import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, Pipeline

def get_llm(model_name: str, tokenizer_name: str, batch_size: int = None, huggingface_or_vllm: str="huggingface"):
    if huggingface_or_vllm == "huggingface":
        return get_hf_llm(model_name, tokenizer_name, batch_size)
    elif huggingface_or_vllm == "vllm":
        return get_vllm_llm(model_name, tokenizer_name, batch_size)
    raise ValueError("huggingface_or_vllm must be either 'huggingface' or 'vllm'")


def get_hf_llm(model_name: str, tokenizer_name: str, batch_size: int) -> Pipeline:
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).half().cuda()
    # Set up text generation pipeline
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto", batch_size=batch_size)
    return lambda x: pipe(x, max_length=2048, min_length=512, use_cache=True, return_full_text=True)

def get_vllm_llm(model_name: str, tokenizer_name: str, batch_size: int) -> Pipeline:

    try:
        from vllm import LLM, SamplingParams
    except:
        raise ImportError("vllm must be installed to use vllm models")
    if tokenizer_name == model_name:
        print("vllm will ignore the tokenizer_name and use the same as model_name")
    if batch_size is None:
        print("vllm does not need the batch_size parameters as it adjusts it dinamically")

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
    
    def generate(messages, llm):
        tokenizer = llm.get_tokenizer()
        prompts = tokenizer.apply_chat_template(
            messages, truncation=None, padding=False,
            add_generation_prompt=True)

        prompts = tokenizer.batch_decode(prompts)
        # print(prompts)
        return llm.generate(prompts, sampling_params=params)
    
    return lambda x: generate(x, llm)

if __name__ == '__main__':
    def main():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sep = "-"*10
        raw_messages = ['tell me a pirate joke', 'tell me a pirate joke']
        messages = [[{"role": "user", "content": f"tell me a pirate joke"}]]*2

        print(sep, "HF Test")
        # llm = get_hf_llm("facebook/opt-125m", "facebook/opt-125m", 1)
        model_name = "/leonardo_scratch/large/userexternal/gpuccett/models/hf_gemma/Gemma-2-9B-It-SPPO-Iter3"
        tokenizer_name = "/leonardo_scratch/large/userexternal/gpuccett/models/hf_gemma/Gemma-2-9B-It-SPPO-Iter3"
        llm = get_llm(model_name, tokenizer_name, 1, "huggingface")
    
        output_text = llm(messages)
        print(sep, len(output_text))
        print(sep, output_text)
        from src.utils import postprocess_results
        hf_out = postprocess_results(output_text)
        print(hf_out)
        print(sep, output_text[0][0]["generated_text"][0])

        del llm
        del output_text
    
        print("VLLM Test")
        # llm = get_vllm_llm("facebook/opt-125m", "facebook/opt-125m")
        llm = get_llm(model_name, tokenizer_name, None, "vllm")
        output_text = llm(raw_messages)
        print(sep, len(output_text))
        print(sep, output_text)
        prompts = []
        for output in output_text:
            prompt = output.prompt
            prompts.append(prompt)
            generated_text = output.outputs[0].text
            print(sep, f"Prompt: {prompt!r}")
            print(sep, f"Generated text: {generated_text!r}")
        vllm_out = postprocess_results(output_text)
        print(sep, vllm_out)
        print("HUGGINGFACE", [i[:100] for i in hf_out])
        print("VLLM", [p + i[:100] for p, i in zip(prompts, vllm_out)])

    main()
