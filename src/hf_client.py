from huggingface_hub import InferenceClient
from src.preprocessing import ProcessedLLMInputs

def get_hf_client(hf_model:str, hf_provider: str, hf_token: str) -> InferenceClient:
    return InferenceClient(model=hf_model,
        provider=hf_provider,
        api_key=hf_token
    )

def get_hf_llm_repsonse(
    client: InferenceClient,
    processed_llm_inp: ProcessedLLMInputs,
    logprobs:bool = True,
    max_tokens:int = 1024,
    temperature:float = 0.0):

    return client.chat_completion(
        messages=processed_llm_inp.message,
        response_format=processed_llm_inp.output_schema,
        logprobs=logprobs,
        max_tokens=max_tokens,
        top_logprobs=len(processed_llm_inp.label_class_code.class_codes),
        temperature=temperature
    )