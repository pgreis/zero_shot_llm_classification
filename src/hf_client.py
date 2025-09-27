import os
from huggingface_hub import InferenceClient
from src.preprocessing import ProcessedLLMInputs

# def get_hf_token(token_name:str = "HF_TOKEN") -> str:
#     hf_token = os.getenv(token_name)
#     if not hf_token:
#         raise KeyError("HF_TOKEN not found in environment. Set HF_TOKEN or use dotenv in bootstrap.")
#     return hf_token

def get_hf_client(hf_provider:str,
                  hf_token:str) -> InferenceClient:
    return InferenceClient(
        provider=hf_provider,
        api_key=hf_token,)

def get_hf_llm_repsonse(client: InferenceClient, processed_llm_inp: ProcessedLLMInputs, config:dict):
    cc = config.get("chat_completion", {})
    if not cc:
        raise KeyError("No chat_completion part in config file")
    return client.chat_completion(
        messages=processed_llm_inp.message,
        response_format=processed_llm_inp.output_schema,
        model=cc.get("model", "meta-llama/Llama-3.1-8B-Instruct"),
        logprobs=cc.get("logprobs", True),
        max_tokens= cc.get("max_tokens", 128),
        top_logprobs=len(processed_llm_inp.label_class_code.class_codes),
        temperature=cc.get("temperature", 0.0)
        )