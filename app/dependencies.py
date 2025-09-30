from fastapi import HTTPException, Query
from huggingface_hub import InferenceClient
from src.hf_client import get_hf_client

def hf_client_dependency(
    hf_token: str = Query(..., description="HuggingFace API token"),
    hf_model: str = Query(..., description="The HuggingFace Model"),
    hf_provider: str = Query(..., description="HuggingFace provider")
    
) -> InferenceClient:

    if not hf_token:
        raise HTTPException(status_code=400, detail="HF token is required")
    
    if not hf_model:
        raise HTTPException(status_code=400, detail="HF model is required")
        
    if not hf_provider:
        raise HTTPException(status_code=400, detail="HF provider is required")

    return get_hf_client(hf_token=hf_token,hf_model=hf_model, hf_provider=hf_provider)
