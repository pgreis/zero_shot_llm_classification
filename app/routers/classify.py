from fastapi import APIRouter, Depends, HTTPException
from app.models.request import ClassificationInput
from app.models.response import ClassificationOutput
from app.dependencies import hf_client_dependency
from app.services.preprocessing import prepare_llm_input
from app.services.postprocessing import process_llm_response
from src.hf_client import get_hf_llm_repsonse

router = APIRouter()

@router.post("/classify", response_model=ClassificationOutput)
async def classify_text(
    request: ClassificationInput,
    hf_client = Depends(hf_client_dependency)
):
    try:
        processed_input, label_mapping = prepare_llm_input(text=request.text,
                                                           topic=request.topic,
                                                           labels=request.labels)
        
        response = get_hf_llm_repsonse(client=hf_client,
                                       processed_llm_inp=processed_input)
        
        processed_response = process_llm_response(response=response,
                                                  label_mapping=label_mapping,
                                                  request_body=request.model_dump())
        return processed_response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")
