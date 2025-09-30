from src.postprocessing import ResponseProcessor, ResponseDFBuilder
from app.config import CONFIG

def process_llm_response(response: dict, label_mapping, request_body: dict):

    processed_response = ResponseProcessor(response=response,
                                           label_to_class_code=label_mapping).processed_response
    
    df = ResponseDFBuilder(response_probs=processed_response,
                                         logprob_mapping=CONFIG["logprob_interpretation"]).df_final
    
    response_input = {
            "topic": request_body.get("topic"),
            "labels": list(request_body.get("labels", [])),
            "text": request_body.get("text")
        }
    
    response_output = df.to_dicts()

    return {
        "input": response_input,
        "outputs": response_output
    }