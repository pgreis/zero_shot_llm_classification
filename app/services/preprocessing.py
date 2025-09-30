from src.preprocessing import LabelToClassCode, UnprocessedLLMInputs, ProcessedLLMInputs
from app.config import SYSTEM_TEMPLATE, USER_TEMPLATE, OUTPUT_SCHEMA

def prepare_llm_input(text: str, topic: str, labels: list):
    label_mapping = LabelToClassCode(labels=labels)

    unprocessed_input = UnprocessedLLMInputs(
        inp_text=text,
        inp_topic=topic,
        system_template=SYSTEM_TEMPLATE,
        user_template=USER_TEMPLATE,
        output_schema_template=OUTPUT_SCHEMA
    )

    processed_input = ProcessedLLMInputs(
        label_to_class_code=label_mapping,
        unprocessed_llm_inputs=unprocessed_input
    )

    return processed_input, label_mapping
