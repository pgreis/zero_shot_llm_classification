import string
from dataclasses import dataclass, field

@dataclass(frozen=False)
class LabelToClassCode:
    labels: list[str]
    class_codes: list[str] = field(init=False)
    class_codes_labels_mapping: dict[str, str] = field(init=False)

    def _get_class_codes(self) -> list[str]:
        n = len(self.labels)
        max_codes = len(string.ascii_uppercase)
        if n > len(string.ascii_uppercase):
            raise ValueError(f"Too many labels ({n}). Max allowed: {max_codes}.")
        return list(string.ascii_uppercase[:n])

    def __post_init__(self) -> None:
        if any(not isinstance(x, str) or not x.strip() for x in self.labels):
            raise ValueError("All labels must be non-empty strings.")
        
        self.class_codes = self._get_class_codes()
        self.class_codes_labels_mapping = dict(zip(self.class_codes, self.labels))

@dataclass(frozen=True)
class UnprocessedLLMInputs:
    inp_text: str
    inp_topic: str
    system_template: str
    user_template: str
    output_schema_template: dict

class ProcessedLLMInputs:
    def __init__(self, label_to_class_code: LabelToClassCode, unprocessed_llm_inputs: UnprocessedLLMInputs):
        self.label_class_code = label_to_class_code
        self.unprocessed_llm_inputs = unprocessed_llm_inputs

        self.inp_text: str = self.unprocessed_llm_inputs.inp_text
        self.inp_topic: str = self.unprocessed_llm_inputs.inp_topic
        self.system_template: str = self.unprocessed_llm_inputs.system_template
        self.user_template: str = self.unprocessed_llm_inputs.user_template
        self.output_schema_template: dict = self.unprocessed_llm_inputs.output_schema_template

        self.class_code_eq_label = self._get_class_code_eq_label_list()
        self.system_msg = self._render_prompt(template=self.system_template,
                                              inp_topic=self.inp_topic,
                                              class_code_eq_label=self.class_code_eq_label)
        self.user_msg = self._render_prompt(template=self.user_template,
                                            inp_text=self.inp_text)

        self.message = [{"role": "system", "content": self.system_msg},
                        {"role": "user", "content": self.user_msg}]

        self.output_schema = self._create_classification_output_schema(min_len_explanation=10,
                                                                       max_len_explanation=200)

    def _get_class_code_eq_label_list(self) -> list[str]:
        return [str(class_code) + "=" + str(label) for class_code, label in self.label_class_code.class_codes_labels_mapping.items()]

    @staticmethod
    def _render_prompt(template, **kwargs) -> str:
        return template.format(**kwargs)
    
    def _create_classification_output_schema(self,
                                             min_len_explanation: int = 10,
                                             max_len_explanation: int = 200) -> dict:
        output_schema = self.output_schema_template.copy()
        class_codes = self.label_class_code.class_codes
        class_code_eq_label = self.class_code_eq_label

        description_filled = f"{', '.join(class_code_eq_label)}"
        allowed_responses_regex_filled = f"^({'|'.join(class_codes)})$"

        output_schema["json_schema"]["schema"]["properties"]["classification"].update({"enum": class_codes,
                                                                                       "description": description_filled,
                                                                                       "pattern": allowed_responses_regex_filled})

        output_schema["json_schema"]["schema"]["properties"]["explanation"].update({"minLength" : min_len_explanation,
                                                                                    "maxLength" : max_len_explanation})

        return output_schema