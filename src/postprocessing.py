import math
import polars as pl
from typing import Dict

# def get_logprobs(response:dict , class_codes:list[str], replacement_mapping:dict={"Ġ" : "", '"' : ""}):

#     res_content = response.choices[0].logprobs.content

#     def _replace_substrings(inp_string:str, replacement_mapping:dict, strip:bool=True) -> str:
#         for key, value in replacement_mapping.items():
#             inp_string = inp_string.replace(key, value)
        
#         if strip :
#             return inp_string.strip()
#         return inp_string 

#     norm = lambda s : _replace_substrings(inp_string=s,
#                                           replacement_mapping=replacement_mapping,
#                                           strip=True)

#     try:
#         first_class_code_i = next(i for i, t in enumerate(res_content) if norm(t.token) in class_codes)
#     except StopIteration:
#         raise ValueError("No label token found in completion tokens.")

#     first_class_code_logprobs = getattr(res_content[first_class_code_i], "top_logprobs", None)
#     if not first_class_code_logprobs:
#         raise ValueError("Missing top_logprobs. Call the API with logprobs=True and a sufficient top_logprobs.")

#     class_codes_logprobs_mapping = {norm(tp.token): tp.logprob for tp in first_class_code_logprobs if norm(tp.token) in class_codes}

#     missing = [lab for lab in class_codes if lab not in class_codes_logprobs_mapping]
#     if missing:
#         raise ValueError(f"Missing label(s) in top_logprobs: {', '.join(missing)}")

#     max_logprobs = max(class_codes_logprobs_mapping.values())
#     exp_total_weight = sum(math.exp(lp - max_logprobs) for lp in class_codes_logprobs_mapping.values())
#     relative_probs = {lab: math.exp(class_codes_logprobs_mapping[lab] - max_logprobs) / exp_total_weight for lab in class_codes}
#     logprobs = {lab: class_codes_logprobs_mapping[lab] for lab in class_codes}


#     return {"rel_probs": relative_probs,
#             "logprobs": logprobs}


# def translate_class_code_keys_in_response(response:dict, class_codes_labels_mapping:dict) -> dict:
    
#     return { key: {class_codes_labels_mapping[k]: v for k, v in value.items()}
#             for key, value in response.items()}




# import polars as pl
# from typing import Dict, List

# class ResponseDFBuilder:
#     def __init__(self, response_probs: Dict[str, Dict[str, float]],
#                  class_mapping: Dict[str, str],
#                  logprob_mapping: Dict[float, str]):
#         self.response_probs = response_probs
#         self.class_mapping = class_mapping
#         self.logprob_mapping = logprob_mapping

#         self.df_final = self._create_base_df()

#         self._add_class_labels()
#         self._add_is_winner_col()
#         self._add_logprob_labels()
#         self._sort_columns()

#     def _create_base_df(self) -> pl.DataFrame:
#         required_keys = {"rel_probs", "logprobs"}
#         if not required_keys.issubset(self.response_probs.keys()):
#             raise KeyError("'rel_probs' and 'logprobs' keys must be provided")
#         return pl.DataFrame({
#             "class": list(self.response_probs['rel_probs'].keys()),
#             "relative_probs": list(self.response_probs['rel_probs'].values()),
#             "logprobs": list(self.response_probs['logprobs'].values())
#         })

#     def _add_class_labels(self) -> None:
#         self.df_final = self.df_final.with_columns(
#             pl.col("class").map_elements(lambda x: self.class_mapping.get(x, x)).alias("label")
#         )

#     def _add_is_winner_col(self, value_col: str = "relative_probs") -> None:
#         self.df_final = self.df_final.with_columns(
#             (pl.col(value_col) == pl.col(value_col).max()).alias("is_winner")
#         )

#     def _add_logprob_labels(self) -> None:
#         bins = sorted(self.logprob_mapping.items(), key=lambda x: x[0], reverse=True)

#         def interpret(val: float) -> str:
#             for cutoff, label in bins:
#                 if val >= cutoff:
#                     return label
#             return bins[-1][1]

#         self.df_final = self.df_final.with_columns(
#             self.df_final["logprobs"].map_elements(interpret).alias("logprob_interp")
#         )

#     def _sort_columns(self, column_order: List[str] = None) -> None:
#         if column_order is None:
#             column_order = ["class", "label", "is_winner", "relative_probs", "logprobs", "logprob_interp"]
#         self.df_final = self.df_final.select(column_order)

import math
import string
from dataclasses import dataclass, field
from typing import List, Dict
import polars as pl

from src.preprocessing import LabelToClassCode

class ResponseProcessor:
    def __init__(self, response:dict, label_to_class_code: LabelToClassCode):
        self.label_to_class_code = label_to_class_code
        self.response = response
        self.logprobs = self.get_logprobs()
        self.processed_response = self.translate_class_code_keys_in_response()

    @staticmethod
    def _replace_substrings(inp_string: str, replacement_mapping: Dict[str, str], strip: bool = True) -> str:
        for key, value in replacement_mapping.items():
            inp_string = inp_string.replace(key, value)
        return inp_string.strip() if strip else inp_string

    def get_logprobs(self, replacement_mapping: Dict[str, str] = {"Ġ": "", '"': ""}) -> dict:
        res_content = self.response.choices[0].logprobs.content
        norm = lambda s: self._replace_substrings(s.token, replacement_mapping)

        try:
            first_class_code_i = next(i for i, t in enumerate(res_content) if norm(t) in self.label_to_class_code.class_codes)
        except StopIteration:
            raise ValueError("No label token found in completion tokens.")

        first_class_code_logprobs = getattr(res_content[first_class_code_i], "top_logprobs", None)
        if not first_class_code_logprobs:
            raise ValueError("Missing top_logprobs. Call the API with logprobs=True and sufficient top_logprobs.")

        class_codes_logprobs_mapping = {
            norm(tp): tp.logprob for tp in first_class_code_logprobs if norm(tp) in self.label_to_class_code.class_codes
        }

        missing = [lab for lab in self.label_to_class_code.class_codes if lab not in class_codes_logprobs_mapping]
        if missing:
            raise ValueError(f"Missing label(s) in top_logprobs: {', '.join(missing)}")

        max_logprobs = max(class_codes_logprobs_mapping.values())
        exp_total_weight = sum(math.exp(lp - max_logprobs) for lp in class_codes_logprobs_mapping.values())
        relative_probs = {lab: math.exp(class_codes_logprobs_mapping[lab] - max_logprobs) / exp_total_weight
                          for lab in self.label_to_class_code.class_codes}
        logprobs = {lab: class_codes_logprobs_mapping[lab] for lab in self.label_to_class_code.class_codes}

        return {"rel_probs": relative_probs, "logprobs": logprobs}

    def translate_class_code_keys_in_response(self) -> dict:
        return {
            key: {self.label_to_class_code.class_codes_labels_mapping[k]: v for k, v in value.items()}
            for key, value in self.logprobs.items()
        }

class ResponseDFBuilder:
    def __init__(self, response_probs: Dict[str, Dict[str, float]], logprob_mapping: Dict[float, str]):

        self.response_probs = response_probs
        self.logprob_mapping = logprob_mapping

        self.df_final = self._create_base_df()
        self._add_is_winner_col()
        self._add_logprob_labels()
        self._sort_columns()

    def _create_base_df(self) -> pl.DataFrame:
        required_keys = {"rel_probs", "logprobs"}
        if not required_keys.issubset(self.response_probs.keys()):
            raise KeyError("'rel_probs' and 'logprobs' keys must be provided")
        return pl.DataFrame({
            "label": list(self.response_probs['rel_probs'].keys()),
            "relative_probs": list(self.response_probs['rel_probs'].values()),
            "logprobs": list(self.response_probs['logprobs'].values())
        })

    def _add_is_winner_col(self, value_col: str = "relative_probs") -> None:
        self.df_final = self.df_final.with_columns(
            (pl.col(value_col) == pl.col(value_col).max()).alias("is_winner")
        )

    def _add_logprob_labels(self) -> None:
        bins = sorted(self.logprob_mapping.items(), key=lambda x: x[0], reverse=True)

        def interpret(val: float) -> str:
            for cutoff, label in bins:
                if val >= cutoff:
                    return label
            return bins[-1][1]

        self.df_final = self.df_final.with_columns(
            self.df_final["logprobs"].map_elements(interpret).alias("logprob_interp")
        )

    def _sort_columns(self, column_order: List[str] = None) -> None:
        if column_order is None:
            column_order = ["label", "is_winner", "logprobs", "logprob_interp", "relative_probs"]
        self.df_final = self.df_final.select(column_order)