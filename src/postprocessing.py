import math

def get_logprobs(response:dict , class_codes:list[str], replacement_mapping:dict={"Ġ" : "", '"' : ""}):

    res_content = response.choices[0].logprobs.content

    def _replace_substrings(inp_string:str, replacement_mapping:dict, strip:bool=True) -> str:
        for key, value in replacement_mapping.items():
            inp_string = inp_string.replace(key, value)
        
        if strip :
            return inp_string.strip()
        return inp_string 

    norm = lambda s : _replace_substrings(inp_string=s,
                                          replacement_mapping=replacement_mapping,
                                          strip=True)

    try:
        first_class_code_i = next(i for i, t in enumerate(res_content) if norm(t.token) in class_codes)
    except StopIteration:
        raise ValueError("No label token found in completion tokens.")

    first_class_code_logprobs = getattr(res_content[first_class_code_i], "top_logprobs", None)
    if not first_class_code_logprobs:
        raise ValueError("Missing top_logprobs. Call the API with logprobs=True and a sufficient top_logprobs.")

    class_codes_logprobs_mapping = {norm(tp.token): tp.logprob for tp in first_class_code_logprobs if norm(tp.token) in class_codes}

    missing = [lab for lab in class_codes if lab not in class_codes_logprobs_mapping]
    if missing:
        raise ValueError(f"Missing label(s) in top_logprobs: {', '.join(missing)}")

    max_logprobs = max(class_codes_logprobs_mapping.values())
    exp_total_weight = sum(math.exp(lp - max_logprobs) for lp in class_codes_logprobs_mapping.values())
    relative_probs = {lab: math.exp(class_codes_logprobs_mapping[lab] - max_logprobs) / exp_total_weight for lab in class_codes}
    logprobs = {lab: class_codes_logprobs_mapping[lab] for lab in class_codes}


    return {"rel_probs": relative_probs,
            "logprobs": logprobs}
