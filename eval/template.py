from typing import List, Dict, Tuple

def parse_example(example: Dict[str, str]) -> Tuple[str, str, str, str]:
    question = example["question"].replace("\n\n", "\n")
    choices_prompt = ""
    for i in range(6):
        choice = chr(i + ord("A"))
        if example[choice] != None:
            choices_prompt += f"({choice}) {example[choice]}\n"
        else:
            break
    answer = example["answer"]
    cot = example["explanation"]
    return question, choices_prompt, answer, cot

def hf_template(example: Dict[str, str], use_cot: bool = False, include_ans: bool = False) -> str:
    question, choices_prompt, answer, cot = parse_example(example)
    
    full_prompt = f"問題：{question}\n{choices_prompt}"
    if include_ans:
        if use_cot:
            full_prompt += f"讓我們一步一步思考。\n{cot}\n正確答案：({answer})"
        else:
            full_prompt += f"正確答案：({answer})"
    else:
        if use_cot:
            full_prompt += f"讓我們一步一步思考。\n"
    return full_prompt

def openai_template(example: Dict[str, str], use_cot: bool = False, include_ans: bool = False) -> str:
    question, choices_prompt, answer, cot = parse_example(example)
    
    full_prompt = f"問題：{question}\n{choices_prompt}"
    if include_ans:
        if use_cot:
            full_prompt += f"讓我們一步一步思考。\n{cot}\n正確答案：({answer})"
        else:
            full_prompt += f"正確答案：({answer})"
    else:
        if use_cot:
            full_prompt += f"讓我們一步一步思考。\n"
        else:
            full_prompt += "正確答案：("
    return full_prompt

def anthropic_template(example: Dict[str, str], use_cot: bool = False, include_ans: bool = False) -> str:
    question, choices_prompt, answer, cot = parse_example(example)
    
    full_prompt = f"問題：{question}\n{choices_prompt}"
    if include_ans:
        if use_cot:
            full_prompt += f"讓我們一步一步思考。\n{cot}\n正確答案：({answer})"
        else:
            full_prompt += f"正確答案：({answer})"
    else:
        if use_cot:
            full_prompt += f"讓我們一步一步思考。\n"
    return full_prompt