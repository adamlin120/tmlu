from typing import List

def hf_template(question: str, choices: List[str], answer: List[str] = None) -> str:
    choices_prompt = ""
    for i, choice in enumerate(choices):
        choices_prompt += f"({chr(i + ord('A'))}) {choice}\n"
    if answer:
        return f"問題：{question}\n{choices_prompt}正確答案：({')('.join(answer)})"
    else:
        return f"問題：{question}\n{choices_prompt}"

def openai_template(question: str, choices: List[str], answer: List[str] = None) -> str:
    choices_prompt = ""
    for i, choice in enumerate(choices):
        choices_prompt += f"({chr(i + ord('A'))}) {choice}\n"
    if answer:
        return f"問題：{question}\n{choices_prompt}正確答案：({')('.join(answer)})"
    else:
        return f"問題：{question}\n{choices_prompt}正確答案：("

def anthropic_template(question: str, choices: List[str], answer: List[str] = None) -> str:
    choices_prompt = ""
    for i, choice in enumerate(choices):
        choices_prompt += f"({chr(i + ord('A'))}) {choice}\n"
    if answer:
        return f"問題：{question}\n{choices_prompt}正確答案：({')('.join(answer)})"
    else:
        return f"問題：{question}\n{choices_prompt}"