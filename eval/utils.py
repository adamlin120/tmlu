from typing import Set

def is_ans_format(text: str):
        if '正確答案' in text:
            return True
        elif '不正確' in text:
            return False
        elif 'A' in text or 'B' in text or 'C' in text or 'D' in text or 'E' in text:
            return True
        else:
            return False

def check_ans(raw_response: str, answer: str):
    raw_response_split = raw_response.strip().split("\n\n")
    if len(raw_response_split[0]) < 50 and is_ans_format(raw_response_split[0]):
        prediction_text = raw_response_split[0]
    else:
        prediction_text = raw_response_split[-1]
    
    prediction: Set[str] = set()
    for i in range(6):
        if f"{chr(ord('A')+i)})" in prediction_text:
            prediction.add(chr(ord('A')+i))
    return prediction == set(answer)

def check_ans_cot(raw_response: str, answer: str):
    raw_response_split = raw_response.strip().split("\n")

    prediction_text = ""
    for i in range(len(raw_response_split)-1, -1, -1):
        if is_ans_format(raw_response_split[i]):
            prediction_text = raw_response_split[i]
            break

    prediction = ""
    ans_pos = prediction_text.find("正確答案")
    if ans_pos != -1:
        for c in prediction_text[ans_pos+4:]:
            if c.isalpha() & c.isascii():
                prediction = c
                break
    
    return prediction == answer
