import argparse
from datasets import load_dataset
from typing import List, Dict, Tuple, Set, Callable
from transformers import AutoTokenizer
from model import HFLM_vLLM, HFLM_transformers, OpenAI_LM, Anthropic_LM
from template import hf_template, openai_template, anthropic_template
import logging
from pprint import pprint
import configparser
import os
import json

config = configparser.ConfigParser()
config.read('config.ini')
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

SUBSETS = [
    'AST_chinese',
    'AST_mathematics',
    'AST_biology',
    'AST_chemistry',
    'AST_physics',
    'AST_civics',
    'AST_geography',
    'AST_history',
    'GSAT_chinese',
    'GSAT_chemistry',
    'GSAT_biology',
    'GSAT_physics',
    'GSAT_earth_science',
    'GSAT_mathematics',
    'GSAT_geography',
    'GSAT_history',
    'GSAT_civics',
    'CAP_mathematics',
    'CAP_biology',
    'CAP_history',
    'CAP_civics',
    'CAP_geography',
    'CAP_physics',
    'CAP_chemistry',
    'CAP_earth_science',
    'driving_rule',
    'basic_traditional_chinese_medicine',
    'clinical_traditional_chinese_medicine',
    'lawyer_qualification',
    'nutritionist',
    'tour_guide',
    'tour_leader',
    'clinical_psychologist',
    'teacher_qualification',
    'accountant'
]

OPENAI_MODELS = {
    "gpt-4-1106-preview",
    "gpt-3.5-turbo-1106",
}

ANTHROPIC_MODELS = {
    "claude-2.0",
    "claude-instant-1.2",
}

def parse_args():
    parser = argparse.ArgumentParser(description="Run TMLU-Eval")
    parser.add_argument(
        "--model",
        type=str,
        default="yentinglin/Taiwan-LLM-7B-v2.0.1-chat",
        help="Model name",
    )
    parser.add_argument(
        "--base_url", type=str, default=None, help="The base url for OpenAI python API library."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Sampling temperature"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=128, help="Max tokens for generation"
    )
    parser.add_argument(
        "--tensor_parallel_size", type=int, default=1, help="Tensor parallel size"
    )
    parser.add_argument(
        "--subsets", type=str, default='ALL', help="The subsets of TMLU (splited by comma). Default is 'ALL'."
    )
    parser.add_argument(
        "--use_logits", action='store_true', default=False, help="Choose the answer base on the logits of each choice."
    )
    parser.add_argument(
        "--revision", type=str, default=None, help="The revision of the huggingface model."
    )
    return parser.parse_args()


def parse_example(example: Dict[str, str]) -> Tuple[str, List[str], List[str]]:
    question = example["question"]
    question = question.replace("\n\n", "\n")
    correct_choices = example["correct_choices"]
    incorrect_choices = example["incorrect_choices"]
    choices: List[Tuple[str, bool]] = []
    for choice in correct_choices:
        choices.append((choice, True))
    for choice in incorrect_choices:
        choices.append((choice, False))
    choices.sort()

    answer: List[str] = []
    for i, choice in enumerate(choices):
        if choice[1]:
            answer.append(chr(i + ord("A")))
    choices = [choice[0] for choice in choices]
    return question, choices, answer

def format_problem(
        example: Dict[str, str], 
        model_template: Callable,
        topic_line: str ="以下選擇題為出自臺灣的考題，答案為其中一個選項。",
        fewshot_examples: List[Dict[str, str]] = None,
    ) -> Tuple[str, List[str]]:
    question, choices, answer = parse_example(example)
    prompt = topic_line + "\n\n"
    if fewshot_examples:
        for fs_ex in fewshot_examples:
            fs_ex_question, fs_ex_choices, fs_ex_answer = parse_example(fs_ex)
            fs_ex_prompt = model_template(fs_ex_question, fs_ex_choices, fs_ex_answer)
            prompt += fs_ex_prompt + "\n\n"
    example_prompt = model_template(question, choices)
    prompt += example_prompt
    example["prompt"] = prompt
    example["answer"] = answer
    example["choice_num"] = len(choices)
    return example

def check_ans(raw_response: str, answer: Set[str]):
    def is_ans_format(text: str):
        if '正確答案' in text:
            return True
        elif '不正確' in text:
            return False
        elif 'A' in text or 'B' in text or 'C' in text or 'D' in text or 'E' in text:
            return True
        else:
            return False
    
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

if __name__ == "__main__":
    args = parse_args()

    is_openai_chat_model = args.model in OPENAI_MODELS or args.base_url
    is_anthropic_chat_model = args.model in ANTHROPIC_MODELS
    assert not ((is_openai_chat_model or is_anthropic_chat_model) and args.use_logits), "API based model doesn't support evaluation based on logits."


    if is_openai_chat_model:
        api_key=config[args.model]["api_key"]
        model = OpenAI_LM(
            args.model, 
            args.max_tokens, 
            args.temperature,
            api_key,
            args.base_url
        )
    elif is_anthropic_chat_model:
        api_key=config[args.model]["api_key"]
        model = Anthropic_LM(
            args.model, 
            args.max_tokens, 
            args.temperature, 
            api_key
        )
    elif args.use_logits:
        model = HFLM_transformers(
            args.model,
            args.max_tokens,
            args.temperature,
            args.revision
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
        )
        model = HFLM_vLLM(
            args.model, 
            args.tensor_parallel_size,
            args.max_tokens,
            args.temperature
        )

    if args.subsets == "ALL":
        subsets = SUBSETS
    else:
        subsets = [x.strip() for x in args.subsets.split(",")]
        for subset in subsets:
            assert(subset in SUBSETS), f"{subset} is not an available subset of TMLU."    

    results = {}
    log_root = os.path.join("log", args.model.replace("/", "_"))
    os.makedirs(log_root, exist_ok=True)
    for subset_name in subsets:
        logging.info(f"Evaluating {subset_name}")
        test_data = load_dataset(
            "miulab/tmlu", 
            subset_name,
            split="test",
        )
        fs_data = load_dataset(
            "miulab/tmlu", 
            subset_name, 
            split="dev",
        )
        if is_openai_chat_model:
            test_data = test_data.map(
                format_problem, 
                fn_kwargs={
                    "model_template": openai_template,
                    "fewshot_examples" : fs_data,
                }
            )
            outputs = model.generate(test_data["prompt"])
        elif is_anthropic_chat_model:
            test_data = test_data.map(
                format_problem, 
                fn_kwargs={
                    "model_template": anthropic_template,
                    "fewshot_examples" : fs_data,
                }
            )
            outputs = model.generate(test_data["prompt"], prefill="正確答案：(")
        elif args.use_logits:
            test_data = test_data.map(
                format_problem, 
                fn_kwargs={
                    "model_template": hf_template,
                    "fewshot_examples" : fs_data,
                }
            )
            outputs = model.generate(test_data["prompt"], test_data["choice_num"])
        else:
            test_data = test_data.map(
                format_problem, 
                fn_kwargs={
                    "model_template": hf_template,
                    "fewshot_examples" : fs_data,
                }
            )
            test_data = test_data.map(
                lambda x: {
                    "prompt": tokenizer.apply_chat_template(
                        [{"role": "user", "content": x["prompt"]}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                }
            )
            outputs = model.generate(test_data["prompt"])
        scores = [
            1 if check_ans(output, row["answer"]) else 0 for output, row in zip(outputs, test_data)
        ]
        avg_score = sum(scores) / len(scores)
        results[subset_name] = avg_score
        with open(os.path.join(log_root, f"{subset_name}_out.jsonl"), "w") as f:
            for output, row in zip(outputs, test_data):
                line = {
                    "id": row["id"],
                    "prompt": row["prompt"],
                    "full_response": output,
                    "gold_answer": row["answer"]
                }
                f.write(json.dumps(line, ensure_ascii=False)+"\n")

    pprint(results)