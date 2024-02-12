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
    'CAP_chinese',
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
    'taiwan_tourist_resources',
    'clinical_psychologist',
    'teacher_qualification',
    'accountant'
]

def parse_args():
    parser = argparse.ArgumentParser(description="Run TMLU-Eval")
    parser.add_argument(
        "--backend",
        choices=['hf', 'vllm', 'openai', 'anthropic', 'custom_api'],
        required=True,
        help="The backend type. Options: ['hf', 'vllm', 'openai', 'anthropic', 'custom_api']"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name.",
    )
    parser.add_argument(
        "--revision", type=str, default=None, help="The revision of the huggingface model."
    )
    parser.add_argument(
        "--dtype", type=str, default=None, help="The dtype of the model."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Sampling temperature."
    )
    parser.add_argument(
        "--max_tokens", type=int, default=128, help="Max new tokens to generate for generation based evalutaion."
    )
    parser.add_argument(
        "--subsets", type=str, default='ALL', help="The subsets of TMLU (splited by comma). Default is 'ALL'."
    )
    parser.add_argument(
        "--base_url", type=str, default=None, help="The base url for OpenAI python API library."
    )
    parser.add_argument(
        "--tensor_parallel_size", type=int, default=1, help="Tensor parallel size for vllm."
    )
    parser.add_argument(
        "--log_dir", type=str, default=None, help="Directory for saving evaluation log."
    )
    parser.add_argument(
        "--overwrite_log_dir", action="store_true", help="Overwrite logs in the directory."
    )
    parser.add_argument(
        "--few_shot_num", type=int, default=5, help="The number for few shot example. Range: [0, 5]"
    )
    parser.add_argument(
        "--timeout", type=float, default=20.0, help="Timeout for API based backend."
    )
    parser.add_argument(
        "--max_retries", type=int, default=100, help="Max retries for API based backend."
    )
    
    return parser.parse_args()


def parse_example(example: Dict[str, str]) -> Tuple[str, List[str], List[str]]:
    question = example["question"]
    question = question.replace("\n\n", "\n")
    choices = []
    for i in range(6):
        if example[chr(i + ord("A"))] != None:
            choices.append(example[chr(i + ord("A"))])
        else:
            break
    answer = example["answer"]
    return question, choices, answer

def format_problem(
        example: Dict[str, str], 
        model_template: Callable,
        topic_line: str ="以下選擇題為出自臺灣的考題，答案為其中一個選項。",
        few_shot_examples: List[Dict[str, str]] = None,
        few_shot_num: int = 0,
    ) -> Tuple[str, List[str]]:
    question, choices, answer = parse_example(example)
    prompt = topic_line + "\n\n"
    if few_shot_examples and few_shot_num:
        for i in range(few_shot_num):
            fs_ex = few_shot_examples[i]
            fs_ex_question, fs_ex_choices, fs_ex_answer = parse_example(fs_ex)
            fs_ex_prompt = model_template(fs_ex_question, fs_ex_choices, fs_ex_answer)
            prompt += fs_ex_prompt + "\n\n"
    example_prompt = model_template(question, choices)
    prompt += example_prompt
    example["prompt"] = prompt
    example["answer"] = answer
    example["choice_num"] = len(choices)
    return example

def check_ans(raw_response: str, answer: List[str]):
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

    if args.backend == 'openai':
        api_key = os.environ.get("OPENAI_API_KEY")
        model = OpenAI_LM(
            model_name=args.model, 
            max_tokens=args.max_tokens, 
            temperature=args.temperature,
            api_key=api_key,
            base_url=args.base_url,
            timeout=args.timeout,
            max_retries=args.max_retries
        )
    elif args.backend == 'anthropic':
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        model = Anthropic_LM(
            model_name=args.model, 
            max_tokens=args.max_tokens, 
            temperature=args.temperature, 
            api_key=api_key,
            timeout=args.timeout,
            max_retries=args.max_retries
        )
    elif args.backend == 'custom_api':
        api_key = "EMPTY"
        model = OpenAI_LM(
            model_name=args.model, 
            max_tokens=args.max_tokens, 
            temperature=args.temperature,
            api_key=api_key,
            base_url=args.base_url,
            timeout=args.timeout,
            max_retries=args.max_retries
        )
    elif args.backend == 'hf':
        model = HFLM_transformers(
            model_name=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            revision=args.revision,
            dtype=args.dtype
        )
        tokenizer = model.get_tokenizer()
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            revision=args.revision
        )
        model = HFLM_vLLM(
            model_name=args.model, 
            tensor_parallel_size=args.tensor_parallel_size,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            revision=args.revision,
            dtype=args.dtype
        )

    if args.subsets == "ALL":
        subsets = SUBSETS
    else:
        subsets = [x.strip() for x in args.subsets.split(",")]
        for subset in subsets:
            assert(subset in SUBSETS), f"{subset} is not an available subset of TMLU."    

    results = {}
    if args.log_dir:
        log_root = args.log_dir
    else:
        if args.backend == "hf":
            log_root = os.path.join("log", f"{args.model.replace('/', '_')}_logits")
        else:
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

        past_scores = []
        past_ids = set()
        if os.path.isfile(os.path.join(log_root, f"{subset_name}_out.jsonl")) and not args.overwrite_log_dir:
            with open(os.path.join(log_root, f"{subset_name}_out.jsonl"), "r") as f:
                lines = f.readlines()
                for line in lines:
                    example = json.loads(line)
                    past_ids.add(example["id"])
                    score = 1 if check_ans(example["full_response"], example["gold_answer"]) else 0
                    past_scores.append(score)
            
            test_data = test_data.filter(lambda example: not example["id"] in past_ids)
        
        if args.backend == 'openai' or args.backend == 'custom_api':
            test_data = test_data.map(
                format_problem, 
                fn_kwargs={
                    "model_template": openai_template,
                    "few_shot_examples" : fs_data,
                    "few_shot_num": args.few_shot_num
                }
            )
            outputs = model.generate(test_data)
        elif args.backend == 'anthropic':
            test_data = test_data.map(
                format_problem, 
                fn_kwargs={
                    "model_template": anthropic_template,
                    "few_shot_examples" : fs_data,
                    "few_shot_num": args.few_shot_num
                }
            )
            outputs = model.generate(test_data, prefill="正確答案：(")
        elif args.backend == 'hf':
            test_data = test_data.map(
                format_problem, 
                fn_kwargs={
                    "model_template": hf_template,
                    "few_shot_examples" : fs_data,
                    "few_shot_num": args.few_shot_num
                }
            )
            test_data = test_data.map(
                lambda x: {
                    "prompt": tokenizer.apply_chat_template(
                        [{"role": "user", "content": x["prompt"]}],
                        tokenize=False,
                        add_generation_prompt=True,
                    ) + "\n正確答案：("
                }
            )
            outputs = model.generate(test_data)
        else:
            test_data = test_data.map(
                format_problem, 
                fn_kwargs={
                    "model_template": hf_template,
                    "few_shot_examples" : fs_data,
                    "few_shot_num": args.few_shot_num
                }
            )
            test_data = test_data.map(
                lambda x: {
                    "prompt": tokenizer.apply_chat_template(
                        [{"role": "user", "content": x["prompt"]}],
                        tokenize=False,
                        add_generation_prompt=True,
                    ) + "\n正確答案：("
                }
            )
            outputs = model.generate(test_data)
        
        if args.overwrite_log_dir:
            output_file_open_type = "w"
        else:
            output_file_open_type = "a"

        with open(os.path.join(log_root, f"{subset_name}_out.jsonl"), output_file_open_type) as f:
            for i in range(len(outputs)):
                line = {
                    "id": test_data[i]["id"],
                    "prompt": test_data[i]["prompt"],
                    "full_response": outputs[i],
                    "gold_answer": test_data[i]["answer"]
                }
                f.write(json.dumps(line, ensure_ascii=False)+"\n")

        assert len(outputs) == len(test_data), f"Error occurs when evaluating {subset_name}"
        scores = [
            1 if check_ans(output, row["answer"]) else 0 for output, row in zip(outputs, test_data)
        ]
        scores += past_scores
        avg_score = sum(scores) / len(scores)
        results[subset_name] = avg_score
    pprint(results)