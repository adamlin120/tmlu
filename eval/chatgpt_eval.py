import openai
import json
import argparse
from tqdm import tqdm
import configparser

config = configparser.ConfigParser()
config.read('config.ini')
openai.api_key = config["OpenAI"]["api_key"]

def query(prompt: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt},
        ]
    )
    return response['choices'][0]['message']['content']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt_jsonl", 
        type=str, 
        required=True,
        help="Path to the jsonl that store generated formated prompts."
    )
    parser.add_argument(
        "--output_jsonl", 
        type=str, 
        required=False, 
        help="Path to the output jsonl for ChatGPT response."
    )

    args = parser.parse_args()
    output = []
    with open(args.prompt_jsonl, "r", encoding="UTF-8") as f:
        input_jsonl = f.readlines()
        for line in tqdm(input_jsonl):
            question = json.loads(line)
            prompt = question["prompt"]
            response = query(prompt)
            prediction = {
                "id" : question["id"],
                "prompt": prompt,
                "full_response": response,
                "gold_answer": question["answer"]
            }
            output.append(prediction)
    
    with open(args.output_jsonl, "w", encoding="UTF-8") as f:
        for prediction in output:
            line = json.dumps(prediction, ensure_ascii=False)
            f.write(line+"\n")

    
    