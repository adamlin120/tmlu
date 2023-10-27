# TMLU Eval

## Prompt generation

Generate prompt for each problem. The output `prompt_jsonl`will contain the prompt for each problem in input jsonl. The order of the choices will be randomly generated.

Usage:

```
usage: gen_problem_prompt.py [-h] --input_jsonl INPUT_JSONL [--example_jsonl EXAMPLE_JSONL] --prompt_jsonl PROMPT_JSONL

optional arguments:
  -h, --help            show this help message and exit
  --input_jsonl INPUT_JSONL
                        Path to the imput jsonl for all questions.
  --example_jsonl EXAMPLE_JSONL
                        Path to the example jsonl for few-shot demonstrations.
  --prompt_jsonl PROMPT_JSONL
                        Output jsonl that will be used to store the formated prompts.
```

Note: if you don't pass example_jsonl, it is the zero-shot setup.



## Test on ChatGPT

### Add config.ini

Add a config.ini under this directory in following format. (Don't forget to fill in the token!)

```
[OpenAI]
api_key=
```

### Query ChatGPT

Usage:

```
usage: chatgpt_eval.py [-h] --input_jsonl INPUT_JSONL [--output_jsonl OUTPUT_JSONL]

optional arguments:
  -h, --help            show this help message and exit
  --input_jsonl INPUT_JSONL
                        Path to the imput jsonl for formated prompts.
  --output_jsonl OUTPUT_JSONL
                        Path to the output jsonl for ChatGPT response.
```


