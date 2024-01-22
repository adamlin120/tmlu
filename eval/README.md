# TMLU Eval

## Open source model

### Probability based evalutaion

Use the probabilities of the option codes as prediction.

```bash
$python3 tmlu_eval.py \
	--model [Model name on huggingface or path] \
	--dtype [torch type for the model, default will follow the model config] \
	--temperature [TEMPERATURE] \
	--prob_based \
	--subsets [Choose subsets of TMLU (names splited by comma) for evalutaion. Default is 'ALL'] \
	--log_dir [Directory for saving evaluation log]
```

#### Example

Command:

```bash
$python3 tmlu_eval.py \
	--model yentinglin/Taiwan-LLM-7B-v2.1-chat \
	--dtype float16 \
	--temperature 0.0 \
	--prob_based \
	--subsets AST_chinese,AST_mathematics \
	--log_dir log/prob_based/Taiwan-LLM-7B-v2.1-chat
```

### Generation based evalutaion

Use the model generate as the prediction.

```bash
$python3 tmlu_eval.py \
	--model [Model name on huggingface or path] \
	--dtype [torch type for the model, default will follow the model config] \
	--temperature [temperature for generation] \
	--max_tokens [max new tokens to generate] \
	--subsets [Choose subsets of TMLU (names splited by comma) for evalutaion. Default is 'ALL']  \
	--tensor_parallel_size [Tensor parallel size for vLLM] \
	--log_dir [Directory for saving evaluation log]
```

#### Example

Command

```bash
$python3 tmlu_eval.py \
	--model yentinglin/Taiwan-LLM-7B-v2.1-chat \
	--dtype float16 \
	--temperature 0.0 \
	--max_tokens 128 \
	--subsets AST_chinese,AST_mathematics \
	--tensor_parallel_size 1 \
	--log_dir log/gen_based/Taiwan-LLM-7B-v2.1-chat
```

## API based model

For API based model, we now only supprt using generation based evaluation.

#### before start

Add a config.ini under this directory in following format.

```
[Model_NAME]
api_key=...
```

### Generation based evalutaion

```bash
$python3 tmlu_eval.py \
	--model [Model name, same as that in the config.ini] \
	--base_url [base url for the API. Default is None (use OpenAI API)]
	--temperature [temperature for generation] \
	--max_tokens [max new tokens to generate] \
	--subsets [Choose subsets of TMLU (names splited by comma) for evalutaion. Default is 'ALL'] \
	--log_dir [Directory for saving evaluation log]
```

#### Example

Config:

```ini
[yentinglin/Taiwan-LLM-MoE-alpha]
api_key=...

[gpt-4-1106-preview]
api_key=...

[claude-instant-1.2]
api_key=...
```

Command:

```bash
$python3 tmlu_eval.py \
	--model yentinglin/Taiwan-LLM-MoE-alpha \
	--base_url http://127.0.0.1:8888
	--temperature 0.0 \
	--max_tokens 128 \
	--subsets AST_chinese,AST_mathematics \
	--log_dir log/gen_based/Taiwan-LLM-MoE-alpha
```

