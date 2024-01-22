import abc
from abc import abstractmethod
from typing import List, Optional, Union
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from openai import OpenAI
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from tqdm import tqdm
import transformers
import torch
import torch.nn.functional as F

def _get_dtype(
    dtype: Union[str, torch.dtype], config: Optional[transformers.AutoConfig] = None
) -> torch.dtype:
    """From https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/models/huggingface.py"""
    if dtype is None and config is not None:
        _torch_dtype = config.torch_dtype
    elif isinstance(dtype, str) and dtype != "auto":
        _torch_dtype = getattr(torch, dtype)
    else:
        _torch_dtype = dtype
    return _torch_dtype

class LM(abc.ABC):

    def __init__(self, max_tokens, temperature):
        self.max_tokens = max_tokens
        self.temperature = temperature

    @abstractmethod
    def generate(self, prompts):
        pass

class HFLM_vLLM(LM):
    def __init__(
        self,
        model_name,
        tensor_parallel_size,
        max_tokens,
        temperature,
        revision=None,
        dtype=None
    ):
        super().__init__(max_tokens, temperature)
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_num_batched_tokens=40960,
            quantization="AWQ" if "awq" in model_name.lower() else None,
            revision=revision,
            dtype=dtype
        )
        self.sampling_params = SamplingParams(
            temperature=self.temperature, max_tokens=self.max_tokens
        )

    def generate(self, prompts):
        outputs = self.llm.generate(prompts, self.sampling_params)
        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        answers = [outputs[i].outputs[0].text for i in range(len(outputs))]
        return answers

class HFLM_transformers(LM):
    def __init__(
        self,
        model_name,
        max_tokens,
        temperature,
        revision=None,
        dtype=None
    ):
        super().__init__(max_tokens, temperature)
        self.config = AutoConfig.from_pretrained(
            model_name,
            revision=revision
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision
        )
        self.tokenizer.truncation_side = 'left'
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype=_get_dtype(dtype, self.config)
        ).cuda()
        self.llm.eval()
    
    def get_tokenizer(self):
        return self.tokenizer

    def encode(self, text):
        encoded = self.tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=True,
            return_tensors="pt"
        )
        return encoded

    def encode_pair(self, context, continuation):
        whole_encoded = self.encode(context + continuation)
        context_encoded = self.encode(context)
        context_encoded_len = context_encoded.shape[1]
        continuation_encoded = whole_encoded[:, context_encoded_len:]
        return context_encoded, continuation_encoded
    
    def generate(self, prompts, choice_nums):
        with torch.no_grad():
            answers = []
            pgbr = tqdm(range(len(prompts)))
            for prompt, choice_num in zip(prompts, choice_nums):
                logits = []
                for i in range(choice_num):
                    choice = chr(i + ord('A'))
                    prompt_encoded, choice_encoded = self.encode_pair(
                        prompt, 
                        choice
                    )
                    choice_encoded_len = choice_encoded.shape[1]
                    logit = self.llm(prompt_encoded[:, :-1].cuda())["logits"]
                    logit = F.log_softmax(logit, dim=-1).cpu()
                    choice_logit = torch.gather(
                        logit[:, -choice_encoded_len:], 
                        2, 
                        choice_encoded.unsqueeze(-1)
                    ).squeeze(dim=-1)
                    logits.append(choice_logit.sum())

                logits = torch.stack(logits)
                answer = torch.argmax(logits)
                answers.append(chr(int(answer) + ord('A'))+")")
                pgbr.update()
        return answers

class OpenAI_LM(LM):
    def __init__(
            self, 
            model_name, 
            max_tokens, 
            temperature,
            api_key,
            base_url=None
        ):
        super().__init__(max_tokens, temperature)

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=20.0,
            max_retries=100
        )
        self.model = model_name

    def query(self, prompt, prefill=""):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": prefill.strip()},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        answer = response.choices[0].message.content
        return answer

    def generate(self, prompts, prefill=""):
        answers = []
        for prompt in tqdm(prompts):
            answer = self.query(prompt, prefill)
            answers.append(answer)
        return answers
    
class Anthropic_LM(LM):
    def __init__(
            self, 
            model_name, 
            max_tokens, 
            temperature,
            api_key,
        ):
        super().__init__(max_tokens, temperature)
        self.client = Anthropic(
            api_key=api_key,
            timeout=20.0,
            max_retries=100
        )
        self.model = model_name

    def query(self, prompt, prefill=""):
        response = self.client.completions.create(
            model=self.model,
            max_tokens_to_sample=self.max_tokens,
            temperature=self.temperature,
            prompt=f"{HUMAN_PROMPT} {prompt}{AI_PROMPT}{prefill.strip()}",
        )
        answer = response.completion
        return answer

    def generate(self, prompts, prefill=""):
        answers = []
        for prompt in tqdm(prompts):
            answer = self.query(prompt, prefill)
            answers.append(answer)
        return answers