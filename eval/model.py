import abc
from abc import abstractmethod
from typing import List
from vllm import LLM, SamplingParams
from openai import OpenAI
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from tqdm import tqdm

class LM(abc.ABC):

    def __init__(self, max_tokens, temperature):
        self.max_tokens = max_tokens
        self.temperature = temperature

    @abstractmethod
    def generate(self, prompts):
        pass

class HFLM(LM):
    def __init__(self, model_name, tensor_parallel_size, max_tokens, temperature):
        super().__init__(max_tokens, temperature)
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_num_batched_tokens=40960,
            quantization="AWQ" if "awq" in model_name.lower() else None,
        )
        self.sampling_params = SamplingParams(
            temperature=self.temperature, max_tokens=self.max_tokens
        )

    def generate(self, prompts):
        outputs = self.llm.generate(prompts, self.sampling_params)
        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        answers = [outputs[i].outputs[0].text for i in range(len(outputs))]
        return answers
    
class OPENAI_LM(LM):
    def __init__(self, model_name, max_tokens, temperature):
        super().__init__(max_tokens, temperature)
        import configparser
        config = configparser.ConfigParser()
        config.read('config.ini')
        self.client = OpenAI(
            api_key=config["OpenAI"]["api_key"],
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
    
class ANTHROPIC_LM(LM):
    def __init__(self, model_name, max_tokens, temperature):
        super().__init__(max_tokens, temperature)
        import configparser
        config = configparser.ConfigParser()
        config.read('config.ini')
        self.client = Anthropic(
            api_key=config["Anthropic"]["api_key"],
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