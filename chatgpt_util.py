from __future__ import annotations
import openai
import os 
from typing import List, Dict
import numpy as np
import backoff 
import json


class ChatGPTUtil:
    
    # make function wrapper to set api key
    def set_api_key(func):
        def wrapper(*args, **kwargs):
            openai.api_key = os.getenv('OPENAI_API_KEY')
            return func(*args, **kwargs)
        return wrapper

    @classmethod
    @backoff.on_exception(backoff.expo, openai.error.RateLimitError)
    @set_api_key
    def get_text_completion(cls, prompt: str, engine: str = 'ada', max_tokens: int=1000, samples: int=1) -> List[str]:
        response = openai.Completion.create(
            engine=engine,
            prompt=prompt,
            max_tokens=max_tokens,
            n=samples)
        if 'choices' not in response:
            raise ValueError(f'No choices in response: {response}')
        return [choice['text'] for choice in response['choices']]
    
    @classmethod
    @backoff.on_exception(backoff.expo, openai.error.RateLimitError)
    @set_api_key
    def get_chat_completion(cls, messages: List[Dict[str, str]], model: str='gpt-3.5-turbo', samples: int=1) -> List[str]:
        """ Messages should be formatted as outlined in:
        https://platform.openai.com/docs/guides/chat """

        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            n=samples)
        
        if 'choices' not in response:
            raise ValueError(f'No choices in response: {response}')
        
        responses: List[str] = []
        for choice in response['choices']:
            if choice['finish_reason'] != 'stop':
                raise ValueError(f'Choice did not finish with stop: {choice}')
            responses.append(choice['message']['content'])
        return responses
    
    @classmethod        
    @backoff.on_exception(backoff.constant, openai.error.RateLimitError, interval=30, jitter=None)
    @set_api_key
    def get_text_embedding(cls, input: str, model: str = 'text-embedding-ada-002') -> np.ndarray:
        response = openai.Embedding.create(
            model=model,
            input=input)
        if 'data' not in response:
            raise ValueError(f'No embedding in response: {response}')
        elif len(response['data']) != 1:
            raise ValueError(f'More than one embedding in response: {response}')
        elif 'embedding' not in response['data'][0]:
            raise ValueError(f'No embedding in response: {response}')
        return np.array(response['data'][0]['embedding'])
       
    @classmethod
    def transform_data_for_finetuning(cls, X: np.ndarray, Y: np.ndarray) -> List[Dict[str, str]]:
       
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f'X and y have different number of rows: {X.shape[0]} and {Y.shape[0]}')
        if Y.shape[1] != 1:
            raise ValueError(f'y has more than one column: {Y.shape[1]}')

        return [
            {"prompt": str(X[i]), "completion": str(Y[i])} 
            for i in range(len(X))]
        
    @classmethod
    def save_data_for_finetuning(cls, data: List[Dict[str, str]], file_path: str):
        with open(file_path, 'w') as f:
            for d in data:
                f.write(json.dumps(d) + '\n')
        
    @classmethod
    def finetune_model(cls, data_file_path: str, validation_data_file_path: str, model_name: str="ada", suffix: str=""):
        os.system(f"""
                  openai api fine_tunes.create -t {data_file_path} 
                    -v {validation_data_file_path} 
                    -m {model_name} 
                    --suffix {suffix}""")
