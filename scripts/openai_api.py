import os
import time
from typing import List, Dict
from openai import OpenAI
import openai
import json
from jinja2 import Environment
from textwrap import dedent
from tqdm import tqdm
import math
from concurrent.futures import ThreadPoolExecutor
import concurrent
import threading
import datetime
from utils import CompareResultObject, calculate_entropy
import numpy as np

openai_api_key = os.environ.get("OPENAI_API_KEY")


# class Timer(object):
#     def __init__(self):
#         self.__start = time.time()

#     def start(self):
#         self.__start = time.time()

#     def get_time(self, restart=True, format=False):
#         end = time.time()
#         span = end - self.__start
#         if restart:
#             self.__start = end
#         if format:
#             return self.format(span)
#         else:
#             return span

#     def format(self, seconds):
#         return datetime.timedelta(seconds=int(seconds))

#     def print(self, name):
#         print(name, self.get_time())


# class OpenAIRequestManager:
#     def __init__(self, response_extractor, api_params={}, api_key=None):
#         if 'engine' not in api_params:
#             api_params['engine'] = 'gpt-3.5-turbo'
#         if 'temperature' not in api_params:
#             api_params['temperature'] = 0.2
#         if 'max_tokens' not in api_params:
#             api_params['max_tokens'] = 128
#         if 'logprobs' not in api_params:
#             api_params['logprobs'] = False
#         if 'top_logprobs' not in api_params:
#             api_params['top_logprobs'] = 5
#         if 'attempt_num' not in api_params:
#             api_params['attempt_num'] = 10        
#         if 'buffer_path' not in api_params:
#             api_params['buffer_path'] = './temp_buffer.jsonl'
#         with open(api_params['buffer_path'], 'w') as f:
#             pass
#         self.response_extractor = response_extractor
#         self.outbuf = open(api_params['buffer_path'], 'a')
#         self.lock = threading.Lock()
#         self.api_params = api_params
#         if api_key:
#             self.client = OpenAI(
#                 api_key=api_key
#             )
#         else:
#             self.client = OpenAI()


#     def write_result(self, result):
#         self.lock.acquire()
#         with open(self.api_params['buffer_path'], 'a') as f:
#             f.write(json.dumps(result, ensure_ascii=False) + '\n')
#         self.outbuf.flush()
#         self.lock.release()
#         # self.outbuf.write(json.dumps(result, ensure_ascii=False) + '\n')
#         # self.outbuf.flush()
#         # self.lock.release()

#     def openai_api_call(self, prompt):
#         id, prompt = prompt
#         messages = [
#             {"role": "system", "content": "You are a helpful AI assistant."},
#             {"role": "user", "content": prompt},
#         ]
#         attempt = 0
#         wait_sec = 0.5
#         while True:
#             try:
#                 response = self.client.chat.completions.create(
#                     model=self.api_params['engine'],
#                     messages=messages,
#                     temperature=self.api_params['temperature'],
#                     max_tokens=self.api_params['max_tokens'],
#                     logprobs=self.api_params['logprobs'],
#                     top_logprobs=self.api_params['top_logprobs'] if self.api_params['logprobs'] else None,
#                 )
#                 result = self.response_extractor(response)
#                 result['id'] = id
#                 self.write_result(result)
#                 break
#             except Exception as e:
#                 print(e)
#                 # if response:
#                 #     print(response.choices[0].message.content.strip())
#                 attempt += 1
#                 if attempt >= self.api_params['attempt_num']:
#                     return None
#                 time.sleep(wait_sec*attempt)

#     def multi_threading_openai_api_call(self, prompts, max_workers=64):
#         timer = Timer()
#         print(f"using model_{self.api_params['engine']}")
#         print('Processing queires')
#         with ThreadPoolExecutor(max_workers=max_workers) as executor:
#             futures = list(
#                 tqdm(
#                     executor.map(self.openai_api_call, enumerate(prompts)), 
#                     total=len(prompts)
#                 )
#             )
#             # futures = [executor.map(self.openai_api_call, idx, prompt) for idx, prompt in enumerate(prompts, start=0)]

#             # for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
#             #     pass
#             # executor.shutdown(wait=True)
#       # for future in futures:
#         #     try:
#         #         result = future.result()
#         #         print(f'Task result: {result}')
#         #     except Exception as e:
#         #         print(f'Task failed: {e}')
#         # for future in futures:
#         #     future.result() 
#         # time.sleep(5)
#         print("Average time after {0} samples: {1}".format(len(prompts), timer.get_time(restart=False) / len(prompts)))
#         print('Processed queries')
#         return futures


def call_openai_chat_completion(prompt, api_params, api_key=None):
    if 'engine' not in api_params:
        api_params['engine'] = 'gpt-3.5-turbo'
    if 'temperature' not in api_params:
        api_params['temperature'] = 0.2
    if 'max_tokens' not in api_params:
        api_params['max_tokens'] = 128
    if 'logprobs' not in api_params:
        api_params['logprobs'] = False
    if 'top_logprobs' not in api_params:
        api_params['top_logprobs'] = 5
    if 'attempt_num' not in api_params:
        api_params['attempt_num'] = 10      
    if api_key:
        client = OpenAI(
            api_key=api_key
        )
    else:
        client = OpenAI(
            api_key=openai_api_key
        )

    msg = [
        {'role': 'system',
        'content': "You are a helpful assistant. Please follow the user's instructions."},
        {'role': 'user', 'content': prompt},
    ]
    attempt = 0
    while True:
        try:
            if 'davinci' in api_params['engine']:
                # text completion model
                response = client.completions.create(
                    model=api_params['engine'],
                    prompt=prompt,
                    temperature=api_params['temperature'],
                    max_tokens=api_params['max_tokens'],
                    logprobs=api_params['top_logprobs'] if api_params['logprobs'] else None,
                    echo=True,
                )
            else:
                # chat model
                response = client.chat.completions.create(
                    model=api_params['engine'],
                    messages=msg,
                    temperature=api_params['temperature'],
                    max_tokens=api_params['max_tokens'],
                    logprobs=api_params['logprobs'],
                    top_logprobs=api_params['top_logprobs'] if api_params['logprobs'] else None,
                )
            return response 

        except Exception as e:
            print(e)
            print(response)
            print(response.choices[0].message.content.strip())
            attempt += 1
            if attempt >= api_params['attempt_num']:
                return None
            wait_sec = 0.1
            time.sleep(wait_sec)



def extract_prob(response, api_params) -> CompareResultObject:
    '''For OpenAI models'''
    if 'instruct' in api_params['engine']:
        # for text completion model
        for idx, token in enumerate(response.choices[0].logprobs.tokens):
            if token in ['A','B']:
                token_prob_candidate = response.choices[0].logprobs.top_logprobs[idx]
                prob_A = np.exp(token_prob_candidate['A']) if 'A' in token_prob_candidate else 0
                prob_B = np.exp(token_prob_candidate['B']) if 'B' in token_prob_candidate else 0
                break
    else: # for chat model
        # if params['eval_method'] == "pairwise with tie":

        prob_A, prob_B, prob_C = 0, 0, 0   
        for token_object in response.choices[0].logprobs.content:
            logprobs = []
            for token_candidate in token_object.top_logprobs:
                logprobs.append(token_candidate.logprob)
                if prob_A==0 and token_candidate.token.strip() == 'A':
                    prob_A = np.exp(token_candidate.logprob)
                elif prob_B==0 and token_candidate.token.strip() == 'B':
                    prob_B = np.exp(token_candidate.logprob)
                elif prob_C==0 and token_candidate.token.strip() == 'C':
                    prob_C = np.exp(token_candidate.logprob)
            if prob_A!=0 or prob_B!=0 or prob_C!=0:
                comparison_result = CompareResultObject(
                        raw_prob_A=prob_A,
                        raw_prob_B=prob_B,
                        raw_prob_C=prob_C,
                        uncertainty=calculate_entropy(logprobs),
                    )
                return comparison_result
                # prob_A, prob_B, prob_C = prob_A/(prob_A+prob_B+prob_C), prob_B/(prob_A+prob_B+prob_C), prob_C/(prob_A+prob_B+prob_C)
                    #= prob_A/(prob_A+prob_B), prob_B/(prob_A+prob_B)
                # return {'prob_A':prob_A, 'prob_B':prob_B, 'prob_C':prob_C, 'uncertainty':calculate_entropy(logprobs)}
        print('Fail case')
        print(response.choices[0])  
        return CompareResultObject(raw_prob_A=0.5, raw_prob_B=0.5, uncertainty=1)
         