import os
import time
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import datetime
import numpy as np
from pairs import CompareResultObject, calculate_uncertainty


openai_api_key = os.environ.get("OPENAI_API_KEY")
base_url = os.environ.get("BASE_URL")
engine = os.environ.get("ENGINE")

class Timer(object):
    def __init__(self):
        self.__start = time.time()

    def start(self):
        self.__start = time.time()

    def get_time(self, restart=True, format=False):
        end = time.time()
        span = end - self.__start
        if restart:
            self.__start = end
        if format:
            return self.format(span)
        else:
            return span

    def format(self, seconds):
        return datetime.timedelta(seconds=int(seconds))

    def print(self, name):
        print(name, self.get_time())


class OpenAIChatModel:
    def __init__(self, params={}, api_key=None):
        self.api_key = api_key
        if 'engine' not in params:
            params['engine'] = engine
        if 'temperature' not in params:
            params['temperature'] = 0
        if 'max_tokens' not in params:
            params['max_tokens'] = 128
        if 'logprobs' not in params:
            params['logprobs'] = True
        if 'top_logprobs' not in params:
            params['top_logprobs'] = 5
        if 'attempt_num' not in params:
            params['attempt_num'] = 10      
        if 'do_sample' not in params:
            params['do_sample'] = False
        if 'top_p' not in params:
            params['top_p'] = 1
        if 'chat_system_instruction' not in params:
            params['chat_system_instruction'] = None
        
        self.params = params
        if not api_key:
            api_key = os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(base_url=base_url, api_key = api_key)


    def compare(self, prompts, max_workers=4):
        result_list = self.multi_threading_openai_chat_completion(
                prompts, 
                self.single_call_compare, 
                max_workers=max_workers
            )
        return result_list


    def call_openai_chat_completion(self, prompt):
        if self.params['chat_system_instruction']:
            msg = [{'role': 'system', 'content': self.params['chat_system_instruction']}]
        else:
            msg = []
        msg.append({'role': 'user', 'content': prompt})

        attempt = 0
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.params['engine'],
                    messages=msg,
                    temperature=self.params['temperature'],
                    max_tokens=self.params['max_tokens'],
                    logprobs=self.params['logprobs'],
                    top_logprobs=self.params['top_logprobs'] if self.params['logprobs'] else None,
                )
                return response 

            except Exception as e:
                print(e)
                attempt += 1
                if attempt >= self.params['attempt_num']:
                    return None
                wait_sec = 1
                time.sleep(wait_sec)


    def multi_threading_openai_chat_completion(self, prompts, single_thread_func_handler, max_workers=4):
        inputs = [{'prompt': prompt} for prompt in prompts]
        timer = Timer()
        # print(f"using model_{self.params['engine']}")
        # print('Processing queires')
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = list(
                    executor.map(lambda x: single_thread_func_handler(x), inputs)
            )
        # print("Average time after {0} samples: {1}".format(len(prompts), timer.get_time(restart=False) / len(prompts)))
        # print('Processed queries')

        result_list = [input['result'] for input in inputs]
        return result_list
    

    def single_call_compare(self, input):        
        response = self.call_openai_chat_completion(input['prompt'])
        compare_result = self.extract_prob(response)
        input['result'] = compare_result


    def extract_prob(self, response) -> CompareResultObject:
        '''For OpenAI models'''
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
                        uncertainty=calculate_uncertainty(logprobs),
                    )
                return comparison_result

        print('Fail case')
        return CompareResultObject(raw_prob_A=0.5, raw_prob_B=0.5, uncertainty=1)
