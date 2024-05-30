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
            params['engine'] = 'gpt-3.5-turbo'
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
        # if 'do_sample' not in params:
        #     params['do_sample'] = False
        if 'chat_system_instruction' not in params:
            params['chat_system_instruction'] = None
        
        self.params = params
        if not api_key:
            api_key = os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(api_key = api_key)


    def compare(self, prompts, max_workers=4):
        result_list = self.multi_threading_openai_chat_completion(
                prompts, 
                self.single_call_compare, 
                max_workers=max_workers
            )
        # result_list = [CompareResultObject.from_json(x['result']) for x in result_list]
        return result_list

    def rate_score(self, prompts, max_workers=4):
        result_list = self.multi_threading_openai_chat_completion(
                prompts, 
                self.single_call_rate_score, 
                max_workers=max_workers
            )
        return result_list, None

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
                print(response)
                attempt += 1
                if attempt >= self.params['attempt_num']:
                    return None
                wait_sec = 1
                time.sleep(wait_sec)


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
                        uncertainty=1,
                    )
                return comparison_result

        print('Fail case')
        print(response.choices[0])  
        return CompareResultObject(raw_prob_A=0.5, raw_prob_B=0.5, uncertainty=1)
    

    def extract_scores(self, response):
        response_string = response.choices[0].message.content
        scores = ['1','2','3','4','5']
        for idx, s in enumerate(scores):
            if s in response_string:
                return idx+1
        print('Fail case, return 3')
        return 3


    def multi_threading_openai_chat_completion(self, prompts, single_thread_func_handler, max_workers=4):
        inputs = [{'prompt': prompt} for prompt in prompts]
        timer = Timer()
        print(f"using model_{self.params['engine']}")
        print('Processing queires')
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = list(
                tqdm(
                    executor.map(lambda x: single_thread_func_handler(x), inputs), 
                    total=len(prompts)
                )
            )
        print("Average time after {0} samples: {1}".format(len(prompts), timer.get_time(restart=False) / len(prompts)))
        print('Processed queries')

        result_list = [input['result'] for input in inputs]
        return result_list


    def single_call_compare(self, input):        
        response = self.call_openai_chat_completion(input['prompt'])
        compare_result = self.extract_prob(response)
        input['result'] = compare_result


    def single_call_rate_score(self, input):
        response = self.call_openai_chat_completion(input['prompt'])
        score = self.extract_scores(response)
        input['result'] = score



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
         


if __name__ == '__main__':

    example_prompt = '''\
Evaluate and compare the coherence of the two following summary candidates for the given input source text.

Input source text: Paul Merson has restarted his row with Andros Townsend after the Tottenham midfielder was brought on with only seven minutes remaining in his team's 0-0 draw with Burnley on Sunday. 'Just been watching the game, did you miss the coach? #RubberDub #7minutes,' Merson put on Twitter. Merson initially angered Townsend for writing in his Sky Sports column that 'if Andros Townsend can get in (the England team) then it opens it up to anybody.' Paul Merson had another dig at Andros Townsend after his appearance for Tottenham against Burnley Townsend was brought on in the 83rd minute for Tottenham as they drew 0-0 against Burnley Andros Townsend scores England's equaliser in their 1-1 friendly draw with Italy in Turin on Tuesday night The former Arsenal man was proven wrong when Townsend hit a stunning equaliser for England against Italy and he duly admitted his mistake. 'It's not as though I was watching hoping he wouldn't score for England, I'm genuinely pleased for him and fair play to him – it was a great goal,' Merson said. 'It's just a matter of opinion, and my opinion was that he got pulled off after half an hour at Manchester United in front of Roy Hodgson, so he shouldn't have been in the squad. 'When I'm wrong, I hold my hands up. I don't have a problem with doing that - I'll always be the first to admit when I'm wrong.' Townsend hit back at Merson on Twitter after scoring for England against Italy Sky Sports pundit  Merson (centre) criticised Townsend's call-up to the England squad last week Townsend hit back at Merson after netting for England in Turin on Wednesday, saying 'Not bad for a player that should be 'nowhere near the squad' ay @PaulMerse?' Any bad feeling between the pair seemed to have passed but Merson was unable to resist having another dig at Townsend after Tottenham drew at Turf Moor.

Compare the following outputs:

Summary candidate A: paul merson was brought on with only seven minutes remaining in his team 's 0-0 draw with burnley . andros townsend scored the tottenham midfielder in the 89th minute . paul merson had another dig at andros townsend after his appearance . the midfielder had been brought on to the england squad last week . click here for all the latest arsenal news news .

Summary candidate B: paul merson has restarted his row with andros townsend . the tottenham midfielder was brought on with only seven minutes remaining in his team 's 0-0 draw with burnley . andros townsend scores england 's equaliser in their 1-1 friendly draw with italy in turin .

Question: Which summary candidate has better coherence? If the candidate A is better, please return 'A'. If the candidate B is better, please return 'B'. You must return the choice only.
Answer: \
'''

    prompts = [example_prompt] * 3 
    model = OpenAIChatModel({'engine':'gpt-3.5-turbo'})
    result = model.compare(prompts)
    print(result)
    print(result.prob_A)
    print(result.uncertainty)