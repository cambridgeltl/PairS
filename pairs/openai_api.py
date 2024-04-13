import os
import time
from openai import OpenAI
from .utils import CompareResultObject, calculate_uncertainty
import numpy as np

openai_api_key = os.environ.get("OPENAI_API_KEY")


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
        print(response.choices[0])  
        return CompareResultObject(raw_prob_A=0.5, raw_prob_B=0.5, uncertainty=1)
         