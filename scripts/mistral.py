from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import os 
import time
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
from utils import CompareResultObject, calculate_uncertainty
import numpy as np


device = 'cuda'

def is_integer_string(s):
    return s.isdigit()


class MistralModelLocal:
    def __init__(self, params):
        self.model_name = params['model']
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, 
                                                            # cache_dir="models",
                                                            device_map=self.device, 
                                                            attn_implementation="flash_attention_2", 
                                                            torch_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                padding_side='left',
                ) #, cache_dir="models")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.A_ids = self.tokenizer.convert_tokens_to_ids(['A','▁A'])   # A: 330
        self.B_ids = self.tokenizer.convert_tokens_to_ids(['B','▁B'])   # B: 365
        self.C_ids = self.tokenizer.convert_tokens_to_ids(['C','▁C'])   # C: 
        self.score_ids = self.tokenizer.convert_tokens_to_ids(['1','2','3','4','5'])


    # def mistral_compare(self, msg)-> CompareResultObject:
    #     # model run locally
    #     sequence, output = self.local_model_chat_completion(msg)
    #     compare_result = self.extract_probs(sequence, output.logits)
    #     return compare_result
    
    def compare(self, prompts):
        '''
        prompts: [batch_size, seq_len]
        output: a list of compare_result_object, [batch_size]
        '''
        sequence, output = self.local_model_chat_completion(prompts)
        compare_results = []
        for idx in range(sequence.shape[0]):
            seq_logits = [logits[idx] for logits in output.logits]      # convert to [seq_len, vocab_size]
            compare_result = self.extract_probs(sequence[idx], seq_logits)
            compare_results.append(compare_result)
        return compare_results
    

    # def mistral_rate(self, msg):
    #     # model run locally
    #     sequence, output = self.local_model_chat_completion(msg)
    #     # print(sequence)
    #     # print(self.tokenizer.batch_decode(sequence))
    #     score, logprobs = self.extract_score(sequence, output.logits)
    #     return score, logprobs
    
    def rate_score(self, prompts):
        sequence, output = self.local_model_chat_completion(prompts)
        # print(output.logits)
        # return sequence, output.logits
        scores, logprobs = [], []
        for idx in range(sequence.shape[0]):
            seq_logits = [logits[idx] for logits in output.logits]      # convert to [seq_len, vocab_size]
            score, logprob = self.extract_score(sequence[idx], seq_logits)
            scores.append(score)
            logprobs.append(logprob)
        return scores, logprobs


    def extract_score(self, sequence, logits):
        '''
        sequence: [batch_size, seq_len]
        logits: seq_len x [batch_size, vocab_size]
        output: int score
        '''
        for idx, token_id in enumerate(sequence):
            logit = logits[idx]
            logprobs = F.log_softmax(logit, dim=-1).cpu()
            score_logprobs = logprobs[self.score_ids].tolist()
            token = self.tokenizer.decode(token_id)
            if is_integer_string(token):
                return int(token), score_logprobs
        print("Failed to extract score")
        print(self.tokenizer.batch_decode(sequence))
        return 3, [np.log(0.2)]*5
        # only string in the response:

        
    def extract_probs(self, sequence, logits)-> CompareResultObject:
        '''
        sequence: [batch_size, seq_len]
        logits: seq_len x [batch_size, vocab_size]
        output: compare_result_object
        '''
        # First token logit 
        # print(self.tokenizer.batch_decode(sequence))
        for idx, token_id in enumerate(sequence):
            if token_id in self.A_ids or token_id in self.B_ids:
                logit = logits[idx]
                probs = F.softmax(logit, dim=-1)
                prob_A = sum([probs[a_id].item() for a_id in self.A_ids])
                prob_B = sum([probs[b_id].item() for b_id in self.B_ids])
                prob_C = sum([probs[c_id].item() for c_id in self.C_ids])
                # print(sequence)
                # print('raw prob_A: ', prob_A, 'raw prob_B: ', prob_B)
                # prob_A, prob_B = prob_A/(prob_A+prob_B), prob_B/(prob_A+prob_B)
                uncertainty = calculate_uncertainty([prob_A, prob_B])
                compare_result = CompareResultObject(raw_prob_A=prob_A, raw_prob_B=prob_B, raw_prob_C=prob_C, uncertainty=uncertainty)
                return compare_result
        print("Failed to extract probs")
        print(self.tokenizer.batch_decode(sequence))
        return CompareResultObject(raw_prob_A=0.5, raw_prob_B=0.5, uncertainty=1)


    # def local_model_chat_completion(self, msg):
    #     msg_encoded = self.tokenizer.apply_chat_template(msg, return_tensors="pt", return_dict=True)
    #     model_inputs = msg_encoded.to(self.device)
    #     output = self.model.generate(**model_inputs, 
    #                             max_new_tokens=64, 
    #                             pad_token_id=self.tokenizer.eos_token_id, 
    #                             do_sample=False, 
    #                             return_dict_in_generate=True, 
    #                             output_logits=True)
    #     newly_generated_tokens = output.sequences[:,model_inputs.input_ids.shape[1]:]
    #     return newly_generated_tokens, output


    def local_model_chat_completion(self, prompts, num_samples=1):
        # if num_samples>1:
        #     prompts = [prompts]*num_samples
        messages = []
        for prompt in prompts:
            msg = MistralModelLocal.get_chat_message(prompt)
            msg = self.tokenizer.apply_chat_template(msg, tokenize=False)# return_tensors="pt", return_dict=True)
            messages.append(msg)

        input = self.tokenizer(messages, return_tensors="pt", padding=True)
        input = input.to(device)
        output = self.model.generate(
                    **input,
                    return_dict_in_generate=True,
                    pad_token_id=self.tokenizer.eos_token_id, 
                    output_logits=True,
                    max_new_tokens=32,
                    do_sample=False,
                    temperature=None,
                    top_p=None
                )

        newly_generated_tokens = output.sequences[:, input.input_ids.shape[-1]:]
        return newly_generated_tokens, output
    

    @staticmethod
    def get_chat_message(prompt):
            return [{'role': 'user', 'content': prompt}]

        



class MistralModel:
    def __init__(self, params, api_key=None):
        self.model_name = params['model']
        self.device = device
        if not api_key:
            self.api_key = os.environ.get('MISTRAL_API_KEY')       
        else:
            self.api_key = api_key 

    def mistral_compare(self, msg):
        # model run through api
        api_params = {
            'model': self.model_name,
            'temperature': 0.0,
            'max_tokens': 32,
            'attempt_num': 10,
        }
        chat_response = MistralModel.call_mistral_chat_completion(msg, api_params, api_key=self.api_key)
        # print(chat_response)
        compare_result = self.extract_choice(chat_response)
        return compare_result


    def extract_choice(self, response) -> CompareResultObject:
        '''
        response: decoded str.
        output: compare_result_object
        '''
        # only string in the response:
        choice = MistralModel.first_appears_first(response, 'A', 'B')
        if choice not in ['A', 'B']:
            print('Failed to extract choice')
            return CompareResultObject(raw_prob_A=0.5, raw_prob_B=0.5, uncertainty=1)
        return CompareResultObject(
            raw_prob_A=float(choice=='A'), 
            raw_prob_B=float(choice=='B'),
            )
        
        
    @staticmethod
    def first_appears_first(string, char1, char2):
        index1 = string.find(char1)
        index2 = string.find(char2)
        if index1 == -1 and index2 == -1:
            return None  # Neither character appears in the string
        elif index1 == -1:
            return char2  # Only char2 appears in the string
        elif index2 == -1:
            return char1  # Only char1 appears in the string
        elif index1 < index2:
            return char1
        else:
            return char2

    @staticmethod
    def call_mistral_chat_completion(msg, api_params, api_key=None):
        if 'model' not in api_params:
            api_params['model'] = "mistral-large-latest"
        if 'temperature' not in api_params:
            api_params['temperature'] = 0.0
        if 'max_tokens' not in api_params:
            api_params['max_tokens'] = 32
        if 'attempt_num' not in api_params:
            api_params['attempt_num'] = 10
        if not api_key:
            api_key = os.environ.get('MISTRAL_API_KEY')

        client = MistralClient(api_key=api_key)

        attempt = 0
        while attempt<api_params['attempt_num']:
            try:
                chat_response = client.chat(
                    model=api_params['model'],
                    messages=msg,
                    temperature=api_params['temperature'],
                    max_tokens=api_params['max_tokens'],
                )
                # For now the api can only return response string.
                return chat_response.choices[0].message.content
            except Exception as e:
                print(e)
                attempt += 1
                api_params['temperature'] += 0.1
                time.sleep(0.2)
        # Fail cases
        print('Fail case: Default randomly selection.')
        return random.choice(['A', 'B'])

    @staticmethod
    def get_mistral_chat_message(prompt, aspect, with_input=False, eval_method='pairwise comparison'):
        if eval_method=='score':
            messages = [
                {"role": "user", "content": prompt},
            ]
            return messages
        else:
            messages = [
                {"role": "user", "content": prompt},
            ]
            return messages
        

