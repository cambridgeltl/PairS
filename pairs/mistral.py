
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
from pairs import CompareResultObject, calculate_uncertainty
import numpy as np


device = 'cuda'

def is_integer_string(s):
    return s.isdigit()


class MistralModelLocal:
    def __init__(self, params):
        self.model_name = params['model']
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, 
                                                            device_map=self.device, 
                                                            # attn_implementation="flash_attention_2",   # flash attention is not easy to install
                                                            torch_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name) #, cache_dir="models")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.A_ids = self.tokenizer.convert_tokens_to_ids(['A','▁A'])   # A: 330
        self.B_ids = self.tokenizer.convert_tokens_to_ids(['B','▁B'])   # B: 365
        self.C_ids = self.tokenizer.convert_tokens_to_ids(['C','▁C'])   # C: 
        self.score_ids = self.tokenizer.convert_tokens_to_ids(['1','2','3','4','5'])


    def mistral_compare(self, msg)-> CompareResultObject:
        # model run locally
        sequence, output = self.local_model_chat_completion(msg)
        compare_result = self.extract_probs(sequence, output.logits)
        return compare_result
    

    # def mistral_rate(self, msg):
    #     # model run locally
    #     sequence, output = self.local_model_chat_completion(msg)
    #     # print(sequence)
    #     # print(self.tokenizer.batch_decode(sequence))
    #     score, logprobs = self.extract_score(sequence, output.logits)
    #     return score, logprobs


    def extract_score(self, sequence, logits):
        '''
        sequence: [batch_size, seq_len]
        logits: seq_len x [batch_size, vocab_size]
        output: int score
        '''
        for idx, token_id in enumerate(sequence[0]):
            logit = logits[idx][0]
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
        for idx, token_id in enumerate(sequence[0]):
            if token_id in self.A_ids or token_id in self.B_ids:
                logit = logits[idx]
                probs = F.softmax(logit, dim=-1)[0]
                prob_A = sum([probs[a_id].item() for a_id in self.A_ids])
                prob_B = sum([probs[b_id].item() for b_id in self.B_ids])
                prob_C = sum([probs[c_id].item() for c_id in self.C_ids])

                uncertainty = calculate_uncertainty([prob_A, prob_B])
                compare_result = CompareResultObject(raw_prob_A=prob_A, raw_prob_B=prob_B, raw_prob_C=prob_C, uncertainty=uncertainty)
                return compare_result
        print("Failed to extract probs")
        print(self.tokenizer.batch_decode(sequence))
        return CompareResultObject(raw_prob_A=0.5, raw_prob_B=0.5, uncertainty=1)


    def local_model_chat_completion(self, msg):
        msg_encoded = self.tokenizer.apply_chat_template(msg, return_tensors="pt", return_dict=True)
        model_inputs = msg_encoded.to(self.device)
        output = self.model.generate(**model_inputs, 
                                max_new_tokens=64, 
                                pad_token_id=self.tokenizer.eos_token_id, 
                                do_sample=False, 
                                return_dict_in_generate=True, 
                                output_logits=True)
        newly_generated_tokens = output.sequences[:,model_inputs.input_ids.shape[1]:]
        return newly_generated_tokens, output
    

    @staticmethod
    def get_mistral_chat_message(prompt, eval_method='pairwise comparison'):
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


