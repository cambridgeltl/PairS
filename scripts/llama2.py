from transformers import LlamaForCausalLM, AutoTokenizer
import torch
import numpy as np
import torch.nn.functional as F
from utils import CompareResultObject, calculate_uncertainty


device = 'cuda'
def is_integer_string(s):
    return s.isdigit()

class Llama2ModelLocal:
    def __init__(self, params):
        self.model_name = params['model']
        self.device = device
        if 'cache_dir' not in params: params['cache_dir'] = None
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            padding_side='left',
            cache_dir=params['cache_dir']
            )   # base_model
        self.tokenizer.pad_token = self.tokenizer.eos_token
        load_in_8bit = True if '13' in self.model_name else False
        self.model = LlamaForCausalLM.from_pretrained(
            self.model_name,
            load_in_8bit=load_in_8bit,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            cache_dir=params['cache_dir'],
            attn_implementation="flash_attention_2", 
        )
        self.model.eval()
        self.A_ids = self.tokenizer.convert_tokens_to_ids(['A','▁A'])   # A: 330
        self.B_ids = self.tokenizer.convert_tokens_to_ids(['B','▁B'])   # B: 365
        self.C_ids = self.tokenizer.convert_tokens_to_ids(['C','▁C'])   # C: 
        self.score_ids = self.tokenizer.convert_tokens_to_ids(['1','2','3','4','5'])



    def rate_score(self, prompts):
        sequence, output = self.local_model_chat_completion(prompts)
        # print(self.tokenizer.batch_decode(sequence))
        # score, logprobs = self.extract_score(sequence, output.logits)
        scores, logprobs = [], []
        for idx in range(sequence.shape[0]):
            seq_logits = [logits[idx] for logits in output.logits]      # convert to [seq_len, vocab_size]
            score, logprob = self.extract_score(sequence[idx], seq_logits)
            scores.append(score)
            logprobs.append(logprob)
        return scores, logprobs


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


    def extract_probs(self, sequence, logits)-> CompareResultObject:
        '''
        sequence: [batch_size, seq_len]
        logits: seq_len x [batch_size, vocab_size]
        output: compare_result_object
        '''
        # First token logit 
        # print(self.tokenizer.batch_decode(sequence))
        # print(self.tokenizer.batch_decode(sequence))
        for idx, token_id in enumerate(sequence):
            if token_id in self.A_ids or token_id in self.B_ids:
                logit = logits[idx]
                probs = F.softmax(logit, dim=-1)
                prob_A = sum([probs[a_id].item() for a_id in self.A_ids])
                prob_B = sum([probs[b_id].item() for b_id in self.B_ids])
                prob_C = sum([probs[c_id].item() for c_id in self.C_ids])
                uncertainty = calculate_uncertainty([prob_A, prob_B])
                compare_result = CompareResultObject(raw_prob_A=prob_A, raw_prob_B=prob_B, raw_prob_C=prob_C, uncertainty=uncertainty)
                return compare_result
        print("Failed to extract probs")
        print(self.tokenizer.batch_decode([sequence]))
        return CompareResultObject(raw_prob_A=0.5, raw_prob_B=0.5, uncertainty=1)



    # def local_model_chat_completion(self, prompt):
    #     msg = Llama2ModelLocal.get_chat_message(prompt)
    #     input = self.tokenizer.apply_chat_template(msg, return_tensors="pt", return_dict=True)
    #     # input = self.tokenizer.apply_chat_template(msg,  tokenize=False)
    #     # input = self.tokenizer(input, return_tensors="pt", return_dict=True)
    #     input = input.to(device)
        
    #     output = self.model.generate(
    #                 inputs=input.input_ids,
    #                 return_dict_in_generate=True,
    #                 output_logits=True,
    #                 max_new_tokens=32,
    #                 do_sample=False,
    #                 temperature=None,
    #                 top_p=None
    #             )

    #     newly_generated_tokens = output.sequences[:,input.input_ids.shape[1]:]
    #     return newly_generated_tokens, output

    def local_model_chat_completion(self, prompts, chat_system_instruction=None, num_samples=1):
        # if num_samples>1:
        #     prompts = [prompts]*num_samples
        messages = []
        for prompt in prompts:
            msg = Llama2ModelLocal.get_chat_message(prompt, chat_system_instruction)
            msg = self.tokenizer.apply_chat_template(msg, tokenize=False)# return_tensors="pt", return_dict=True)
            messages.append(msg)

        input = self.tokenizer(messages, return_tensors="pt", padding=True)
        input = input.to(device)
        output = self.model.generate(
                    **input,
                    return_dict_in_generate=True,
                    output_logits=True,
                    max_new_tokens=32,
                    do_sample=False,
                    temperature=None,
                )

        newly_generated_tokens = output.sequences[:, input.input_ids.shape[-1]:]
        return newly_generated_tokens, output

    
    @staticmethod
    def get_chat_message(prompt, chat_system_instruction=None):
        if chat_system_instruction:
            message = [
                {'role': 'system', 'content': chat_system_instruction},
                {'role': 'user', 'content': prompt},
            ]
        else:
            message = [{'role': 'user', 'content': prompt}]
        return message


if __name__=='__main__':

    params = {
        'model': "meta-llama/Llama-2-7b-chat-hf",
        'cache_dir': None,
        'eval_size': 50,
        'template': 'score',
        'aspect': 'coherence',
        'dataset': 'SumEval',
        'with_input': True,
    }

    prompt = "The quick brown fox jumps over the lazy dog."
    model = Llama2ModelLocal(params)

    score, logprobs = model.rate_score(prompt)
