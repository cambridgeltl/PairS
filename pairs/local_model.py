
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
from pairs import CompareResultObject, calculate_uncertainty


device = 'cuda'
class LocalModel:
    def __init__(self, params):
        self.model_name = params['model']
        self.temperature = params['temperature'] if 'temperature' in params else 0
        self.max_tokens = params['max_tokens'] if 'max_tokens' in params else 64
        self.do_sample = params['do_sample'] if 'do_sample' in params else False
        self.top_p = params['top_p'] if 'top_p' in params else 1
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, 
                                                            device_map=self.device, 
                                                            # attn_implementation="flash_attention_2",   # flash attention is not easy to install
                                                            torch_dtype=torch.bfloat16)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            padding_side='left',
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        if 'mistral' in self.model_name or 'Llama-2' in self.model_name or \
           'vicuna' in self.model_name or 'zephyr' in self.model_name or \
           'Phi' in self.model_name or 'gemma' in self.model_name:
            self.A_ids = self.tokenizer.convert_tokens_to_ids(['A','▁A'])   # A: 330
            self.B_ids = self.tokenizer.convert_tokens_to_ids(['B','▁B'])   # B: 365
            self.C_ids = self.tokenizer.convert_tokens_to_ids(['C','▁C'])   # C: 
        elif 'Llama-3' in self.model_name:
            self.A_ids = self.tokenizer.convert_tokens_to_ids(['A','ĠA'])   
            self.B_ids = self.tokenizer.convert_tokens_to_ids(['B','ĠB'])   
            self.C_ids = self.tokenizer.convert_tokens_to_ids(['C','ĠC'])  
        self.score_ids = self.tokenizer.convert_tokens_to_ids(['1','2','3','4','5'])


    def compare(self, prompts):
        sequence, output = self.local_model_chat_completion(prompts)
        compare_results = []
        for idx in range(sequence.shape[0]):
            seq_logits = [logits[idx] for logits in output.logits]      # convert to [seq_len, vocab_size]
            compare_result = self.extract_probs(sequence[idx], seq_logits)
            # print(compare_result.prob_A)
            compare_results.append(compare_result)
        return compare_results
        
        
    def extract_probs(self, sequence, logits)-> CompareResultObject:
        '''
        sequence: [batch_size, seq_len]
        logits: seq_len x [batch_size, vocab_size]
        output: compare_result_object
        '''
        # First token logit 
        for idx, token_id in enumerate(sequence):
            if token_id in self.A_ids or token_id in self.B_ids:
                logit = logits[idx]
                probs = F.softmax(logit, dim=-1)
                prob_A = max([probs[a_id].item() for a_id in self.A_ids])
                prob_B = max([probs[b_id].item() for b_id in self.B_ids])
                prob_C = max([probs[c_id].item() for c_id in self.C_ids])

                uncertainty = calculate_uncertainty([prob_A, prob_B])
                compare_result = CompareResultObject(raw_prob_A=prob_A, raw_prob_B=prob_B, raw_prob_C=prob_C, uncertainty=uncertainty)
                return compare_result
        print("Failed to extract probs")
        print(self.tokenizer.decode(sequence))
        return CompareResultObject(raw_prob_A=0.5, raw_prob_B=0.5, uncertainty=1)


    def local_model_chat_completion(self, prompts):
        if 'vicuna' in self.model_name:
            input = self.tokenizer(prompts, return_tensors="pt", padding=True)
        else:
            messages = []
            for prompt in prompts:
                msg = LocalModel.get_chat_message(prompt)
                msg = self.tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)# return_tensors="pt", return_dict=True)
                messages.append(msg)

            input = self.tokenizer(messages, return_tensors="pt", padding=True)
        if self.model_name == 'google/gemma-2-9b-it':
            input.pop('attention_mask')
        input = input.to(device)
        output = self.model.generate(
                    # input_ids=input.input_ids,
                    # attention_mask=input.attention_mask,
                    **input,
                    return_dict_in_generate=True,
                    pad_token_id=self.tokenizer.eos_token_id, 
                    output_logits=True,
                    max_new_tokens=self.max_tokens,
                    do_sample=self.do_sample,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )

        newly_generated_tokens = output.sequences[:, input.input_ids.shape[-1]:]
        return newly_generated_tokens, output
    

    @staticmethod
    def get_chat_message(prompt, chat_system_instruction=None):
        if chat_system_instruction:
            message = [
                # {'role': 'assistant', 'content': chat_system_instruction},
                {'role': 'user', 'content': prompt},
            ]
        else:
            message = [{'role': 'user', 'content': prompt}]
        return message

