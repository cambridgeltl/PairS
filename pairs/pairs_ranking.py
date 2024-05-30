from jinja2 import Environment
from pairs import get_general_prompt_template
import numpy as np
from pairs import Llama2ModelLocal, MistralModelLocal, call_openai_chat_completion, extract_prob
import copy
from tqdm import tqdm


class BeamItem:
    def __init__(self, index_pathway=[], cum_prob=1, pointer_A=-1, pointer_B=-1):
        self.index_pathway = index_pathway
        self.cum_prob = cum_prob
        self.pointer_A = pointer_A
        self.pointer_B = pointer_B

    def __str__(self):
        return f'index_pathway: {self.index_pathway}, cum_prob: {self.cum_prob}'
    

def moving_average(sum,val,idx):
    return sum*idx/(idx+1) + val/(idx+1)


def is_better_than_prob(id1, id2, input, output, params, api_key=None):    
    prompt_template = get_general_prompt_template(with_input=params['with_input'])
    environment = Environment()
    prompt_template = environment.from_string(prompt_template)

    prompt = prompt_template.render(
        input=input,
        output_1=output[id1],
        output_2=output[id2],
    )

    # print(prompt)
    # assert False
    api_params = {
    'engine': params['engine'], 
    'temperature':0, 
    'logprobs':True, 
    'top_logprobs':5, 
    'max_tokens': 16,
    'wait_sec': 0.3,
    'attempt_num': 10,
    }

    
    if 'mistral' in params['engine'] or 'mixtral' in params['engine']:
        msg = MistralModelLocal.get_mistral_chat_message(prompt)
        compare_result = params['model'].mistral_compare(msg)
        # if params['calibration']:
        #     calibration_shifts = get_calibration_shift(params['engine'], params['dataset'], params['aspect'])
        #     compare_result.calibraet_shift(calibration_shifts)
        return compare_result
    
    if 'llama' in params['engine']:
        compare_result = params['model'].compare(prompt)
        return compare_result
    
    else:  # for OpenAI model
        retry, max_retry = 0, 5
        while retry < max_retry:
            try:
                response = call_openai_chat_completion(prompt, api_params, api_key)
                compare_result = extract_prob(response, api_params)
                return compare_result      # prob_A is the probability of A being better than B
            except Exception as e:
                print(e)
                retry += 1
                if retry >= max_retry:
                    print('Fail case')
                    return True
                
                

def merge(indices, left, mid, right, input, output, params):
    left_copy = indices[left:mid]
    right_copy = indices[mid:right]

    i = 0
    j = 0
    k = left
    
    while i < len(left_copy) and j < len(right_copy):
        if 'progress_bar' in params: params['progress_bar'].update(1)
        compare_result = is_better_than_prob(left_copy[i], right_copy[j], input, output, params)
        # params['compare_log'][(left_copy[i], right_copy[j])] = compare_result.to_json() #compare_result['prob_A']
        params['api_call'] += 1
        # print(compare_result)
        if compare_result['prob_A']>compare_result['prob_B']:
            indices[k] = right_copy[j]
            j += 1
        else:
            indices[k] = left_copy[i]
            i += 1
        k += 1

    while i < len(left_copy):
        indices[k] = left_copy[i]
        i += 1
        k += 1

    while j < len(right_copy):
        indices[k] = right_copy[j]
        j += 1
        k += 1


def get_likelihood_coefficient(N, p):
    x = [0, (N-1)/2, N-1]
    # y = [1, np.log2(N/2), 1]
    y = [1, 1, 1]

    coefficients = np.polyfit(x, y, 2)  # Fit a 3rd-degree polynomial curve
    func = np.poly1d(coefficients)
    return func(p)


def merge_with_confidence_beam(indices, left, mid, right, input, output, params):
    def get_probA(i, j):
        if prob_A_matrix[i, j] == 0:
            compare_result = is_better_than_prob(left_copy[i], right_copy[j], input, output, params)
            # print(compare_result)
            prob_A_matrix[i,j] = compare_result['prob_A']
            # uncertainty_matrix[i,j] = compare_result['uncertainty']
            params['api_call'] += 1
            if 'progress_bar' in params: params['progress_bar'].update(1)
            # params['compare_log'][(left_copy[i], right_copy[j])] = compare_result.to_json()
        return prob_A_matrix[i, j]#, uncertainty_matrix[i,j]

    left_copy = indices[left:mid]
    right_copy = indices[mid:right]

    beam_size = params['beam_size']
    prob_A_matrix = np.zeros((len(left_copy), len(right_copy)))  # prob_A_matrix[i, j] is the probability of A better than B 
    # uncertainty_matrix = np.ones_like(prob_A_matrix)  # uncertainty_matrix[i, j] is the uncertainty of A is better
    prob_A_matrix[0, 0] = get_probA(0, 0)
    prob_gap = params['prob_gap']

    coef = get_likelihood_coefficient(right-left, 0)
    if prob_A_matrix[0, 0] > 0.5+prob_gap:
        beam = [
            BeamItem(index_pathway=[('B', 0)], cum_prob=np.log(prob_A_matrix[0, 0]+1e-9)*coef, pointer_B=0),
        ]
    elif prob_A_matrix[0, 0] < 0.5-prob_gap:
        beam = [
            BeamItem(index_pathway=[('A', 0)], cum_prob=np.log(1-prob_A_matrix[0, 0]+1e-9)*coef, pointer_A=0),
        ]
    else:
        beam = [
            BeamItem(index_pathway=[('B', 0)], cum_prob=np.log(prob_A_matrix[0, 0]+1e-9)*coef, pointer_B=0),
            BeamItem(index_pathway=[('A', 0)], cum_prob=np.log(1-prob_A_matrix[0, 0]+1e-9)*coef, pointer_A=0),
        ]

    for i in range(len(left_copy)+len(right_copy)-1):
        coef = np.round(get_likelihood_coefficient(right-left, i+1),5)
        new_beam = []
        for beam_item in beam:
            for choice in ['A', 'B']:
                beam_item_copy = copy.deepcopy(beam_item)
                if (beam_item_copy.pointer_A < len(left_copy)-1 and beam_item_copy.pointer_B < len(right_copy)-1) \
                    and not (i==len(left_copy)+len(right_copy)-2):
                    prob_A = get_probA(
                            min(beam_item_copy.pointer_A+1, len(left_copy)-1),
                            min(beam_item_copy.pointer_B+1, len(right_copy)-1), 
                        )
                    if (choice == 'A' and prob_A>0.5+prob_gap) or (choice == 'B' and 1-prob_A > 0.5+prob_gap):
                        continue
                    # beam_item_copy.cum_prob *= 1-prob_A if choice == 'A' else prob_A
                    logprob = np.log(1-prob_A+1e-9) if choice == 'A' else np.log(prob_A+1e-9)
                    beam_item_copy.cum_prob = moving_average(beam_item_copy.cum_prob, logprob*coef, i+1)

                beam_item_copy.pointer_A += 1 if choice == 'A' else 0
                beam_item_copy.pointer_B += 1 if choice == 'B' else 0

                if (beam_item_copy.pointer_A >= len(left_copy)) or \
                    (beam_item_copy.pointer_B >= len(right_copy)):
                    continue

                current_step = (choice, beam_item_copy.pointer_A if choice == 'A' else beam_item_copy.pointer_B)
                beam_item_copy.index_pathway.append(current_step)
                new_beam.append(beam_item_copy)

        # reduce beam
        new_beam.sort(key=lambda x: x.cum_prob, reverse=True)
        beam = new_beam[:beam_size]

    best_candidate = beam[0]
    sorted_index = []
    for item in best_candidate.index_pathway:
        if item[0] == 'A':
            sorted_index.append(left_copy[item[1]])
        else:
            sorted_index.append(right_copy[item[1]])
    indices[left:right] = sorted_index



def PairsGreedy(input, output, params):
    '''
    Rank the output summaries in ascending order, based on pairwise comparison using greedy algorithm.
    input: str, the input source text
    output: list of str, the list of output summaries
    params: dict, the hyperparameters
    '''

    def rank_greedy(indices, left, right, input, output, params):
        if right - left > 1:
            mid = (left + right) // 2
            rank_greedy(indices, left, mid, input, output, params)
            rank_greedy(indices, mid, right, input, output, params)
            merge(indices, left, mid, right, input, output, params)
        return indices


    if params['engine'] in ["mistralai/Mistral-7B-Instruct-v0.1", "mistralai/Mistral-7B-Instruct-v0.2"]:
        params['model'] = MistralModelLocal(params={'model': params['engine']})
    elif 'llama' in params['engine']:
        params['model'] = Llama2ModelLocal(params={'model': params['engine']})
    
    params['progress_bar'] = tqdm(total=int(len(output) * np.log2(len(output))), desc='Processing')

    indices = list(range(len(output)))
    indices = rank_greedy(indices, 0, len(indices), input, output, params)
    return indices



def PairsBeam(input, output, params):
    '''
    Rank the output summaries in ascending order, based on pairwise comparison using beam search algorithm with uncertainty prunning.
    input: str, the input source text
    output: list of str, the list of output summaries
    params: dict, the hyperparameters
    '''

    def rank_beam(indices, left, right, input, output, params):
        if right - left > 1:
            mid = (left + right) // 2
            rank_beam(indices, left, mid, input, output, params)
            rank_beam(indices, mid, right, input, output, params)
            merge_with_confidence_beam(indices, left, mid, right, input, output, params)
        return indices
    
    if params['engine'] in ["mistralai/Mistral-7B-Instruct-v0.1", "mistralai/Mistral-7B-Instruct-v0.2"]:
        params['model'] = MistralModelLocal(params={'model': params['engine']})
    elif 'llama' in params['engine']:
        params['model'] = Llama2ModelLocal(params={'model': params['engine']})
    
    params['progress_bar'] = tqdm(total=int(len(output)**2), desc='Processing')

    indices = list(range(len(output)))
    indices = rank_beam(indices, 0, len(indices), input, output, params)
    return indices