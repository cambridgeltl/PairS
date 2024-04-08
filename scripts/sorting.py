from tqdm import tqdm
from openai_api import call_openai_chat_completion, extract_prob
from jinja2 import Environment
from prompts import get_prompt_template, get_aspect_instruction
import numpy as np
from collections import Counter
from utils import shuffle_lists, calculate_correlation, load_newsroom, load_summEval, calculate_uncertainty, load_hanna
from utils import CompareResultObject, insert_index_to_anchors, get_calibration_shift, calculate_entropy
from mistral import MistralModel, MistralModelLocal
import json
import copy
import random


def moving_average(sum,val,idx):
    return sum*idx/(idx+1) + val/(idx+1)


class BeamItem:
    def __init__(self, index_pathway=[], cum_prob=1, pointer_A=-1, pointer_B=-1):
        self.index_pathway = index_pathway
        self.cum_prob = cum_prob
        self.pointer_A = pointer_A
        self.pointer_B = pointer_B

    def __str__(self):
        return f'index_pathway: {self.index_pathway}, cum_prob: {self.cum_prob}'


def is_better_than_prob(id1, id2, input, output, params, api_key=None):
    prompt_instruction = get_aspect_instruction(params['aspect'], eval_method=params['eval_method'], dataset=params['dataset'])
    
    prompt_template = get_prompt_template(prompt_name=params['eval_method'], 
                                          model_name=params['engine'], 
                                          aspect=params['aspect'], 
                                          dataset=params['dataset'],
                                          with_input=params['with_input'])
    environment = Environment()
    prompt_template = environment.from_string(prompt_template)

    prompt = prompt_template.render(
        instruction=prompt_instruction,
        input_1=input[id1],
        input_2=input[id2],
        output_1=output[id1],
        output_2=output[id2],
        aspect=params['aspect'],
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
        msg = MistralModel.get_mistral_chat_message(prompt, aspect=params['aspect'], with_input=params['with_input'])
        # [print(m['content']) for m in msg]
        # assert False
        compare_result = params['model'].mistral_compare(msg)
        if params['calibration']:
            calibration_shifts = get_calibration_shift(params['engine'], params['dataset'], params['aspect'])
            compare_result.calibraet_shift(calibration_shifts)
        return compare_result
    
    elif 'llama' in params['engine']:
        compare_result = params['model'].compare(prompt)
        # print(compare_result)
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


def merge_sort_indices(input, output, params):
    if 'model' not in params:
        if params['engine'] in ["mistralai/Mistral-7B-Instruct-v0.1", "mistralai/Mistral-7B-Instruct-v0.2"]:
            from mistral import MistralModelLocal
            params['model'] = MistralModelLocal(params={'model': params['engine']})

        elif 'mistral' in params['engine'] or 'mixtral' in params['engine']:
            from mistral import MistralModel
            params['model'] = MistralModel(params={'model': params['engine']})

        elif 'llama' in params['engine']:
            from llama2 import Llama2ModelLocal
            params['model'] = Llama2ModelLocal(params={'model': params['engine']})

    indices = list(range(len(output)))
    indices = merge_sort(indices, 0, len(indices), input, output, params)
    return indices


def merge_sort(indices, left, right, input, output, params):
    if right - left > 1:
        mid = (left + right) // 2
        merge_sort(indices, left, mid, input, output, params)
        merge_sort(indices, mid, right, input, output, params)
        if params['confidence_beam']:
            merge_with_confidence_beam(indices, left, mid, right, input, output, params)
        else:
            merge(indices, left, mid, right, input, output, params)
    return indices


def merge(indices, left, mid, right, input, output, params):
    left_copy = indices[left:mid]
    right_copy = indices[mid:right]

    i = 0
    j = 0
    k = left
    
    while i < len(left_copy) and j < len(right_copy):
        if 'progress_bar' in params: params['progress_bar'].update(1)
        compare_result = is_better_than_prob(left_copy[i], right_copy[j], input, output, params)
        params['compare_log'][(left_copy[i], right_copy[j])] = compare_result.to_json() #compare_result['prob_A']
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
            params['compare_log'][(left_copy[i], right_copy[j])] = compare_result.to_json()
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
        # print(coef)
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
                    beam_item_copy.cum_prob = moving_average(beam_item_copy.cum_prob, logprob*coef, i)

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


def binary_search_insert_index(input, output, params, anchors_idx, target_idx):
    left = 0
    right = len(anchors_idx) - 1

    while left <= right:
        mid = (left + right) // 2
        if 'progress_bar' in params: params['progress_bar'].update(1)
        compare_result = is_better_than_prob(anchors_idx[mid], target_idx, input, output, params=params)
        params['compare_log'][(anchors_idx[mid], target_idx)] = compare_result.to_json()
        params['api_call'] += 1
        
        if compare_result['prob_A'] > compare_result['prob_B']:
            right = mid - 1
        else:
            left = mid + 1
    return left 



def merge_sort_with_scale(input, output, scores, params, sort_size=100):
    # step 1 sort initial subset
    sorted_anchor_indices = merge_sort_indices(input[:sort_size], output[:sort_size], params)
    initial_scores = np.array(scores[:sort_size])
    calculate_correlation(initial_scores[sorted_anchor_indices], list(range(sort_size)))
    params['progress_bar'].close()

    # step 2: Get anchor examples index
    # attemp 1, Use all initial indices as anchor

    # step 3: determine the rest of the data, binary search
    progress_bar = tqdm(total=len(input)-sort_size, desc='Processing')
    searech_result = []
    for idx in range(sort_size, len(input)):
        insert_index = binary_search_insert_index(input, output, params, sorted_anchor_indices, idx)
        searech_result.append(insert_index)
        progress_bar.update(1)
    progress_bar.close()

    # step 4: insert the rest of the data to the sorted indices
    sorted_full_indices = insert_index_to_anchors(sorted_anchor_indices, searech_result, sort_size)

    return sorted_full_indices




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='SumEval')
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--aspect', type=str, default='coherence')
    parser.add_argument('--eval_method', type=str, default='pairwise comparison')
    parser.add_argument('--scaling_anchor_size', type=int, default=0)
    parser.add_argument('--eval_size', type=int, default=300)
    parser.add_argument('--engine', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--confidence_beam', action="store_true")
    parser.add_argument('--prob_gap', type=float, default=0.15)
    parser.add_argument('--beam_size', type=int, default=100)
    parser.add_argument('--with_input', action="store_true")
    parser.add_argument('--calibration', action="store_true")
    args = parser.parse_args()

    print('aspect:', args.aspect)
    print('engine:', args.engine)
    print('dataset:', args.dataset)
    print('confidence_beam:', args.confidence_beam)
    print('beam_size:', args.beam_size)
    print('calibration:', args.calibration)

    params = {
        'dataset': args.dataset,
        'engine': args.engine,
        'aspect': args.aspect,
        'eval_method': args.eval_method,
        'confidence_beam': args.confidence_beam,
        'beam_size': args.beam_size,
        'api_call': 0,
        'prob_gap': args.prob_gap,
        'with_input': args.with_input,
        'calibration': args.calibration,
        'compare_log': {},
    }
    # Load the dataset
    if args.dataset == 'SumEval':
        summ_eval_path = 'data/SummEval/model_annotations.aligned.paired.jsonl'
        input, output, scores = load_summEval(summ_eval_path)
    elif args.dataset == 'newsroom':
        newsroom_path = 'data/newsroom/newsroom.json'
        input, output, scores = load_newsroom(newsroom_path)
    elif args.dataset == 'hanna':
        hanna_path = 'data/hanna/hanna_stories_annotations.csv'
        input, output, scores = load_hanna(hanna_path)
    else:
        print('Dataset not supported.')
        assert False
    
    scores = scores[args.aspect]
    intput, output, scores = shuffle_lists(input, output, scores)

    input, output, scores = input[:args.eval_size], output[:args.eval_size], scores[:args.eval_size]

    # Initialize the progress bar
    if params['confidence_beam']:
        params['progress_bar'] = tqdm(total=int(len(input)**2), desc='Processing')
    else:
        params['progress_bar'] = tqdm(total=int(len(input) * np.log2(len(input))), desc='Processing')

    # Run the sorting algorithm
    if args.scaling_anchor_size==0:
        ranking_indices = merge_sort_indices(input, output, params)
    else:
        ranking_indices = merge_sort_with_scale(input, output, scores, params, sort_size=args.scaling_anchor_size)


    human_scores = np.array(scores)

    print(
        'dataset:', args.dataset, 
        'aspect:', args.aspect, 
        'eval_method:', args.eval_method, 
        'beam_search', args.confidence_beam, 
        'engine:', args.engine,
        'api_call:', params['api_call'],
        'beam_size:', params['beam_size'],
        'prob_gap:', params['prob_gap'],
        'scaling_anchor_size:', args.scaling_anchor_size,
        'eval_size:', len(scores),
        'with_input', args.with_input,

    )
    calculate_correlation(human_scores[ranking_indices], list(range(len(human_scores))))

    score_cnter = Counter(human_scores)
    modified_scores = []
    for s in sorted(score_cnter.keys()):
        modified_scores += [s]*Counter(human_scores)[s]

    calculate_correlation(predicted_score=human_scores[ranking_indices], reference_score=modified_scores)


    params['progress_bar'].close()

    # Save the result
    if args.save_path is not None:
        results = {
            'aspect': args.aspect,
            'confidence_beam': args.confidence_beam,
            'beam_size': params['beam_size'],
            'engine': args.engine,
            'dataset': args.dataset,
            'human_scores': scores,
            'gpt_ranking': ranking_indices,
            'compare_log': {str(key):val for key, val in params['compare_log'].items()},
        }

        with open(args.save_path, 'a') as f:
            json.dump(results, f)
            f.write('\n')
