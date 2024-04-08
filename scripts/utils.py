import numpy as np
from collections import Counter
from scipy.stats import norm, rankdata
import random
import scipy
from sklearn.metrics import mean_absolute_error
# from cdf_transform import transform_cdf_matching
import json
import pandas as pd
import math


class CompareResultObject:
    def __init__(self, raw_prob_A=0, raw_prob_B=0, raw_prob_C=0, uncertainty=1):
        # self.prob_A = prob_A
        # self.prob_B = prob_B
        # self.prob_C = prob_C
        self.raw_prob_A = raw_prob_A
        self.raw_prob_B = raw_prob_B
        self.raw_prob_C = raw_prob_C
        prob_sum = raw_prob_A + raw_prob_B + raw_prob_C
        self.prob_A = raw_prob_A/prob_sum
        self.prob_B = raw_prob_B/prob_sum
        self.prob_C = raw_prob_C/prob_sum
        self.uncertainty = uncertainty

    def calibraet_shift(self, shifts):
        shifted_prob_A = self.raw_prob_A/np.exp(shifts['A'])
        shifted_prob_B = self.raw_prob_B/np.exp(shifts['B'])
        shifted_prob_C = self.raw_prob_C/np.exp(shifts['C'])
        prob_sum = shifted_prob_A + shifted_prob_B + shifted_prob_C
        self.prob_A = shifted_prob_A/prob_sum
        self.prob_B = shifted_prob_B/prob_sum
        self.prob_C = shifted_prob_C/prob_sum

    def __str__(self) -> str:
        string = f'prob_A: {round(self.prob_A,2)}, prob_B: {round(self.prob_B,2)}, prob_C: {round(self.prob_C,2)}, uncertainty: {round(self.uncertainty,3)} \n'
        string += f'raw_prob_A: {round(self.raw_prob_A,2)}, raw_prob_B: {round(self.raw_prob_B,2)}, raw_prob_C: {round(self.raw_prob_C,2)}'
        return string
    
    def __getitem__(self, key):
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def to_json(self):
        return {'prob_A': float(self.prob_A), 'prob_B': float(self.prob_B), 'prob_C': float(self.prob_C), 'uncertainty': float(self.uncertainty),
                'raw_prob_A': float(self.raw_prob_A), 'raw_prob_B': float(self.raw_prob_B), 'raw_prob_C': float(self.raw_prob_C)}


def get_calibration_shift(model_name, dataset, aspect):
    calibration_shift_file = f'./calibration_shift.json'
    with open(calibration_shift_file, 'r') as file:
        calibration_shift = json.load(file)
    file.close()
    shifts = calibration_shift[model_name][dataset][aspect]
    return {
        'A': shifts['logprobA'] if 'logprobA' in shifts else 0,
        'B': shifts['logprobB'] if 'logprobB' in shifts else 0,
        'C': shifts['logprobC'] if 'logprobC' in shifts else 0,
    }
    

def shuffle_lists(*args):
    """Shuffle multiple lists together and return the shuffled lists."""
    # Check if all lists are of the same length
    if len(set(map(len, args))) != 1:
        raise ValueError("All lists must be of the same length")

    # Combine the lists element-wise
    combined_lists = list(zip(*args))
    random.shuffle(combined_lists)

    # Unzip the combined list into separate lists
    shuffled_lists = zip(*combined_lists)
    return [list(lst) for lst in shuffled_lists]


def get_score_dist(scores):
    cnter = Counter(scores)
    return [cnter[i]/cnter.total() for i in range(1,6)]


def float_to_int(num):
    return int(round(float(num), 0))


def calculate_uncertainty(probablities):
    probablities = np.array(probablities)
    entropy = -np.sum(probablities * np.log(probablities))
    return entropy


def calculate_entropy(logprobs):
    '''
    logprobs: a list of logprobs
    '''    
    return -np.sum(np.exp(logprobs)* logprobs)


def insert_index_to_anchors(original_list, insert_elements, index_offset=0):
    """
    original_list: Anchor list
    insert_elements: List of elements to be inserted
    The goal is to insert the index of the insert_elements at the position of the value of the insert_elements 
    to the original_list.  
    For example:
    original_list = ['a', 'b', 'c','d','e','f']
    insert_elements = [4,2,1,3,5,3,1]
    index_offset = 1
    Result List: ['a', 3, 7, 'b', 2, 'c', 4, 6, 'd', 1, 'e', 5, 'f']
    """
    original_list = original_list.copy()
    insert_positions = np.sort(insert_elements)[::-1] 
    insert_val = np.argsort(insert_elements)[::-1]

    for index, val in zip(insert_positions, insert_val):
        original_list.insert(index, val+index_offset)
    return [int(num) for num in original_list]


############################################################################
######   Load datasets
############################################################################
def load_jsonl(file_path):
    with open(file_path) as f:
        lines = f.readlines()
    return [json.loads(line) for line in lines]


def load_json(file_path):
    with open(file_path) as f:
        return json.load(f)
    

def load_summEval(path, flat_output=True):
    data_summ_eval = load_jsonl(path)

    input = []
    for i in range(len(data_summ_eval)):
        input.append(data_summ_eval[i]['text'])

    output = []
    for i in range(len(data_summ_eval)):
        output.append(data_summ_eval[i]['decoded'])

    # coherence
    coherence_scores = []
    for i in range(len(data_summ_eval)):
        coherence = [anootation['coherence'] for anootation in data_summ_eval[i]['expert_annotations']]
        coherence_scores.append(round(sum(coherence)/len(coherence),1))
    # turker_annotations
    # fluency
    fluency_scores = []
    for i in range(len(data_summ_eval)):
        fluency = [anootation['fluency'] for anootation in data_summ_eval[i]['expert_annotations']]
        fluency_scores.append(round(sum(fluency)/len(fluency),1))

    # relevance
    relevance_scores = []
    for i in range(len(data_summ_eval)):
        relevance = [anootation['relevance'] for anootation in data_summ_eval[i]['expert_annotations']]
        relevance_scores.append(round(sum(relevance)/len(relevance),1))

    # consistency
    consistency_scores = []
    for i in range(len(data_summ_eval)):
        consistency = [anootation['consistency'] for anootation in data_summ_eval[i]['expert_annotations']]
        consistency_scores.append(round(sum(consistency)/len(consistency),1))

    if flat_output:
        return input, output, {'coherence': coherence_scores, 'fluency': fluency_scores, 'relevance': relevance_scores, 'consistency': consistency_scores}
    else:
        candidate_num = 16
        input_doc, output_doc, coherence_doc, fluency_doc, relevance_doc, consistency_doc = [], [], [], [], [], []
        for i in range(0, len(input), candidate_num):
            input_doc.append(input[i:i+candidate_num])
            output_doc.append(output[i:i+candidate_num])
            coherence_doc.append(coherence_scores[i:i+candidate_num])
            fluency_doc.append(fluency_scores[i:i+candidate_num])
            relevance_doc.append(relevance_scores[i:i+candidate_num])
            consistency_doc.append(consistency_scores[i:i+candidate_num])

        return input_doc, output_doc, {'coherence': coherence_doc, 'fluency': fluency_doc, 'relevance': relevance_doc, 'consistency': consistency_doc}
    

def load_newsroom(path, flat_output=True):
    with open(path, 'r') as file:
        newsroom = json.load(file)
    file.close()

    data=newsroom
    input = [dp['source'].replace('</p><p>', ' ') for dp in data]
    output = [dp['system_output'] for dp in data]
    coherence = [round(dp['scores']['coherence'],1) for dp in data]
    fluency = [round(dp['scores']['fluency'],1) for dp in data]
    informativeness = [round(dp['scores']['informativeness'],1) for dp in data]
    relevance = [round(dp['scores']['relevance'],1) for dp in data]
    if flat_output:
        return input, output, {'coherence': coherence, 'fluency': fluency, 'informativeness': informativeness, 'relevance':relevance}
    else:
        candidate_num = 7
        input_doc, output_doc, coherence_doc, fluency_doc, informativeness_doc, relevance_doc = [], [], [], [], [], []
        for i in range(0, len(input), candidate_num):
            input_doc.append(input[i:i+candidate_num])
            output_doc.append(output[i:i+candidate_num])
            coherence_doc.append(coherence[i:i+candidate_num])
            fluency_doc.append(fluency[i:i+candidate_num])
            informativeness_doc.append(informativeness[i:i+candidate_num])
            relevance_doc.append(relevance[i:i+candidate_num])

        return input_doc, output_doc, {'coherence': coherence_doc, 'fluency': fluency_doc, 'informativeness': informativeness_doc, 'relevance': relevance_doc}


def load_sf_data(file_path):
    '''
    Load SFHOT/SFRES data
    '''
    data = load_json(file_path)
    input = [dp['source'] for dp in data]
    output = [dp['system_output'] for dp in data]
    naturalness = [dp['scores']['naturalness'] for dp in data]
    informativeness = [dp['scores']['informativeness'] for dp in data]
    overall = [dp['scores']['overall'] for dp in data]
    return input, output, {'naturalness': naturalness, 'informativeness': informativeness, 'overall': overall}


def load_hanna(file_path):
    '''
    Load Hanna data
    '''
    try:
        dataset = pd.read_csv(file_path)
    except:
        file_path = "data/hanna_stories_annotations.csv"
        dataset = pd.read_csv(file_path)

    processed_df = {}
    for i in range(dataset.shape[0]):
        idx = dataset['Story ID'][i]
        if str(idx) not in processed_df:
            processed_df[str(idx)] = {
                'input': dataset['Prompt'][i],
                'output': dataset['Prompt'][i] + ' ' + dataset['Story'][i],
                'relevance': [dataset['Relevance'][i]],
                'coherence': [dataset['Coherence'][i]],
                'empathy': [dataset['Empathy'][i]],
                'surprise': [dataset['Surprise'][i]],
                'engagement': [dataset['Engagement'][i]],
                'complexity': [dataset['Complexity'][i]]
            }
        else:
            processed_df[str(idx)]['relevance'].append(dataset['Relevance'][i])
            processed_df[str(idx)]['coherence'].append(dataset['Coherence'][i])
            processed_df[str(idx)]['empathy'].append(dataset['Empathy'][i])
            processed_df[str(idx)]['surprise'].append(dataset['Surprise'][i])
            processed_df[str(idx)]['engagement'].append(dataset['Engagement'][i])
            processed_df[str(idx)]['complexity'].append(dataset['Complexity'][i])

    input = [dp['input'] for dp in list(processed_df.values())]
    output = [dp['output'] for dp in list(processed_df.values())]
    relevance = [round(np.mean(dp['relevance']),1) for dp in list(processed_df.values())]
    coherence = [round(np.mean(dp['coherence']),1) for dp in list(processed_df.values())]
    empathy = [round(np.mean(dp['empathy']),1) for dp in list(processed_df.values())]
    surprise = [round(np.mean(dp['surprise']),1) for dp in list(processed_df.values())]
    engagement = [round(np.mean(dp['engagement']),1) for dp in list(processed_df.values())]
    complexity = [round(np.mean(dp['complexity']),1) for dp in list(processed_df.values())]
    scores = {
        'relevance': relevance,   #  how well the story matches its prompt
        'sensible': coherence,  # how much the story makes sense
        'empathy': empathy, # how well the reader understood the character’s emotions, derived from the importance of emotional commentary
        'surprise': surprise,   # how surprising the end of the story was, derived from the importance of schema violation, or unexpectedness
        'engagement': engagement,   # how much the reader engaged with the story;
        'complexity': complexity    #  how elaborate the story is; derived from the importance of detailed descriptions and sophisticated problem-solving
    }
    return input, output, scores



############################################################################
######   Correlation Analysis    
############################################################################

def calculate_correlation(reference_score, predicted_score):
    spearman_corr, _ = scipy.stats.spearmanr(reference_score, predicted_score)

    if math.isnan(spearman_corr):
        # print(reference_score, predicted_score)
        # print(sum(reference_score), sum(predicted_score))
        spearman_corr = 1 if all(element==reference_score[0] for element in reference_score) else 0
    print('Spearmans correlation: %.3f' % spearman_corr)
    kendall_tau, _ = scipy.stats.kendalltau(reference_score, predicted_score)
    print('Kendall tau: %.3f' % kendall_tau)
    mae = mean_absolute_error(reference_score, predicted_score)
    print('MAE: %.3f' % mae)
    return spearman_corr, kendall_tau, mae


def correlation_analysis(results):
    print('Uncalibrated scores:')
    spearman_original,_,_ = calculate_correlation(results['human_scores'], results['pred_scores'])
    print('------------------')

    print('Uncalibrated G-Eval:')
    weights = np.array([1, 2, 3, 4, 5])
    weighted_gpt_scores = np.exp(results['pred_logprob']).T @ weights
    weighted_gpt_scores = np.round(weighted_gpt_scores, decimals=1)
    spearman_geval,_,_ = calculate_correlation(results['human_scores'], weighted_gpt_scores)
    print('------------------')

    return spearman_original, spearman_geval

