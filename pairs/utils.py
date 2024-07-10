import numpy as np
import random
import json


class CompareResultObject:
    def __init__(self, raw_prob_A=0, raw_prob_B=0, raw_prob_C=0, uncertainty=1):
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
    

def calculate_uncertainty(probablities):
    probablities = np.array(probablities)
    entropy = -np.sum(probablities * np.log(probablities))
    return entropy


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
