import numpy as np


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

