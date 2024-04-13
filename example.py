from pairs import PairsGreedy, PairsBeam
from scripts.utils import shuffle_lists, load_summEval


# Load example data
summ_eval_path = 'data/SummEval/model_annotations.aligned.paired.jsonl'
input_doc, output_doc, _ = load_summEval(summ_eval_path, flat_output=False)

doc_id = 42
input, output = input_doc[doc_id], output_doc[doc_id]
input, output = shuffle_lists(input, output)

# The same input source text corresponds to multiple output summaries
print('Number of summary candidates:', len(output))


method = 'PairsGreedy'
if method == 'PairsGreedy':
    # Set hyperparameters
    params = {
        # 'engine': "mistralai/Mistral-7B-Instruct-v0.1",
        'engine': "meta-llama/Llama-2-7b-chat-hf",
        'api_call': 0,
        'with_input': True,
    }
    # Rank the output summaries
    indices = PairsGreedy(input[0], output, params)
    print(indices)


elif method == 'PairsBeam':
    # Set hyperparameters
    params = {
        'engine': "mistralai/Mistral-7B-Instruct-v0.1",
        'beam_size': 2000,
        'api_call': 0,
        'prob_gap': 0.1,
        'with_input': True,
    }
    # Rank the output summaries
    indices = PairsBeam(input[0], output, params)
    print(indices)