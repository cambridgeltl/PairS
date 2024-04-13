# Code for Aligning with Human Judgement: The Role of Pairwise Preference in Large Language Model Evaluators

![pairs](figs/pairs.png)
**Link to paper**:
[Aligning with Human Judgement: The Role of Pairwise Preference in Large Language Model Evaluators](https://arxiv.org/abs/2403.16950) (arXiv preprint arXiv:2403.16950)


## Code

### Evaluate on Datasets
For NewsRoom and SummEval
```python
bash pairs_run.sh
```

For Hanna
```python
bash pairs_flat_run
```

### Notebook Demo
We provide a Notebook demonstrations in ```notebooks/```.

### Break downs
**Load dataset**: We put all datasets loading in ```scripts/utils.py```.  

**Prompts**: We put all prompts and instructions in ```scripts/prompts.py```.  

**Base models**: We supports the following base models, ```mistralai/Mistral-7B-Instruct-v0.1```, ```meta-llama/Llama-2-7b-chat-hf```, all versions of ```GPT-3.5-turbo``` and ```GPT-4-turbo```.  

**Hyper-parameters**:
  - ```dataset```: We support 3 datasets, 'newsroom', 'SummEval' and 'hanna'.
  - ```eval_method```: For all PairS method, we use 'pairwise comparison'.
  - ```engine```: The base models.
  - ```with_input```: If the data format has input text. For example, the summarization task has source text as input, but story writing task has no input text.
  - ```confidence_beam```: ```True``` for PairS-beam and ```False``` for PairS-greedy.
  - ```prob_gap```: The uncertainty tolerance. $0.1$ represents we will create beam candidates for both A and B if $0.5-0.1 < P(A\succ B) < 0.5+0.1$.

More details and comments will be added soon.


## Citation
If you find our work helpful, please consider citing our paper:

```
@article{liu2024aligning,
  title={Aligning with Human Judgement: The Role of Pairwise Preference in Large Language Model Evaluators},
  author={Liu, Yinhong and Zhou, Han and Guo, Zhijiang and Shareghi, Ehsan and Vulic, Ivan and Korhonen, Anna and Collier, Nigel},
  journal={arXiv preprint arXiv:2403.16950},
  year={2024}
}
```
