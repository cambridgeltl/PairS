from textwrap import dedent


def get_prompt_template(prompt_name, aspect='coherence', dataset='SummEval', model_name=None, with_input=False):
    adj_lookup = {'coherence': 'coherent',
                  'fluency': 'fluent',
                  'relevance': 'relevant',
                  'informativeness': 'informative',
                  'overall': 'overall high-quality',
                  'naturalness': 'natural',
                  'sensible': 'sensible',
                  'surprise': 'surprising',
                  'complexity': 'complex',
                  'consistency': 'consistent'}
                  
    if dataset in ['SummEval', 'newsroom']:
        task = 'summarization'
        description = 'summarization'
        text_name = ('summary', 'Summary')
        text_names = 'summaries'        
        input_name = ('source text', 'Source text')
    elif dataset in ['sfhot','sfres']:
        task = 'd2t'
        description = 'data-to-text generation'
        text_name = ('text', 'Text')
        text_names = 'texts'
        input_name = ('data', 'Data')
    elif dataset in ['hanna']:
        task = 'd2t'
        description = 'creative writing'
        text_name = ('story', 'Story')
        text_names = 'stories'


    ###################################################### Pairwise comparison ######################################################
    if prompt_name == "pairwise comparison":
        if with_input:
            prompt = dedent(f"""\
            {input_name[1]}: {{{{ input }}}}

            {{{{instruction}}}}
            {text_name[1]} candidate A: {{{{ output_1 }}}}
            {text_name[1]} candidate B: {{{{ output_2 }}}}
                                    
            Question: Which {text_name[0]} candidate is more {adj_lookup[aspect]}? \
If the {text_name[0]} A is more {adj_lookup[aspect]}, please return 'A'. \
If the {text_name[0]} B is more {adj_lookup[aspect]}, please return 'B'. \
Plese only return the choice.
            Answer: """)
        else:
            prompt = dedent(f"""\
            {{{{instruction}}}}
            Which {text_name[0]} is more {adj_lookup[aspect]}?
                            
            {text_name[1]} A: {{{{ output_1 }}}}
                            
            {text_name[1]} B: {{{{ output_2 }}}}
                                    
            Question: If the {text_name[1]} A is more {adj_lookup[aspect]}, please return "A". \
If the {text_name[1]} B is more {adj_lookup[aspect]}, please return "B". You must only return the choice.
            Answer: """)

    ###################################################### Pairwise comparison 3-way ######################################################
    elif prompt_name == "pairwise comparison 3-way":
        if with_input:
            prompt = dedent(f"""\
            {{{{instruction}}}}

            {input_name[1]}: {{{{ input_1 }}}}

            Evaluate and compare the following {text_names}:
            
            {text_name[1]} A: {{{{ output_1 }}}}

            {text_name[1]} B: {{{{ output_2 }}}}
                                    
            Question: Which {text_name[0]} is more {adj_lookup[aspect]}? \
If the {text_name[0]} A is more {adj_lookup[aspect]}, please return 'A'. If the {text_name[0]} B is more {adj_lookup[aspect]}, please return 'B'. \
If both {text_names} are equally {adj_lookup[aspect]}, please return 'C'. Plese only return the choice.
            Answer: """)
        else:
            prompt = dedent(f"""\
            {{{{instruction}}}}
            Which {text_name[0]} is more {adj_lookup[aspect]}?
                            
            {text_name[1]} A: {{{{ output_1 }}}}
                            
            {text_name[1]} B: {{{{ output_2 }}}}
                                    
            Question: If the {text_name[1]} A is more {adj_lookup[aspect]}, please return "A". \
If the {text_name[1]} B is more {adj_lookup[aspect]}, please return "B". You must only return the choice.
            Answer: """)

    ###################################################### Baseline score prompts ######################################################
    elif prompt_name == "score":
        if with_input:
            prompt = dedent(f"""\
            {{{{instruction}}}}

            {input_name[1]}: {{{{ input }}}}            

            {text_name[1]}: {{{{ output }}}}

            Please rate on a scale from 1 to 5, where 1 represents very low {adj_lookup[aspect]}, \
and 5 indicates excellent {adj_lookup[aspect]}. You must only return an int score.
            Score: """)
        else:
            prompt = dedent(f"""\
            {{{{instruction}}}}

            Evaluate the following {text_name[1]}.
            {text_name[1]}: {{{{ output }}}}

            Question: Please rate on a scale from 1 to 5, where 1 represents very low {adj_lookup[aspect]}, \
and 5 indicates excellent {adj_lookup[aspect]}. You must only return the int score.
            Score: """)

    return prompt


def get_aspect_instruction(aspect, eval_method='pairwise comparison', dataset='SummEval'):

    if dataset in ['SummEval', 'newsroom']:
        task = 'summarization'
        description = 'summarization'
        text_name = 'summary'
        text_names = 'summaries'
        
    elif dataset in ['sfhot','sfres']:
        task = 'd2t'
        description = 'data-to-text generation'
        text_name = 'text'
        text_names = 'texts'

    elif dataset in ['hanna']:
        task = 'd2t'
        description = 'creative story writing'
        text_name = 'story'
        text_names = 'stories'

    else:
        print('Dataset not support')
        assert False

    instructions = {
        'coherence': {
            'score': f'Please evaluate the coherence of the following {text_name}. ',
            'pairwise comparison': f'Compare the coherence of the two following {text_names}. '
                        f'Consider aspects such as clarity and logical flow. '
                        f"A {text_name} is coherent if it accurately captures the key information from the article, "
                        "and presents them in a clear manner."
        },
        'fluency': {
            'score': f'Please evaluate the fluency of the following {text_name}. ',
            'pairwise comparison': f'Evaluate and compare the fluency of the two following {text_names}. ',
                        # f'A fluent {text_name} should use clear language that avoids redundancy and errors. '
                        # f'A fluent {text_name} should use appropriate transition words, connectors, and avoid abrupt. '
                        # f'A fluent {text_name} should use correct spelling, punctuation, and capitalization throughout the summary, '
                        # 'and follow the conventions of standard written English.',
        },
        'relevance': {
            'score': f'Please evaluate the relevance of the following {text_name}. '
                f'A {text_name} is relevant if it captures the main points from the article, without leaving out any crucial details or adding any unnecessary or inaccurate ones. '
                f'A {text_name} is more relevant if it uses the same or similar terms and expressions as the article. '
                f'A {text_name} is less relevant if it omits some of the key facts from the article, or if it introduces irrelevant information that is not supported by the article.',
            'pairwise comparison': f'Evaluate and compare the relevance level of two {text_names}. '
                f'A {text_name} is relevant if it captures the main points from the article, without leaving out any crucial details or adding any unnecessary or inaccurate ones. '
                f'A {text_name} is more relevant if it uses the same or similar terms and expressions as the article. '
                f'A {text_name} is less relevant if it omits some of the key facts from the article, or if it introduces irrelevant information that is not supported by the article.'
        },
        'informativeness': {
            'score': f'Please evaluate the informativeness of the following {text_name}. ',
            'pairwise comparison': f"Compare the performance of two {description} examples, especially focusing on informativeness. "
                                    f"Evaluate how each {text_name} converts their input text to natural language text, without omitting, adding, or distorting any facts."
        },
        'consistency': {
            'score': f'Please evaluate the consistency of the following {text_name}. '
                f'A {text_name} is consistent with the article if it faithfully reflects the main points, facts, and tone of the article. '
                f'A {text_name} is inconsistent if it introduces any errors, contradictions, or distortions of the original article.',
            'pairwise comparison': f'Evaluate and compare how two {text_names} consistently follow the source text. '
                f'A {text_name} is consistent with the article if it faithfully reflects the main points, facts, and tone of the article. '
                f'A {text_name} is inconsistent if it introduces any errors, contradictions, or distortions of the original article.'
        },
        'naturalness': {
            'score': 'Please evaluate the informativeness of the following passages. '
                     'Please rate on a scale from 1 to 5, where 1 represents very low informativeness, '
                     'and 5 indicates excellent informativeness. Your response should be in the format of a list of float numbers. '
                     'For example: "[2, 4, 3]"',
            'pairwise comparison': f'Evaluate the naturalness of two {text_names}. '
                'A sentence is natural if it is fluent, coherent, grammatical, and human-like.'
        },
        'overall':{
            'score': None,
            'pairwise comparison': f'Please evaluate the overall quality of the {description} '
                f'Consider the coherence, fluency, relevance, and informativeness of the {text_names}. '
                f'If you think {text_name} A is better, please return "A". If you think {text_name} B is better, please return "B".'
        },
        'sensible':{
            'score': f'Please evaluate the sensibility of the following {text_name}. '
                f'A {text_name} is sensible if the events are consistent and align with the context they are set in. '
                'A sensible story has good believability. '
                f'A {text_name} is not sensible if there are contradictions.',
            'pairwise comparison': f'Please evaluate and compare the sensibility of the following {text_names}. '
                f'A {text_name} is sensible if the events within each {text_name} are consistent and align with the context they are set in. '
                'A sensible story has good believability. '
                f'A {text_name} is not sensible if there are contradictions.'
        },
        'surprise':{
            'score': f"Assess the given story based on its capacity to generate surprise. ",
            'pairwise comparison': f"Please evaluate and compare two {text_names} in terms of their ability "
                "to evoke surprise and unexpected plot twists. Consider the effectiveness of building suspense "
                "and anticipation, and the manipulation of reader expectations. "
        },
        'complexity':{
            'score': "Please evaluate the narrative complexity of the following creative story. "
                "Consider the complexity from the aspect of structure, character development, thematic depth, and stylistic elements employed in the story. ",
            'pairwise comparison': "Please evaluate and compare the narrative complexity of the following creative stories. "
                "Consider the complexity from the aspects of structure, character development, thematic depth, and stylistic elements employed in each story. "
        }
    }

    if (aspect in instructions) and (eval_method in instructions[aspect]):
        return instructions[aspect][eval_method]
    else:
        print('Aspect or evaluation method not supported.')
        return None


if __name__ == "__main__":
    prompt = get_aspect_instruction('overall', eval_method='pairwise', dataset='sfhot')
    print(prompt)

    prompt_instruction = get_prompt_template("pairwise comparison", 'any', aspect='overall', dataset='sfhot')

    print(prompt_instruction)