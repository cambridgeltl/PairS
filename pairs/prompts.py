from textwrap import dedent


def get_general_prompt_template(with_input):
    if with_input:
        prompt = dedent("""\
        Evaluate and compare the overall quality of the two following output candidates for the given input.

        Input: {{ input }}

        Compare the following outputs:

        Output candidate A: {{ output_1 }}

        Output candidate B: {{ output_2 }}
                                            
        Question: Which output candidate has better overall quality? \
If the candidate A is better, please return 'A'. \
If the candidate B is better, please return 'B'. \
You must return the choice only.
        Answer: """)

    else:
        prompt = dedent("""\
        Evaluate and compare the overall quality of the two following output candidates.

        Output candidate A: {{ output_1 }}

        Output candidate B: {{ output_2 }}
                                            
        Question: Which output candidate has better overall quality? \
If the candidate A is better, please return 'A'. \
If the candidate B is better, please return 'B'. \
You must return the choice only.
        Answer: """)

    return prompt