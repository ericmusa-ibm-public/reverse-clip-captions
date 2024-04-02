from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from genetic import generate_text, trim_incomplete_response, extract_strings


def make_crossover_prompt(mother_text, father_text):
    return \
f'''
Take two sentences and combine them in multiple new ways:

1. "A black horse on a white background." + "A silver fish traveling upstream."
    - "The silhouette of a fish over a silver background." 
    - "A black horse and silver fish." 
    - "A dark colored horse traveling up a stream."
    - "A black fish swimming up a white river."

2. "Two businessmen shaking hands on a sidewalk." + "A graph showing the impact of various pesticides on the Colorado Potato Beetle."
    - "A graph showing businessmen spraying pesticides." 
    - "Two businessmen discussing a graph about Colorado Potato Beetle populations." 
    - "A Colorado Potato Beetle on the sidewalk has died due to pesticides."
    - "A Colorado Potato Beetle shakes hands with a businessman."

3. "{mother_text}" + "{father_text}"'''


crossover_params = {
    GenParams.DECODING_METHOD: 'sample',
    GenParams.MIN_NEW_TOKENS: 20,
    GenParams.MAX_NEW_TOKENS: 120,
    GenParams.TEMPERATURE: 0.50,
    # GenParams.RANDOM_SEED: 42,
    GenParams.REPETITION_PENALTY: 1.1,
}

def crossover(mother_text, father_text, new_params={}, verbose=False, trim=2, extract=True):
    params = dict(crossover_params)
    params.update(new_params)
    crossover_prompt = make_crossover_prompt(mother_text, father_text)
    if verbose: print(crossover_prompt, end='')
    crossover_response = generate_text(crossover_prompt, new_params=params)
    if verbose: print(crossover_response)
    if trim > 0: 
        crossover_response = trim_incomplete_response(crossover_response, delim_follows_text=trim, strip_ws=False)
    if verbose: print('trimmed:', crossover_response)
    return extract_strings(crossover_response) if extract else crossover_response


if __name__ == '__main__':
    import dotenv
    dotenv.load_dotenv()
    
    import random

    set_a = [
        "A black horse on a white background.", 
        "Two businessmen shaking hands on a sidewalk."
        ]
    
    set_b = [
        "The endless sky begets fathomless depths, unknowable secrets.", 
        "A graph showing the impact of various pesticides on the Colorado Potato Beetle."
        ]
    
    combinations = [
        [i, j, set_a[i], set_b[j]] for i in range(len(set_a)) for j in range(len(set_b))
    ]
    random.shuffle(combinations)


    for ai, bj, a, b in combinations:
        print(f'mother {ai}:', a)
        print(f'father {bj}:', b)
        crossover_result = crossover(a, b)
        print(crossover_result)
        print()
        # input()
        