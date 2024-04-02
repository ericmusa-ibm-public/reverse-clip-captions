from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from utils import generate_text, trim_incomplete_response, extract_strings

# STRONG = 'Take the following text and change it as much as possible while retaining the same meaning.'
# MEDIUM = 'Take the following text and change some of the words to rephrase the same meaning.'
# WEAK = 'Take the following text and tweak some of the wording while retaining the meaning.'
STRONG = ' significantly'
MEDIUM = ''
WEAK = ' a little bit'
strengths = [WEAK, MEDIUM, STRONG]


def make_mutation_prompt(orig_text, strength=STRONG):
    assert strength in strengths
    return \
f'''
Take the following text and change some of the words{strength}:

1. "A black horse on a white background"
    - "The silhouette of a horse over a white background" 
    - "A dark colored horse on a light colored background"

2. "Two businessmen shaking hands on a sidewalk"
    - "One businessman shaking another businessman's hand on the street" 
    - "Two businessmen greeting each other on the sidewalk"

3. "{orig_text}"'''

mutate_params = {
    GenParams.DECODING_METHOD: 'sample',
    GenParams.MIN_NEW_TOKENS: 20,
    GenParams.MAX_NEW_TOKENS: 50,
    GenParams.TEMPERATURE: 0.60,
    # GenParams.RANDOM_SEED: 42,
    GenParams.REPETITION_PENALTY: 1.0,
}


def mutate(original_text, new_params={}, verbose=False, trim=2, extract=True):
    params = dict(mutate_params)
    params.update(new_params)
    mutation_prompt = make_mutation_prompt(original_text)
    if verbose: print(mutation_prompt, end='')
    mutated_response = generate_text(mutation_prompt, new_params=params)
    if verbose: print(mutated_response)
    if trim > 0: 
        mutated_response = trim_incomplete_response(mutated_response, delim_follows_text=trim, strip_ws=False)
    if verbose: print('trimmed:', mutated_response)
    return extract_strings(mutated_response) if extract else mutated_response


if __name__ == '__main__':
    import dotenv
    dotenv.load_dotenv()
    
    from utils import ask
    
    question = 'why does the sea sometimes glow around me when I wade through it at night?'
    print(question)

    answer = ask(question, new_params={
        GenParams.MIN_NEW_TOKENS: 10,
        GenParams.MAX_NEW_TOKENS: 60,
        })
    print(answer)

    rephrasings = mutate(answer, new_params={
        GenParams.MIN_NEW_TOKENS: 10,
        GenParams.MAX_NEW_TOKENS: 100,
        })
    for rephrasing in rephrasings:
        print(rephrasing)
