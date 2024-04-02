from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from utils import generate_text, trim_incomplete_response, extract_strings


random_sample_prompt = \
f'''
Write some random sentences:
 - "A black horse on a white background."
 - "Two businessmen shaking hands on a sidewalk."
 - "A graph showing the impact of various pesticides on the Colorado Potato Beetle."
 - "The endless sky begets fathomless depths, unknowable secrets."
 - '''

random_sample_params = {
    GenParams.DECODING_METHOD: 'sample',
    GenParams.MIN_NEW_TOKENS: 5,
    GenParams.MAX_NEW_TOKENS: 50,
    GenParams.TEMPERATURE: 0.75,
    # GenParams.RANDOM_SEED: 42,
    GenParams.REPETITION_PENALTY: 1.3,
}

def generate_random_samples(new_params={}, trim=2, extract=True, min_length=10):
    params = dict(random_sample_params)
    params.update(new_params)
    random_sample = generate_text(random_sample_prompt, new_params=params)
    if trim > 0:
        random_sample = trim_incomplete_response(random_sample)
    if extract:
        random_sample = extract_strings(random_sample)
        random_sample = [_ for _ in random_sample if len(_) >= min_length]
    return random_sample


if __name__ == '__main__':
    import dotenv
    dotenv.load_dotenv()
        
    for _ in generate_random_samples(): print(_)


