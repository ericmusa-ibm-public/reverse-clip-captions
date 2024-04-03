B_INST, E_INST = "<s>[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SYSTEM_PROMPT = """\
You are an integral part of a word-searching algorithm. \
In essence, you are a linguistic expert being tasked with \
modifying and combining text in creative but coherent new ways. \
"""


def get_llama_prompt(prompt, sys_prompt=None):
    sys_prompt = sys_prompt or SYSTEM_PROMPT
    prompt_template =  B_INST + B_SYS + sys_prompt + E_SYS + prompt + E_INST
    return prompt_template


import os
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams


default_params = {
    GenParams.DECODING_METHOD: 'sample',
    GenParams.MIN_NEW_TOKENS: 10,
    GenParams.MAX_NEW_TOKENS: 25,
    GenParams.TEMPERATURE: 0.25,
    # GenParams.RANDOM_SEED: 42,
    GenParams.REPETITION_PENALTY: 1.05,
}

supported_models = [
    'bigcode/starcoder', 
    'bigscience/mt0-xxl', 
    'codellama/codellama-34b-instruct-hf', 
    'google/flan-t5-xl', 
    'google/flan-t5-xxl', 
    'google/flan-ul2', 
    'ibm-mistralai/mixtral-8x7b-instruct-v01-q', 
    'ibm/granite-13b-chat-v1', 
    'ibm/granite-13b-chat-v2', 
    'ibm/granite-13b-instruct-v1', 
    'ibm/granite-13b-instruct-v2', 
    'ibm/granite-20b-multilingual', 
    'meta-llama/llama-2-13b-chat', 
    'meta-llama/llama-2-70b-chat'
]


def generate_text(prompt, model='meta-llama/llama-2-13b-chat', new_params={}, sys_prompt=None):
    if not prompt.startswith(B_INST) and prompt.endswith(E_INST):
        prompt = get_llama_prompt(prompt, sys_prompt)
    
    params = dict(default_params)
    params.update(new_params)

    llm = Model(
        model_id=model,
        params=params,
        credentials={
            'apikey' : os.environ['WATSONX_API_KEY'], 
            'url' : os.environ['WATSONX_URL']
        },
        project_id=os.environ['WATSONX_PROJECT_ID']
    )
    return llm.generate_text(prompt)


import re
import string

def trim_incomplete_response(
        response, 
        delimiters=('.', '!', '?'), 
        comma_is_delimiter=False, 
        delim_follows_text=0,
        include_quotes=True,
        cutoff_str='...', 
        strip_ws=True,
        ):

    trimmed = str(response)
    # copy the response, allowing indexing past the cutoff for the trailing quote

    if not response.endswith(delimiters):  
        # if the response already ends with a delimiter, all good, skip to end and return original response

        assert delim_follows_text < len(response), f'delim_follows_text ({delim_follows_text}) must be '
        # `delim_follows_text` must be less than the length of the response
        # if it were longer, we would attempt to slice the response string past its length

        i = len(response)
        while i > 0:
        # now we iterate backwards through the response, doing checks as nec. acc. to the function args
            i -= 1  
            # decrement first because index of len(response) is out of bounds

            if response[i] in delimiters:  
            # check if this char is in delimiters

                if delim_follows_text > 0:
                # if so, then if we need to also check if preceding characters are text...
                    
                    if not all(char in string.ascii_letters for char in response[i-delim_follows_text:i]):
                        continue
                    # then check if the preceding N=`delim_follow_text` characters are letters. if not, skip to next i

                trimmed = response[:i+1]
                break
                # here, the response[i] is a delimiter, its preceding characters are letters (if req.'d)
                # set `trimmed`, the output string, to `response` up to and including response[i]

            elif response[i] == ',' and comma_is_delimiter:
            # if response[i] is not a regular delim character but we want to treat commas as a delimiter...

                trimmed = response[:i] + cutoff_str
                break
                # then `trimmed` is set up to But Not Including the "," response[i], 
                # and the `cutoff_str` is appended to the end

        if include_quotes:
            # if responses will be sentences in quotes, then we want to add back 
            # quotes immediately following a delimiter that were trimmed off
            if response[i+1] == "'":
                trimmed += "'"
            elif response[i+1] == "\"":
                trimmed += "\""

        if strip_ws:
            # strip white space and newline characters
            trimmed = trimmed.strip()

    return trimmed


def extract_strings(mutated_text):
    '''
    extracts strings in the following format:

        """
        1. "A black horse on a white background." + "A silver fish traveling upstream."
        - "The silhouette of a fish over a silver background." 
        - "A black horse and silver fish." 
        - "A dark colored horse traveling up a stream."
        - "A black fish swimming up a white river."

        """

        This would return:
        ['The silhouette of a fish over a silver background.', 
        'A black horse and silver fish.', 'A dark colored horse traveling up a stream.', 
        'A black fish swimming up a white river.',]
    '''
    return re.findall(r"(\w+[\w| |']*.?)", mutated_text)


question_params = {
    GenParams.DECODING_METHOD: 'sample',
    GenParams.MIN_NEW_TOKENS: 10,
    GenParams.MAX_NEW_TOKENS: 25,
    GenParams.TEMPERATURE: 0.25,
    # GenParams.RANDOM_SEED: 42,
    GenParams.REPETITION_PENALTY: 1.05,
}


def ask(question, new_params={}, verbose=False):
    params = dict(question_params)
    params.update(new_params)
    if verbose: print(question)

    response = generate_text(question, new_params=params)
    if verbose: print(response)

    trimmed_response = trim_incomplete_response(
        response, 
        comma_is_delimiter=True, 
        cutoff_str='.', 
        delim_follows_text=2
        )
    if verbose: print(trimmed_response)
    return trimmed_response


if __name__ == '__main__':
    import dotenv
    dotenv.load_dotenv()
    
    question_a = 'why is the sky blue?'
    print(question_a)

    answer_a = ask(question_a)
    print(answer_a)

    question_b = 'why does the sea sometimes glow around me when I wade through it at night?'
    print(question_b)

    answer_b = ask(question_b, new_params={
        GenParams.MIN_NEW_TOKENS: 10,
        GenParams.MAX_NEW_TOKENS: 60,
        })
    print(answer_b)
