
import json
import openai
from tqdm import tqdm
import random
import time
import requests
import os
from pathlib import Path


def get_oauth_token(p_token_url, p_client_id, p_client_secret, p_scope):
    file_name = "py_llm_oauth_token.json"
    try:
        base_path = Path(__file__).parent
        file_path = Path.joinpath(base_path, file_name)
    except Exception as e:
        print(f"Error occurred while setting file path: {e}")
        return None
 
    try:
        # Check if the token is cached
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                token = json.load(f)
        else:
            # Get a new token from the OAuth server
            response = requests.post(
                p_token_url,
                data={"grant_type": "client_credentials", "client_id": p_client_id,
                      "client_secret": p_client_secret, "scope": p_scope}
            )
            response.raise_for_status()
            token = response.json()
            # with open(file_path, "w") as f:
            #     json.dump(token, f)
    except Exception as e:
        print(f"Error occurred while getting OAuth token: {e}")
        return None
 
    try:
        # Check if the token is expired
        expires_in = time.time() + token["expires_in"]
        if time.time() > expires_in:
            # Refresh the token
            token = get_oauth_token(p_token_url, p_client_id,
                                    p_client_secret, p_scope)
    except Exception as e:
        print(f"Error occurred while while getting OAuth token: {e}")
        return None
 
    authToken = token["access_token"]
    return authToken


# define a retry decorator
def retry_with_exponential_backoff(
    func,
    initial_delay: float = 3,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 20,
    errors: tuple = (openai.error.RateLimitError,),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            # Retry on specified errors
            except errors as e:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper


@retry_with_exponential_backoff
def chatcompletions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


def openai_generate_answer(prompt_list, MODEL):
    
    answer_list = []
    start_idx = 10
    try:
        for idx, prompt in enumerate(prompt_list[start_idx:1000]):
            # print("prompt:")
            # print(prompt)
            # print(len(prompt.split()))
            if idx % 200 == 0:
                openai_setup()
            
            try:
                completion = chatcompletions_with_backoff(
                    engine=MODEL,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                )
            except:
                completion = None
            # print("completion:")
            # print(completion)
            if completion is None or completion.choices[0]['finish_reason'] == "content_filter":
                answer = "got content_filter. needs to regenerate"
            else:
                answer = completion.choices[0].message.content
                answer = answer.replace("\n", " ")

            print("idx:", idx+start_idx)
            print(answer)
            answer_list.append(answer)

    except Exception as error:
        print("error occurs. current idx:", idx+start_idx)
        print("error msg:", error)

    return answer_list
    

def format_prompt(ctxs, question, topk=5, max_length=2800, single_turn=False):
    system = "System: This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context.\n\n"

    ## for sandia (not working)
    # system = "System: This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should always give an answer even though the answer cannot be found from the context.\n\n"

    ## get top-5 context
    neighbours = ["title: " + ctx["title"] + ", source: " + ctx["text"] for ctx in ctxs[:topk]]
    context = "\n\n".join(neighbours) + "\n\n"

    ## get prompt
    if single_turn:
        ## for chatgpt
        question = "User: Please give an answer in 1-2 sentences. " + question + "\n\nAssistant:"

        ## for gpt-4
        # question = "User: Please give an answer in one sentences. " + question + "\n\nAssistant:"

        ## for gpt-4 in sandia (not working)
        # question = "User: Please give an answer in one sentence. I want to emphasize that you need to always give an answer even though you cannot find it from the aboved context. " + question + "\n\nAssistant:"
    else:
        tmp_list = question.split("User:", 1)  # split will stop at the first "User:"
        
        ## for chatgpt
        # question = "User: Please give an answer in 1-2 sentences." + tmp_list[1]
        # question = "User: For this and next following questions, please give an answer in 1-2 sentences." + tmp_list[1]

        ## for gpt-4
        question = "User: Please give an answer in just one sentence." + tmp_list[1]
        # question = "User: For this and next following questions, please give an answer in just one sentence." + tmp_list[1]

    prompt = system + context + question

    prompt_length = len(prompt.split())
    if prompt_length > max_length:
        # cut the context
        token_list = context.split()
        dist = prompt_length - max_length
        context = " ".join(token_list[:-dist])
        prompt = system + context + question

        assert len(prompt.split()) <= max_length

    return prompt


def load_data(datapath, single_turn=False):
    print("loading data from %s" % datapath)
    with open(datapath, "r") as f:
        data_list = json.load(f)

    prompt_list = []
    for item in tqdm(data_list):
        prompt = format_prompt(item['ctxs'], item['question'], single_turn=single_turn)
        prompt_list.append(prompt)
    
    return prompt_list


def write_to_files(output_datapath, answer_list):

    print("writing to %s" % output_datapath)    
    with open(output_datapath, "a+") as f:
        for answer in answer_list:
            f.write(answer + "\n")


def main_multi_turn():
    ## doc2dial
    input_datapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/doc2dial/doc2dial_ftdragon_chatgptgen7k_chunk150_QA_test.json"
    
    ## quac
    # input_datapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/quac/quac_ftdragon_chatgptgen7k_chunk150_QA_test.json"

    # ## qrecc
    # input_datapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/qrecc/qrecc_ftdragon_chatgptgen7k_chunk150_QA_test.json"

    ## get prompt_list
    prompt_list = load_data(input_datapath)
    # print(len(prompt_list[95].split()))
    # print(prompt_list[95])

    ## get answer_list
    # MODEL = "gpt-3.5-turbo"
    MODEL = "gpt-4"
    answer_list = openai_generate_answer(prompt_list, MODEL)

    ### write to output files
    ## doc2dial
    output_datapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/doc2dial/gpt_4_based_on_ftdragon_chatgptgen7k_chunk150.txt"

    ## quac
    # output_datapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/quac/gpt_4_based_on_ftdragon_chatgptgen7k_chunk150.txt"
    
    ## qrecc
    # output_datapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/qrecc/gpt_4_based_on_ftdragon_chatgptgen7k_chunk150.txt"

    write_to_files(output_datapath, answer_list)


def check_content_filter():
    # datapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/quac/gpt_4_based_on_ftdragon_chatgptgen7k_chunk150.txt"
    datapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/doc2dial/gpt_4_based_on_ftdragon_chatgptgen7k_chunk150.txt"

    with open(datapath, "r") as f:
        data_list = f.readlines()
    print("total length of data_list:", len(data_list))

    regenerate_idx_list = []
    for idx, item in enumerate(data_list):
        item = item.strip()
        if item == "got content_filter. needs to regenerate":
            regenerate_idx_list.append(idx)
    
    print("length of regenerate_idx_list:", len(regenerate_idx_list))
    print("regenerate_idx_list:", regenerate_idx_list)


def merge_generation_files():
    ## for merging files with content_filter
    ## doc2dial
    datapath_main = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/doc2dial/gpt_4_based_on_ftdragon_chatgptgen7k_chunk150.txt"
    datapath_content_filter = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/doc2dial/regenerate_content_filter_cases.txt"

    ## quac
    # datapath_main = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/quac/gpt_4_based_on_ftdragon_chatgptgen7k_chunk150.txt"
    # datapath_content_filter = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/quac/regenerate_content_filter_cases.txt"

    with open(datapath_main, "r") as f:
        data_list_main = f.readlines()
    
    with open(datapath_content_filter, "r") as f:
        data_list_content_filter = f.readlines()

    final_data_list = []
    content_filter_idx = 0
    for item in data_list_main:
        item = item.strip()
        if item == "got content_filter. needs to regenerate":
            item = data_list_content_filter[content_filter_idx].strip()
            content_filter_idx += 1
        final_data_list.append(item)

    assert content_filter_idx == len(data_list_content_filter)
    
    ## doc2dial
    output_datapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/doc2dial/gpt_4_based_on_ftdragon_chatgptgen7k_chunk150_final.txt"

    ## quac
    # output_datapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/quac/gpt_4_based_on_ftdragon_chatgptgen7k_chunk150_final.txt"
    
    print("writing final_data_list to %s" % output_datapath)
    with open(output_datapath, "w") as f:
        for item in final_data_list:
            f.write(item + "\n")


def openai_setup():
    # print("====== starting openai_setup =====")
    # Define your credentials and URL
    client_id = "nvssa-prd-e1acndDubakb4dxMJJNhNL-S7va1z5iR5iKFsLKa31M"
    client_secret = "ssap-jxPmcpzSHyt8oe3"
    # Please use this URL for retrieving token https://prod.api.nvidia.com/oauth/api/v1/ssa/default/token
    token_url = "https://prod.api.nvidia.com/oauth/api/v1/ssa/default/token"
    # Please use this Scope for Azure OpenAI: azureopenai-readwrite
    scope = "azureopenai-readwrite"
    token = get_oauth_token(token_url, client_id, client_secret, scope)
    # print("token:", token)

    openai.api_type = "azure"
    openai.api_base = "https://prod.api.nvidia.com/llm/v1/azure/"
    openai.api_version = "2023-08-01-preview"
    openai.api_key = token

if __name__ == "__main__":

    # openai_setup()
    # main_multi_turn()


    ## regenerate content_filter parts
    # check_content_filter()
    merge_generation_files()
