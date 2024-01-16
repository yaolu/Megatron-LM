
import json
import openai
from tqdm import tqdm
import random
import time


API_KEY="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"

# define a retry decorator
def retry_with_exponential_backoff(
    func,
    initial_delay: float = 3,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 5,
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
    start_idx = 0
    try:
        for idx, prompt in enumerate(prompt_list[start_idx:]):
            # print("prompt:")
            # print(prompt)
            openai.api_key = API_KEY

            # MODEL = "gpt-3.5-turbo"
            # MODEL = "gpt-4"
            
            completion = chatcompletions_with_backoff(
                model=MODEL,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                request_timeout=60
            )
            # print("completion:")
            # print(completion)

            answer = completion.choices[0].message.content
            answer = answer.replace("\n", " ")

            print("idx:", idx+start_idx)
            print(answer)
            answer_list.append(answer)
            time.sleep(0.5)

    except Exception as error:
        print("error occurs. current idx:", idx+start_idx)
        print("error msg:", error)

    return answer_list
    

def format_prompt(ctxs, question, topk=5, max_length=2500, single_turn=False, iscoqa=False):
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
        if iscoqa:
            ## for chatgpt on coqa
            tmp_list = question.split("User:")
            tmp_list = tmp_list[1:]

            question = ""
            if len(tmp_list) > 1:
                for item in tmp_list[:-1]:
                    question += "User:" + item
            question += "User: Answer the following question with a short span. The answer needs to be just in a few words." + tmp_list[-1]

        else:
            tmp_list = question.split("User:", 1)  # split will stop at the first "User:"
            
            ## for chatgpt on (doc2dial, qrecc)
            question = "User: Please give an answer in 1-2 sentences." + tmp_list[1]

            ## for chatgpt on (doqa, quac)
            # question = "User: Please give an answer in just one sentences." + tmp_list[1]

            ## for gpt-4
            # question = "User: Please give an answer in just one sentence." + tmp_list[1]

    prompt = system + context + question
    # prompt = system + question

    prompt_length = len(prompt.split())
    if prompt_length > max_length:
        # cut the context
        token_list = context.split()
        dist = prompt_length - max_length
        context = " ".join(token_list[:-dist])
        prompt = system + context + question

        assert len(prompt.split()) <= max_length

    return prompt


def load_data(datapath, single_turn=False, iscoqa=False):
    print("loading data from %s" % datapath)
    with open(datapath, "r") as f:
        data_list = json.load(f)

    prompt_list = []
    for item in tqdm(data_list):
        prompt = format_prompt(item['ctxs'], item['question'], single_turn=single_turn, iscoqa=iscoqa)
        prompt_list.append(prompt)
    
    return prompt_list


def write_to_files(output_datapath, answer_list):

    print("writing to %s" % output_datapath)    
    with open(output_datapath, "a+") as f:
        for answer in answer_list:
            f.write(answer + "\n")


def main_multi_turn():
    ## doc2dial
    # input_datapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/doc2dial/doc2dial_ftdragon_chatgptgen7k_chunk150_QA_test.json"
    
    ## quac
    # input_datapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/quac/quac_ftdragon_chatgptgen7k_chunk150_QA_test.json"

    ## qrecc
    input_datapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/qrecc/qrecc_ftdragon_chatgptgen7k_chunk150_QA_test.json"

    ## coqa
    # input_datapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/coqa/coqa_QA_dev.json"

    ## doqa_cooking
    # input_datapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/doqa/doqa_cooking_QA_test.json"
    ## doqa_movies
    # input_datapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/doqa/doqa_movies_QA_test.json"
    ## doqa_travel
    # input_datapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/doqa/doqa_travel_QA_test.json"

    ## get prompt_list
    # prompt_list = load_data(input_datapath, iscoqa=True)
    prompt_list = load_data(input_datapath)
    print("total length of prompt_list:", len(prompt_list))
    # print(len(prompt_list[95].split()))
    # print(prompt_list[95])


    ## if generate for content_filter_idx_list
    # content_filter_idx_list = [507, 717, 755, 758, 759, 760, 761, 769, 771, 1655, 1658, 2046]
    # content_filter_prompt_list = [prompt_list[idx] for idx in content_filter_idx_list]
    # print("length of content_filter_prompt_list:", len(content_filter_prompt_list))

    ## get answer_list
    MODEL = "gpt-3.5-turbo"
    # MODEL = "gpt-4"

    answer_list = openai_generate_answer(prompt_list, MODEL)
    ## for regenerating content_filtered data
    # answer_list = openai_generate_answer(content_filter_prompt_list, MODEL)

    ## write to output files

    ## doc2dial
    # output_datapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/doc2dial/chatgpt_3.5_turbo_based_on_ftdragon_chatgptgen7k_chunk150.txt"
    # output_datapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/doc2dial/gpt_4_based_on_ftdragon_chatgptgen7k_chunk150.txt"

    ## quac
    # output_datapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/quac/chatgpt_3.5_turbo_based_on_ftdragon_chatgptgen7k_chunk150.txt"
    # output_datapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/quac/gpt_4_based_on_ftdragon_chatgptgen7k_chunk150.txt"
    # output_datapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/quac/regenerate_content_filter_cases.txt"
    
    ## qrecc
    output_datapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/qrecc/chatgpt_3.5_turbo_based_on_ftdragon_chatgptgen7k_chunk150.txt"
    # output_datapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/qrecc/gpt_4_based_on_ftdragon_chatgptgen7k_chunk150.txt"

    ## coqa
    # output_datapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/coqa/chatgpt_3.5_turbo_gen.txt"
    
    ## doqa
    # output_datapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/doqa/chatgpt_3.5_turbo_gen_doqa_cooking.txt"
    # output_datapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/doqa/chatgpt_3.5_turbo_gen_doqa_movies.txt"
    # output_datapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/doqa/chatgpt_3.5_turbo_gen_doqa_travel.txt"

    write_to_files(output_datapath, answer_list)


def main_single_turn():
    ## ford
    # input_datapath = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved/test.json"

    ## nvit
    # input_datapath = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved/test.json"

    ## landrover
    # input_datapath = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved/test.json"

    ## sandia
    input_datapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/sandia_e5_unsupervised_retriever_chunkbysents300_retrieved/test.json"

    ## get prompt_list
    prompt_list = load_data(input_datapath, single_turn=True)
    print("length of the total prompt_list:", len(prompt_list))
    # print(len(prompt_list[107].split()))


    ## get answer_list
    # MODEL = "gpt-4"
    MODEL = "gpt-3.5-turbo"
    answer_list = openai_generate_answer(prompt_list, MODEL)

    ## write to output files
    ## ford
    # output_datapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/ford/gpt_4_answers_v2.txt"

    ## nvit
    # output_datapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/nvit/gpt_4_answers_v2.txt"

    ## landrover
    # output_datapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/landrover/gpt_4_answers_v2.txt"

    ## sandia
    output_datapath = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/sandia_e5_unsupervised_retriever_chunkbysents300_retrieved/gpt_4_answers_v2.txt"
    write_to_files(output_datapath, answer_list)


if __name__ == "__main__":
    main_multi_turn()
    # main_single_turn()
