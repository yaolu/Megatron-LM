
from tqdm import tqdm
import os
import json


def format_qa_history_to_question(qa_history, num_turn=7, all_turns=False):
    question = ""
    if not all_turns:
        qa_history = qa_history[-num_turn:]
    for item in qa_history:
        if item[0] == "user":
            question += "User: " + item[1] + "\n\n"
        else:
            assert item[0] == "agent"
            question += "Assistant: " + item[1] + "\n\n"

    question += "Assistant:"
    
    return question


def parse_coqa(input_datapath, output_datapath):
    with open(input_datapath, "r") as f:
        data_list = json.load(f)['data']

    print("parsing from %s" % input_datapath)
    print("length of data_list:", len(data_list))

    output_datalist = []
    num_unknown = 0
    num_all_unknown = 0
    for item in tqdm(data_list):
        document = item['story']
        doc_dict = {"title": "", "text": document}

        questions = item['questions']
        answers = item["answers"]
        additional_answers_0 = item['additional_answers']['0']
        additional_answers_1 = item['additional_answers']['1']
        additional_answers_2 = item['additional_answers']['2']

        assert len(questions) == len(answers) == len(additional_answers_0) == \
                    len(additional_answers_1) == len(additional_answers_2)
        
        qa_history = []
        for question, answer, answer_0, answer_1, answer_2 in zip(
                questions, answers, additional_answers_0, 
                additional_answers_1, additional_answers_2):
            
            question = question['input_text']
            answer = answer['input_text']
            answer_0 = answer_0['input_text']
            answer_1 = answer_1['input_text']
            answer_2 = answer_2['input_text']

            qa_history.append(("user", question))
            formated_question = format_qa_history_to_question(qa_history)
            qa_history.append(("agent", answer))

            count_unknown_each_turn = 0
            if answer == "unknown":
                count_unknown_each_turn += 1
                answer = "Sorry. I cannot find the answer based on the context."
            if answer_0 == "unknown":
                count_unknown_each_turn += 1
                answer_0 = "Sorry. I cannot find the answer based on the context."
            if answer_1 == "unknown":
                count_unknown_each_turn += 1
                answer_1 = "Sorry. I cannot find the answer based on the context."
            if answer_2 == "unknown":
                count_unknown_each_turn += 1
                answer_2 = "Sorry. I cannot find the answer based on the context."
            
            if count_unknown_each_turn > 0:
                print("="*80)
                print("count_unknown_each_turn:", count_unknown_each_turn)
                print(question)
                num_unknown += 1
                if count_unknown_each_turn == 4:
                    num_all_unknown += 1

            answer_list_each_turn = [answer, answer_0, answer_1, answer_2]

            data_dict = {
                "ctxs": [doc_dict],
                "sub-paragraphs": document,
                "answers": answer_list_each_turn,
                "question": formated_question
            }
            output_datalist.append(data_dict)

    print("num_unknown: %d; num_all_unknown: %d" % (num_unknown, num_all_unknown))
    print("length of output_datalist:", len(output_datalist))
    with open(output_datapath, "w") as f:
        json.dump(output_datalist, f, indent=2)


def parse_doqa(input_datapath, output_datapath):

    with open(input_datapath, "r") as f:
        data_list = json.load(f)['data']

    print("parsing from %s" % input_datapath)
    print("length of data_list:", len(data_list))

    output_datalist = []
    num_unknown = 0
    num_all_unknown = 0

    for item in tqdm(data_list):
        title = item['title']
        document = item['paragraphs'][0]['context']

        ## remove CANNOTANSWER
        document = document.replace("CANNOTANSWER", "").strip()
        doc_dict = {"title": title, "text": document}

        qa_pairs = item['paragraphs'][0]['qas']
        qa_history = []
        for qa in qa_pairs:
            question = qa['question']
            answers = qa['answers']
            orig_answer = qa['orig_answer']['text']
            
            count_unknown_each_turn = 0

            if orig_answer == "CANNOTANSWER":
                count_unknown_each_turn += 1
                orig_answer = "Sorry. I cannot find the answer based on the context."

            answer_list_each_turn = [orig_answer]
            for answer_dict in answers:
                if answer_dict['text'] == "CANNOTANSWER":
                    count_unknown_each_turn += 1
                    answer = "Sorry. I cannot find the answer based on the context."
                else:
                    answer = answer_dict['text']
                answer_list_each_turn.append(answer)

            if count_unknown_each_turn > 0:
                num_unknown += 1
                if count_unknown_each_turn == 1+len(answers):
                    num_all_unknown += 1

            qa_history.append(("user", question))
            formated_question = format_qa_history_to_question(qa_history)
            qa_history.append(("agent", orig_answer))
            
            data_dict = {
                "ctxs": [doc_dict],
                "sub-paragraphs": document,
                "answers": answer_list_each_turn,
                "question": formated_question
            }
            output_datalist.append(data_dict)


    print("num_unknown: %d; num_all_unknown: %d" % (num_unknown, num_all_unknown))
    print("length of output_datalist:", len(output_datalist))
    with open(output_datapath, "w") as f:
        json.dump(output_datalist, f, indent=2)


def main_coqa():
    datafolder = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/coqa"

    input_devpath = os.path.join(datafolder, "CoQA/coqa-dev-v1.0.json")
    output_devpath = os.path.join(datafolder, "coqa_QA_dev.json")
    parse_coqa(input_devpath, output_devpath)

    # input_trainpath = os.path.join(datafolder, "CoQA/coqa-train-v1.0.json")
    # output_trainpath = os.path.join(datafolder, "coqa_QA_train.json")
    # parse_coqa(input_trainpath, output_trainpath)


def main_doqa():
    datafolder = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/doqa"

    input_testpath_cooking = os.path.join(datafolder, "doqa-v2.1/doqa_dataset/doqa-cooking-test-v2.1.json")
    output_testpath_cooking = os.path.join(datafolder, "doqa_cooking_QA_test.json")
    parse_doqa(input_testpath_cooking, output_testpath_cooking)
    
    input_testpath_movies = os.path.join(datafolder, "doqa-v2.1/doqa_dataset/doqa-movies-test-v2.1.json")
    output_testpath_movies = os.path.join(datafolder, "doqa_movies_QA_test.json")
    parse_doqa(input_testpath_movies, output_testpath_movies)

    input_testpath_travel = os.path.join(datafolder, "doqa-v2.1/doqa_dataset/doqa-travel-test-v2.1.json")
    output_testpath_travel = os.path.join(datafolder, "doqa_travel_QA_test.json")
    parse_doqa(input_testpath_travel, output_testpath_travel)



if __name__ == "__main__":
    # main_coqa()
    main_doqa()

