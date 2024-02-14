# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.


import sys
import os
from tqdm import tqdm
import string
import json
import regex
import numpy as np

sys.path.append(os.path.abspath(os.path.join(
    os.path.join(os.path.dirname(__file__), "../../../../"))))
from tools.retro.text_generation.metrics import F1Metric


def normalize_answer(s):
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        if type(text) == dict:
            return text['text'].lower()
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_f1_score(predicted_answers, groundtruth_answer, exp_name="default"):
    """Evaluating F1 Score"""
    print(len(predicted_answers), len(groundtruth_answer))
    if len(predicted_answers) != len(groundtruth_answer):
        groundtruth_answer = groundtruth_answer[:len(predicted_answers)]

    guess_list = []
    answer_list = []

    assert len(guess_list) == len(answer_list), \
        "lengths of guess and answer are different!"

    for pred, ans in zip(predicted_answers, groundtruth_answer):
        pred = pred.strip()
        if type(ans) == str:
            ans = ans.strip()
        elif type(ans) == dict:
            ans = ans['text'].strip()
        elif ans == None:
            continue
        if "<|endoftext|>" in pred:
            pred = pred.replace("<|endoftext|>", "")
        if ans == "no_passages_used":
            ans = ""
        guess_list.append(pred)
        answer_list.append(ans)

    precision, recall, f1 = F1Metric.compute_all_pairs(guess_list, answer_list)
    print('Method: %s; Precision: %.4f; recall: %.4f; f1: %.4f' % ( \
        exp_name, precision, recall, f1))


def load_groundtruth_file(data_file):
    with open(data_file, "r") as f:
        nq_examples = json.load(f)

    data = []
    for instance in nq_examples:
        if "answers" in instance:
            answers = instance["answers"]
            if len(answers) < 1:
                answers = [None]
        elif "answer" in instance:
            if type(instance["answer"]) is str:
                answers = [instance["answer"]]
            elif type(instance["answer"]) is list:
                answers = instance["answer"]
            else:
                answers = [str(instance["answer"])]
        else:
            raise ValueError("need to have answer or answers")
        data.append(answers[0])

    return data


def read_prediction(prediction_file):
    prediction_list = []
    print('reading %s' % prediction_file)
    with open(prediction_file, "r") as f:
        for i, line in enumerate(tqdm(f)):
            if prediction_file.endswith("jsonl"):
                line = json.loads(line)["pred"]
                # print(line)
            line = line.replace("Answer:", "")
            line = line.replace("Answer: ", "")
            line = line.replace('????  ', "")
            line = line.replace('A: ', "")
            line = line.replace("A:", "")

            line = line.strip()

            if "<|endoftext|>" in line:
                line = line.replace("<|endoftext|>", "")
            line = normalize_answer(line)  # normalize the answer
            prediction_list.append(line)

    return prediction_list


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def ems(prediction, ground_truths):
    return max([exact_match_score(prediction, gt) for gt in ground_truths])


def evaluate_ems(prediction_file, ground_truth_file, dev_num=3000):
    prediction_list = read_prediction(prediction_file)
    ground_truths_list = []

    if ground_truth_file.endswith(('txt', 'lst')):
        raw_data = open(ground_truth_file, 'r')
    else:
        with open(ground_truth_file, 'r') as f:
            raw_data = json.load(f)
    if "dev" in ground_truth_file:
        raw_data = raw_data[:dev_num]
        prediction_list = prediction_list[:dev_num]

    for each in raw_data:
        if ground_truth_file.endswith('txt'):
            each = json.loads(each)

        if 'answers' in each:
            ground_truths_list.append(each['answers'])
        elif 'answer' in each:
            ground_truths_list.append(each['answer'])
        else:
            ground_truths_list.append([each])

    exactmatch = []

    good_example_list = []
    for i, each in enumerate(prediction_list):
        try:
            score = ems(each, ground_truths_list[i])
        except ValueError as e:
            continue
        exactmatch.append(score)
        if score:
            good_example_list.append(i)

    final_em_score = np.mean(exactmatch)

    print('Exact Match: %.4f;' % final_em_score)

    print('done :-)')

    return final_em_score, exactmatch


def load_prediction(data_file):
    data = []
    with open(data_file, "r") as f:
        for line in f.readlines():
            data.append(line.strip())

    return data


def evaluate_f1(ground_truth_file, prediction_file, reduced_test_only=False):
    groundtruth_answer = load_groundtruth_file(ground_truth_file)
    predicted_answers = load_prediction(prediction_file)
    if not reduced_test_only:
        compute_f1_score(predicted_answers, groundtruth_answer)


if __name__ == "__main__":
    model_names = []
    # model_names += "retro-open_inst_pp1_same_format_ctx1_843m_128_5e-6",
    # model_names += "retro-qc_pp1_same_format_ctx1_843m_128_5e-6",
    #
    # model_names += "retro-open_inst_pp1_same_format_ctx1_43b_128_5e-6",
    # model_names += "retro-qc_pp1_same_format_ctx1_43b_128_5e-6",
    # model_names += "llama2_text_70b_with_qc",
    # model_names += "llama2_text_70b_pp1",
    # model_names += "llama2_chat_70b_pp1",
    model_names += "llama2_text_13b",

    for model_name in model_names:
        ckpt_path = "/lustre/fsw/adlr/adlr-nlp/boxinw/llama-megatron/checkpoints/applications/{}/".format(model_name)
        ckpt_path = "/lustre/fs4/portfolios/adlr/users/boxinw/llama-megatron/checkpoints/applications/{}/".format(model_name)

        n_ctx = 5
        n_enc = 2
        iter = 1000
        model_param = "70b"
        model_param = "13b"
        # iter = "release"

        # # prediction_file = ckpt_path + "nq_{}_changeformat_generate_70b_test_greedy_0_20000_ret.txt.v2".format(n_ctx)
        # prediction_file = ckpt_path + "nq_{}_changeformat_generate_13b_test_greedy_0_200_ret.txt.v2".format(n_ctx)
        # ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/instruction_tuning_data/NQ/test.json"
        # ground_truth_file = "/lustre/fs4/portfolios/adlr/users/boxinw/instruction_tuning_data/NQ/test.json"
        #
        # print(prediction_file)
        # print(ground_truth_file)
        # evaluate_f1(ground_truth_file, prediction_file)
        # evaluate_ems(prediction_file, ground_truth_file)
        #
        #
        # print("=====================================")
        #
        # n_ctx = 1
        # n_enc = 1
        #
        # prediction_file = ckpt_path + "/doc2dial_{}_changeformat_generate_70b_test_greedy_0_20000.txt.v2".format(
        #     n_ctx)
        # ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/doc2dial/doc2dial_ftdragon_chatgptgen7k_chunk150_QA_test.json"
        # print(prediction_file)
        # print(ground_truth_file)
        # evaluate_f1(ground_truth_file, prediction_file)
        #
        # print("=====================================")
        #
        # n_ctx = 5
        # n_enc = 2
        # prediction_file = ckpt_path + "tqa_{}_changeformat_generate_70b_test_greedy_0_20000_ret.txt.v2".format(
        #     n_ctx)
        # # prediction_file = ckpt_path + "/flex_gate_0_reuse_foundational_qa_nq_{}_{}_{}_test_greedy_0_20000_{}.txt".format(
        # #     n_ctx, n_enc, model_param, iter)
        # ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/instruction_tuning_data/TQA/test.json"
        #
        # print(prediction_file)
        # print(ground_truth_file)
        # evaluate_f1(ground_truth_file, prediction_file)
        # evaluate_ems(prediction_file, ground_truth_file)
        #
        # print("=====================================")
        #
        # n_ctx = 1
        # n_enc = 1
        # tasks = ["newsqa", "squad2.0", "squad1.1", "Quoref", "NarrativeQA", "drop"] #  "ROPES",
        # for task in tasks:
        #     prediction_file = ckpt_path + "/{}_{}_changeformat_generate_70b_test_greedy_0_20000.txt.v2".format(
        #         task, n_ctx)
        #     ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/instruction_tuning_data/{}/test.json".format(task)
        #
        #     print(task)
        #     print(prediction_file)
        #     print(ground_truth_file)
        #     evaluate_f1(ground_truth_file, prediction_file)
        #
        #     if "squad" in task:
        #         evaluate_ems(prediction_file, ground_truth_file)
        #     print("=====================================")

        n_ctx = 5
        n_enc = 2
        tasks = ["ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved",
                 "NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved",
                 "landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved"]
        for task in tasks:
            prediction_file = ckpt_path + "/{}_{}_changeformat_generate_70b_test_greedy_0_250_ret.txt.v2".format(
                task, n_ctx)
            ground_truth_file = "/lustre/fs4/portfolios/adlr/users/boxinw/instruction_tuning_data/{}/test.json".format(task)

            print(task)
            print(prediction_file)
            print(ground_truth_file)
            evaluate_f1(ground_truth_file, prediction_file)
            print("=====================================")

