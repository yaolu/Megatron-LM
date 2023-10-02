from tqdm import tqdm
import string
import json
from metrics import F1Metric
import numpy as np


def compute_f1_score(predicted_answers, groundtruth_answer, exp_name="default"):
    """Evaluating F1 Score"""
    print(len(predicted_answers), len(groundtruth_answer))
    if len(predicted_answers) != len(groundtruth_answer):
        groundtruth_answer = groundtruth_answer[:len(predicted_answers)]


    assert len(predicted_answers) == len(groundtruth_answer), \
        "lengths of guess and answer are different!"

    f1_scores = []
    for pred, answers in zip(predicted_answers, groundtruth_answer):
        pred = pred.strip()

        f1_score = []
        for ans in answers:
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
            # _, _, f1 = F1Metric.compute_all_pairs([pred], [ans])
            precision, recall, f1 = F1Metric.compute_each_pair(pred, ans, 1)
            f1_score.append(f1)
        f1_score = max(f1_score)
        f1_scores.append(f1_score)
    print('Method: %s; best avg f1: %.4f' % ( \
        exp_name, np.mean(f1_scores)))


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
        data.append(answers)
    return data


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
        return compute_f1_score(predicted_answers, groundtruth_answer)
    # groundtruth_answer, predicted_answers = groundtruth_answer[:43], predicted_answers[:43]
    # compute_f1_score(predicted_answers, groundtruth_answer)
        
if __name__ == "__main__":
    # model_name = "qa_blendv12_pp1_same_format_ctx1_43b_64_3e-7"
    # model_name = "gpt3-43b-pretraining-gpt-fitting-tp8pp1"
    # model_name = "sft_gpt-fitting-pp1_same_format_ctx1_43b_128_5e-6"
    # model_name = "retro-sft_pp1_same_format_ctx1_43b_128_5e-6"
    model_name = "retro-multiturn_qa_blendv2_retro_1e-8_conv_quiet_cockatoo_pp1_addmultiturn_same_format_ctx1_43b_64_3e-7"
    model_name = "multiturn_qa_blendv2_gpt_1e-8_conv_quiet_cockatoo_pp1_addmultiturn_same_format_ctx1_43b_64_3e-7"
    model_name = "multiturn_qa_blendv2_nemo_gpt_1e-8_conv_quiet_cockatoo_pp1_addmultiturn_same_format_ctx1_43b_64_3e-7"
    model_name = "sft_pp1_same_format_ctx1_43b_128_5e-6"
    model_name = "megatron_sft_quiet_cockatoo_tp8_pp1"
    # model_name = "multiturn_qa_blend_commercial_v5_gpt_1e-8_conv_quiet_cockatoo_pp1_addmultiturn_same_format_ctx1_43b_64_3e-7"
    # model_name = "multiturn_qa_blend_commercial_v5_gpt_1e-8_conv_quiet_cockatoo_pp1_addmultiturn_bak_same_format_ctx1_43b_64_3e-7"
    model_name = "multiturn_qa_blend_commercial_v5_nemo_gpt_1e-8_conv_quiet_cockatoo_pp1_addmultiturn_same_format_ctx1_43b_64_3e-7"
    model_name = "sft_full-qc-pp1_same_format_ctx1_43b_128_5e-6"
    # model_name = "sft_gpt-fitting-full-qc-pp1_same_format_ctx1_43b_128_5e-6"
    # model_name = "retro-sft_full-qc-pp1_same_format_ctx1_43b_128_5e-6"
    # model_name = "retro-sft_full-qc-pp1-seed-2333_same_format_ctx1_43b_128_5e-6"
    ckpt_path = "/lustre/fsw/adlr/adlr-nlp/boxinw/sft-megatron-lm/checkpoints/applications/{}/".format(model_name)
    n_ctx = 2
    n_enc = 2
    iter = 3000
    iter = 1000
    # iter = 1
    # iter = 6000
    # iter = 1500
    # iter = 4500

    if 'retro' in model_name:
        prediction_file = ckpt_path + "/flex_reuse_foundational_qa_ELI5-oracle_{}_{}_43b_test_greedy_0_1000_{}.txt".format(
            n_ctx, n_enc, iter)
        ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/instruction_tuning_data/ELI5-oracle/test.json"
        print(prediction_file)
        print(ground_truth_file)
        evaluate_f1(ground_truth_file, prediction_file)

        prediction_file = ckpt_path + "/flex_gate_0_reuse_foundational_qa_ELI5-oracle_{}_{}_43b_test_greedy_0_1000_{}.txt".format(
            n_ctx, n_enc, iter)
        ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/instruction_tuning_data/ELI5-oracle/test.json"
        print(prediction_file)
        print(ground_truth_file)
        evaluate_f1(ground_truth_file, prediction_file)

        for i in range(1, 8):
            prediction_file = ckpt_path + "/template_{}_flex_reuse_foundational_qa_ELI5-oracle_{}_{}_43b_test_greedy_0_1000_{}.txt".format(
                i, n_ctx, n_enc, iter)
            ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/instruction_tuning_data/ELI5-oracle/test.json"
            print(prediction_file)
            print(ground_truth_file)
            evaluate_f1(ground_truth_file, prediction_file)
    else:
        prediction_file = ckpt_path + "/ELI5-oracle_{}_generate_43b_test_greedy_0_1000_{}_ret.txt.v2".format(
            n_ctx, iter)
        ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/instruction_tuning_data/ELI5-oracle/test.json"
        print(prediction_file)
        print(ground_truth_file)
        evaluate_f1(ground_truth_file, prediction_file)

        for i in range(1, 8):
            prediction_file = ckpt_path + "/template_{}_ELI5-oracle_{}_generate_43b_test_greedy_0_1000_{}_ret.txt.v2".format(
                i, n_ctx, iter)
            ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/instruction_tuning_data/ELI5-oracle/test.json"
            print(prediction_file)
            print(ground_truth_file)
            evaluate_f1(ground_truth_file, prediction_file)


