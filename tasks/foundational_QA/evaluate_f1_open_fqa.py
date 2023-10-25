from tqdm import tqdm
import string
import json
from metrics import F1Metric
from evaluate import read_prediction_withprob, read_prediction
import regex
import numpy as np

from evaluate_nlg import evaluate_nlg


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


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def ems(prediction, ground_truths):
    return max([exact_match_score(prediction, gt) for gt in ground_truths])


def evaluate_ems(prediction_file, ground_truth_file, dev_num=3000):
    if prediction_file.endswith('withprob.txt'):
        prediction_list, _ = read_prediction_withprob(prediction_file)
    else:
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
        # print("=============")
        # print(each)
        # print(ground_truths_list[i])
        try:
            score = ems(each, ground_truths_list[i])
        except ValueError as e:
            continue
        # print(score)
        exactmatch.append(score)
        if score:
            good_example_list.append(i)

    print("len of valid answers", len(exactmatch))
    final_em_score = np.mean(exactmatch)

    print('Exact Match: %.4f;' % final_em_score)

    print('done :-)')

    return final_em_score, exactmatch
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
    # model_name = "qa_blendv12_pp1_same_format_ctx1_8b_64_3e-7"
    # model_name = "gpt3-8b-pretraining-gpt-fitting-tp8pp1"
    # model_name = "sft_gpt-fitting-pp1_same_format_ctx1_8b_128_5e-6"
    # model_name = "retro-sft_pp1_same_format_ctx1_8b_128_5e-6"
    model_name = "retro-multiturn_qa_blendv2_retro_1e-8_conv_quiet_cockatoo_pp1_addmultiturn_same_format_ctx1_8b_64_3e-7"
    model_name = "multiturn_qa_blendv2_gpt_1e-8_conv_quiet_cockatoo_pp1_addmultiturn_same_format_ctx1_8b_64_3e-7"
    model_name = "multiturn_qa_blendv2_nemo_gpt_1e-8_conv_quiet_cockatoo_pp1_addmultiturn_same_format_ctx1_8b_64_3e-7"
    model_name = "sft_pp1_same_format_ctx1_8b_128_5e-6"
    model_name = "megatron_sft_quiet_cockatoo_tp8_pp1"
    model_name = "multiturn_qa_blend_commercial_v5_gpt_1e-8_conv_quiet_cockatoo_pp1_addmultiturn_same_format_ctx1_8b_64_3e-7"
    model_name = "multiturn_qa_blend_commercial_v5_gpt_1e-8_conv_quiet_cockatoo_pp1_addmultiturn_bak_same_format_ctx1_8b_64_3e-7"
    model_name = "multiturn_qa_blend_commercial_v5_nemo_gpt_1e-8_conv_quiet_cockatoo_pp1_addmultiturn_same_format_ctx1_8b_64_3e-7"
    model_name = "sft_full-qc-pp1_same_format_ctx1_8b_128_5e-6"
    model_name = "sft_gpt-fitting-full-qc-pp1_same_format_ctx1_8b_128_5e-6"
    model_name = "sft_pp1-3.5t_same_format_ctx1_8b_128_5e-6"
    model_name = "sft_gpt-fitting-pp1-3.5t_same_format_ctx1_8b_128_5e-6"
    # model_name = "multiturn_qa_blend_commercial_v5_gpt_fitting_1e-8_conv_full_quiet_cockatoo_pp1_addmultiturn_same_format_ctx1_8b_64_3e-7"
    # model_name = "multiturn_qa_blend_commercial_v5_gpt_1e-8_conv_full_quiet_cockatoo_pp1_addmultiturn_same_format_ctx1_8b_64_3e-7"
    model_name = "multiturn_qa_blendv2_gpt-fitting_1e-8_conv_quiet_cockatoo_pp1_addmultiturn-3.5t_same_format_ctx1_8b_64_3e-7"
    # model_name = "multiturn_qa_blendv2_gpt_1e-8_conv_quiet_cockatoo_pp1_addmultiturn-3.5t_same_format_ctx1_8b_64_3e-7"
    # model_name = "multiturn_qa_blend_commercial_v15_gpt-fitting_1e-8_conv_quiet_cockatoo_pp1_addmultiturn-commercial-3.5t_same_format_ctx1_8b_64_3e-7"
    # model_name = "multiturn_qa_blend_commercial_v15_gpt_1e-8_conv_quiet_cockatoo_pp1_addmultiturn-commercial-3.5t_same_format_ctx1_8b_64_3e-7"

    ckpt_path = "/lustre/fsw/adlr/adlr-nlp/boxinw/sft-megatron-lm/checkpoints/applications/{}/".format(model_name)
    
    n_ctx = 5
    iter = 3000
    # iter = 1000

    prediction_file = ckpt_path + "/qrecc_1_generate_8b_test_greedy_0_20000_{}.txt.v2".format(iter)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/qrecc/qrecc_ftdragon_chatgptgen7k_chunk150_QA_test.json"
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)
    evaluate_ems(prediction_file, ground_truth_file)

    prediction_file = ckpt_path + "/nq_5_generate_8b_test_greedy_0_20000_{}_ret.txt.v2".format(iter)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/instruction_tuning_data/NQ/test.json"
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)
    evaluate_ems(prediction_file, ground_truth_file)

    prediction_file = ckpt_path + "/tqa_5_generate_8b_test_greedy_0_20000_{}_ret.txt.v2".format(iter)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/instruction_tuning_data/TQA/test.json"
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)
    evaluate_ems(prediction_file, ground_truth_file)

    prediction_file = ckpt_path + "/newsqa_1_generate_8b_test_greedy_0_20000_{}.txt.v2".format(iter)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/instruction_tuning_data/newsqa/test.json"
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)

    prediction_file = ckpt_path + "/squad2.0_1_generate_8b_test_greedy_0_20000_{}.txt.v2".format(iter)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/instruction_tuning_data/squad2.0/test.json"
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)
    evaluate_ems(prediction_file, ground_truth_file)

    prediction_file = ckpt_path + "/squad1.1_1_generate_8b_test_greedy_0_20000_{}.txt.v2".format(iter)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/instruction_tuning_data/squad1.1/test.json"
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)
    evaluate_ems(prediction_file, ground_truth_file)

    prediction_file = ckpt_path + "/ROPES_1_generate_8b_test_greedy_0_20000_{}.txt.v2".format(iter)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/instruction_tuning_data/ROPES/test.json"
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)

    prediction_file = ckpt_path + "/Quoref_1_generate_8b_test_greedy_0_20000_{}.txt.v2".format(iter)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/instruction_tuning_data/Quoref/test.json"
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)

    prediction_file = ckpt_path + "/NarrativeQA_1_generate_8b_test_greedy_0_20000_{}.txt.v2".format(iter)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/instruction_tuning_data/NarrativeQA/test.json"
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)

    prediction_file = ckpt_path + "/drop_1_generate_8b_test_greedy_0_20000_{}.txt.v2".format(iter)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/instruction_tuning_data/drop/test.json"
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)

    prediction_file = ckpt_path + "//doc2dial_1_generate_8b_test_greedy_0_20000_{}.txt.v2".format(iter)
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed/test.txt"
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/doc2dial/doc2dial_ftdragon_chatgptgen7k_chunk150_QA_test.json"
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)
    evaluate_nlg(ground_truth_file, prediction_file)


