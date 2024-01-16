
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from tqdm import tqdm
import string
import json
from msdp.metrics import F1Metric
from evaluate_f1_fqa_zeroshot import evaluate_cannot_answer_acc

def compute_f1_score(predicted_answers, groundtruth_answer, exp_name="default"):
    """Evaluating F1 Score"""
    print(len(predicted_answers), len(groundtruth_answer))
    if len(predicted_answers) != len(groundtruth_answer):
        groundtruth_answer = groundtruth_answer[:len(predicted_answers)]

    # predicted_answers = predicted_answers[:1000]
    # groundtruth_answer = groundtruth_answer[:1000]

    guess_list = []
    for answer in predicted_answers:
        answer = answer.strip()
        if "<|endoftext|>" in answer:
            answer = answer.replace("<|endoftext|>", "")
        guess_list.append(answer)

    answer_list = []
    for answer in groundtruth_answer:
        # answer = answer.strip()
        # if answer == "no_passages_used":
        #     answer = ""
        answer_list.append(answer)

    assert len(guess_list) == len(answer_list), \
        "lengths of guess and answer are different!"

    precision, recall, f1 = F1Metric.compute_all_pairs(guess_list, answer_list)
    print('Method: %s; Precision: %.4f; recall: %.4f; f1: %.4f' % (\
        exp_name, precision, recall, f1))


def load_groundtruth_file(data_file):
    
    with open(data_file, "r") as f:
        nq_examples = json.load(f)

    data = []
    for instance in nq_examples:
        if "answers" in instance:
            answers = instance["answers"]
        elif "answer" in instance:
            if type(instance["answer"]) is str:
                answers = [instance["answer"]]
            elif type(instance["answer"]) is list:
                answers = instance["answer"]
            else:
                answers = [str(instance["answer"])]
        else:
            raise ValueError("need to have answer or answers")
        # data.append(answers[0])
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
    
    if "llama2_chat" in prediction_file and ("quac" in prediction_file or "doqa" in prediction_file):
        predicted_answers_new = []
        for pred in predicted_answers:
            pred = pred.lower()
            if "i'm not sure" in pred or "cannot find" in pred or "not able to" in pred or "unable to" in pred or "does not provide" in pred or "cannot provide" in pred or "cannot answer" in pred or "cannot be found" in pred or "cannot be determined" in pred or "don't have information" in pred or "do not have information" in pred or "couldn't find" in pred or "no information in the context" in pred or "does not mention" in pred or "not explicitly mentioned" in pred or "i don't have any" in pred or "i do not have any" in pred or "does not specify" in pred or "doesn't provide" in pred or "doesn't specify" in pred or "there is no information" in pred or "there is no mention" in pred or "not mentioned" in pred or "i don't have enough information" in pred or "there is no specific information" in pred or "there is no specific mention" in pred or "no information found" in pred or "I don't have that information" in pred:
                pred = "Sorry. I cannot find the answer based on the context."

            predicted_answers_new.append(pred)
        predicted_answers = predicted_answers_new

    compute_f1_score(predicted_answers, groundtruth_answer)


def separate_cannot_answer(ground_truth_file, prediction_file, topk=5, is_doqa=False):
    # load ground truth
    with open(ground_truth_file, "r") as f:
        groundtruth_answers = json.load(f)
    # load prediction
    predicted_answers = load_prediction(prediction_file)
    print(len(predicted_answers), len(groundtruth_answers))
    if len(predicted_answers) != len(groundtruth_answers):
        groundtruth_answers = groundtruth_answers[:len(predicted_answers)]

    predicted_answers_new = []
    for pred in predicted_answers:
        pred = pred.lower()
        if "i'm not sure" in pred or "cannot find" in pred or "not able to" in pred or "unable to" in pred or "does not provide" in pred or "cannot provide" in pred or "cannot answer" in pred or "cannot be found" in pred or "cannot be determined" in pred or "don't have information" in pred or "do not have information" in pred or "couldn't find" in pred or "no information in the context" in pred or "does not mention" in pred or "not explicitly mentioned" in pred or "i don't have any" in pred or "i do not have any" in pred or "does not specify" in pred or "doesn't provide" in pred or "doesn't specify" in pred or "there is no information" in pred or "there is no mention" in pred or "not mentioned" in pred or "i don't have enough information" in pred or "there is no specific information" in pred or "there is no specific mention" in pred or "no information found" in pred or "I don't have that information" in pred:
            pred = "Sorry. I cannot find the answer based on the context."
        predicted_answers_new.append(pred)
    predicted_answers = predicted_answers_new

    cannot_answer_idx_list = []
    answerable_idx_list = []
    for idx, item in enumerate(groundtruth_answers):
        if is_doqa:
            answer = item["answers"][0]
        else:
            answer = item['answer']
        question = item['question']
        noanswer_response = "Sorry. I cannot find the answer based on the context."
        # if noanswer_response in question:
        #     # we only evaluate the case where question doesn't have this noanswer turn
        #     continue
        if answer == noanswer_response:
            cannot_answer_idx_list.append(idx)
            continue

        answerable_idx_list.append(idx)

        # if is_doqa:
        #     answerable_idx_list.append(idx)
        # else:
        #     ctx_list = []
        #     for ctx_dict in item['ctxs'][:topk]:
        #         ctx_list.append(ctx_dict['text'])
        #     sub_paragraph = item['sub-paragraphs']
        #     if sub_paragraph in ctx_list:
        #         answerable_idx_list.append(idx)

    print("number of cannot answer cases: %d (out of %d)" % (len(cannot_answer_idx_list), len(groundtruth_answers)))
    print("number of answerable cases: %d (out of %d)" % (len(answerable_idx_list), len(groundtruth_answers)))

    return predicted_answers, cannot_answer_idx_list, answerable_idx_list

def evaluate_cannot_answer_and_answerable_acc(predicted_answers, cannot_answer_idx_list, answerable_idx_list):
    # cannot answer
    noanswer_count = 0
    for idx in cannot_answer_idx_list:
        prediction = predicted_answers[idx]
        prediction = prediction.lower()
        # print(prediction)
        if "sorry" in prediction and "cannot find the answer" in prediction:
            # print(prediction)
            noanswer_count += 1
    cannot_answer_acc = noanswer_count / len(cannot_answer_idx_list)
    print("accuracy of cannot answer cases: %.4f" % cannot_answer_acc)

    # answerable
    answerable_count = 0
    for idx in answerable_idx_list:
        prediction = predicted_answers[idx]
        prediction = prediction.lower()
        if "sorry" in prediction and "cannot find the answer" in prediction:
            # print(prediction)
            continue
        answerable_count += 1
    answerable_acc = answerable_count / len(answerable_idx_list)
    print("accuracy of answerable cases: %.4f" % answerable_acc)


def evaluate_cannot_answer_acc(ground_truth_file, prediction_file, is_doqa=False):
    predicted_answers, cannot_answer_idx_list, answerable_idx_list = \
                                separate_cannot_answer(ground_truth_file, prediction_file, is_doqa=is_doqa)

    evaluate_cannot_answer_and_answerable_acc(predicted_answers, cannot_answer_idx_list, answerable_idx_list)


def evaluate_finqa_and_convfinqa(ground_truth_file, prediction_file):

    groundtruth_answers = load_groundtruth_file(ground_truth_file)
    predicted_answers = load_prediction(prediction_file)

    print(len(predicted_answers), len(groundtruth_answers))
    if len(predicted_answers) != len(groundtruth_answers):
        groundtruth_answers = groundtruth_answers[:len(predicted_answers)]

    count_exact_match = 0
    for pred, gold in zip(predicted_answers, groundtruth_answers):
        pred = pred.strip()
        gold = gold.strip()
        if pred == gold:
            count_exact_match += 1
    
    print("accuracy of exact match: %.4f" % (count_exact_match/len(predicted_answers)))


def evaluate_finqa_and_convfinqa_llama2(ground_truth_file, prediction_file):

    def _is_float(string):
        try:
            float(string)
            return True
        except ValueError:
            return False

    with open(ground_truth_file, "r") as f:
        gold_list = json.load(f)
    
    groundtruth_answers = [item['exe_answer'] for item in gold_list]
    predicted_answers = load_prediction(prediction_file)

    print(len(predicted_answers), len(groundtruth_answers))
    if len(predicted_answers) != len(groundtruth_answers):
        groundtruth_answers = groundtruth_answers[:len(predicted_answers)]

    count_exact_match = 0
    for pred, gold in zip(predicted_answers, groundtruth_answers):
        # print("="*80)
        # print("original pred:", pred)
        tok_list = pred.split()
        while "is" in tok_list or "was" in tok_list or "be" in tok_list or "were" in tok_list or "are" in tok_list:
            if "is" in tok_list:
                idx = tok_list.index("is")
                pred = " ".join(tok_list[idx+1:])
            if "was" in tok_list:
                idx = tok_list.index("was")
                pred = " ".join(tok_list[idx+1:])
            if "be" in tok_list:
                idx = tok_list.index("be")
                pred = " ".join(tok_list[idx+1:])
            if "were" in tok_list:
                idx = tok_list.index("were")
                pred = " ".join(tok_list[idx+1:])
            if "are" in tok_list:
                idx = tok_list.index("are")
                pred = " ".join(tok_list[idx+1:])
            tok_list = pred.split()

        if "+" in pred or "-" in pred or "*" in pred or "/" in pred:
            if "=" in pred:
                idx = pred.index("=")
                pred = pred[0:idx].strip()

            pred = pred.replace(",", "").replace("$", "").replace("million", "").replace("billion", "").strip()
            pred = pred.replace("percent", "").replace("%", "").strip()
            if pred.endswith("."):
                pred = pred[:-1].strip()
            
            # print("original pred v2:", pred)
            try:
                eval(pred)
            except:
                if "(" in pred:
                    # print("before:", pred)
                    pred_tmp = pred.split("(")[1]
                    pred_tmp = pred_tmp.replace("(", "").replace(")", "").strip()
                    if pred_tmp != "":
                        pred = pred_tmp
                    # print("after:", pred)

            try:
                pred = round(eval(pred), 3)
            except:
                # print("prediction convert fail:", pred)
                ### keep original pred
                if _is_float(pred.split()[-1]):
                    pred = pred.split()[-1]
                else:
                    for tok in pred.split():
                        if _is_float(tok):
                            pred = tok
                            break
            
            if _is_float(gold):
                gold = round(float(gold), 3)
            
            # print("final pred v1:", pred)
            # print("gold:", gold)
        else:
            pred = pred.replace(",", "").replace("$", "").replace("million", "").replace("billion", "").replace("%", "").replace("percent", "").strip()
            if pred.endswith("."):
                pred = pred[:-1].strip()

            for tok in pred.split():
                if _is_float(tok):
                    pred = tok
                    break

            if _is_float(pred):
                pred = float(pred)

            # print("final pred v2:", pred)
            # print("gold:", gold)
        if pred == gold:
            count_exact_match += 1
        elif type(pred) == float and round(pred/100, 3) == gold:
            # for percentage case
            count_exact_match += 1
    
    print("accuracy of exact match: %.4f" % (count_exact_match/len(predicted_answers)))


if __name__ == "__main__":

    # model_name = "llama2_chat_70b_our_format_multiturn"
    model_name = "llama2_chat_70b"
    # model_name = "llama2_text_70b_with_qc"

    ckpt_path="/lustre/fsw/adlr/adlr-nlp/zihanl/inform/foundational-qa/llama-2/checkpoints/applications/{}".format(model_name)
    n_ctx=5
    
    print("evaluating %s ..." % model_name)
    
    ### multi-turn
    prediction_file = ckpt_path + "/doc2dial_{}_changeformat_generate_70b_test_greedy_0_4000_ret.txt.v2".format(n_ctx)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/doc2dial/doc2dial_ftdragon_chatgptgen7k_chunk150_QA_test.json"
    print("-"*80)
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)

    prediction_file = ckpt_path + "/quac_{}_changeformat_generate_70b_test_greedy_0_8000_ret.txt.v2".format(n_ctx)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/quac/quac_ftdragon_chatgptgen7k_chunk150_QA_test.json"
    print("-"*80)
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)
    print("accuracy on cannot answer:")
    evaluate_cannot_answer_acc(ground_truth_file, prediction_file)

    prediction_file = ckpt_path + "/qrecc_{}_changeformat_generate_70b_test_greedy_0_4000_ret.txt.v2".format(n_ctx)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/qrecc/qrecc_ftdragon_chatgptgen7k_chunk150_QA_test.json"
    print("-"*80)
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)

    ## coqa
    prediction_file = ckpt_path + "/coqa_{}_changeformat_generate_70b_test_greedy_0_8000_ret.txt.v2".format(n_ctx)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/coqa/coqa_QA_dev.json"
    print("-"*80)
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)

    ## doqa_cooking
    prediction_file = ckpt_path + "/doqa_cooking_{}_changeformat_generate_70b_test_greedy_0_2000_ret.txt.v2".format(n_ctx)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/doqa/doqa_cooking_QA_test.json"
    print("-"*80)
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)
    evaluate_cannot_answer_acc(ground_truth_file, prediction_file, is_doqa=True)

    ## doqa_travel
    prediction_file = ckpt_path + "/doqa_travel_{}_changeformat_generate_70b_test_greedy_0_2000_ret.txt.v2".format(n_ctx)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/doqa/doqa_travel_QA_test.json"
    print("-"*80)
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)
    evaluate_cannot_answer_acc(ground_truth_file, prediction_file, is_doqa=True)

    ## doqa_movies
    prediction_file = ckpt_path + "/doqa_movies_{}_changeformat_generate_70b_test_greedy_0_2000_ret.txt.v2".format(n_ctx)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/doqa/doqa_movies_QA_test.json"
    print("-"*80)
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)
    evaluate_cannot_answer_acc(ground_truth_file, prediction_file, is_doqa=True)

    ## convfinqa
    prediction_file = ckpt_path + "/convfinqav3_{}_changeformat_generate_70b_test_greedy_0_1500_ret.txt.v2".format(n_ctx)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/convfinqav3/convfinqav3_QA_dev.json"
    print("-"*80)
    print(prediction_file)
    print(ground_truth_file)
    evaluate_finqa_and_convfinqa_llama2(ground_truth_file, prediction_file)

    # ## finqa
    # prediction_file = ckpt_path + "/finqav2_{}_changeformat_generate_70b_test_greedy_0_1500_ret.txt.v2".format(n_ctx)
    # ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/finqav2/finqav2_QA_dev.json"
    # print("-"*80)
    # print(prediction_file)
    # print(ground_truth_file)
    # evaluate_finqa_and_convfinqa_llama2(ground_truth_file, prediction_file)

    ## sqa
    prediction_file = ckpt_path + "/sqa_{}_changeformat_generate_70b_test_greedy_0_3100_ret.txt.v2".format(n_ctx)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/sqa/sqa_QA_test.json"
    print("-"*80)
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)

    ## topiocqa
    prediction_file = ckpt_path + "/topiocqa_20_changeformat_generate_70b_test_greedy_0_2600_ret.txt.v2"
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/topiocqa/topiocqa_dev_retrieval_dragon_ft_chatgptgen7k.json"
    print("-"*80)
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)

    ## hybriddial
    prediction_file = ckpt_path + "/hybriddial_5_changeformat_generate_70b_test_greedy_0_1200_ret.txt.v2"
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/HybridDial/HybridDial_fqa_test.json"
    print("-"*80)
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)

    ## inscit
    prediction_file = ckpt_path + "/inscit_20_changeformat_generate_70b_test_greedy_0_550_ret.txt.v2"
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/inscit/inscit_dev_retrieval_dragon_ft_chatgptgen7k_with_topic.json"
    print("-"*80)
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)

