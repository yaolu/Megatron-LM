
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from tqdm import tqdm
import string
import json
from msdp.metrics import F1Metric
from nltk.tokenize import sent_tokenize
import copy
import re

def compute_f1_score(predicted_answers, groundtruth_answer, exp_name="default"):
    """Evaluating F1 Score"""
    print(len(predicted_answers), len(groundtruth_answer))
    if len(predicted_answers) != len(groundtruth_answer):
        groundtruth_answer = groundtruth_answer[:len(predicted_answers)]

    # predicted_answers = predicted_answers[:150]
    # groundtruth_answer = groundtruth_answer[:150]

    guess_list = []
    for answer in predicted_answers:
        answer = answer.strip()
        if "<|endoftext|>" in answer:
            answer = answer.replace("<|endoftext|>", "")

        ## only take 2 sentences (it gets similar or even lower if we just take the first sentence)
        # if len(sent_tokenize(answer)) > 2:
        #     answer = " ".join(sent_tokenize(answer)[:2])
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

    groundtruth_answers = load_groundtruth_file(ground_truth_file)
    if "inscit" in ground_truth_file:
        groundtruth_answers_update = []
        for answers in groundtruth_answers:
            answers_update = []
            for ans in answers:
                if ans != "Sorry. I cannot find the answer based on the context.":
                    answers_update.append(ans)
            assert len(answers_update) > 0
            groundtruth_answers_update.append(copy.deepcopy(answers_update))
        groundtruth_answers = groundtruth_answers_update

    predicted_answers = load_prediction(prediction_file)
    if ("quac" in prediction_file or "doqa" in prediction_file) and ("gpt_4" in prediction_file or "chatgpt_3.5" in prediction_file):
        predicted_answers_new = []
        for pred in predicted_answers:
            pred = pred.lower()
            if "cannot find" in pred or "not able to" in pred or "unable to" in pred or "does not provide" in pred or "cannot provide" in pred or "cannot answer" in pred or "cannot be found" in pred or "cannot be determined" in pred or "don't have information" in pred or "do not have information" in pred or "couldn't find" in pred or "no information in the context" in pred or "does not mention" in pred or "not explicitly mentioned" in pred or "i don't have any" in pred or "i do not have any" in pred or "does not specify" in pred or "doesn't provide" in pred or "doesn't specify" in pred or "there is no information" in pred or "there is no mention" in pred or "not mentioned" in pred or "i don't have enough information" in pred or "there is no specific information" in pred or "there is no specific mention" in pred or "no information found" in pred or "I don't have that information" in pred:
                # print(pred)
                pred = "Sorry. I cannot find the answer based on the context."
            predicted_answers_new.append(pred)
        predicted_answers = predicted_answers_new
    # if not reduced_test_only:
    #     compute_f1_score(predicted_answers, groundtruth_answer)
    # groundtruth_answer, predicted_answers = groundtruth_answer[:43], predicted_answers[:43]
    compute_f1_score(predicted_answers, groundtruth_answers)

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
        if "cannot find" in pred or "not able to" in pred or "unable to" in pred or "does not provide" in pred or "cannot provide" in pred or "cannot answer" in pred or "cannot be found" in pred or "cannot be determined" in pred or "don't have information" in pred or "do not have information" in pred or "couldn't find" in pred or "no information in the context" in pred or "does not mention" in pred or "not explicitly mentioned" in pred or "i don't have any" in pred or "i do not have any" in pred or "does not specify" in pred or "doesn't provide" in pred or "doesn't specify" in pred or "there is no information" in pred or "there is no mention" in pred or "not mentioned" in pred or "i don't have enough information" in pred or "there is no specific information" in pred or "there is no specific mention" in pred or "no information found" in pred or "I don't have that information" in pred:
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


def evaluate_finqa_and_convfinqa_v2(ground_truth_file, prediction_file):

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
        if "+" in pred or "-" in pred or "*" in pred or "/" in pred:
            pred = pred.replace(",", "").replace("$", "").replace("million", "").replace("billion", "").strip()
            pred = pred.replace("percent", "").replace("%", "").strip()
            try:
                pred = round(eval(pred), 3)
            except:
                # print("prediction convert fail:", pred)
                ### keep original pred
                pass
            try:
                gold = round(float(gold), 3)
            except:
                pass
            
            # print("final pred:", pred)
            # print("gold:", gold)
        else:
            pred = pred.replace(",", "").replace("$", "").replace("million", "").replace("billion", "").replace("%", "").replace("percent", "").strip()
            
            try:
                pred = float(pred)
            except:
                # print("prediction convert fail:", pred)
                ### keep original pred
                pass

            # print("="*80)
            # print("final pred:", pred)
            # print("gold:", gold)
        if pred == gold:
            count_exact_match += 1
    
    print("accuracy of exact match: %.4f" % (count_exact_match/len(predicted_answers)))


def evaluate_finqa_and_convfinqa_chatgpt(ground_truth_file, prediction_file):

    def _is_float(string):
        try:
            float(string)
            return True
        except ValueError:
            return False

    with open(ground_truth_file, "r") as f:
        gold_list = json.load(f)
    
    groundtruth_answers = [item['exe_answer'] for item in gold_list]
    question_list = [item['question'].split("User: ")[-1].replace("Assistant:","").strip() for item in gold_list]
    predicted_answers = load_prediction(prediction_file)

    print(len(predicted_answers), len(groundtruth_answers))
    if len(predicted_answers) != len(groundtruth_answers):
        groundtruth_answers = groundtruth_answers[:len(predicted_answers)]

    count_exact_match = 0
    for question, pred, gold in zip(question_list, predicted_answers, groundtruth_answers):
        # print("="*80)
        # print("question:", question)
        # print("original pred:", pred)

        pred = sent_tokenize(pred)[-1]
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
        
        # print("original pred v2:", pred)
        if "+" in pred or "-" in pred or "*" in pred or "/" in pred:
            if "=" in pred:
                idx = pred.index("=")
                pred = pred[0:idx].strip()
            while ":" in pred:
                idx = pred.index(":")
                pred = pred[idx+1:].strip()

            # print("original pred v3:", pred)
            pred = pred.replace(",", "").replace("$", "").replace("million", "").replace("billion", "").strip()
            pred = pred.replace("percent", "").replace("%", "").strip()
            if pred.endswith("."):
                pred = pred[:-1].strip()

            try:
                eval(pred)
            except:
                # print("="*80)
                # print("before:", pred)

                ## remove any tokens that are not numbers or arithmetic symbols
                new_tok_list = []
                for tok in pred.split():
                    if _is_float(tok) or "+" in tok or "-" in tok or "*" in tok or "/" in tok or "(" in tok or ")" in tok:
                        new_tok_list.append(tok)
                pred = " ".join(new_tok_list)

                ## remove brakcets
                pred = re.sub(r'\([^)]*\)', '', pred)
                pred = pred.replace("(", "").replace(")", "").strip()
                # print("after:", pred)
                if pred == "":
                    pred = "EMPTY_ANSWER"
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
            
            if _is_float(pred):
                pred = round(float(pred), 3)
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

    # model_name = "multiturn_qa_blend_finance_v6_1_llama2_text_13b_with_qc_multiturn_same_format_ctx1_13b_64_3e-7"
    # model_name = "multiturn_qa_blend_finance_v6_llama2_text_13b_with_qc_multiturn_same_format_ctx1_13b_64_3e-7"
    # model_name = "multiturn_qa_blend_finance_v6_1_llama2_text_7b_with_qc_multiturn_same_format_ctx1_7b_64_3e-7"
    model_name = "multiturn_qa_blend_commercial_v23_1_llama2_text_7b_with_qc_multiturn_same_format_ctx1_7b_64_3e-7"

    # model_name = "multiturn_qa_blend_commercial_v19_1_llama2_text_13b_with_qc_multiturn_same_format_ctx1_13b_64_3e-7"
    # model_name = "multiturn_qa_blend_commercial_v19_llama2_text_13b_with_qc_multiturn_same_format_ctx1_13b_64_3e-7"
    # model_name = "multiturn_qa_blend_commercial_v19_1_llama2_chat_13b_multiturn_same_format_ctx1_13b_64_3e-7"
    # model_name = "multiturn_qa_blend_commercial_v23_1_llama2_text_13b_with_qc_multiturn_same_format_ctx1_13b_64_3e-7"

    ckpt_path="/lustre/fsw/adlr/adlr-nlp/zihanl/inform/foundational-qa/llama-2/checkpoints/applications/{}".format(model_name)
    n_ctx=5

    # ### multi-turn

    ## doc2dial
    prediction_file = ckpt_path + "/doc2dial_{}_generate_7b_test_greedy_0_4000_3333_ret.txt.v2".format(n_ctx)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/doc2dial/doc2dial_ftdragon_chatgptgen7k_chunk150_QA_test.json"
    print("-"*80)
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)

    ## quac
    prediction_file = ckpt_path + "/quac_{}_generate_7b_test_greedy_0_7500_3333_ret.txt.v2".format(n_ctx)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/quac/quac_ftdragon_chatgptgen7k_chunk150_QA_test.json"
    print("-"*80)
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)
    print("accuracy on cannot answer:")
    evaluate_cannot_answer_acc(ground_truth_file, prediction_file)

    ## qrecc
    prediction_file = ckpt_path + "/qrecc_{}_generate_7b_test_greedy_0_3000_3333_ret.txt.v2".format(n_ctx)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/qrecc/qrecc_ftdragon_chatgptgen7k_chunk150_QA_test.json"
    print("-"*80)
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)


    ## coqa
    prediction_file = ckpt_path + "/coqa_{}_generate_7b_test_greedy_0_8000_3333_ret.txt.v2".format(n_ctx)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/coqa/coqa_QA_dev.json"
    print("-"*80)
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)

    ## doqa_cooking
    prediction_file = ckpt_path + "/doqa_cooking_{}_generate_7b_test_greedy_0_2000_3333_ret.txt.v2".format(n_ctx)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/doqa/doqa_cooking_QA_test.json"
    print("-"*80)
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)
    evaluate_cannot_answer_acc(ground_truth_file, prediction_file, is_doqa=True)

    ## doqa_travel
    prediction_file = ckpt_path + "/doqa_travel_{}_generate_7b_test_greedy_0_2000_3333_ret.txt.v2".format(n_ctx)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/doqa/doqa_travel_QA_test.json"
    print("-"*80)
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)
    evaluate_cannot_answer_acc(ground_truth_file, prediction_file, is_doqa=True)

    ## doqa_movies
    prediction_file = ckpt_path + "/doqa_movies_{}_generate_7b_test_greedy_0_2000_3333_ret.txt.v2".format(n_ctx)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/doqa/doqa_movies_QA_test.json"
    print("-"*80)
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)
    evaluate_cannot_answer_acc(ground_truth_file, prediction_file, is_doqa=True)

    ## convfinqa
    prediction_file = ckpt_path + "/convfinqav3_{}_generate_7b_test_greedy_0_1500_3333_ret.txt.v2".format(n_ctx)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/convfinqav3/convfinqav3_QA_dev.json"
    print("-"*80)
    print(prediction_file)
    print(ground_truth_file)
    evaluate_finqa_and_convfinqa_v2(ground_truth_file, prediction_file)

    ## sqa
    prediction_file = ckpt_path + "/sqa_{}_generate_7b_test_greedy_0_3100_3333_ret.txt.v2".format(n_ctx)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/sqa/sqa_QA_test.json"
    print("-"*80)
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)

    ## topiocqa
    prediction_file = ckpt_path + "/topiocqa_20_generate_7b_test_greedy_0_2600_3333_ret.txt.v2"
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/topiocqa/topiocqa_dev_retrieval_dragon_ft_chatgptgen7k.json"
    print("-"*80)
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)

    ## hybriddial
    prediction_file = ckpt_path + "/hybriddial_5_generate_7b_test_greedy_0_1200_3333_ret.txt.v2"
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/HybridDial/HybridDial_fqa_test.json"
    print("-"*80)
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)

    ## inscit
    prediction_file = ckpt_path + "/inscit_20_generate_7b_test_greedy_0_550_3600_ret.txt.v2"
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/inscit/inscit_dev_retrieval_dragon_ft_chatgptgen7k_with_topic.json"
    print("-"*80)
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)
    
    
    # ## llmware
    # prediction_file = ckpt_path + "/llmware_{}_generate_13b_test_greedy_0_500_3600_ret.txt.v2".format(n_ctx)
    # ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/llmware/test.json"
    # print("-"*80)
    # print(prediction_file)
    # print(ground_truth_file)
    # evaluate_f1(ground_truth_file, prediction_file)


