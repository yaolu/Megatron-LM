
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from tqdm import tqdm
import string
import json
from msdp.metrics import F1Metric
from evaluate_f1_fqa_zeroshot_multiturn import evaluate_finqa_and_convfinqa_v2

def compute_f1_score(predicted_answers, groundtruth_answer, exp_name="default"):
    """Evaluating F1 Score"""
    print(len(predicted_answers), len(groundtruth_answer))
    if len(predicted_answers) != len(groundtruth_answer):
        groundtruth_answer = groundtruth_answer[:len(predicted_answers)]

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

    groundtruth_answers = load_groundtruth_file(ground_truth_file)
    predicted_answers = load_prediction(prediction_file)
    # if not reduced_test_only:
    #     compute_f1_score(predicted_answers, groundtruth_answer)
    # groundtruth_answer, predicted_answers = groundtruth_answer[:43], predicted_answers[:43]
    compute_f1_score(predicted_answers, groundtruth_answers)

def separate_cannot_answer(ground_truth_file, prediction_file, topk=5):
    # load ground truth
    with open(ground_truth_file, "r") as f:
        groundtruth_answers = json.load(f)
    # load prediction
    predicted_answers = load_prediction(prediction_file)
    print(len(predicted_answers), len(groundtruth_answers))
    if len(predicted_answers) != len(groundtruth_answers):
        groundtruth_answers = groundtruth_answers[:len(predicted_answers)]

    cannot_answer_idx_list = []
    answerable_idx_list = []
    for idx, item in enumerate(groundtruth_answers):
        answer = item['answer']
        question = item['question']
        noanswer_response = "Sorry. I cannot find the answer based on the context."
        if noanswer_response in question:
            # we only evaluate the case where question doesn't have this noanswer turn
            continue
        if answer == noanswer_response:
            cannot_answer_idx_list.append(idx)
            continue

        ctx_list = []
        for ctx_dict in item['ctxs'][:topk]:
            ctx_list.append(ctx_dict['text'])
        sub_paragraph = item['sub-paragraphs']
        if sub_paragraph in ctx_list:
            answerable_idx_list.append(idx)

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


def evaluate_cannot_answer_acc(ground_truth_file, prediction_file):
    predicted_answers, cannot_answer_idx_list, answerable_idx_list = \
                                separate_cannot_answer(ground_truth_file, prediction_file)

    evaluate_cannot_answer_and_answerable_acc(predicted_answers, cannot_answer_idx_list, answerable_idx_list)


def evaluate_exact_match(ground_truth_file, prediction_file):

    groundtruth_answers = load_groundtruth_file(ground_truth_file)
    predicted_answers = load_prediction(prediction_file)

    print(len(predicted_answers), len(groundtruth_answers))
    if len(predicted_answers) != len(groundtruth_answers):
        groundtruth_answers = groundtruth_answers[:len(predicted_answers)]

    count_exact_match = 0
    for pred, gold in zip(predicted_answers, groundtruth_answers):
        pred = pred.strip()
        gold = gold[0].strip()
        if pred == gold:
            count_exact_match += 1
    
    print("accuracy of exact match: %.4f" % (count_exact_match/len(predicted_answers)))


def calculate_arithmetic_results(input_list, question_list):

    def _get_number(number, score_list):
        if number.startswith("#"):
            score_idx = int(number[1:])
            number = score_list[score_idx]
        elif number.startswith("const_"):
            number = number.replace("const_", "")
            number = float(number)
        elif "%" in number:
            number = number.replace("%", "")
            number = float(number) * 0.01
        else:
            ## convert number like 11,223 to 11223
            number = number.replace(",", "")
            try:
                number = float(number)
            except:
                # print(number)
                number = 0
        
        return number

    output_list = []
    ## calculate the score when there is add, subtract, multiply, divide. if not, just keep the original
    for item, question in zip(input_list, question_list):
        if "add(" in item or "subtract(" in item or "divide(" in item or "multiply(" in item:
            splits = item.split("),")
            score_list = []
            for i, split_item in enumerate(splits):
                split_item.strip()
                assert "add" in split_item or "subtract" in split_item or \
                            "divide" in split_item or "multiply" in split_item

                if "add" in split_item:
                    split_item = split_item.replace("add(", "").replace(")", "")
                    numbers = split_item.split(", ")
                    if len(numbers) == 1:
                        score = float(numbers[0].replace(",",""))
                    else:
                        number0, number1 = numbers[0].strip(), numbers[1].strip()
                        number0 = _get_number(number0, score_list)
                        number1 = _get_number(number1, score_list)
                        score = number0 + number1
                    score_list.append(score)
                
                elif "subtract" in split_item:
                    split_item = split_item.replace("subtract(", "").replace(")", "")
                    numbers = split_item.split(", ")
                    if len(numbers) == 1:
                        score = float(numbers[0].replace(",",""))
                    else:
                        number0, number1 = numbers[0].strip(), numbers[1].strip()
                        number0 = _get_number(number0, score_list)
                        number1 = _get_number(number1, score_list)
                        score = number0 - number1
                    score_list.append(score)

                elif "divide" in split_item:
                    split_item = split_item.replace("divide(", "").replace(")", "")
                    numbers = split_item.split(", ")
                    if len(numbers) == 1:
                        score = float(numbers[0].replace(",",""))
                    else:
                        number0, number1 = numbers[0].strip(), numbers[1].strip()
                        number0 = _get_number(number0, score_list)
                        number1 = _get_number(number1, score_list)
                        if number1 == 0:
                            score = 0
                        else:
                            score = number0 / number1
                    score_list.append(score)

                else:
                    split_item = split_item.replace("multiply(", "").replace(")", "")
                    numbers = split_item.split(", ")
                    if len(numbers) == 1:
                        score = float(numbers[0].replace(",",""))
                    else:
                        number0, number1 = numbers[0].strip(), numbers[1].strip()
                        number0 = _get_number(number0, score_list)
                        number1 = _get_number(number1, score_list)
                        score = number0 * number1
                    score_list.append(score)
            
            if "percentage" in question:
                output_list.append(str(round(score_list[-1]*100, 2)))
            else:
                output_list.append(str(round(score_list[-1], 2)))
        else:
            output_list.append(item)

    return output_list


def evaluate_tatqa(ground_truth_file, prediction_file):
    
    groundtruth_answers = load_groundtruth_file(ground_truth_file)
    predicted_answers = load_prediction(prediction_file)

    print(len(predicted_answers), len(groundtruth_answers))
    if len(predicted_answers) != len(groundtruth_answers):
        groundtruth_answers = groundtruth_answers[:len(predicted_answers)]
    
    with open(ground_truth_file, "r") as f:
        data_list = json.load(f)
        data_list = data_list[:len(predicted_answers)]
    question_list = [item['question'] for item in data_list]

    predicted_answers_updated = calculate_arithmetic_results(predicted_answers, question_list)

    def _processing(answer_list):
        ## convert like $32 -> $ 32; 32% -> 32 %
        updated_answer_list = []
        for answer in answer_list:
            if type(answer) == list:
                answer = answer[0].replace("$", "$ ").replace("%", " %")
                updated_answer_list.append([answer])
            else:
                answer = answer.replace("$", "$ ").replace("%", " %")
                updated_answer_list.append(answer)
        return updated_answer_list

    predicted_answers_updated = _processing(predicted_answers_updated)
    groundtruth_answers = _processing(groundtruth_answers)
    compute_f1_score(predicted_answers_updated, groundtruth_answers)

    ## measure the exact match accuracy of arithmetic cases
    def _is_number(string):
        try:
            float(string)
            return True
        except:
            return False

    # gold_list = []
    arithmetic_idx_list = []
    for idx, item in enumerate(data_list):
        if item['arithmetic']:
            arithmetic_idx_list.append(idx)
            # print("="*80)
            # print("arithmetic case")
            # print("question:", item['question'])
            # print("gold:", item['answer'])
            # print("pred:", predicted_answers_updated[idx])
            # print("pred origin:", predicted_answers[idx])
            # gold_list.append(item['answer'])
        # else:
        #     print("="*80)
        #     print("non-arithmetic case")
        #     print("gold:", item['answer'])
        #     print("pred:", predicted_answers_updated[idx])

    # pred_list = [predicted_answers_updated[idx] for idx in arithmetic_idx_list]
    # assert len(pred_list) == len(gold_list)
    num_exact_match = 0
    for arithmetic_idx in arithmetic_idx_list:
        pred = predicted_answers_updated[arithmetic_idx]
        gold = groundtruth_answers[arithmetic_idx][0]
        # print("="*80)
        # print(pred, gold)
        if _is_number(pred) and float(pred) == float(gold):
            num_exact_match += 1

    acc = num_exact_match / len(arithmetic_idx_list)
    print("exact match accuracy for arithmetic cases: %.4f" % acc)            


if __name__ == "__main__":

    # model_name = "multiturn_qa_blendv2_llama2_text_70b_with_qc_multiturn_same_format_ctx1_70b_64_3e-7"
    # model_name = "multiturn_qa_blendv2_llama2_chat_70b_multiturn_same_format_ctx1_70b_64_3e-7"
    # model_name = "multiturn_qa_blendv2_llama2_chat_70b_multiturn_same_format_ctx1_70b_64_6e-7"
    # model_name = "multiturn_qa_blend_commercial_v9_llama2_text_70b_with_qc_multiturn_same_format_ctx1_70b_64_3e-7"
    # model_name = "multiturn_qa_blend_commercial_v10_llama2_text_70b_with_qc_multiturn_same_format_ctx1_70b_64_3e-7"
    # model_name = "multiturn_qa_blend_commercial_v12_llama2_text_70b_with_qc_multiturn_same_format_ctx1_70b_64_3e-7"
    
    # model_name = "multiturn_qa_blendv5_llama2_text_70b_with_qc_multiturn_same_format_ctx1_70b_64_3e-7"
    # model_name = "multiturn_qa_blendv6_llama2_text_70b_with_qc_multiturn_same_format_ctx1_70b_64_3e-7"
    # model_name = "multiturn_qa_blendv7_llama2_text_70b_with_qc_multiturn_same_format_ctx1_70b_64_3e-7"
    # model_name = "multiturn_qa_blend_commercial_v15_llama2_text_70b_with_qc_multiturn_same_format_ctx1_70b_64_3e-7"

    # model_name = "sft_blend_llama2_text_70b_same_format_ctx1_70b_128_5e-6"

    ## finance / tabular checkpoints
    # model_name = "multiturn_qa_blend_commercial_finance_v2_llama2_text_70b_with_qc_multiturn_same_format_ctx1_70b_64_3e-7"
    # model_name = "multiturn_qa_blend_finance_v2_llama2_text_70b_with_qc_multiturn_same_format_ctx1_70b_64_3e-7"
    # model_name = "multiturn_qa_blend_finance_v3_llama2_text_70b_with_qc_multiturn_same_format_ctx1_70b_64_3e-7"
    # model_name = "multiturn_qa_blend_finance_v4_llama2_text_70b_with_qc_multiturn_same_format_ctx1_70b_64_3e-7"
    # model_name = "multiturn_qa_blend_finance_v5_llama2_text_70b_with_qc_multiturn_same_format_ctx1_70b_64_3e-7"
    # model_name = "multiturn_qa_blend_finance_v6_llama2_text_70b_with_qc_multiturn_same_format_ctx1_70b_64_3e-7"
    # model_name = "multiturn_qa_blend_commercial_v19_llama2_text_70b_with_qc_multiturn_same_format_ctx1_70b_64_3e-7"
    model_name = "multiturn_qa_blend_commercial_v19_1_llama2_text_13b_with_qc_multiturn_same_format_ctx1_13b_64_3e-7"


    ckpt_path="/lustre/fsw/adlr/adlr-nlp/zihanl/inform/foundational-qa/llama-2/checkpoints/applications/{}".format(model_name)
    n_ctx=5


    ## single-turn (batch-1)
    prediction_file = ckpt_path + "/att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_{}_generate_13b_test_greedy_0_250_3600_ret.txt.v2".format(n_ctx)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/data/att/att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved/test.json"
    print("-"*80)
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)

    prediction_file = ckpt_path + "/ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved_{}_generate_13b_test_greedy_0_250_3600_ret.txt.v2".format(n_ctx)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved/test.json"
    print("-"*80)
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)

    prediction_file = ckpt_path + "/nq_{}_generate_13b_test_greedy_0_200_3600_ret.txt.v2".format(n_ctx)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/NQ/test.json"
    print("-"*80)
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)

    prediction_file = ckpt_path + "/nv_benefits_dragon_retriever300_retrieved_generic_{}_generate_13b_test_greedy_0_250_3600_ret.txt.v2".format(n_ctx)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/nv_benefits_dragon_retriever300_retrieved_generic/test.json"
    print("-"*80)
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)

    prediction_file = ckpt_path + "/NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_{}_generate_13b_test_greedy_0_250_3600_ret.txt.v2".format(n_ctx)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved/test.json"
    print("-"*80)
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)

    prediction_file = ckpt_path + "/Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_{}_generate_13b_test_greedy_0_250_3600_ret.txt.v2".format(n_ctx)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved/test.json"
    print("-"*80)
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)

    prediction_file = ckpt_path + "/landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved_{}_generate_13b_test_greedy_0_250_3600_ret.txt.v2".format(n_ctx)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved/test.json"
    print("-"*80)
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)

    prediction_file = ckpt_path + "/sandia_{}_generate_13b_test_greedy_0_250_3600_ret.txt.v2".format(n_ctx)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/sandia_e5_unsupervised_retriever_chunkbysents300_retrieved/test.json"
    print("-"*80)
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)

    # ## finance / tabular data
    # prediction_file = ckpt_path + "/fetaqa_{}_generate_70b_test_greedy_0_1001_3437_ret.txt.v2".format(n_ctx)
    # ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/fetaqa/fetaqa_QA_dev.json"
    # print("-"*80)
    # print(prediction_file)
    # print(ground_truth_file)
    # evaluate_f1(ground_truth_file, prediction_file)

    # # prediction_file = ckpt_path + "/tatqav2_{}_generate_70b_test_greedy_0_1000_3437_ret.txt.v2".format(n_ctx)
    # # ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/tatqav2/dev.json"
    # # print("-"*80)
    # # print(prediction_file)
    # # print(ground_truth_file)
    # # evaluate_tatqa(ground_truth_file, prediction_file)

    # prediction_file = ckpt_path + "/WikiTableQuestions_{}_generate_70b_test_greedy_0_2200_3437_ret.txt.v2".format(n_ctx)
    # ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/WikiTableQuestions/WikiTableQuestions_QA_test.json"
    # print("-"*80)
    # print(prediction_file)
    # print(ground_truth_file)
    # ## evaluate_exact_match(ground_truth_file, prediction_file)
    # evaluate_f1(ground_truth_file, prediction_file)

    # ## financial datasets
    # prediction_file = ckpt_path + "/finqav2_{}_generate_70b_test_greedy_0_1500_3437_ret.txt.v2".format(n_ctx)
    # ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/finqav2/finqav2_QA_dev.json"
    # print("-"*80)
    # print(prediction_file)
    # print(ground_truth_file)
    # evaluate_finqa_and_convfinqa_v2(ground_truth_file, prediction_file)

    # prediction_file = ckpt_path + "/convfinqav3_{}_generate_70b_test_greedy_0_1500_3437_ret.txt.v2".format(n_ctx)
    # ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/convfinqav3/convfinqav3_QA_dev.json"
    # print("-"*80)
    # print(prediction_file)
    # print(ground_truth_file)
    # evaluate_finqa_and_convfinqa_v2(ground_truth_file, prediction_file)
    
    # prediction_file = ckpt_path + "/HybridQA_{}_generate_70b_test_greedy_0_1500_3437_ret.txt.v2".format(n_ctx)
    # ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/HybridQA/HybridQA_QA_dev.json"
    # print("-"*80)
    # print(prediction_file)
    # print(ground_truth_file)
    # evaluate_f1(ground_truth_file, prediction_file)

    
    # ## single-turn (batch-2)
    # prediction_file = ckpt_path + "/BioASQ_{}_generate_70b_test_greedy_0_1000_3437_ret.txt.v2".format(n_ctx)
    # ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/BioASQ/test.json"
    # print("-"*80)
    # print(prediction_file)
    # print(ground_truth_file)
    # evaluate_f1(ground_truth_file, prediction_file)

    # prediction_file = ckpt_path + "/DuoRC_ParaphraseRC_{}_generate_70b_test_greedy_0_1000_3437_ret.txt.v2".format(n_ctx)
    # ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/DuoRC_ParaphraseRC/test.json"
    # print("-"*80)
    # print(prediction_file)
    # print(ground_truth_file)
    # evaluate_f1(ground_truth_file, prediction_file)

    # prediction_file = ckpt_path + "/boolq_{}_generate_70b_test_greedy_0_1000_3437_ret.txt.v2".format(n_ctx)
    # ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/boolq/test.json"
    # print("-"*80)
    # print(prediction_file)
    # print(ground_truth_file)
    # evaluate_f1(ground_truth_file, prediction_file)

    # prediction_file = ckpt_path + "/msmarco_{}_generate_70b_test_greedy_0_1000_3437_ret.txt.v2".format(n_ctx)
    # ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/msmarco/test.json"
    # print("-"*80)
    # print(prediction_file)
    # print(ground_truth_file)
    # evaluate_f1(ground_truth_file, prediction_file)

    # prediction_file = ckpt_path + "/multirc_{}_generate_70b_test_greedy_0_1000_3437_ret.txt.v2".format(n_ctx)
    # ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/multirc/test.json"
    # print("-"*80)
    # print(prediction_file)
    # print(ground_truth_file)
    # evaluate_f1(ground_truth_file, prediction_file)

    # prediction_file = ckpt_path + "/race_{}_generate_70b_test_greedy_0_1000_3437_ret.txt.v2".format(n_ctx)
    # ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/race/test.json"
    # print("-"*80)
    # print(prediction_file)
    # print(ground_truth_file)
    # evaluate_f1(ground_truth_file, prediction_file)

    # prediction_file = ckpt_path + "/TextbookQA_{}_generate_70b_test_greedy_0_1000_3437_ret.txt.v2".format(n_ctx)
    # ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/TextbookQA/test.json"
    # print("-"*80)
    # print(prediction_file)
    # print(ground_truth_file)
    # evaluate_f1(ground_truth_file, prediction_file)


    # ## multi-turn
    # prediction_file = ckpt_path + "/doc2dial_{}_generate_70b_test_greedy_0_1000_3437_ret.txt.v2".format(n_ctx)
    # ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/doc2dial/doc2dial_ftdragon_chatgptgen7k_chunk150_QA_test.json"
    # print("-"*80)
    # print(prediction_file)
    # print(ground_truth_file)
    # evaluate_f1(ground_truth_file, prediction_file)

    # prediction_file = ckpt_path + "/quac_{}_generate_70b_test_greedy_0_1000_3437_ret.txt.v2".format(n_ctx)
    # ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/quac/quac_ftdragon_chatgptgen7k_chunk150_QA_test.json"
    # print("-"*80)
    # print(prediction_file)
    # print(ground_truth_file)
    # evaluate_f1(ground_truth_file, prediction_file)
    # print("accuracy on cannot answer:")
    # evaluate_cannot_answer_acc(ground_truth_file, prediction_file)

    # prediction_file = ckpt_path + "/qrecc_{}_generate_70b_test_greedy_0_1000_3437_ret.txt.v2".format(n_ctx)
    # ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/qrecc/qrecc_ftdragon_chatgptgen7k_chunk150_QA_test.json"
    # print("-"*80)
    # print(prediction_file)
    # print(ground_truth_file)
    # evaluate_f1(ground_truth_file, prediction_file)


    # prediction_file = ckpt_path + "/sharc_{}_generate_70b_test_greedy_0_1000_3437_ret.txt.v2".format(n_ctx)
    # ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/sharc/sharc_ftdragon_chatgptgen7k_chunk150_QA_test.json"
    # print("-"*80)
    # print(prediction_file)
    # print(ground_truth_file)
    # evaluate_f1(ground_truth_file, prediction_file)
    # print("accuracy on cannot answer:")
    # evaluate_cannot_answer_acc(ground_truth_file, prediction_file)


    ### evaluate on nvolve retrieval ctxs

    # ## att_nvolve
    # prediction_file = ckpt_path + "/att_nvolve_{}_generate_70b_test_greedy_0_200_3442_ret.txt.v2".format(n_ctx)
    # ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/data/att/att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved/test.json"
    # print("-"*80)
    # print(prediction_file)
    # print(ground_truth_file)
    # evaluate_f1(ground_truth_file, prediction_file)

    # ## iternal_nvolve
    # prediction_file = ckpt_path + "/iternal_nvolve_{}_generate_70b_test_greedy_0_250_3442_ret.txt.v2".format(n_ctx)
    # ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved/test.json"
    # print("-"*80)
    # print(prediction_file)
    # print(ground_truth_file)
    # evaluate_f1(ground_truth_file, prediction_file)

    # ## nvit_nvolve
    # prediction_file = ckpt_path + "/nvit_nvolve_{}_generate_70b_test_greedy_0_250_3442_ret.txt.v2".format(n_ctx)
    # ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved/test.json"
    # print("-"*80)
    # print(prediction_file)
    # print(ground_truth_file)
    # evaluate_f1(ground_truth_file, prediction_file)

    # ## sandia_nvolve
    # prediction_file = ckpt_path + "/sandia_nvolve_{}_generate_70b_test_greedy_0_250_3442_ret.txt.v2".format(n_ctx)
    # ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/sandia_e5_unsupervised_retriever_chunkbysents300_retrieved/test.json"
    # print("-"*80)
    # print(prediction_file)
    # print(ground_truth_file)
    # evaluate_f1(ground_truth_file, prediction_file)
    