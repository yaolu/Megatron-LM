from tqdm import tqdm
import string
import json
from metrics import F1Metric


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
    # model_name = "qa_blendv12_pp1_same_format_ctx1_43b_64_3e-7"
    # model_name = "gpt3-43b-pretraining-gpt-fitting-tp8pp1"
    # model_name = "sft_gpt-fitting-pp1_same_format_ctx1_43b_128_5e-6"
    # model_name = "retro-sft_pp1_same_format_ctx1_43b_128_5e-6"
    model_name = "retro-multiturn_qa_blendv2_retro_1e-8_conv_quiet_cockatoo_pp1_addmultiturn_same_format_ctx1_43b_64_3e-7"
    model_name = "multiturn_qa_blendv2_gpt_1e-8_conv_quiet_cockatoo_pp1_addmultiturn_same_format_ctx1_43b_64_3e-7"
    model_name = "multiturn_qa_blendv2_nemo_gpt_1e-8_conv_quiet_cockatoo_pp1_addmultiturn_same_format_ctx1_43b_64_3e-7"
    model_name = "sft_pp1_same_format_ctx1_43b_128_5e-6"
    # model_name = "megatron_sft_quiet_cockatoo_tp8_pp1"

    ckpt_path = "/lustre/fsw/adlr/adlr-nlp/boxinw/sft-megatron-lm/checkpoints/applications/{}/".format(model_name)
    n_ctx = 5
    iter = 3000
    iter = 1000
    # iter = 1
    # iter = 6000
    # iter = 1500
    # iter = 4500

    prediction_file = ckpt_path + "/att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_{}_generate_43b_test_greedy_0_250_1000_ret.txt.v2".format(
        n_ctx)
    prediction_file = ckpt_path + "/att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_{}_generate_43b_test_greedy_0_250_{}_ret.txt.v2".format(
        n_ctx, iter)
    # prediction_file = ckpt_path + "/att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_{}_generate_43b_test_greedy_0_250_0_ret.txt.v2".format(n_ctx)
    # prediction_file = ckpt_path + "/att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_{}_generate_43b_test_greedy_0_250_1_ret.txt.v2".format(n_ctx)
    # prediction_file = ckpt_path + "/att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_{}_generate_43b_test_greedy_0_250_32552_ret.txt.v2".format(n_ctx)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/data/att/att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved/test.json"
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)

    prediction_file = ckpt_path + "/ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved_{}_generate_43b_test_greedy_0_250_1000_ret.txt.v2".format(
        n_ctx)
    prediction_file = ckpt_path + "/ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved_{}_generate_43b_test_greedy_0_250_{}_ret.txt.v2".format(
        n_ctx, iter)
    # prediction_file = ckpt_path + "/ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved_{}_generate_43b_test_greedy_0_250_0_ret.txt.v2".format(n_ctx)
    # prediction_file = ckpt_path + "/ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved_{}_generate_43b_test_greedy_0_250_1_ret.txt.v2".format(n_ctx)
    # prediction_file = ckpt_path + "/ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved_{}_generate_43b_test_greedy_0_250_32552_ret.txt.v2".format(n_ctx)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved/test.json"
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)
    #
    prediction_file = ckpt_path + "/nq_{}_generate_43b_test_greedy_0_200_1000_ret.txt.v2".format(n_ctx)
    prediction_file = ckpt_path + "/nq_{}_generate_43b_test_greedy_0_200_{}_ret.txt.v2".format(n_ctx, iter)
    # prediction_file = ckpt_path + "/nq_{}_generate_43b_test_greedy_0_200_0_ret.txt.v2".format(n_ctx)
    # prediction_file = ckpt_path + "/nq_{}_generate_43b_test_greedy_0_200_1_ret.txt.v2".format(n_ctx)
    # prediction_file = ckpt_path + "/nq_{}_generate_43b_test_greedy_0_200_32552_ret.txt.v2".format(n_ctx)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/NQ/test.json"
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)

    prediction_file = ckpt_path + "/nv_benefits_dragon_retriever300_retrieved_generic_{}_generate_43b_test_greedy_0_250_1000_ret.txt.v2".format(
        n_ctx)
    prediction_file = ckpt_path + "/nv_benefits_dragon_retriever300_retrieved_generic_{}_generate_43b_test_greedy_0_250_{}_ret.txt.v2".format(
        n_ctx, iter)
    # prediction_file = ckpt_path + "/nv_benefits_dragon_retriever300_retrieved_generic_{}_generate_43b_test_greedy_0_250_0_ret.txt.v2".format(n_ctx)
    # prediction_file = ckpt_path + "/nv_benefits_dragon_retriever300_retrieved_generic_{}_generate_43b_test_greedy_0_250_1_ret.txt.v2".format(n_ctx)
    # prediction_file = ckpt_path + "/nv_benefits_dragon_retriever300_retrieved_generic_{}_generate_43b_test_greedy_0_250_32552_ret.txt.v2".format(n_ctx)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/nv_benefits_dragon_retriever300_retrieved_generic/test.json"  # for single-turn
    # ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa//nv_benefits_dragon_retriever300_retrieved_generic/test.json"
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)

    prediction_file = ckpt_path + "/NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_{}_generate_43b_test_greedy_0_250_1000_ret.txt.v2".format(
        n_ctx)
    prediction_file = ckpt_path + "/NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_{}_generate_43b_test_greedy_0_250_{}_ret.txt.v2".format(
        n_ctx, iter)
    # prediction_file = ckpt_path + "/NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_{}_generate_43b_test_greedy_0_250_0_ret.txt.v2".format(n_ctx)
    # prediction_file = ckpt_path + "/NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_{}_generate_43b_test_greedy_0_250_1_ret.txt.v2".format(n_ctx)
    # prediction_file = ckpt_path + "/NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_{}_generate_43b_test_greedy_0_250_32552_ret.txt.v2".format(n_ctx)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved/test.json"
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)


    prediction_file = ckpt_path + "/Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_{}_generate_43b_test_greedy_0_250_1000_ret.txt.v2".format(
        n_ctx)
    prediction_file = ckpt_path + "/Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_{}_generate_43b_test_greedy_0_250_{}_ret.txt.v2".format(
        n_ctx, iter)
    # prediction_file = ckpt_path + "/Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_{}_generate_43b_test_greedy_0_250_0_ret.txt.v2".format(n_ctx)
    # prediction_file = ckpt_path + "/Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_{}_generate_43b_test_greedy_0_250_1_ret.txt.v2".format(n_ctx)
    # prediction_file = ckpt_path + "/Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_{}_generate_43b_test_greedy_0_250_32552_ret.txt.v2".format(n_ctx)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved/test.json"
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)


    prediction_file = ckpt_path + "/landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved_{}_generate_43b_test_greedy_0_250_1000_ret.txt.v2".format(
        n_ctx)
    prediction_file = ckpt_path + "/landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved_{}_generate_43b_test_greedy_0_250_{}_ret.txt.v2".format(
        n_ctx, iter)
    # prediction_file = ckpt_path + "/landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved_{}_generate_43b_test_greedy_0_250_0_ret.txt.v2".format(n_ctx)
    # prediction_file = ckpt_path + "/landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved_{}_generate_43b_test_greedy_0_250_1_ret.txt.v2".format(n_ctx)
    # prediction_file = ckpt_path + "/landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved_{}_generate_43b_test_greedy_0_250_32552_ret.txt.v2".format(n_ctx)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved/test.json"
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)


    # prediction_file = ckpt_path + "/inference_input_retriever_dragon_msmarcominilm_doc2dial_{}_generate_43b_test_greedy_0_250_1000_ret.txt.v2".format(n_ctx)
    prediction_file = ckpt_path + "/inference_input_retriever_dragon_msmarcominilm_doc2dial_{}_generate_43b_test_greedy_0_250_{}_ret.txt.v2".format(
        n_ctx, iter)
    # prediction_file = ckpt_path + "/inference_input_retriever_dragon_msmarcominilm_doc2dial_{}_generate_43b_test_greedy_0_250_0_ret.txt.v2".format(n_ctx)
    # prediction_file = ckpt_path + "/inference_input_retriever_dragon_msmarcominilm_doc2dial_{}_generate_43b_test_greedy_0_250_1_ret.txt.v2".format(n_ctx)
    # prediction_file = ckpt_path + "/inference_input_retriever_dragon_msmarcominilm_doc2dial_{}_generate_43b_test_greedy_0_250_32552_ret.txt.v2".format(n_ctx)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/inference_input_retriever_dragon_msmarcominilm_doc2dial/test.json"
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)


    prediction_file = ckpt_path + "/newsqa_1_generate_43b_test_greedy_0_250_{}.txt.v2".format(iter)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/instruction_tuning_data/newsqa/test.json"
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)

    prediction_file = ckpt_path + "/squad2.0_1_generate_43b_test_greedy_0_250_{}.txt.v2".format(iter)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/instruction_tuning_data/squad2.0/test.json"
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)

    prediction_file = ckpt_path + "/squad1.1_1_generate_43b_test_greedy_0_250_{}.txt.v2".format(iter)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/instruction_tuning_data/squad1.1/test.json"
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)

    prediction_file = ckpt_path + "/ROPES_1_generate_43b_test_greedy_0_250_{}.txt.v2".format(iter)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/instruction_tuning_data/ROPES/test.json"
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)

    prediction_file = ckpt_path + "/Quoref_1_generate_43b_test_greedy_0_250_{}.txt.v2".format(iter)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/instruction_tuning_data/Quoref/test.json"
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)

    prediction_file = ckpt_path + "/NarrativeQA_1_generate_43b_test_greedy_0_250_{}.txt.v2".format(iter)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/instruction_tuning_data/NarrativeQA/test.json"
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)

    prediction_file = ckpt_path + "/drop_1_generate_43b_test_greedy_0_250_{}.txt.v2".format(iter)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/instruction_tuning_data/drop/test.json"
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)
