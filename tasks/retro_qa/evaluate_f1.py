from tqdm import tqdm
import string
import json
from metrics import F1Metric
from evaluate import evaluate_ems

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
        compute_f1_score(predicted_answers, groundtruth_answer)
    # groundtruth_answer, predicted_answers = groundtruth_answer[:43], predicted_answers[:43]
    # compute_f1_score(predicted_answers, groundtruth_answer)


if __name__ == "__main__":
    model_name = "gpt3-43b-multi-1.1t-gtc/tp8pp1"
    model_name = "gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed"
    # model_name = "gpt3-43b-pretraining-gpt-fitting-tp8pp1"
    model_name = "retro-sft_pp1_same_format_ctx1_43b_128_5e-6"
    model_name = "retro-sft_full-qc-pp1_same_format_ctx1_43b_128_5e-6"
    # model_name = "retro-sft_full-qc-pp1-seed-2333_same_format_ctx1_43b_128_5e-6"
    # model_name = "retro-qa_blendv12_pp1_same_format_ctx1_43b_64_3e-7"
    # ckpt_path = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/{}/".format(model_name)
    # model_name = "retro-multiturn_qa_blendv2_retro_1e-8_conv_quiet_cockatoo_pp1_addmultiturn_same_format_ctx1_43b_64_3e-7"
    # model_name = "retro-multiturn_qa_blend_commercial_v5_retro_1e-8_conv_quiet_cockatoo_pp1_addmultiturn_same_format_ctx1_43b_64_3e-7"
    model_name = "retro-multiturn_qa_blend_commercial_v5_retro_1e-8_conv_full_quiet_cockatoo_pp1_addmultiturn_same_format_ctx1_43b_64_3e-7"
    model_name = "retro-multiturn_qa_blend_commercial_v5_retro_1e-8_conv_full_quiet-seed-2333_cockatoo_pp1_addmultiturn_same_format_ctx1_43b_64_3e-7"
    model_name = "retro-sft_full-qc-pp1-seed-2333_same_format_ctx1_43b_128_5e-6"
    model_name = "retro-sft_full-qc-pp1_same_format_ctx1_43b_128_5e-6"
    ckpt_path = "/lustre/fsw/adlr/adlr-nlp/boxinw/sft-megatron-lm/checkpoints/applications/{}/".format(model_name)
    n_ctx = 5
    n_enc = 2
    iter = 1000
    # iter = 1500
    # iter = 3000


    prediction_file = ckpt_path + "/generate_att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_43b_test_greedy_0_250_472541.txt"
    prediction_file = ckpt_path + "/generate_att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_43b_test_greedy_0_250_472541.txt.period.txt"
    prediction_file = ckpt_path + "/generate_att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_43b_test_greedy_0_250_32552.txt"
    prediction_file = ckpt_path + "/generate_att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_43b_test_greedy_0_250_32552.txt.period.txt"
    prediction_file = ckpt_path + "/chunk_1k_generate_att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_43b_test_greedy_0_250_32552.txt"
    prediction_file = ckpt_path + "/chunk_1k_generate_att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_43b_test_greedy_0_250_32552.txt.period.txt"
    prediction_file = ckpt_path + "/gate_0_chunk_1k_generate_att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_43b_test_greedy_0_250_32552.txt"
    # prediction_file = ckpt_path + "/gate_0_chunk_1k_generate_att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_43b_test_greedy_0_250_32552.txt.period.txt"

    # prediction_file = ckpt_path + "/foundational_qa_att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_5_2_43b_test_greedy_0_250_32552.txt"
    # prediction_file = ckpt_path + "/reuse_foundational_qa_att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_{}_{}_43b_test_greedy_0_250_{}.txt".format(n_ctx, n_enc, iter)
    prediction_file = ckpt_path + "/flex_reuse_foundational_qa_att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_{}_{}_43b_test_greedy_0_250_{}.txt".format(n_ctx, n_enc, iter)
    # prediction_file = ckpt_path + "/flex_gate_0_reuse_foundational_qa_att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_{}_{}_43b_test_greedy_0_250_{}.txt".format(n_ctx, n_enc, iter)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/data/att/att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved/test.json"
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)

    prediction_file = ckpt_path + "/generate_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved_43b_test_greedy_0_250_472541.txt"
    prediction_file = ckpt_path + "/generate_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved_43b_test_greedy_0_250_472541.txt.period.txt"
    prediction_file = ckpt_path + "/generate_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved_43b_test_greedy_0_250_32552.txt"
    prediction_file = ckpt_path + "/generate_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved_43b_test_greedy_0_250_32552.txt.period.txt"
    prediction_file = ckpt_path + "/chunk_1k_generate_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved_43b_test_greedy_0_250_32552.txt"
    prediction_file = ckpt_path + "/chunk_1k_generate_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved_43b_test_greedy_0_250_32552.txt.period.txt"
    prediction_file = ckpt_path + "/gate_0_chunk_1k_generate_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved_43b_test_greedy_0_250_32552.txt"
    # prediction_file = ckpt_path + "/gate_0_chunk_1k_generate_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved_43b_test_greedy_0_250_32552.txt.period.txt"

    # prediction_file = ckpt_path + "/foundational_qa_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved_5_2_43b_test_greedy_0_250_32552.txt"
    # prediction_file = ckpt_path + "/reuse_foundational_qa_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved_{}_{}_43b_test_greedy_0_250_{}.txt".format(n_ctx, n_enc, iter)
    prediction_file = ckpt_path + "/flex_reuse_foundational_qa_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved_{}_{}_43b_test_greedy_0_250_{}.txt".format(n_ctx, n_enc, iter)
    # prediction_file = ckpt_path + "/flex_gate_0_reuse_foundational_qa_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved_{}_{}_43b_test_greedy_0_250_{}.txt".format(n_ctx, n_enc, iter)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved/test.json"
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)

    #
    prediction_file = ckpt_path + "/generate_nq_43b_test_greedy_0_200_472541.txt"
    prediction_file = ckpt_path + "/generate_nq_43b_test_greedy_0_200_472541.txt.period.txt"
    prediction_file = ckpt_path + "/generate_nq_43b_test_greedy_0_200_32552.txt"
    prediction_file = ckpt_path + "/generate_nq_43b_test_greedy_0_200_32552.txt.period.txt"
    prediction_file = ckpt_path + "/chunk_1k_generate_nq_43b_test_greedy_0_200_32552.txt"
    prediction_file = ckpt_path + "/chunk_1k_generate_nq_43b_test_greedy_0_200_32552.txt.period.txt"
    prediction_file = ckpt_path + "/gate_0_chunk_1k_generate_nq_43b_test_greedy_0_200_32552.txt"
    # prediction_file = ckpt_path + "/gate_0_chunk_1k_generate_nq_43b_test_greedy_0_200_32552.txt.period.txt"

    # prediction_file = ckpt_path + "/foundational_qa_nq_5_2_43b_test_greedy_0_200_32552.txt"
    # prediction_file = ckpt_path + "/reuse_foundational_qa_nq_{}_{}_43b_test_greedy_0_200_{}.txt".format(n_ctx, n_enc, iter)
    prediction_file = ckpt_path + "/flex_reuse_foundational_qa_nq_{}_{}_43b_test_greedy_0_200_{}.txt".format(n_ctx, n_enc, iter)
    # prediction_file = ckpt_path + "/flex_gate_0_reuse_foundational_qa_nq_{}_{}_43b_test_greedy_0_200_{}.txt".format(n_ctx, n_enc, iter)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/NQ/test.json"
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)


    prediction_file = ckpt_path + "generate_nv_benefits_dragon_retriever300_retrieved_generic_43b_test_greedy_0_250_472541.txt"
    prediction_file = ckpt_path + "generate_nv_benefits_dragon_retriever300_retrieved_generic_43b_test_greedy_0_250_472541.txt.period.txt"
    prediction_file = ckpt_path + "generate_nv_benefits_dragon_retriever300_retrieved_generic_43b_test_greedy_0_250_32552.txt"
    prediction_file = ckpt_path + "generate_nv_benefits_dragon_retriever300_retrieved_generic_43b_test_greedy_0_250_32552.txt.period.txt"
    prediction_file = ckpt_path + "chunk_1k_generate_nv_benefits_dragon_retriever300_retrieved_generic_43b_test_greedy_0_250_32552.txt"
    prediction_file = ckpt_path + "chunk_1k_generate_nv_benefits_dragon_retriever300_retrieved_generic_43b_test_greedy_0_250_32552.txt.period.txt"
    prediction_file = ckpt_path + "gate_0_chunk_1k_generate_nv_benefits_dragon_retriever300_retrieved_generic_43b_test_greedy_0_250_32552.txt"
    # prediction_file = ckpt_path + "gate_0_chunk_1k_generate_nv_benefits_dragon_retriever300_retrieved_generic_43b_test_greedy_0_250_32552.txt.period.txt"

    # prediction_file = ckpt_path + "foundational_qa_nv_benefits_dragon_retriever300_retrieved_generic_5_2_43b_test_greedy_0_250_32552.txt"
    # prediction_file = ckpt_path + "reuse_foundational_qa_nv_benefits_dragon_retriever300_retrieved_generic_{}_{}_43b_test_greedy_0_250_{}.txt".format(n_ctx, n_enc, iter)
    prediction_file = ckpt_path + "flex_reuse_foundational_qa_nv_benefits_dragon_retriever300_retrieved_generic_{}_{}_43b_test_greedy_0_250_{}.txt".format(n_ctx, n_enc, iter)
    # prediction_file = ckpt_path + "flex_gate_0_reuse_foundational_qa_nv_benefits_dragon_retriever300_retrieved_generic_{}_{}_43b_test_greedy_0_250_{}.txt".format(n_ctx, n_enc, iter)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/nv_benefits_dragon_retriever300_retrieved_generic/test.json"
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa//nv_benefits_dragon_retriever300_retrieved_generic/test.json"  # multiturn datasets

    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)


    prediction_file = ckpt_path + "/generate_NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_43b_test_greedy_0_250_472541.txt"
    prediction_file = ckpt_path + "/generate_NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_43b_test_greedy_0_250_472541.txt.period.txt"
    prediction_file = ckpt_path + "/generate_NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_43b_test_greedy_0_250_32552.txt"
    prediction_file = ckpt_path + "/generate_NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_43b_test_greedy_0_250_32552.txt.period.txt"
    prediction_file = ckpt_path + "/chunk_1k_generate_NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_43b_test_greedy_0_250_32552.txt"
    prediction_file = ckpt_path + "/chunk_1k_generate_NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_43b_test_greedy_0_250_32552.txt.period.txt"
    prediction_file = ckpt_path + "/gate_0_chunk_1k_generate_NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_43b_test_greedy_0_250_32552.txt"
    # prediction_file = ckpt_path + "/gate_0_chunk_1k_generate_NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_43b_test_greedy_0_250_32552.txt.period.txt"

    # prediction_file = ckpt_path + "/foundational_qa_NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_5_2_43b_test_greedy_0_250_32552.txt"
    # prediction_file = ckpt_path + "/reuse_foundational_qa_NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_{}_{}_43b_test_greedy_0_250_{}.txt".format(n_ctx, n_enc, iter)
    prediction_file = ckpt_path + "/flex_reuse_foundational_qa_NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_{}_{}_43b_test_greedy_0_250_{}.txt".format(n_ctx, n_enc, iter)
    # prediction_file = ckpt_path + "/flex_gate_0_reuse_foundational_qa_NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_{}_{}_43b_test_greedy_0_250_{}.txt".format(n_ctx, n_enc, iter)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved/test.json"
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)


    prediction_file = ckpt_path + "generate_Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_43b_test_greedy_0_250_472541.txt"
    prediction_file = ckpt_path + "generate_Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_43b_test_greedy_0_250_472541.txt.period.txt"
    prediction_file = ckpt_path + "generate_Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_43b_test_greedy_0_250_32552.txt"
    prediction_file = ckpt_path + "generate_Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_43b_test_greedy_0_250_32552.txt.period.txt"
    prediction_file = ckpt_path + "chunk_1k_generate_Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_43b_test_greedy_0_250_32552.txt"
    prediction_file = ckpt_path + "chunk_1k_generate_Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_43b_test_greedy_0_250_32552.txt.period.txt"
    prediction_file = ckpt_path + "gate_0_chunk_1k_generate_Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_43b_test_greedy_0_250_32552.txt"
    # prediction_file = ckpt_path + "gate_0_chunk_1k_generate_Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_43b_test_greedy_0_250_32552.txt.period.txt"

    # prediction_file = ckpt_path + "foundational_qa_Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_5_2_43b_test_greedy_0_250_32552.txt"
    # prediction_file = ckpt_path + "reuse_foundational_qa_Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_{}_{}_43b_test_greedy_0_250_{}.txt".format(n_ctx, n_enc, iter)
    prediction_file = ckpt_path + "flex_reuse_foundational_qa_Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_{}_{}_43b_test_greedy_0_250_{}.txt".format(n_ctx, n_enc, iter)
    # prediction_file = ckpt_path + "flex_gate_0_reuse_foundational_qa_Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved_{}_{}_43b_test_greedy_0_250_{}.txt".format(n_ctx, n_enc, iter)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved/test.json"
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)

    prediction_file = ckpt_path + "/generate_landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved_43b_test_greedy_0_250_472541.txt"
    prediction_file = ckpt_path + "/generate_landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved_43b_test_greedy_0_250_472541.txt.period.txt"
    prediction_file = ckpt_path + "/generate_landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved_43b_test_greedy_0_250_32552.txt"
    prediction_file = ckpt_path + "/generate_landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved_43b_test_greedy_0_250_32552.txt.period.txt"
    prediction_file = ckpt_path + "/chunk_1k_generate_landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved_43b_test_greedy_0_250_32552.txt"
    prediction_file = ckpt_path + "/chunk_1k_generate_landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved_43b_test_greedy_0_250_32552.txt.period.txt"
    prediction_file = ckpt_path + "/gate_0_chunk_1k_generate_landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved_43b_test_greedy_0_250_32552.txt"
    # prediction_file = ckpt_path + "/gate_0_chunk_1k_generate_landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved_43b_test_greedy_0_250_32552.txt.period.txt"

    # prediction_file = ckpt_path + "/generate_landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved_43b_test_greedy_0_250_32552.txt"
    # prediction_file = ckpt_path + "/foundational_qa_landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved_5_2_43b_test_greedy_0_250_32552.txt"
    # prediction_file = ckpt_path + "/reuse_foundational_qa_landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved_{}_{}_43b_test_greedy_0_250_{}.txt".format(n_ctx, n_enc, iter)
    prediction_file = ckpt_path + "/flex_reuse_foundational_qa_landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved_{}_{}_43b_test_greedy_0_250_{}.txt".format(n_ctx, n_enc, iter)
    # prediction_file = ckpt_path + "/flex_gate_0_reuse_foundational_qa_landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved_{}_{}_43b_test_greedy_0_250_{}.txt".format(n_ctx, n_enc, iter)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved/test.json"
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)

    prediction_file = ckpt_path + "/generate_inference_input_retriever_dragon_msmarcominilm_doc2dial_43b_test_greedy_0_250_472541.txt"
    prediction_file = ckpt_path + "/generate_inference_input_retriever_dragon_msmarcominilm_doc2dial_43b_test_greedy_0_250_472541.txt.period.txt"
    prediction_file = ckpt_path + "/generate_inference_input_retriever_dragon_msmarcominilm_doc2dial_43b_test_greedy_0_250_32552.txt"
    prediction_file = ckpt_path + "/generate_inference_input_retriever_dragon_msmarcominilm_doc2dial_43b_test_greedy_0_250_32552.txt.period.txt"
    prediction_file = ckpt_path + "/chunk_1k_generate_inference_input_retriever_dragon_msmarcominilm_doc2dial_43b_test_greedy_0_250_32552.txt"
    prediction_file = ckpt_path + "/chunk_1k_generate_inference_input_retriever_dragon_msmarcominilm_doc2dial_43b_test_greedy_0_250_32552.txt.period.txt"
    prediction_file = ckpt_path + "/gate_0_chunk_1k_generate_inference_input_retriever_dragon_msmarcominilm_doc2dial_43b_test_greedy_0_250_32552.txt"
    # prediction_file = ckpt_path + "/gate_0_chunk_1k_generate_inference_input_retriever_dragon_msmarcominilm_doc2dial_43b_test_greedy_0_250_32552.txt.period.txt"

    # prediction_file = ckpt_path + "/foundational_qa_inference_input_retriever_dragon_msmarcominilm_doc2dial_5_2_43b_test_greedy_0_250_32552.txt"
    # prediction_file = ckpt_path + "/reuse_foundational_qa_inference_input_retriever_dragon_msmarcominilm_doc2dial_{}_{}_43b_test_greedy_0_250_{}.txt".format(n_ctx, n_enc, iter)
    prediction_file = ckpt_path + "/flex_reuse_foundational_qa_inference_input_retriever_dragon_msmarcominilm_doc2dial_{}_{}_43b_test_greedy_0_250_{}.txt".format(n_ctx, n_enc, iter)
    # prediction_file = ckpt_path + "/flex_gate_0_reuse_foundational_qa_inference_input_retriever_dragon_msmarcominilm_doc2dial_{}_{}_43b_test_greedy_0_250_{}.txt".format(n_ctx, n_enc, iter)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/inference_input_retriever_dragon_msmarcominilm_doc2dial/test.json"
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)

    prediction_file = ckpt_path + "/flex_reuse_foundational_qa_nq_5_2_43b_test_greedy_0_2500_{}.txt".format(iter)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/instruction_tuning_data/NQ/test.json"
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)
    evaluate_ems(prediction_file, ground_truth_file)

    prediction_file = ckpt_path + "/flex_reuse_foundational_qa_tqa_5_2_43b_test_greedy_0_2500_{}.txt".format(iter)
    # prediction_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed/test.txt"
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/instruction_tuning_data/TQA/test.json"
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)
    evaluate_ems(prediction_file, ground_truth_file)

    prediction_file = ckpt_path + "/flex_reuse_foundational_qa_newsqa_1_1_43b_test_greedy_0_2500_{}.txt".format(iter)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/instruction_tuning_data/newsqa/test.json"
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)

    prediction_file = ckpt_path + "/flex_reuse_foundational_qa_squad2.0_1_1_43b_test_greedy_0_2500_{}.txt".format(iter)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/instruction_tuning_data/squad2.0/test.json"
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)

    prediction_file = ckpt_path + "/flex_reuse_foundational_qa_squad1.1_1_1_43b_test_greedy_0_2500_{}.txt".format(iter)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/instruction_tuning_data/squad1.1/test.json"
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)

    prediction_file = ckpt_path + "/flex_reuse_foundational_qa_ROPES_1_1_43b_test_greedy_0_2500_{}.txt".format(iter)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/instruction_tuning_data/ROPES/test.json"
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)

    prediction_file = ckpt_path + "/flex_reuse_foundational_qa_Quoref_1_1_43b_test_greedy_0_2500_{}.txt".format(iter)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/instruction_tuning_data/Quoref/test.json"
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)

    prediction_file = ckpt_path + "/flex_reuse_foundational_qa_NarrativeQA_1_1_43b_test_greedy_0_2500_{}.txt".format(iter)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/instruction_tuning_data/NarrativeQA/test.json"
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)

    prediction_file = ckpt_path + "/flex_reuse_foundational_qa_drop_1_1_43b_test_greedy_0_2500_{}.txt".format(iter)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/instruction_tuning_data/drop/test.json"
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)


    prediction_file = ckpt_path + "/flex_reuse_foundational_qa_ELI5_{}_{}_43b_test_greedy_0_1000_{}.txt".format(
        n_ctx, n_enc, iter)
    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/boxinw/instruction_tuning_data/ELI5/test.json"
    print(prediction_file)
    print(ground_truth_file)
    evaluate_f1(ground_truth_file, prediction_file)
