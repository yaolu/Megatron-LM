
import json
import random
import os

random.seed(1234)
data_folder = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa"
ours_output_folder = "/lustre/fsw/adlr/adlr-nlp/zihanl/inform/foundational-qa/llama-2/checkpoints/applications/multiturn_qa_blend_finance_v6_llama2_text_70b_with_qc_multiturn_same_format_ctx1_70b_64_3e-7"

def get_question_context_gold_list(datapath):
    print("="*80)
    print("processing %s" % datapath)
    with open(datapath, "r") as f:
        data_list = json.load(f)
    
    question_list = []
    context_list = []
    gold_list = []
    topk = 5
    for item in data_list:
        ctxs = item['ctxs']
        neighbours = [ctx["text"] for ctx in ctxs[:topk]]
        context = "\n\n".join(neighbours)
        context_list.append(context)

        question_list.append(item['question'])
        if 'answers' in item:
            gold_list.append(item['answers'][0])
        else:
            gold_list.append(item['answer'])

    return question_list, context_list, gold_list


def get_output_list(datapath):
    with open(datapath, "r") as f:
        output_list = f.readlines()
    return output_list


def write_human_eval_files(question_list, context_list, gold_list, ours_output_list, \
                baseline_output_list, baseline_model_name, output_datapath, n_samples=200):
    
    output_list = []
    for question, context, gold, ours_output, baseline_output in zip(
            question_list, context_list, gold_list, ours_output_list, baseline_output_list):
        
        data_dict = {
            "context": context,
            "question": question,
            "ours_output": ours_output,
            "baseline_output": baseline_output,
            "gold": gold
        }

        output_list.append(data_dict)

    ## shuffle
    random.shuffle(output_list)
    output_list = output_list[:n_samples]

    print("dumping %d samples to %s" % (n_samples, output_datapath))
    with open(output_datapath, "w") as f:
        json.dump(output_list, f, indent=2)


def get_human_eval_doc2dial():
    
    ## doc2dial
    dataset_path = os.path.join(data_folder, "doc2dial/doc2dial_ftdragon_chatgptgen7k_chunk150_QA_test.json")
    ours_output_path = os.path.join(ours_output_folder, "doc2dial_5_generate_70b_test_greedy_0_4000_3435_ret.txt.v2")
    chatgpt_output_path = os.path.join(data_folder, "doc2dial/chatgpt_3.5_turbo_based_on_ftdragon_chatgptgen7k_chunk150.txt")
    gpt4_output_path = os.path.join(data_folder, "doc2dial/gpt_4_based_on_ftdragon_chatgptgen7k_chunk150_final.txt")

    question_list, context_list, gold_list = get_question_context_gold_list(dataset_path)
    ours_output_list = get_output_list(ours_output_path)
    chatgpt_output_list = get_output_list(chatgpt_output_path)
    gpt4_output_list = get_output_list(gpt4_output_path)

    assert len(ours_output_list) == len(chatgpt_output_list) == len(gpt4_output_list) == \
                len(question_list) == len(context_list) == len(gold_list)

    output_datapath_ours_vs_chatgpt = os.path.join(data_folder, "human_eval/doc2dial_ours_vs_chatgpt.json")
    output_datapath_ours_vs_gpt4 = os.path.join(data_folder, "human_eval/doc2dial_ours_vs_gpt4.json")

    write_human_eval_files(question_list, context_list, gold_list, ours_output_list, \
                     chatgpt_output_list, "chatgpt", output_datapath_ours_vs_chatgpt)
    write_human_eval_files(question_list, context_list, gold_list, ours_output_list, \
                     gpt4_output_list, "gpt4", output_datapath_ours_vs_gpt4)


def get_human_eval_quac():
    
    ## quac
    dataset_path = os.path.join(data_folder, "quac/quac_ftdragon_chatgptgen7k_chunk150_QA_test.json")
    ours_output_path = os.path.join(ours_output_folder, "quac_5_generate_70b_test_greedy_0_8000_3435_ret.txt.v2")
    chatgpt_output_path = os.path.join(data_folder, "quac/chatgpt_3.5_turbo_based_on_ftdragon_chatgptgen7k_chunk150.txt")
    gpt4_output_path = os.path.join(data_folder, "quac/gpt_4_based_on_ftdragon_chatgptgen7k_chunk150_final.txt")

    question_list, context_list, gold_list = get_question_context_gold_list(dataset_path)
    ours_output_list = get_output_list(ours_output_path)
    chatgpt_output_list = get_output_list(chatgpt_output_path)
    gpt4_output_list = get_output_list(gpt4_output_path)

    assert len(ours_output_list) == len(chatgpt_output_list) == len(gpt4_output_list) == \
                len(question_list) == len(context_list) == len(gold_list)

    output_datapath_ours_vs_chatgpt = os.path.join(data_folder, "human_eval/quac_ours_vs_chatgpt.json")
    output_datapath_ours_vs_gpt4 = os.path.join(data_folder, "human_eval/quac_ours_vs_gpt4.json")

    write_human_eval_files(question_list, context_list, gold_list, ours_output_list, \
                     chatgpt_output_list, "chatgpt", output_datapath_ours_vs_chatgpt)
    write_human_eval_files(question_list, context_list, gold_list, ours_output_list, \
                     gpt4_output_list, "gpt4", output_datapath_ours_vs_gpt4)


def get_human_eval_qrecc():
    
    ## qrecc
    dataset_path = os.path.join(data_folder, "qrecc/qrecc_ftdragon_chatgptgen7k_chunk150_QA_test.json")
    ours_output_path = os.path.join(ours_output_folder, "qrecc_5_generate_70b_test_greedy_0_4000_3435_ret.txt.v2")
    chatgpt_output_path = os.path.join(data_folder, "qrecc/chatgpt_3.5_turbo_based_on_ftdragon_chatgptgen7k_chunk150.txt")
    gpt4_output_path = os.path.join(data_folder, "qrecc/gpt_4_based_on_ftdragon_chatgptgen7k_chunk150.txt")

    question_list, context_list, gold_list = get_question_context_gold_list(dataset_path)
    ours_output_list = get_output_list(ours_output_path)
    chatgpt_output_list = get_output_list(chatgpt_output_path)
    gpt4_output_list = get_output_list(gpt4_output_path)
    
    gpt4_out_len = len(gpt4_output_list)
    question_list = question_list[:gpt4_out_len]
    context_list = context_list[:gpt4_out_len]
    gold_list = gold_list[:gpt4_out_len]
    ours_output_list = ours_output_list[:gpt4_out_len]
    chatgpt_output_list = chatgpt_output_list[:gpt4_out_len]

    assert len(ours_output_list) == len(chatgpt_output_list) == len(gpt4_output_list) == \
                len(question_list) == len(context_list) == len(gold_list)

    output_datapath_ours_vs_chatgpt = os.path.join(data_folder, "human_eval/qrecc_ours_vs_chatgpt.json")
    output_datapath_ours_vs_gpt4 = os.path.join(data_folder, "human_eval/qrecc_ours_vs_gpt4.json")

    write_human_eval_files(question_list, context_list, gold_list, ours_output_list, \
                     chatgpt_output_list, "chatgpt", output_datapath_ours_vs_chatgpt)
    write_human_eval_files(question_list, context_list, gold_list, ours_output_list, \
                     gpt4_output_list, "gpt4", output_datapath_ours_vs_gpt4)


def get_human_eval_coqa():
    
    ## coqa
    dataset_path = os.path.join(data_folder, "coqa/coqa_QA_dev.json")
    ours_output_path = os.path.join(ours_output_folder, "coqa_5_generate_70b_test_greedy_0_8000_3435_ret.txt.v2")
    chatgpt_output_path = os.path.join(data_folder, "coqa/chatgpt_3.5_turbo_gen.txt")
    gpt4_output_path = os.path.join(data_folder, "coqa/gpt_4_based_on_ftdragon_chatgptgen7k_chunk150_coqa_final.txt")

    question_list, context_list, gold_list = get_question_context_gold_list(dataset_path)
    ours_output_list = get_output_list(ours_output_path)
    chatgpt_output_list = get_output_list(chatgpt_output_path)
    gpt4_output_list = get_output_list(gpt4_output_path)

    assert len(ours_output_list) == len(chatgpt_output_list) == len(gpt4_output_list) == \
                len(question_list) == len(context_list) == len(gold_list)

    output_datapath_ours_vs_chatgpt = os.path.join(data_folder, "human_eval/coqa_ours_vs_chatgpt.json")
    output_datapath_ours_vs_gpt4 = os.path.join(data_folder, "human_eval/coqa_ours_vs_gpt4.json")

    write_human_eval_files(question_list, context_list, gold_list, ours_output_list, \
                     chatgpt_output_list, "chatgpt", output_datapath_ours_vs_chatgpt)
    write_human_eval_files(question_list, context_list, gold_list, ours_output_list, \
                     gpt4_output_list, "gpt4", output_datapath_ours_vs_gpt4)


def get_human_eval_doqa():
    
    ## doqa
    dataset_path_cooking = os.path.join(data_folder, "doqa/doqa_cooking_QA_test.json")
    ours_output_path_cooking = os.path.join(ours_output_folder, "doqa_cooking_5_generate_70b_test_greedy_0_2000_3435_ret.txt.v2")
    chatgpt_output_path_cooking = os.path.join(data_folder, "doqa/chatgpt_3.5_turbo_gen_doqa_cooking.txt")
    gpt4_output_path_cooking = os.path.join(data_folder, "doqa/gpt_4_based_on_ftdragon_chatgptgen7k_chunk150_doqa_cooking_final.txt")
    
    dataset_path_movies = os.path.join(data_folder, "doqa/doqa_movies_QA_test.json")
    ours_output_path_movies = os.path.join(ours_output_folder, "doqa_movies_5_generate_70b_test_greedy_0_2000_3435_ret.txt.v2")
    chatgpt_output_path_movies = os.path.join(data_folder, "doqa/chatgpt_3.5_turbo_gen_doqa_movies.txt")
    gpt4_output_path_movies = os.path.join(data_folder, "doqa/gpt_4_based_on_ftdragon_chatgptgen7k_chunk150_doqa_movies_final.txt")
    
    dataset_path_travel = os.path.join(data_folder, "doqa/doqa_travel_QA_test.json")
    ours_output_path_travel = os.path.join(ours_output_folder, "doqa_travel_5_generate_70b_test_greedy_0_2000_3435_ret.txt.v2")
    chatgpt_output_path_travel = os.path.join(data_folder, "doqa/chatgpt_3.5_turbo_gen_doqa_travel.txt")
    gpt4_output_path_travel = os.path.join(data_folder, "doqa/gpt_4_based_on_ftdragon_chatgptgen7k_chunk150_doqa_travel_final.txt")

    question_list_cooking, context_list_cooking, gold_list_cooking = get_question_context_gold_list(dataset_path_cooking)
    ours_output_list_cooking = get_output_list(ours_output_path_cooking)
    chatgpt_output_list_cooking = get_output_list(chatgpt_output_path_cooking)
    gpt4_output_list_cooking = get_output_list(gpt4_output_path_cooking)

    question_list_movies, context_list_movies, gold_list_movies = get_question_context_gold_list(dataset_path_movies)
    ours_output_list_movies = get_output_list(ours_output_path_movies)
    chatgpt_output_list_movies = get_output_list(chatgpt_output_path_movies)
    gpt4_output_list_movies = get_output_list(gpt4_output_path_movies)

    question_list_travel, context_list_travel, gold_list_travel = get_question_context_gold_list(dataset_path_travel)
    ours_output_list_travel = get_output_list(ours_output_path_travel)
    chatgpt_output_list_travel = get_output_list(chatgpt_output_path_travel)
    gpt4_output_list_travel = get_output_list(gpt4_output_path_travel)

    question_list = question_list_cooking + question_list_movies + question_list_travel
    context_list = context_list_cooking + context_list_movies + context_list_travel
    gold_list = gold_list_cooking + gold_list_movies + gold_list_travel
    ours_output_list = ours_output_list_cooking + ours_output_list_movies + ours_output_list_travel
    chatgpt_output_list = chatgpt_output_list_cooking + chatgpt_output_list_movies + chatgpt_output_list_travel
    gpt4_output_list = gpt4_output_list_cooking + gpt4_output_list_movies + gpt4_output_list_travel

    assert len(ours_output_list) == len(chatgpt_output_list) == len(gpt4_output_list) == \
                len(question_list) == len(context_list) == len(gold_list)

    output_datapath_ours_vs_chatgpt = os.path.join(data_folder, "human_eval/doqa_ours_vs_chatgpt.json")
    output_datapath_ours_vs_gpt4 = os.path.join(data_folder, "human_eval/doqa_ours_vs_gpt4.json")

    write_human_eval_files(question_list, context_list, gold_list, ours_output_list, \
                     chatgpt_output_list, "chatgpt", output_datapath_ours_vs_chatgpt)
    write_human_eval_files(question_list, context_list, gold_list, ours_output_list, \
                     gpt4_output_list, "gpt4", output_datapath_ours_vs_gpt4)


def get_human_eval_convfinqav3():
    
    ## convfinqav3
    dataset_path = os.path.join(data_folder, "convfinqav3/convfinqav3_QA_dev.json")
    ours_output_path = os.path.join(ours_output_folder, "convfinqav3_5_generate_70b_test_greedy_0_1500_3435_ret.txt.v2")
    chatgpt_output_path = os.path.join(data_folder, "convfinqav3/chatgpt_3.5_turbo_gen_convfinqav3.txt")
    gpt4_output_path = os.path.join(data_folder, "convfinqav3/gpt_4_convfinqa_final.txt")

    question_list, context_list, gold_list = get_question_context_gold_list(dataset_path)
    ours_output_list = get_output_list(ours_output_path)
    chatgpt_output_list = get_output_list(chatgpt_output_path)
    gpt4_output_list = get_output_list(gpt4_output_path)

    assert len(ours_output_list) == len(chatgpt_output_list) == len(gpt4_output_list) == \
                len(question_list) == len(context_list) == len(gold_list)

    output_datapath_ours_vs_chatgpt = os.path.join(data_folder, "human_eval/convfinqav3_ours_vs_chatgpt.json")
    output_datapath_ours_vs_gpt4 = os.path.join(data_folder, "human_eval/convfinqav3_ours_vs_gpt4.json")

    write_human_eval_files(question_list, context_list, gold_list, ours_output_list, \
                     chatgpt_output_list, "chatgpt", output_datapath_ours_vs_chatgpt)
    write_human_eval_files(question_list, context_list, gold_list, ours_output_list, \
                     gpt4_output_list, "gpt4", output_datapath_ours_vs_gpt4)


def get_human_eval_sqa():
    
    ## sqa
    dataset_path = os.path.join(data_folder, "sqa/sqa_QA_test.json")
    ours_output_path = os.path.join(ours_output_folder, "sqa_5_generate_70b_test_greedy_0_3100_3435_ret.txt.v2")
    chatgpt_output_path = os.path.join(data_folder, "sqa/chatgpt_3.5_turbo_gen_sqa.txt")
    gpt4_output_path = os.path.join(data_folder, "sqa/gpt_4_sqa_final.txt")

    question_list, context_list, gold_list = get_question_context_gold_list(dataset_path)
    ours_output_list = get_output_list(ours_output_path)
    chatgpt_output_list = get_output_list(chatgpt_output_path)
    gpt4_output_list = get_output_list(gpt4_output_path)

    assert len(ours_output_list) == len(chatgpt_output_list) == len(gpt4_output_list) == \
                len(question_list) == len(context_list) == len(gold_list)

    output_datapath_ours_vs_chatgpt = os.path.join(data_folder, "human_eval/sqa_ours_vs_chatgpt.json")
    output_datapath_ours_vs_gpt4 = os.path.join(data_folder, "human_eval/sqa_ours_vs_gpt4.json")

    write_human_eval_files(question_list, context_list, gold_list, ours_output_list, \
                     chatgpt_output_list, "chatgpt", output_datapath_ours_vs_chatgpt)
    write_human_eval_files(question_list, context_list, gold_list, ours_output_list, \
                     gpt4_output_list, "gpt4", output_datapath_ours_vs_gpt4)


def get_human_eval_topiocqa():
    
    ## topiocqa
    dataset_path = os.path.join(data_folder, "topiocqa/topiocqa_dev_retrieval_dragon_ft_chatgptgen7k.json")
    ours_output_path = os.path.join(ours_output_folder, "topiocqa_20_generate_70b_test_greedy_0_2600_3435_ret.txt.v2")
    chatgpt_output_path = os.path.join(data_folder, "topiocqa/chatgpt_3.5_turbo_gen_topiocqa.txt")
    gpt4_output_path = os.path.join(data_folder, "topiocqa/gpt_4_topiocqa_final.txt")

    question_list, context_list, gold_list = get_question_context_gold_list(dataset_path)
    ours_output_list = get_output_list(ours_output_path)
    chatgpt_output_list = get_output_list(chatgpt_output_path)
    gpt4_output_list = get_output_list(gpt4_output_path)

    assert len(ours_output_list) == len(chatgpt_output_list) == len(gpt4_output_list) == \
                len(question_list) == len(context_list) == len(gold_list)

    output_datapath_ours_vs_chatgpt = os.path.join(data_folder, "human_eval/topiocqa_ours_vs_chatgpt.json")
    output_datapath_ours_vs_gpt4 = os.path.join(data_folder, "human_eval/topiocqa_ours_vs_gpt4.json")

    write_human_eval_files(question_list, context_list, gold_list, ours_output_list, \
                     chatgpt_output_list, "chatgpt", output_datapath_ours_vs_chatgpt)
    write_human_eval_files(question_list, context_list, gold_list, ours_output_list, \
                     gpt4_output_list, "gpt4", output_datapath_ours_vs_gpt4)


def get_human_eval_hybriddial():
    
    ## hybriddial
    dataset_path = os.path.join(data_folder, "HybridDial/HybridDial_fqa_test.json")
    ours_output_path = os.path.join(ours_output_folder, "hybriddial_5_generate_70b_test_greedy_0_1200_3435_ret.txt.v2")
    chatgpt_output_path = os.path.join(data_folder, "HybridDial/chatgpt_3.5_turbo_gen_hybriddial.txt")
    gpt4_output_path = os.path.join(data_folder, "HybridDial/gpt_4_hybriddial.txt")

    question_list, context_list, gold_list = get_question_context_gold_list(dataset_path)
    ours_output_list = get_output_list(ours_output_path)
    chatgpt_output_list = get_output_list(chatgpt_output_path)
    gpt4_output_list = get_output_list(gpt4_output_path)

    assert len(ours_output_list) == len(chatgpt_output_list) == len(gpt4_output_list) == \
                len(question_list) == len(context_list) == len(gold_list)

    output_datapath_ours_vs_chatgpt = os.path.join(data_folder, "human_eval/HybridDial_ours_vs_chatgpt.json")
    output_datapath_ours_vs_gpt4 = os.path.join(data_folder, "human_eval/HybridDial_ours_vs_gpt4.json")

    write_human_eval_files(question_list, context_list, gold_list, ours_output_list, \
                     chatgpt_output_list, "chatgpt", output_datapath_ours_vs_chatgpt)
    write_human_eval_files(question_list, context_list, gold_list, ours_output_list, \
                     gpt4_output_list, "gpt4", output_datapath_ours_vs_gpt4)


def get_human_eval_inscit():
    
    ## inscit
    dataset_path = os.path.join(data_folder, "inscit/inscit_dev_retrieval_dragon_ft_chatgptgen7k_with_topic.json")
    ours_output_path = os.path.join(ours_output_folder, "inscit_20_generate_70b_test_greedy_0_550_3435_ret.txt.v2")
    chatgpt_output_path = os.path.join(data_folder, "inscit/chatgpt_3.5_turbo_gen_inscit.txt")
    gpt4_output_path = os.path.join(data_folder, "inscit/gpt_4_inscit_final.txt")

    question_list, context_list, gold_list = get_question_context_gold_list(dataset_path)
    ours_output_list = get_output_list(ours_output_path)
    chatgpt_output_list = get_output_list(chatgpt_output_path)
    gpt4_output_list = get_output_list(gpt4_output_path)

    assert len(ours_output_list) == len(chatgpt_output_list) == len(gpt4_output_list) == \
                len(question_list) == len(context_list) == len(gold_list)

    output_datapath_ours_vs_chatgpt = os.path.join(data_folder, "human_eval/inscit_ours_vs_chatgpt.json")
    output_datapath_ours_vs_gpt4 = os.path.join(data_folder, "human_eval/inscit_ours_vs_gpt4.json")

    write_human_eval_files(question_list, context_list, gold_list, ours_output_list, \
                     chatgpt_output_list, "chatgpt", output_datapath_ours_vs_chatgpt)
    write_human_eval_files(question_list, context_list, gold_list, ours_output_list, \
                     gpt4_output_list, "gpt4", output_datapath_ours_vs_gpt4)


def main():
    # get_human_eval_doc2dial()
    # get_human_eval_quac()
    # get_human_eval_qrecc()
    # get_human_eval_coqa()
    # get_human_eval_doqa()
    # get_human_eval_convfinqav3()
    # get_human_eval_sqa()
    # get_human_eval_topiocqa()
    # get_human_eval_hybriddial()

    get_human_eval_inscit()

if __name__ == "__main__":
    main()
