
import json
import random


def get_question_context_gold_list(datapath):
    with open(datapath, "r") as f:
        data_list = json.load(f)

    question_list = []
    context_list = []
    gold_list = []
    topk = 5
    for item in data_list:
        ctxs = item['ctxs']
        neighbours = ["title: " + ctx["title"] + ", source: " + ctx["text"] for ctx in ctxs[:topk]]
        context = "\n\n".join(neighbours)
        context_list.append(context)

        question_list.append(item['question'])
        if 'answers' in item:
            gold_list.append(item['answers'][0])
        else:
            gold_list.append(item['answer'])

    return question_list, context_list, gold_list

def get_output_list(datapath):
    with open(datapath) as f:
        output_list = f.readlines()
    return output_list

def main():
    ## doqa
    dataset_path = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/doqa/doqa_travel_QA_test.json"
    ours_path = "/lustre/fsw/adlr/adlr-nlp/zihanl/inform/foundational-qa/llama-2/checkpoints/applications/multiturn_qa_blend_finance_v6_llama2_text_70b_with_qc_multiturn_same_format_ctx1_70b_64_3e-7/doqa_travel_5_generate_70b_test_greedy_0_2000_3435_ret.txt.v2"
    ours_13b_path = "/lustre/fsw/adlr/adlr-nlp/zihanl/inform/foundational-qa/llama-2/checkpoints/applications/multiturn_qa_blend_finance_v6_1_llama2_text_13b_with_qc_multiturn_same_format_ctx1_13b_64_3e-7/doqa_travel_5_generate_13b_test_greedy_0_2000_3600_ret.txt.v2"
    gpt4_path = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/doqa/gpt_4_based_on_ftdragon_chatgptgen7k_chunk150_doqa_travel_final.txt"
    chatgpt_path = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/doqa/chatgpt_3.5_turbo_gen_doqa_travel.txt"

    # ## qrecc
    # dataset_path = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/qrecc/qrecc_ftdragon_chatgptgen7k_chunk150_QA_test.json"
    # ours_path = "/lustre/fsw/adlr/adlr-nlp/zihanl/inform/foundational-qa/llama-2/checkpoints/applications/multiturn_qa_blend_finance_v6_llama2_text_70b_with_qc_multiturn_same_format_ctx1_70b_64_3e-7/qrecc_5_generate_70b_test_greedy_0_4000_3435_ret.txt.v2"
    # ours_13b_path = "/lustre/fsw/adlr/adlr-nlp/zihanl/inform/foundational-qa/llama-2/checkpoints/applications/multiturn_qa_blend_finance_v6_1_llama2_text_13b_with_qc_multiturn_same_format_ctx1_13b_64_3e-7/qrecc_5_generate_13b_test_greedy_0_3000_3600_ret.txt.v2"
    # gpt4_path = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/qrecc/gpt_4_based_on_ftdragon_chatgptgen7k_chunk150.txt"
    # chatgpt_path = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/qrecc/chatgpt_3.5_turbo_based_on_ftdragon_chatgptgen7k_chunk150.txt"

    # ## inscit
    # dataset_path = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/inscit/inscit_dev_retrieval_dragon_ft_chatgptgen7k_with_topic.json"
    # ours_path = "/lustre/fsw/adlr/adlr-nlp/zihanl/inform/foundational-qa/llama-2/checkpoints/applications/multiturn_qa_blend_finance_v6_llama2_text_70b_with_qc_multiturn_same_format_ctx1_70b_64_3e-7/inscit_20_generate_70b_test_greedy_0_550_3435_ret.txt.v2"
    # ours_13b_path = "/lustre/fsw/adlr/adlr-nlp/zihanl/inform/foundational-qa/llama-2/checkpoints/applications/multiturn_qa_blend_finance_v6_1_llama2_text_13b_with_qc_multiturn_same_format_ctx1_13b_64_3e-7/inscit_20_generate_13b_test_greedy_0_550_3600_ret.txt.v2"
    # gpt4_path = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/inscit/gpt_4_inscit_final.txt"
    # chatgpt_path = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/inscit/chatgpt_3.5_turbo_gen_inscit.txt"

    # ## doc2dial
    # dataset_path = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/doc2dial/doc2dial_ftdragon_chatgptgen7k_chunk150_QA_test.json"
    # ours_path = "/lustre/fsw/adlr/adlr-nlp/zihanl/inform/foundational-qa/llama-2/checkpoints/applications/multiturn_qa_blendv2_llama2_text_70b_with_qc_multiturn_same_format_ctx1_70b_64_3e-7/doc2dial_5_generate_70b_test_greedy_0_4000_3600_ret.txt.v2"
    # gpt4_path = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/doc2dial/gpt_4_based_on_ftdragon_chatgptgen7k_chunk150_final.txt"
    # chatgpt_path = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/doc2dial/chatgpt_3.5_turbo_based_on_ftdragon_chatgptgen7k_chunk150.txt"

    # ## topiocqa
    # dataset_path = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/topiocqa/topiocqa_dev_retrieval_dragon_ft_chatgptgen7k.json"
    # ours_path = "/lustre/fsw/adlr/adlr-nlp/zihanl/inform/foundational-qa/llama-2/checkpoints/applications/multiturn_qa_blend_finance_v6_llama2_text_70b_with_qc_multiturn_same_format_ctx1_70b_64_3e-7/topiocqa_20_generate_70b_test_greedy_0_2600_3435_ret.txt.v2"
    # ours_13b_path = "/lustre/fsw/adlr/adlr-nlp/zihanl/inform/foundational-qa/llama-2/checkpoints/applications/multiturn_qa_blend_finance_v6_1_llama2_text_13b_with_qc_multiturn_same_format_ctx1_13b_64_3e-7/topiocqa_20_generate_13b_test_greedy_0_2600_3600_ret.txt.v2"
    # gpt4_path = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/topiocqa/gpt_4_topiocqa_final.txt"
    # chatgpt_path = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/topiocqa/chatgpt_3.5_turbo_gen_topiocqa.txt"

    # ## sqa
    # dataset_path = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/sqa/sqa_QA_test.json"
    # ours_path = "/lustre/fsw/adlr/adlr-nlp/zihanl/inform/foundational-qa/llama-2/checkpoints/applications/multiturn_qa_blend_finance_v6_llama2_text_70b_with_qc_multiturn_same_format_ctx1_70b_64_3e-7/sqa_5_generate_70b_test_greedy_0_3100_3435_ret.txt.v2"
    # ours_13b_path = "/lustre/fsw/adlr/adlr-nlp/zihanl/inform/foundational-qa/llama-2/checkpoints/applications/multiturn_qa_blend_finance_v6_1_llama2_text_13b_with_qc_multiturn_same_format_ctx1_13b_64_3e-7/sqa_5_generate_13b_test_greedy_0_3100_3600_ret.txt.v2"
    # gpt4_path = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/sqa/gpt_4_sqa_final.txt"
    # chatgpt_path = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/sqa/chatgpt_3.5_turbo_gen_sqa.txt"

    # ## convfinqa
    # dataset_path = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/convfinqav3/convfinqav3_QA_dev.json"
    # ours_path = "/lustre/fsw/adlr/adlr-nlp/zihanl/inform/foundational-qa/llama-2/checkpoints/applications/multiturn_qa_blend_finance_v6_1_llama2_text_70b_with_qc_multiturn_same_format_ctx1_70b_64_3e-7/convfinqav3_5_generate_70b_test_greedy_0_1500_3435_ret.txt.v2"
    # ours_13b_path = "/lustre/fsw/adlr/adlr-nlp/zihanl/inform/foundational-qa/llama-2/checkpoints/applications/multiturn_qa_blend_finance_v6_1_llama2_text_13b_with_qc_multiturn_same_format_ctx1_13b_64_3e-7/convfinqav3_5_generate_13b_test_greedy_0_1500_3600_ret.txt.v2"
    # gpt4_path = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/convfinqav3/gpt_4_convfinqa_final.txt"
    # chatgpt_path = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/convfinqav3/chatgpt_3.5_turbo_gen_convfinqav3.txt"

    # ## hybriddial
    # dataset_path = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/HybridDial/HybridDial_fqa_test.json"
    # ours_path = "/lustre/fsw/adlr/adlr-nlp/zihanl/inform/foundational-qa/llama-2/checkpoints/applications/multiturn_qa_blend_finance_v6_llama2_text_70b_with_qc_multiturn_same_format_ctx1_70b_64_3e-7/hybriddial_5_generate_70b_test_greedy_0_1200_3435_ret.txt.v2"
    # ours_13b_path = "/lustre/fsw/adlr/adlr-nlp/zihanl/inform/foundational-qa/llama-2/checkpoints/applications/multiturn_qa_blend_finance_v6_1_llama2_text_13b_with_qc_multiturn_same_format_ctx1_13b_64_3e-7/hybriddial_5_generate_13b_test_greedy_0_1200_3600_ret.txt.v2"
    # gpt4_path = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/HybridDial/gpt_4_hybriddial.txt"
    # chatgpt_path = "/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/HybridDial/chatgpt_3.5_turbo_gen_hybriddial.txt"

    question_list, context_list, gold_list = get_question_context_gold_list(dataset_path)
    output_ours_list = get_output_list(ours_path)
    output_ours_13b_list = get_output_list(ours_13b_path)
    output_chatgpt_list = get_output_list(chatgpt_path)
    output_gpt4_list = get_output_list(gpt4_path)
    
    # question_list = question_list[:1000]
    # context_list = context_list[:1000]
    # gold_list = gold_list[:1000]
    # output_ours_list = output_ours_list[:1000]
    # output_ours_13b_list = output_ours_13b_list[:1000]
    # output_chatgpt_list = output_chatgpt_list[:1000]
    # output_gpt4_list = output_gpt4_list[:1000]

    assert len(question_list) == len(context_list) == len(gold_list) == len(output_ours_list) == \
             len(output_ours_13b_list) == len(output_chatgpt_list) == len(output_gpt4_list)

    # random.seed(1234)
    # random.seed(4567)
    random.seed(6789)
    idx_list = [i for i in range(len(question_list))]
    random.shuffle(idx_list)

    num_sample = 200
    for idx in idx_list[:num_sample*2]:
        
        if len(question_list[idx].split("\n\n")) <= 2: continue

        # if "does not provide" in output_gpt4_list[idx].strip():
        print("="*80)
        print("context:")
        print(context_list[idx])
        
        print("-"*80)
        print("question:")
        print(question_list[idx])
        
        print("-"*80)
        print("ours 70b:")
        print(output_ours_list[idx].strip())
        
        print("-"*80)
        print("ours 13b:")
        print(output_ours_13b_list[idx].strip())
        
        print("-"*80)
        print("gpt-3.5:")
        print(output_chatgpt_list[idx].strip())
        
        print("-"*80)
        print("gpt-4:")
        print(output_gpt4_list[idx].strip())
        
        print("-"*80)
        print("gold:")
        print(gold_list[idx])


if __name__ == "__main__":
    main()

