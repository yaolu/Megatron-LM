model_name=multiturn_qa_blendv2_gpt-fitting_1e-8_conv_quiet_cockatoo_pp1_addmultiturn-3.5t_same_format_ctx1_8b_64_3e-7
num_ctxs=5
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nq 8b greedy test  0 200 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 8b greedy test  0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 8b greedy test  0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh inference_input_retriever_dragon_msmarcominilm_doc2dial 8b greedy test 0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 8b greedy test 0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nv_benefits_dragon_retriever300_retrieved_generic 8b greedy test 0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 8b greedy test 0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 8b greedy test 0 250 3000 $num_ctxs $model_name true
#
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nq  8b greedy test 0 20000 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh tqa  8b greedy test 0 20000 3000 $num_ctxs $model_name true

num_ctxs=1
bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh doc2dial_gold  8b greedy test 0 20000 3000 $num_ctxs $model_name

model_name=multiturn_qa_blend_commercial_v15_gpt-fitting_1e-8_conv_quiet_cockatoo_pp1_addmultiturn-commercial-3.5t_same_format_ctx1_8b_64_3e-7
num_ctxs=5
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nq 8b greedy test  0 200 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 8b greedy test  0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 8b greedy test  0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh inference_input_retriever_dragon_msmarcominilm_doc2dial 8b greedy test 0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 8b greedy test 0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nv_benefits_dragon_retriever300_retrieved_generic 8b greedy test 0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 8b greedy test 0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 8b greedy test 0 250 3000 $num_ctxs $model_name true
#
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nq  8b greedy test 0 20000 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh tqa  8b greedy test 0 20000 3000 $num_ctxs $model_name true
#
num_ctxs=1
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh newsqa  8b greedy test 0 20000 3000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh squad2.0 8b greedy test 0 20000 3000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh squad1.1 8b greedy test 0 20000 3000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh ROPES  8b greedy test 0 20000 3000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh Quoref  8b greedy test 0 20000 3000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh NarrativeQA 8b greedy test 0 20000 3000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh drop  8b greedy test 0 20000 3000 $num_ctxs $model_name
#
bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh doc2dial_gold  8b greedy test 0 20000 3000 $num_ctxs $model_name
#
#model_name=multiturn_qa_blendv2_gpt_1e-8_conv_quiet_cockatoo_pp1_addmultiturn-3.5t_same_format_ctx1_8b_64_3e-7
#num_ctxs=5
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nq 8b greedy test  0 200 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 8b greedy test  0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 8b greedy test  0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh inference_input_retriever_dragon_msmarcominilm_doc2dial 8b greedy test 0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 8b greedy test 0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nv_benefits_dragon_retriever300_retrieved_generic 8b greedy test 0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 8b greedy test 0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 8b greedy test 0 250 3000 $num_ctxs $model_name true
#
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nq  8b greedy test 0 20000 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh tqa  8b greedy test 0 20000 3000 $num_ctxs $model_name true
#
num_ctxs=1
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh newsqa  8b greedy test 0 20000 3000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh squad2.0 8b greedy test 0 20000 3000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh squad1.1 8b greedy test 0 20000 3000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh ROPES  8b greedy test 0 20000 3000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh Quoref  8b greedy test 0 20000 3000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh NarrativeQA 8b greedy test 0 20000 3000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh drop  8b greedy test 0 20000 3000 $num_ctxs $model_name
#
bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh doc2dial_gold  8b greedy test 0 20000 3000 $num_ctxs $model_name

model_name=multiturn_qa_blend_commercial_v15_gpt_1e-8_conv_quiet_cockatoo_pp1_addmultiturn-commercial-3.5t_same_format_ctx1_8b_64_3e-7
num_ctxs=5
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nq 8b greedy test  0 200 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 8b greedy test  0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 8b greedy test  0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh inference_input_retriever_dragon_msmarcominilm_doc2dial 8b greedy test 0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 8b greedy test 0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nv_benefits_dragon_retriever300_retrieved_generic 8b greedy test 0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 8b greedy test 0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 8b greedy test 0 250 3000 $num_ctxs $model_name true
#
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nq  8b greedy test 0 20000 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh tqa  8b greedy test 0 20000 3000 $num_ctxs $model_name true

num_ctxs=1
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh newsqa  8b greedy test 0 20000 3000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh squad2.0 8b greedy test 0 20000 3000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh squad1.1 8b greedy test 0 20000 3000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh ROPES  8b greedy test 0 20000 3000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh Quoref  8b greedy test 0 20000 3000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh NarrativeQA 8b greedy test 0 20000 3000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh drop  8b greedy test 0 20000 3000 $num_ctxs $model_name
#
bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh doc2dial_gold  8b greedy test 0 20000 3000 $num_ctxs $model_name

model_name=sft_gpt-fitting-pp1-3.5t_same_format_ctx1_8b_128_5e-6
num_ctxs=5
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nq 8b greedy test  0 200 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 8b greedy test  0 250 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 8b greedy test  0 250 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh inference_input_retriever_dragon_msmarcominilm_doc2dial 8b greedy test 0 250 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 8b greedy test 0 250 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nv_benefits_dragon_retriever300_retrieved_generic 8b greedy test 0 250 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 8b greedy test 0 250 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 8b greedy test 0 250 1000 $num_ctxs $model_name true

#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nq  8b greedy test 0 20000 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh tqa  8b greedy test 0 20000 1000 $num_ctxs $model_name true
#
num_ctxs=1
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh newsqa  8b greedy test 0 20000 1000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh squad2.0 8b greedy test 0 20000 1000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh squad1.1 8b greedy test 0 20000 1000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh ROPES  8b greedy test 0 20000 1000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh Quoref  8b greedy test 0 20000 1000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh NarrativeQA 8b greedy test 0 20000 1000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh drop  8b greedy test 0 20000 1000 $num_ctxs $model_name
#
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh doc2dial  8b greedy test 0 20000 1000 $num_ctxs $model_name
bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh doc2dial_gold  8b greedy test 0 20000 1000 $num_ctxs $model_name
#
model_name=sft_pp1-3.5t_same_format_ctx1_8b_128_5e-6
num_ctxs=5
##bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nq 8b greedy test  0 200 1000 $num_ctxs $model_name true
##bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 8b greedy test  0 250 1000 $num_ctxs $model_name true
##bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 8b greedy test  0 250 1000 $num_ctxs $model_name true
##bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh inference_input_retriever_dragon_msmarcominilm_doc2dial 8b greedy test 0 250 1000 $num_ctxs $model_name true
##bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 8b greedy test 0 250 1000 $num_ctxs $model_name true
##bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nv_benefits_dragon_retriever300_retrieved_generic 8b greedy test 0 250 1000 $num_ctxs $model_name true
##bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 8b greedy test 0 250 1000 $num_ctxs $model_name true
##bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 8b greedy test 0 250 1000 $num_ctxs $model_name true
#
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nq  8b greedy test 0 20000 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh tqa  8b greedy test 0 20000 1000 $num_ctxs $model_name true
#
num_ctxs=1
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh newsqa  8b greedy test 0 20000 1000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh squad2.0 8b greedy test 0 20000 1000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh squad1.1 8b greedy test 0 20000 1000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh ROPES  8b greedy test 0 20000 1000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh Quoref  8b greedy test 0 20000 1000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh NarrativeQA 8b greedy test 0 20000 1000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh drop  8b greedy test 0 20000 1000 $num_ctxs $model_name
#
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh doc2dial  8b greedy test 0 20000 1000 $num_ctxs $model_name
bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh doc2dial_gold  8b greedy test 0 20000 1000 $num_ctxs $model_name




model_name=sft_gpt-fitting-full-qc-pp1_same_format_ctx1_43b_128_5e-6
num_ctxs=5
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nq 43b greedy test  0 200 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test  0 250 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 43b greedy test  0 250 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh inference_input_retriever_dragon_msmarcominilm_doc2dial 43b greedy test 0 250 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test 0 250 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nv_benefits_dragon_retriever300_retrieved_generic 43b greedy test 0 250 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 43b greedy test 0 250 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test 0 250 1000 $num_ctxs $model_name true

#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nq  43b greedy test 0 2500 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh tqa  43b greedy test 0 2500 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh ELI5  43b greedy test 0 1000 1000 $num_ctxs $model_name true

#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nq  43b greedy test 0 20000 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh tqa  43b greedy test 0 20000 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh ELI5  43b greedy test 0 20000 1000 $num_ctxs $model_name true

#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh doc2dial  43b greedy test 0 20000 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh quac  43b greedy test 0 20000 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh qrecc  43b greedy test 0 20000 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh sharc  43b greedy test 0 20000 1000 $num_ctxs $model_name true


# template ablation
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh tqa  43b greedy test 0 20000 1000 $num_ctxs $model_name  1 true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh tqa  43b greedy test 0 20000 1000 $num_ctxs $model_name  2 true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh tqa  43b greedy test 0 20000 1000 $num_ctxs $model_name  3 true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh tqa  43b greedy test 0 20000 1000 $num_ctxs $model_name  4 true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh tqa  43b greedy test 0 20000 1000 $num_ctxs $model_name  5 true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh tqa  43b greedy test 0 20000 1000 $num_ctxs $model_name  6 true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh tqa  43b greedy test 0 20000 1000 $num_ctxs $model_name  7 true
##
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh nq  43b greedy test 0 20000 1000 $num_ctxs $model_name  1 true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh nq  43b greedy test 0 20000 1000 $num_ctxs $model_name  2 true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh nq  43b greedy test 0 20000 1000 $num_ctxs $model_name  3 true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh nq  43b greedy test 0 20000 1000 $num_ctxs $model_name  4 true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh nq  43b greedy test 0 20000 1000 $num_ctxs $model_name  5 true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh nq  43b greedy test 0 20000 1000 $num_ctxs $model_name  6 true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh nq  43b greedy test 0 20000 1000 $num_ctxs $model_name  7 true
#
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh ELI5  43b greedy test 0 1000 1000 $num_ctxs $model_name  1 true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh ELI5  43b greedy test 0 1000 1000 $num_ctxs $model_name  2 true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh ELI5  43b greedy test 0 1000 1000 $num_ctxs $model_name  3 true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh ELI5  43b greedy test 0 1000 1000 $num_ctxs $model_name  4 true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh ELI5  43b greedy test 0 1000 1000 $num_ctxs $model_name  5 true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh ELI5  43b greedy test 0 1000 1000 $num_ctxs $model_name  6 true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh ELI5  43b greedy test 0 1000 1000 $num_ctxs $model_name  7 true

num_ctxs=2
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh ELI5-oracle  43b greedy test 0 1000 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh ELI5-oracle  43b greedy test 0 20000 1000 $num_ctxs $model_name true

# template ablation
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh ELI5-oracle  43b greedy test 0 1000 1000 $num_ctxs $model_name  1 true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh ELI5-oracle  43b greedy test 0 1000 1000 $num_ctxs $model_name  2 true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh ELI5-oracle  43b greedy test 0 1000 1000 $num_ctxs $model_name  3 true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh ELI5-oracle  43b greedy test 0 1000 1000 $num_ctxs $model_name  4 true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh ELI5-oracle  43b greedy test 0 1000 1000 $num_ctxs $model_name  5 true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh ELI5-oracle  43b greedy test 0 1000 1000 $num_ctxs $model_name  6 true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh ELI5-oracle  43b greedy test 0 1000 1000 $num_ctxs $model_name  7 true

num_ctxs=1
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh newsqa  43b greedy test 0 2500 1000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh squad2.0 43b greedy test 0 2500 1000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh squad1.1 43b greedy test 0 2500 1000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh ROPES  43b greedy test 0 2500 1000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh Quoref  43b greedy test 0 2500 1000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh NarrativeQA 43b greedy test 0 2500 1000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh drop  43b greedy test 0 2500 1000 $num_ctxs $model_name

#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh newsqa  43b greedy test 0 20000 1000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh squad2.0 43b greedy test 0 20000 1000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh squad1.1 43b greedy test 0 20000 1000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh ROPES  43b greedy test 0 20000 1000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh Quoref  43b greedy test 0 20000 1000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh NarrativeQA 43b greedy test 0 20000 1000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh drop  43b greedy test 0 20000 1000 $num_ctxs $model_name

#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh doc2dial  43b greedy test 0 20000 1000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh quac     43b greedy test 0 3000 1000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh qrecc   43b greedy test 0 20000 1000 $num_ctxs $model_name

# template ablation
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh squad2.0  43b greedy test 0 20000 1000 $num_ctxs $model_name 1
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh squad2.0  43b greedy test 0 20000 1000 $num_ctxs $model_name 2
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh squad2.0  43b greedy test 0 20000 1000 $num_ctxs $model_name 3
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh squad2.0  43b greedy test 0 20000 1000 $num_ctxs $model_name 4
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh squad2.0  43b greedy test 0 20000 1000 $num_ctxs $model_name 5
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh squad2.0  43b greedy test 0 20000 1000 $num_ctxs $model_name 6
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh squad2.0  43b greedy test 0 20000 1000 $num_ctxs $model_name 7
#
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh squad1.1  43b greedy test 0 20000 1000 $num_ctxs $model_name 1
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh squad1.1  43b greedy test 0 20000 1000 $num_ctxs $model_name 2
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh squad1.1  43b greedy test 0 20000 1000 $num_ctxs $model_name 3
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh squad1.1  43b greedy test 0 20000 1000 $num_ctxs $model_name 4
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh squad1.1  43b greedy test 0 20000 1000 $num_ctxs $model_name 5
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh squad1.1  43b greedy test 0 20000 1000 $num_ctxs $model_name 6
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh squad1.1  43b greedy test 0 20000 1000 $num_ctxs $model_name 7
#
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh newsqa  43b greedy test 0 20000 1000 $num_ctxs $model_name 1
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh newsqa  43b greedy test 0 20000 1000 $num_ctxs $model_name 2
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh newsqa  43b greedy test 0 20000 1000 $num_ctxs $model_name 3
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh newsqa  43b greedy test 0 20000 1000 $num_ctxs $model_name 4
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh newsqa  43b greedy test 0 20000 1000 $num_ctxs $model_name 5
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh newsqa  43b greedy test 0 20000 1000 $num_ctxs $model_name 6
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh newsqa  43b greedy test 0 20000 1000 $num_ctxs $model_name 7
#
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh Quoref  43b greedy test 0 20000 1000 $num_ctxs $model_name 1
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh Quoref  43b greedy test 0 20000 1000 $num_ctxs $model_name 2
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh Quoref  43b greedy test 0 20000 1000 $num_ctxs $model_name 3
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh Quoref  43b greedy test 0 20000 1000 $num_ctxs $model_name 4
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh Quoref  43b greedy test 0 20000 1000 $num_ctxs $model_name 5
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh Quoref  43b greedy test 0 20000 1000 $num_ctxs $model_name 6
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh Quoref  43b greedy test 0 20000 1000 $num_ctxs $model_name 7
#
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh NarrativeQA  43b greedy test 0 20000 1000 $num_ctxs $model_name 1
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh NarrativeQA  43b greedy test 0 20000 1000 $num_ctxs $model_name 2
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh NarrativeQA  43b greedy test 0 20000 1000 $num_ctxs $model_name 3
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh NarrativeQA  43b greedy test 0 20000 1000 $num_ctxs $model_name 4
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh NarrativeQA  43b greedy test 0 20000 1000 $num_ctxs $model_name 5
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh NarrativeQA  43b greedy test 0 20000 1000 $num_ctxs $model_name 6
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh NarrativeQA  43b greedy test 0 20000 1000 $num_ctxs $model_name 7
#
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh drop  43b greedy test 0 20000 1000 $num_ctxs $model_name 1
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh drop  43b greedy test 0 20000 1000 $num_ctxs $model_name 2
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh drop  43b greedy test 0 20000 1000 $num_ctxs $model_name 3
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh drop  43b greedy test 0 20000 1000 $num_ctxs $model_name 4
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh drop  43b greedy test 0 20000 1000 $num_ctxs $model_name 5
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh drop  43b greedy test 0 20000 1000 $num_ctxs $model_name 6
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh drop  43b greedy test 0 20000 1000 $num_ctxs $model_name 7

#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh doc2dial  43b greedy test 0 20000 1000 $num_ctxs $model_name 1
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh doc2dial  43b greedy test 0 20000 1000 $num_ctxs $model_name 2
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh doc2dial  43b greedy test 0 20000 1000 $num_ctxs $model_name 3
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh doc2dial  43b greedy test 0 20000 1000 $num_ctxs $model_name 4
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh doc2dial  43b greedy test 0 20000 1000 $num_ctxs $model_name 5
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh doc2dial  43b greedy test 0 20000 1000 $num_ctxs $model_name 6
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh doc2dial  43b greedy test 0 20000 1000 $num_ctxs $model_name 7
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh doc2dial  43b greedy test 0 20000 1000 $num_ctxs $model_name 7

model_name=sft_full-qc-pp1_same_format_ctx1_43b_128_5e-6
num_ctxs=5
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nq 43b greedy test  0 200 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test  0 250 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 43b greedy test  0 250 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh inference_input_retriever_dragon_msmarcominilm_doc2dial 43b greedy test 0 250 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test 0 250 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nv_benefits_dragon_retriever300_retrieved_generic 43b greedy test 0 250 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 43b greedy test 0 250 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test 0 250 1000 $num_ctxs $model_name true

#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nq  43b greedy test 0 2500 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh tqa  43b greedy test 0 2500 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh ELI5  43b greedy test 0 1000 1000 $num_ctxs $model_name true

#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nq  43b greedy test 0 20000 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh tqa  43b greedy test 0 20000 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh ELI5  43b greedy test 0 20000 1000 $num_ctxs $model_name true

#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh doc2dial  43b greedy test 0 20000 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh quac     43b greedy test 0 20000 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh qrecc     43b greedy test 0 20000 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh sharc     43b greedy test 0 20000 1000 $num_ctxs $model_name true

# template ablation
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh tqa  43b greedy test 0 20000 1000 $num_ctxs $model_name  1 true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh tqa  43b greedy test 0 20000 1000 $num_ctxs $model_name  2 true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh tqa  43b greedy test 0 20000 1000 $num_ctxs $model_name  3 true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh tqa  43b greedy test 0 20000 1000 $num_ctxs $model_name  4 true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh tqa  43b greedy test 0 20000 1000 $num_ctxs $model_name  5 true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh tqa  43b greedy test 0 20000 1000 $num_ctxs $model_name  6 true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh tqa  43b greedy test 0 20000 1000 $num_ctxs $model_name  7 true
##
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh nq  43b greedy test 0 20000 1000 $num_ctxs $model_name  1 true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh nq  43b greedy test 0 20000 1000 $num_ctxs $model_name  2 true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh nq  43b greedy test 0 20000 1000 $num_ctxs $model_name  3 true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh nq  43b greedy test 0 20000 1000 $num_ctxs $model_name  4 true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh nq  43b greedy test 0 20000 1000 $num_ctxs $model_name  5 true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh nq  43b greedy test 0 20000 1000 $num_ctxs $model_name  6 true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh nq  43b greedy test 0 20000 1000 $num_ctxs $model_name  7 true

num_ctxs=2
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh ELI5-oracle  43b greedy test 0 1000 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh ELI5-oracle  43b greedy test 0 20000 1000 $num_ctxs $model_name true

# template ablation
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh ELI5-oracle  43b greedy test 0 1000 1000 $num_ctxs $model_name  1 true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh ELI5-oracle  43b greedy test 0 1000 1000 $num_ctxs $model_name  2 true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh ELI5-oracle  43b greedy test 0 1000 1000 $num_ctxs $model_name  3 true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh ELI5-oracle  43b greedy test 0 1000 1000 $num_ctxs $model_name  4 true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh ELI5-oracle  43b greedy test 0 1000 1000 $num_ctxs $model_name  5 true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh ELI5-oracle  43b greedy test 0 1000 1000 $num_ctxs $model_name  6 true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh ELI5-oracle  43b greedy test 0 1000 1000 $num_ctxs $model_name  7 true

num_ctxs=1
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh newsqa  43b greedy test 0 2500 1000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh squad2.0 43b greedy test 0 2500 1000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh squad1.1 43b greedy test 0 2500 1000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh ROPES  43b greedy test 0 2500 1000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh Quoref  43b greedy test 0 2500 1000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh NarrativeQA 43b greedy test 0 2500 1000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh drop  43b greedy test 0 2500 1000 $num_ctxs $model_name

#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh newsqa  43b greedy test 0 20000 1000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh squad2.0 43b greedy test 0 20000 1000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh squad1.1 43b greedy test 0 20000 1000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh ROPES  43b greedy test 0 20000 1000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh Quoref  43b greedy test 0 20000 1000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh NarrativeQA 43b greedy test 0 20000 1000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh drop  43b greedy test 0 20000 1000 $num_ctxs $model_name

#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh doc2dial  43b greedy test 0 20000 1000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh quac  43b greedy test 0 3000 1000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh qrecc  43b greedy test 0 20000 1000 $num_ctxs $model_name

# template ablation
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh squad2.0  43b greedy test 0 20000 1000 $num_ctxs $model_name 1
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh squad2.0  43b greedy test 0 20000 1000 $num_ctxs $model_name 2
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh squad2.0  43b greedy test 0 20000 1000 $num_ctxs $model_name 3
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh squad2.0  43b greedy test 0 20000 1000 $num_ctxs $model_name 4
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh squad2.0  43b greedy test 0 20000 1000 $num_ctxs $model_name 5
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh squad2.0  43b greedy test 0 20000 1000 $num_ctxs $model_name 6
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh squad2.0  43b greedy test 0 20000 1000 $num_ctxs $model_name 7
#
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh squad1.1  43b greedy test 0 20000 1000 $num_ctxs $model_name 1
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh squad1.1  43b greedy test 0 20000 1000 $num_ctxs $model_name 2
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh squad1.1  43b greedy test 0 20000 1000 $num_ctxs $model_name 3
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh squad1.1  43b greedy test 0 20000 1000 $num_ctxs $model_name 4
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh squad1.1  43b greedy test 0 20000 1000 $num_ctxs $model_name 5
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh squad1.1  43b greedy test 0 20000 1000 $num_ctxs $model_name 6
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh squad1.1  43b greedy test 0 20000 1000 $num_ctxs $model_name 7
#
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh newsqa  43b greedy test 0 20000 1000 $num_ctxs $model_name 1
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh newsqa  43b greedy test 0 20000 1000 $num_ctxs $model_name 2
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh newsqa  43b greedy test 0 20000 1000 $num_ctxs $model_name 3
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh newsqa  43b greedy test 0 20000 1000 $num_ctxs $model_name 4
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh newsqa  43b greedy test 0 20000 1000 $num_ctxs $model_name 5
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh newsqa  43b greedy test 0 20000 1000 $num_ctxs $model_name 6
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh newsqa  43b greedy test 0 20000 1000 $num_ctxs $model_name 7
#
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh Quoref  43b greedy test 0 20000 1000 $num_ctxs $model_name 1
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh Quoref  43b greedy test 0 20000 1000 $num_ctxs $model_name 2
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh Quoref  43b greedy test 0 20000 1000 $num_ctxs $model_name 3
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh Quoref  43b greedy test 0 20000 1000 $num_ctxs $model_name 4
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh Quoref  43b greedy test 0 20000 1000 $num_ctxs $model_name 5
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh Quoref  43b greedy test 0 20000 1000 $num_ctxs $model_name 6
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh Quoref  43b greedy test 0 20000 1000 $num_ctxs $model_name 7
#
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh NarrativeQA  43b greedy test 0 20000 1000 $num_ctxs $model_name 1
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh NarrativeQA  43b greedy test 0 20000 1000 $num_ctxs $model_name 2
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh NarrativeQA  43b greedy test 0 20000 1000 $num_ctxs $model_name 3
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh NarrativeQA  43b greedy test 0 20000 1000 $num_ctxs $model_name 4
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh NarrativeQA  43b greedy test 0 20000 1000 $num_ctxs $model_name 5
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh NarrativeQA  43b greedy test 0 20000 1000 $num_ctxs $model_name 6
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh NarrativeQA  43b greedy test 0 20000 1000 $num_ctxs $model_name 7
#
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh drop  43b greedy test 0 20000 1000 $num_ctxs $model_name 1
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh drop  43b greedy test 0 20000 1000 $num_ctxs $model_name 2
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh drop  43b greedy test 0 20000 1000 $num_ctxs $model_name 3
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh drop  43b greedy test 0 20000 1000 $num_ctxs $model_name 4
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh drop  43b greedy test 0 20000 1000 $num_ctxs $model_name 5
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh drop  43b greedy test 0 20000 1000 $num_ctxs $model_name 6
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh drop  43b greedy test 0 20000 1000 $num_ctxs $model_name 7

#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh doc2dial  43b greedy test 0 20000 1000 $num_ctxs $model_name 1
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh doc2dial  43b greedy test 0 20000 1000 $num_ctxs $model_name 2
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh doc2dial  43b greedy test 0 20000 1000 $num_ctxs $model_name 3
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh doc2dial  43b greedy test 0 20000 1000 $num_ctxs $model_name 4
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh doc2dial  43b greedy test 0 20000 1000 $num_ctxs $model_name 5
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh doc2dial  43b greedy test 0 20000 1000 $num_ctxs $model_name 6
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_template.sh doc2dial  43b greedy test 0 20000 1000 $num_ctxs $model_name 7

#
model_name=multiturn_qa_blend_commercial_v5_gpt_fitting_1e-8_conv_full_quiet_cockatoo_pp1_addmultiturn_same_format_ctx1_43b_64_3e-7
num_ctxs=5
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nq 43b greedy test  0 200 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test  0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 43b greedy test  0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh inference_input_retriever_dragon_msmarcominilm_doc2dial 43b greedy test 0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test 0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nv_benefits_dragon_retriever300_retrieved_generic 43b greedy test 0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 43b greedy test 0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test 0 250 3000 $num_ctxs $model_name true
#
model_name=multiturn_qa_blend_commercial_v5_gpt_1e-8_conv_full_quiet_cockatoo_pp1_addmultiturn_same_format_ctx1_43b_64_3e-7
num_ctxs=5
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nq 43b greedy test  0 200 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test  0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 43b greedy test  0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh inference_input_retriever_dragon_msmarcominilm_doc2dial 43b greedy test 0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test 0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nv_benefits_dragon_retriever300_retrieved_generic 43b greedy test 0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 43b greedy test 0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test 0 250 3000 $num_ctxs $model_name true


model_name=multiturn_qa_blend_commercial_v5_nemo_gpt_1e-8_conv_quiet_cockatoo_pp1_addmultiturn_same_format_ctx1_43b_64_3e-7
num_ctxs=5
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nq 43b greedy test  0 200 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test  0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 43b greedy test  0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh inference_input_retriever_dragon_msmarcominilm_doc2dial 43b greedy test 0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test 0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nv_benefits_dragon_retriever300_retrieved_generic 43b greedy test 0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 43b greedy test 0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test 0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh doc2dial 43b greedy test 0 1000 3000 $num_ctxs $model_name true


model_name=multiturn_qa_blend_commercial_v5_gpt_1e-8_conv_quiet_cockatoo_pp1_addmultiturn_bak_same_format_ctx1_43b_64_3e-7
num_ctxs=5
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nq 43b greedy test  0 200 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test  0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 43b greedy test  0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh inference_input_retriever_dragon_msmarcominilm_doc2dial 43b greedy test 0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test 0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nv_benefits_dragon_retriever300_retrieved_generic 43b greedy test 0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 43b greedy test 0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test 0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh doc2dial 43b greedy test 0 1000 3000 $num_ctxs $model_name true


model_name=multiturn_qa_blend_commercial_v5_gpt_1e-8_conv_quiet_cockatoo_pp1_addmultiturn_same_format_ctx1_43b_64_3e-7
num_ctxs=5
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nq 43b greedy test  0 200 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test  0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 43b greedy test  0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh inference_input_retriever_dragon_msmarcominilm_doc2dial 43b greedy test 0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test 0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nv_benefits_dragon_retriever300_retrieved_generic 43b greedy test 0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 43b greedy test 0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test 0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh doc2dial 43b greedy test 0 1000 3000 $num_ctxs $model_name true


model_name=multiturn_qa_blendv2_gpt_1e-8_conv_quiet_cockatoo_pp1_addmultiturn_same_format_ctx1_43b_64_3e-7
# model_name=multiturn_qa_blend_commercial_gpt_1e-8_conv_quiet_cockatoo_pp1_addmultiturn_same_format_ctx1_43b_64_3e-7
model_name=multiturn_qa_blendv2_nemo_gpt_1e-8_conv_quiet_cockatoo_pp1_addmultiturn_same_format_ctx1_43b_64_3e-7
num_ctxs=5
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nq 43b greedy test  0 200 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test  0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 43b greedy test  0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh inference_input_retriever_dragon_msmarcominilm_doc2dial 43b greedy test 0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test 0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nv_benefits_dragon_retriever300_retrieved_generic 43b greedy test 0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 43b greedy test 0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test 0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh doc2dial 43b greedy test 0 1000 3000 $num_ctxs $model_name true

## single-turn qa (batch-1)
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nq 43b greedy test  0 200 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test  0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 43b greedy test  0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test 0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nv_benefits_dragon_retriever300_retrieved_generic 43b greedy test 0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 43b greedy test 0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test 0 250 3000 $num_ctxs $model_name true
#
### single-turn-qa (batch-2)
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh BioASQ 43b greedy test 0 1000 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh DuoRC_ParaphraseRC 43b greedy test 0 1000 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh boolq 43b greedy test 0 1000 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh msmarco 43b greedy test 0 1000 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh multirc 43b greedy test 0 1000 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh race 43b greedy test 0 1000 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh TextbookQA 43b greedy test 0 1000 3000 $num_ctxs $model_name true
#
### multi-turn qa
## doc2dial 3939 samples
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh doc2dial 43b greedy test 0 1000 3000 $num_ctxs $model_name true
## quac 7354 samples
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh quac 43b greedy test 0 1000 3000 $num_ctxs $model_name true
## qrecc 2805 samples
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh qrecc 43b greedy test 0 1000 3000 $num_ctxs $model_name true
## sharc 10000 samples
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh sharc 43b greedy test 0 1000 3000 $num_ctxs $model_name true
#

model_name=retro-multiturn_qa_blendv2_retro_1e-8_conv_quiet_cockatoo_pp1_addmultiturn_same_format_ctx1_43b_64_3e-7
num_ctxs=5
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_retro.sh nq 43b greedy test  0 200 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_retro.sh att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test  0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_retro.sh ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 43b greedy test  0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_retro.sh inference_input_retriever_dragon_msmarcominilm_doc2dial 43b greedy test 0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_retro.sh NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test 0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_retro.sh nv_benefits_dragon_retriever300_retrieved_generic 43b greedy test 0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_retro.sh landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 43b greedy test 0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_retro.sh Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test 0 250 3000 $num_ctxs $model_name true
#


# model_name=qa_blendv12_gpt_1e-8_conv_quiet_cockatoo_pp1_fixed_newsqa_same_format_ctx1_43b_64_3e-7
model_name=sft_pp1_same_format_ctx1_8b_128_5e-6
num_ctxs=5
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nq 8b greedy test  0 3610 1000 $num_ctxs $model_name true

model_name=retro-sft_pp1_same_format_ctx1_8b_128_5e-6
num_ctxs=5
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_retro.sh nq 8b greedy test  0 3610 1000 $num_ctxs $model_name true


model_name=sft_pp1_same_format_ctx1_43b_128_5e-6
num_ctxs=5
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nq 43b greedy test  0 200 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test  0 250 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 43b greedy test  0 250 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh inference_input_retriever_dragon_msmarcominilm_doc2dial 43b greedy test 0 250 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test 0 250 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nv_benefits_dragon_retriever300_retrieved_generic 43b greedy test 0 250 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 43b greedy test 0 250 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test 0 250 1000 $num_ctxs $model_name true

# open qa
num_ctxs=1
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh newsqa  43b greedy test 0 250 1000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh squad2.0 43b greedy test 0 250 1000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh squad1.1 43b greedy test 0 250 1000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh ROPES  43b greedy test 0 250 1000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh Quoref  43b greedy test 0 250 1000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh NarrativeQA 43b greedy test 0 250 1000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh drop  43b greedy test 0 250 1000 $num_ctxs $model_name

num_ctxs=5
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nq  43b greedy test 0 2500 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh tqa  43b greedy test 0 2500 1000 $num_ctxs $model_name true
#num_ctxs=1
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh newsqa  43b greedy test 0 2500 1000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh squad2.0 43b greedy test 0 2500 1000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh squad1.1 43b greedy test 0 2500 1000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh ROPES  43b greedy test 0 2500 1000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh Quoref  43b greedy test 0 2500 1000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh NarrativeQA 43b greedy test 0 2500 1000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh drop  43b greedy test 0 2500 1000 $num_ctxs $model_name

#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_l_64.sh newsqa 43b greedy test  0 250 1000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_l_64.sh squad2.0 43b greedy test  0 250 1000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_l_64.sh squad1.1 43b greedy test  0 250 1000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_l_64.sh ROPES 43b greedy test 0 250 1000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_l_64.sh Quoref 43b greedy test 0 250 1000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_l_64.sh NarrativeQA 43b greedy test 0 250 1000 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_l_64.sh drop 43b greedy test 0 250 1000 $num_ctxs $model_name

model_name=megatron_sft_quiet_cockatoo_tp8_pp1
num_ctxs=1
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh newsqa 43b greedy test  0 250 1 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh squad2.0 43b greedy test  0 250 1 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh squad1.1 43b greedy test  0 250 1 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh ROPES 43b greedy test 0 250 1 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh Quoref 43b greedy test 0 250 1 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh NarrativeQA 43b greedy test 0 250 1 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh drop 43b greedy test 0 250 1 $num_ctxs $model_name

num_ctxs=5
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nq  43b greedy test 0 2500 1 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh tqa  43b greedy test 0 2500 1 $num_ctxs $model_name true
#num_ctxs=1
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh newsqa 43b greedy test  0 2500 1 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh squad2.0 43b greedy test  0 2500 1 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh squad1.1 43b greedy test  0 2500 1 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh ROPES 43b greedy test 0 2500 1 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh Quoref 43b greedy test 0 2500 1 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh NarrativeQA 43b greedy test 0 2500 1 $num_ctxs $model_name
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh drop 43b greedy test 0 2500 1 $num_ctxs $model_name

#

#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_l_64.sh nq 43b greedy test  0 200 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_l_64.sh att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test  0 250 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_l_64.sh ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 43b greedy test  0 250 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_l_64.sh inference_input_retriever_dragon_msmarcominilm_doc2dial 43b greedy test 0 250 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_l_64.sh NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test 0 250 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_l_64.sh nv_benefits_dragon_retriever300_retrieved_generic 43b greedy test 0 250 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_l_64.sh landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 43b greedy test 0 250 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_l_64.sh Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test 0 250 1000 $num_ctxs $model_name true

model_name=sft_gpt-fitting-pp1_same_format_ctx1_43b_128_5e-6
num_ctxs=5
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nq 43b greedy test  0 200 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test  0 250 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 43b greedy test  0 250 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh inference_input_retriever_dragon_msmarcominilm_doc2dial 43b greedy test 0 250 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test 0 250 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nv_benefits_dragon_retriever300_retrieved_generic 43b greedy test 0 250 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 43b greedy test 0 250 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test 0 250 1000 $num_ctxs $model_name true

model_name=retro-sft_pp1_same_format_ctx1_43b_128_5e-6
num_ctxs=5
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_retro.sh nq 43b greedy test  0 200 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_retro.sh att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test  0 250 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_retro.sh ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 43b greedy test  0 250 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_retro.sh inference_input_retriever_dragon_msmarcominilm_doc2dial 43b greedy test 0 250 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_retro.sh NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test 0 250 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_retro.sh nv_benefits_dragon_retriever300_retrieved_generic 43b greedy test 0 250 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_retro.sh landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 43b greedy test 0 250 1000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv_retro.sh Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test 0 250 1000 $num_ctxs $model_name true



#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nq 43b greedy test  0 200 0 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test  0 250 0 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 43b greedy test  0 250 0 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh inference_input_retriever_dragon_msmarcominilm_doc2dial 43b greedy test 0 250 0 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test 0 250 0 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nv_benefits_dragon_retriever300_retrieved_generic 43b greedy test 0 250 0 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 43b greedy test 0 250 0 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test 0 250 0 $num_ctxs $model_name true

#model_name=megatron_sft_quiet_cockatoo_tp8_pp1
#
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nq 43b greedy test  0 200 1 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test  0 250 1 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 43b greedy test  0 250 1 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh inference_input_retriever_dragon_msmarcominilm_doc2dial 43b greedy test 0 250 1 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test 0 250 1 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nv_benefits_dragon_retriever300_retrieved_generic 43b greedy test 0 250 1 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 43b greedy test 0 250 1 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test 0 250 1 $num_ctxs $model_name true


#model_name=gpt3-43b-pretraining-gpt-fitting-tp8pp1
#
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nq 43b greedy test  0 200 32552 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test  0 250 32552 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 43b greedy test  0 250 32552 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh inference_input_retriever_dragon_msmarcominilm_doc2dial 43b greedy test 0 250 32552 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test 0 250 32552 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nv_benefits_dragon_retriever300_retrieved_generic 43b greedy test 0 250 32552 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 43b greedy test 0 250 32552 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test 0 250 32552 $num_ctxs $model_name true

model_name=qa_blendv12_pp1_same_format_ctx1_43b_64_3e-7
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nq 43b greedy test  0 200 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test  0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 43b greedy test  0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh inference_input_retriever_dragon_msmarcominilm_doc2dial 43b greedy test 0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test 0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nv_benefits_dragon_retriever300_retrieved_generic 43b greedy test 0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 43b greedy test 0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test 0 250 3000 $num_ctxs $model_name true

#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nq 43b greedy test  0 200 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test  0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 43b greedy test  0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh inference_input_retriever_dragon_msmarcominilm_doc2dial 43b greedy test 0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test 0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nv_benefits_dragon_retriever300_retrieved_generic 43b greedy test 0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 43b greedy test 0 250 3000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test 0 250 3000 $num_ctxs $model_name true
#
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nq 43b greedy test  0 200 6000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test  0 250 6000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 43b greedy test  0 250 6000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh inference_input_retriever_dragon_msmarcominilm_doc2dial 43b greedy test 0 250 6000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test 0 250 6000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nv_benefits_dragon_retriever300_retrieved_generic 43b greedy test 0 250 6000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 43b greedy test 0 250 6000 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test 0 250 6000 $num_ctxs $model_name true

#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nq 43b greedy test  0 200 4500 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test  0 250 4500 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 43b greedy test  0 250 4500 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh inference_input_retriever_dragon_msmarcominilm_doc2dial 43b greedy test 0 250 4500 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test 0 250 4500 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nv_benefits_dragon_retriever300_retrieved_generic 43b greedy test 0 250 4500 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 43b greedy test 0 250 4500 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test 0 250 4500 $num_ctxs $model_name true

#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nq 43b greedy test  0 200 1500 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test  0 250 1500 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 43b greedy test  0 250 1500 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh inference_input_retriever_dragon_msmarcominilm_doc2dial 43b greedy test 0 250 1500 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test 0 250 1500 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh nv_benefits_dragon_retriever300_retrieved_generic 43b greedy test 0 250 1500 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 43b greedy test 0 250 1500 $num_ctxs $model_name true
#bash examples/foundational_qa/generate_multijob_ckpt_step_same_format_cross_fqa_conv.sh Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 43b greedy test 0 250 1500 $num_ctxs $model_name true

#bash examples/foundational_qa/retro_generate_multijob_ckpt_step_same_format_reuse_flex.sh nq 43b greedy test  32 1e-6  0 200 32552 2 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed 2
