
# model_name=llama2_chat_7b
# model_name=llama2_chat_70b
#model_name=llama2_text_70b_with_qc
model_name=llama2_text_70b_pp1
#model_name=llama2_chat_70b_pp1
model_name=llama2_text_13b
num_ctxs=5


## tabular qa
# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh fetaqa 70b greedy test 0 1001 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh tatqav2 70b greedy test 0 1000 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh WikiTableQuestions 70b greedy test 0 2200 $num_ctxs $model_name true
#bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh WikiTableQuestions 70b greedy test 2200 2200 $num_ctxs $model_name true


## finance qa
# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh finqa 70b greedy test 0 1000 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh convfinqa 70b greedy test 0 1000 $num_ctxs $model_name true

# # ## single-turn qa (batch-1)

#for (( i=0; i<=4000; i+=1000 )); do
#    echo $i
#    bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh nq 70b greedy test  $i 1000 $num_ctxs $model_name true
#done

#for (( i=0; i<=4000; i+=1000 )); do
#    echo $i
#    cat nq_5_changeformat_generate_70b_test_greedy_${i}_1000_ret.txt.v2 >>  nq_5_changeformat_generate_70b_test_greedy_0_20000_ret.txt.v2
#done

#for (( i=0; i<=12000; i+=1000 )); do
#    echo $i
#    bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh tqa 70b greedy test  $i 1000 $num_ctxs $model_name true
#done

#for (( i=0; i<=12000; i+=1000 )); do
#    echo $i
#    cat tqa_5_changeformat_generate_70b_test_greedy_${i}_1000_ret.txt.v2 >>  tqa_5_changeformat_generate_70b_test_greedy_0_20000_ret.txt.v2
#done

#
bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh nq 13b greedy test  0 200 $num_ctxs $model_name true
## bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh tqa 70b greedy test  0 20000 $num_ctxs $model_name true
## bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh tqa 70b greedy test  0 5000 $num_ctxs $model_name true
## bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh tqa 70b greedy test  5000 5000 $num_ctxs $model_name true
## bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh tqa 70b greedy test  10000 20000 $num_ctxs $model_name true

#bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 70b greedy test  0 250 $num_ctxs $model_name true
#bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh NVIT_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 70b greedy test 0 250 $num_ctxs $model_name true
#bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh landrover_plus_benz_clean_plus_ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 70b greedy test 0 250 $num_ctxs $model_name true


num_ctxs=1
#
#for x in "squad2.0" "squad1.1"; do
#    for (( i=0; i<=12000; i+=1000 )); do
#        echo "$x: $i"
#        bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh $x 70b greedy test  $i 1000 $num_ctxs $model_name
#    done
#done

#for x in "squad2.0" "squad1.1"; do
#    for (( i=0; i<=12000; i+=1000 )); do
#        echo "$x: $i"
#        cat ${x}_1_changeformat_generate_70b_test_greedy_${i}_1000.txt.v2 >>  ${x}_1_changeformat_generate_70b_test_greedy_0_20000.txt.v2
#    done
#    wc -l ${x}_1_changeformat_generate_70b_test_greedy_0_20000.txt.v2
#done

#for x in "newsqa" "Quoref" "NarrativeQA" "drop" "doc2dial"; do
#    for (( i=0; i<=6000; i+=1000 )); do
#        echo "$x: $i"
#        bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh $x 70b greedy test  $i 1000 $num_ctxs $model_name
#    done
#done

#for x in "newsqa" "Quoref" "NarrativeQA" "drop" "doc2dial"; do
#    for (( i=0; i<=6000; i+=1000 )); do
#        echo "$x: $i"
#        cat ${x}_1_changeformat_generate_70b_test_greedy_${i}_1000.txt.v2 >>  ${x}_1_changeformat_generate_70b_test_greedy_0_20000.txt.v2
#    done
#    wc -l ${x}_1_changeformat_generate_70b_test_greedy_0_20000.txt.v2
#done

#for x in "doc2dial"; do
#    for (( i=0; i<=3000; i+=1000 )); do
#        echo "$x: $i"
#        bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh $x 70b greedy test  $i 1000 $num_ctxs $model_name
#    done
#done

#for x in "doc2dial"; do
#    for (( i=0; i<=3000; i+=1000 )); do
#        echo "$x: $i"
#        cat ${x}_1_changeformat_generate_70b_test_greedy_${i}_1000.txt.v2 >>  ${x}_1_changeformat_generate_70b_test_greedy_0_20000.txt.v2
#    done
#    wc -l ${x}_1_changeformat_generate_70b_test_greedy_0_20000.txt.v2
#done

#bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh qmsum.dragon_retriever_chunkbysents300  70b greedy test  0 200 $num_ctxs $model_name
#bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh gov_report.dragon_retriever_chunkbysents300 70b greedy test  0 200 $num_ctxs $model_name
#bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh summ_screen_fd.dragon_retriever_chunkbysents300 70b greedy test  0 200 $num_ctxs $model_name


# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh squad2.0 70b greedy test  0 5000 $num_ctxs $model_name
# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh squad2.0 70b greedy test  5000 5000 $num_ctxs $model_name
# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh squad2.0 70b greedy test  10000 20000 $num_ctxs $model_name

# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh squad1.1 70b greedy test  0 5000 $num_ctxs $model_name
# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh squad1.1 70b greedy test  5000 5000 $num_ctxs $model_name
# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh squad1.1 70b greedy test  10000 20000 $num_ctxs $model_name

# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh doc2dial 70b greedy test  0 2000 $num_ctxs $model_name
# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh doc2dial 70b greedy test  2000 20000 $num_ctxs $model_name

# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh newsqa  70b greedy test  0 20000 $num_ctxs $model_name
# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh squad2.0 70b greedy test  0 20000 $num_ctxs $model_name
# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh squad1.1 70b greedy test  0 20000 $num_ctxs $model_name
# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh ROPES   70b greedy test  0 20000 $num_ctxs $model_name
# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh Quoref  70b greedy test  0 20000 $num_ctxs $model_name
# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh NarrativeQA 70b greedy test  0 20000 $num_ctxs $model_name
# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh drop 70b greedy test  0 20000 $num_ctxs $model_name

# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh doc2dial 70b greedy test  0 20000 $num_ctxs $model_name
# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh att_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 70b greedy test  0 250 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh nv_benefits_dragon_retriever300_retrieved_generic 70b greedy test 0 250 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh Iternal_dragon_retriever_msmarcominilm_reranker_chunkbysents300_retrieved 70b greedy test 0 250 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh sandia 70b greedy test 0 250 $num_ctxs $model_name true



# ## single-turn qa (batch-2)
# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh BioASQ 70b greedy test 0 1000 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh DuoRC_ParaphraseRC 70b greedy test 0 1000 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh boolq 70b greedy test 0 1000 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh msmarco 70b greedy test 0 1000 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh multirc 70b greedy test 0 1000 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh race 70b greedy test 0 1000 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh TextbookQA 70b greedy test 0 1000 $num_ctxs $model_name true

# ## multi-turn qa
# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh doc2dial 70b greedy test 0 1000 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh quac 70b greedy test 0 1000 $num_ctxs $model_name true
# bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh qrecc 70b greedy test 0 1000 $num_ctxs $model_name true
# # bash examples/fqa_llama2/generate_llama2_sft_zeroshot.sh sharc 70b greedy test 0 1000 $num_ctxs $model_name true
