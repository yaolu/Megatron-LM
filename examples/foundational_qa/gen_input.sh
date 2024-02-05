
#sample_input_file="/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/$TASK/${split}.json"
#DATA_FOLDER="/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/$TASK"
#FEWSHOT_INPUT_FOLDER="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa"

sample_input_file="/lustre/fsw/adlr/adlr-nlp/boxinw/instruction_tuning_data/$TASK/${split}.json"
DATA_FOLDER="/lustre/fs4/portfolios/adlr/users/boxinw/instruction_tuning_data/$TASK"
FEWSHOT_INPUT_FOLDER="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa"

if [[ $TASK == "nq" ]]; then
    sample_input_file="/lustre/fs4/portfolios/adlr/users/boxinw/instruction_tuning_data/NQ/${split}.json"
    fewshot_input_file="${FEWSHOT_INPUT_FOLDER}/single-turn-qa/NQ/fewshot_samples.json"
    DATA_FOLDER="/lustre/fs4/portfolios/adlr/users/boxinw/instruction_tuning_data/NQ"
fi

if [[ $TASK == "tqa" ]]; then
    sample_input_file="/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/TQA/${split}.json"
    DATA_FOLDER="/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/TQA"
fi

if [[ $TASK == att* ]]; then
    DATA_FOLDER="/lustre/fsw/adlr/adlr-nlp/pengx/data/att/$TASK"
    sample_input_file="/lustre/fsw/adlr/adlr-nlp/pengx/data/att/$TASK/${split}.json"
    fewshot_input_file="${FEWSHOT_INPUT_FOLDER}/single-turn-qa/att/fewshot_samples.json"
fi

if [[ $TASK == nv_benefits* ]]; then
    DATA_FOLDER="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/$TASK"
    sample_input_file="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/$TASK/${split}.json"
    fewshot_input_file="${FEWSHOT_INPUT_FOLDER}/single-turn-qa/nv_benefits/fewshot_samples.json"
fi

if [[ $TASK == Iternal* ]]; then
    fewshot_input_file="${FEWSHOT_INPUT_FOLDER}/single-turn-qa/iternal/fewshot_samples.json"
fi

if [[ $TASK == NVIT* ]]; then
    fewshot_input_file="${FEWSHOT_INPUT_FOLDER}/single-turn-qa/nvit/fewshot_samples.json"
fi

if [[ $TASK == ford* ]]; then
    fewshot_input_file="${FEWSHOT_INPUT_FOLDER}/single-turn-qa/ford/fewshot_samples.json"
fi

if [[ $TASK == landrover* ]]; then
    fewshot_input_file="${FEWSHOT_INPUT_FOLDER}/single-turn-qa/landrover/fewshot_samples.json"
fi

if [[ $TASK == "sandia" ]]; then
    sample_input_file="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/sandia_e5_unsupervised_retriever_chunkbysents300_retrieved/test.json"
    ## Need to work on fewshot inputs
    fewshot_input_file=""
fi

if [[ $TASK == "llmware" ]]; then
    sample_input_file="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/llmware/test.json"
    ## Need to work on fewshot inputs
    fewshot_input_file=""
fi

if [[ $TASK == "llmware_unanswerable" ]]; then
    sample_input_file="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/llmware/test_unanswerable.json"
    ## Need to work on fewshot inputs
    fewshot_input_file=""
fi

if [[ $TASK == "convfinqa" ]]; then
    sample_input_file="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/convfinqa/test.json"
    ## Need to work on fewshot inputs
    fewshot_input_file=""
fi
if [[ $TASK == "convfinqav2" ]]; then
    sample_input_file="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/convfinqav2/convfinqav2_QA_dev.json"
    ## Need to work on fewshot inputs
    fewshot_input_file=""
fi
if [[ $TASK == "convfinqav3" ]]; then
    sample_input_file="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/convfinqav3/convfinqav3_QA_dev.json"
    ## Need to work on fewshot inputs
    fewshot_input_file=""
fi

if [[ $TASK == "finqa" ]]; then
    sample_input_file="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/finqa/test.json"
    ## Need to work on fewshot inputs
    fewshot_input_file=""
fi
if [[ $TASK == "finqav2" ]]; then
    sample_input_file="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/finqav2/finqav2_QA_dev.json"
    ## Need to work on fewshot inputs
    fewshot_input_file=""
fi

if [[ $TASK == "tatqav2" ]]; then
    sample_input_file="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/tatqav2/dev.json"
    ## Need to work on fewshot inputs
    fewshot_input_file=""
fi

if [[ $TASK == "fetaqa" ]]; then
    sample_input_file="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/fetaqa/fetaqa_QA_dev.json"
    ## Need to work on fewshot inputs
    fewshot_input_file=""
fi

if [[ $TASK == "WikiTableQuestions" ]]; then
    sample_input_file="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/WikiTableQuestions/WikiTableQuestions_QA_test.json"
    ## Need to work on fewshot inputs
    fewshot_input_file=""
fi

if [[ $TASK == "sqa" ]]; then
    sample_input_file="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/sqa/sqa_QA_test.json"
    ## Need to work on fewshot inputs
    fewshot_input_file=""
fi

if [[ $TASK == "HybridQA" ]]; then
    sample_input_file="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/HybridQA/HybridQA_QA_dev.json"
    ## Need to work on fewshot inputs
    fewshot_input_file=""
fi

if [[ $TASK == "BioASQ" ]]; then
    DATA_FOLDER="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/$TASK"
    sample_input_file="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/$TASK/test.json"
    fewshot_input_file="${FEWSHOT_INPUT_FOLDER}/single-turn-qa/$TASK/fewshot_samples.json"
fi

if [[ $TASK == "DuoRC_ParaphraseRC" ]]; then
    DATA_FOLDER="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/$TASK"
    sample_input_file="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/$TASK/test.json"
    fewshot_input_file="${FEWSHOT_INPUT_FOLDER}/single-turn-qa/$TASK/fewshot_samples.json"
fi

if [[ $TASK == "boolq" ]]; then
    DATA_FOLDER="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/$TASK"
    sample_input_file="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/$TASK/test.json"
    fewshot_input_file="${FEWSHOT_INPUT_FOLDER}/single-turn-qa/$TASK/fewshot_samples.json"
fi

if [[ $TASK == "msmarco" ]]; then
    DATA_FOLDER="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/$TASK"
    sample_input_file="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/$TASK/test.json"
    fewshot_input_file="${FEWSHOT_INPUT_FOLDER}/single-turn-qa/$TASK/fewshot_samples.json"
fi

if [[ $TASK == "multirc" ]]; then
    DATA_FOLDER="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/$TASK"
    sample_input_file="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/$TASK/test.json"
    fewshot_input_file="${FEWSHOT_INPUT_FOLDER}/single-turn-qa/$TASK/fewshot_samples.json"
fi

if [[ $TASK == "race" ]]; then
    DATA_FOLDER="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/$TASK"
    sample_input_file="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/$TASK/test.json"
    fewshot_input_file="${FEWSHOT_INPUT_FOLDER}/single-turn-qa/$TASK/fewshot_samples.json"
fi

if [[ $TASK == "TextbookQA" ]]; then
    DATA_FOLDER="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/$TASK"
    sample_input_file="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/$TASK/test.json"
    fewshot_input_file="${FEWSHOT_INPUT_FOLDER}/single-turn-qa/$TASK/fewshot_samples.json"
fi

if [[ $TASK == "doc2dial" ]]; then
    DATA_FOLDER="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/$TASK"
    sample_input_file="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/$TASK/${TASK}_ftdragon_chatgptgen7k_chunk150_QA_test.json"
    fewshot_input_file="${FEWSHOT_INPUT_FOLDER}/multi-turn-qa/doc2dial/fewshot_samples.json"
fi

if [[ $TASK == "quac" ]]; then
    DATA_FOLDER="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/$TASK"
    sample_input_file="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/$TASK/${TASK}_ftdragon_chatgptgen7k_chunk150_QA_test.json"
    fewshot_input_file="${FEWSHOT_INPUT_FOLDER}/multi-turn-qa/quac/fewshot_samples.json"
fi

if [[ $TASK == "qrecc" ]]; then
    DATA_FOLDER="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/$TASK"
    sample_input_file="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/$TASK/${TASK}_ftdragon_chatgptgen7k_chunk150_QA_test.json"
    fewshot_input_file="${FEWSHOT_INPUT_FOLDER}/multi-turn-qa/qrecc/fewshot_samples.json"
fi

if [[ $TASK == "sharc" ]]; then
    DATA_FOLDER="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/$TASK"
    sample_input_file="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/$TASK/${TASK}_ftdragon_chatgptgen7k_chunk150_QA_test.json"
    fewshot_input_file="${FEWSHOT_INPUT_FOLDER}/multi-turn-qa/sharc/fewshot_samples.json"
fi

if [[ $TASK == "coqa" ]]; then
    DATA_FOLDER="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/$TASK"
    sample_input_file="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/$TASK/coqa_QA_dev.json"
    fewshot_input_file=""
fi

if [[ $TASK == "doqa_cooking" ]]; then
    DATA_FOLDER="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/doqa"
    sample_input_file="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/doqa/doqa_cooking_QA_test.json"
    fewshot_input_file=""
fi

if [[ $TASK == "doqa_movies" ]]; then
    DATA_FOLDER="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/doqa"
    sample_input_file="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/doqa/doqa_movies_QA_test.json"
    fewshot_input_file=""
fi

if [[ $TASK == "doqa_travel" ]]; then
    DATA_FOLDER="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/doqa"
    sample_input_file="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/doqa/doqa_travel_QA_test.json"
    fewshot_input_file=""
fi

if [[ $TASK == "topiocqa" ]]; then
    DATA_FOLDER="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/topiocqa"
    sample_input_file="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/topiocqa/topiocqa_dev_retrieval_dragon_ft_chatgptgen7k.json"
    fewshot_input_file=""
fi

if [[ $TASK == "hybriddial" ]]; then
    DATA_FOLDER="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/HybridDial"
    sample_input_file="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/HybridDial/HybridDial_fqa_test.json"
    fewshot_input_file=""
fi

if [[ $TASK == "inscit" ]]; then
    DATA_FOLDER="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/inscit"
    sample_input_file="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/inscit/inscit_dev_retrieval_dragon_ft_chatgptgen7k_with_topic.json"
    fewshot_input_file=""
fi


## nvolve retrieved
if [[ $TASK == "att_nvolve" ]]; then
    sample_input_file="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/pac_squad_nvolve40k_retrieved_passages/att_test.json"
    fewshot_input_file=""
    DATA_FOLDER="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/pac_squad_nvolve40k_retrieved_passages"
fi
if [[ $TASK == "iternal_nvolve" ]]; then
    sample_input_file="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/pac_squad_nvolve40k_retrieved_passages/iternal_test.json"
    fewshot_input_file=""
    DATA_FOLDER="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/pac_squad_nvolve40k_retrieved_passages"
fi
if [[ $TASK == "nvit_nvolve" ]]; then
    sample_input_file="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/pac_squad_nvolve40k_retrieved_passages/nvit_test.json"
    fewshot_input_file=""
    DATA_FOLDER="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/pac_squad_nvolve40k_retrieved_passages"
fi
if [[ $TASK == "sandia_nvolve" ]]; then
    sample_input_file="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/pac_squad_nvolve40k_retrieved_passages/sandia_test.json"
    fewshot_input_file=""
    DATA_FOLDER="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/single-turn-qa/pac_squad_nvolve40k_retrieved_passages"
fi


## dragon retrieval baselines
if [[ $TASK == "doc2dial_dragon" ]]; then
    DATA_FOLDER="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/doc2dial"
    sample_input_file="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/doc2dial/doc2dial_dragon_QA_test.json"
    fewshot_input_file=""
fi

if [[ $TASK == "quac_dragon" ]]; then
    DATA_FOLDER="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/quac"
    sample_input_file="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/quac/quac_dragon_QA_test.json"
    fewshot_input_file=""
fi

if [[ $TASK == "qrecc_dragon" ]]; then
    DATA_FOLDER="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/qrecc"
    sample_input_file="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/qrecc/qrecc_dragon_QA_test.json"
    fewshot_input_file="${FEWSHOT_INPUT_FOLDER}/multi-turn-qa/qrecc/fewshot_samples.json"
fi

if [[ $TASK == "topiocqa_dragon" ]]; then
    DATA_FOLDER="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/topiocqa"
    sample_input_file="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/topiocqa/topiocqa_dev_retrieval_dragon.json"
    fewshot_input_file=""
fi

if [[ $TASK == "inscit_dragon" ]]; then
    DATA_FOLDER="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/inscit"
    sample_input_file="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/inscit/inscit_dev_retrieval_dragon_with_topic.json"
    fewshot_input_file=""
fi
