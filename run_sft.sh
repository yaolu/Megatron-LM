## instructions
# qa blends: https://gitlab-master.nvidia.com/ADLR/megatron-lm/-/blob/main_sft/examples/foundational_qa/qa_blendv12.sh
# finetuning command: bash examples/foundational_qa/finetune_normal_lm.sh qa_blendv12 43b  64 3e-7 1 gpt_1e-8_conv_quiet_cockatoo_pp1
# all data under: /lustre/fsw/adlr/adlr-nlp/pengx/data/foundational_qa/s3_data/


bash examples/foundational_qa/finetune_normal_lm.sh qa_blendv12 2b  64 3e-7 1 pp1

# where is the quietcockatoo data?

# constant learning rate?

# how the training works?

bash examples/foundational_qa/sft_normal_lm.sh sft 2b   128 5e-6 1 pp1
bash examples/foundational_qa/sft_normal_lm.sh sft 8b   128 5e-6 1 pp1
bash examples/foundational_qa/sft_normal_lm.sh sft 43b  128 5e-6 1 pp1

bash examples/foundational_qa/sft_normal_lm.sh sft 43b  128 5e-6 1 gpt-fitting-pp1

## How about retro?

#1. no neighbors
#2. get the current smallest input len in the batch and set the chunk size = seq-len - smallest input
#3. (ablation) pad tokens to the left

bash examples/foundational_qa/sft_retro_lm.sh sft 2b  128 5e-6 1 pp1

bash examples/foundational_qa/sft_retro_lm.sh sft 8b  128 5e-6 1 pp1

bash examples/foundational_qa/sft_retro_lm.sh sft 43b  128 5e-6 1 pp1

# 100% quiet cockatoo

bash examples/foundational_qa/sft_retro_lm.sh sft 43b  128 5e-6 1 full-qc-pp1

bash examples/foundational_qa/sft_retro_lm.sh sft 43b  128 5e-6 1 full-qc-pp1-seed-2333

bash examples/foundational_qa/sft_retro_lm.sh sft 43b  128 5e-6 1 full-qc-pp1-seed-1234

bash examples/foundational_qa/sft_normal_lm.sh sft 43b 128 5e-6 1 full-qc-pp1

bash examples/foundational_qa/sft_normal_lm.sh sft 43b 128 5e-6 1 gpt-fitting-full-qc-pp1



# run for second time
bash examples/foundational_qa/sft_normal_lm.sh sft 43b  128 5e-6 1 pp1


# Phase II: QA-tuning
bash examples/foundational_qa/finetune_normal_lm.sh qa_blendv12 43b 64 3e-7 1 pp1  /lustre/fsw/adlr/adlr-nlp/boxinw/sft-megatron-lm/checkpoints/applications/sft_pp1_same_format_ctx1_43b_128_5e-6


bash examples/foundational_qa/finetune_retro_lm.sh qa_blendv12 43b 64 3e-7 1 pp1  /lustre/fsw/adlr/adlr-nlp/boxinw/sft-megatron-lm/checkpoints/applications/retro-sft_pp1_same_format_ctx1_43b_128_5e-6

bash examples/foundational_qa/finetune_retro_lm.sh qa_blendv12 2b 64 3e-7 1 pp1  /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-2b-pretraining-retro-fitting

## generation

bash run_gen_blends.sha

## Evaluation

python tasks/foundational_QA/evaluate_f1_fqa.py

## Phase III: multi-turn QA

bash examples/foundational_qa/finetune_normal_lm.sh multiturn_qa_blendv2 43b 64 3e-7 1 gpt_1e-8_conv_quiet_cockatoo_pp1_addmultiturn  /lustre/fsw/adlr/adlr-nlp/boxinw/sft-megatron-lm/checkpoints/applications/sft_pp1_same_format_ctx1_43b_128_5e-6

bash examples/foundational_qa/finetune_normal_lm.sh multiturn_qa_blendv2 43b 64 3e-7 1 nemo_gpt_1e-8_conv_quiet_cockatoo_pp1_addmultiturn  /lustre/fsw/adlr/adlr-nlp/pengx/shared_ckpts/megatron_sft_quiet_cockatoo_tp8_pp1/

bash examples/foundational_qa/finetune_normal_lm.sh multiturn_qa_blendv2 2b 64 3e-7 1 gpt_1e-8_conv_quiet_cockatoo_pp1_addmultiturn  /lustre/fsw/adlr/adlr-nlp/boxinw/sft-megatron-lm/checkpoints/applications/sft_pp1_same_format_ctx1_2b_128_5e-6

bash examples/foundational_qa/finetune_retro_lm.sh multiturn_qa_blendv2 43b 64 3e-7 1 retro_1e-8_conv_quiet_cockatoo_pp1_addmultiturn  /lustre/fsw/adlr/adlr-nlp/boxinw/sft-megatron-lm/checkpoints/applications/retro-sft_pp1_same_format_ctx1_43b_128_5e-6

bash examples/foundational_qa/finetune_retro_lm.sh multiturn_qa_blendv2 2b 64 3e-7 1 retro_1e-8_conv_quiet_cockatoo_pp1_addmultiturn  /lustre/fsw/adlr/adlr-nlp/boxinw/sft-megatron-lm/checkpoints/applications/retro-sft_pp1_same_format_ctx1_2b_128_5e-6

### Phase IV: multi-turn QA v5
bash examples/foundational_qa/finetune_retro_lm.sh multiturn_qa_blend_commercial_v5 2b 64 3e-7 1 retro_1e-8_conv_quiet_cockatoo_pp1_addmultiturn  /lustre/fsw/adlr/adlr-nlp/boxinw/sft-megatron-lm/checkpoints/applications/retro-sft_pp1_same_format_ctx1_2b_128_5e-6
bash examples/foundational_qa/finetune_retro_lm.sh multiturn_qa_blend_commercial_v5 43b 64 3e-7 1 retro_1e-8_conv_quiet_cockatoo_pp1_addmultiturn  /lustre/fsw/adlr/adlr-nlp/boxinw/sft-megatron-lm/checkpoints/applications/retro-sft_pp1_same_format_ctx1_43b_128_5e-6
bash examples/foundational_qa/finetune_retro_lm.sh multiturn_qa_blend_commercial_v5 43b 64 3e-7 1 retro_1e-8_conv_full_quiet_cockatoo_pp1_addmultiturn  /lustre/fsw/adlr/adlr-nlp/boxinw/sft-megatron-lm/checkpoints/applications/retro-sft_full-qc-pp1_same_format_ctx1_43b_128_5e-6
bash examples/foundational_qa/finetune_retro_lm.sh multiturn_qa_blend_commercial_v5 43b 64 3e-7 1 retro_1e-8_conv_full_quiet-seed-2333_cockatoo_pp1_addmultiturn  /lustre/fsw/adlr/adlr-nlp/boxinw/sft-megatron-lm/checkpoints/applications/retro-sft_full-qc-pp1-seed-2333_same_format_ctx1_43b_128_5e-6
#bash examples/foundational_qa/finetune_retro_lm.sh multiturn_qa_blend_commercial_v5 43b 64 3e-7 1 retro_1e-8_conv_full_quiet-seed-1234_cockatoo_pp1_addmultiturn  /lustre/fsw/adlr/adlr-nlp/boxinw/sft-megatron-lm/checkpoints/applications/retro-sft_full-qc-pp1-seed-1234_same_format_ctx1_43b_128_5e-6



bash examples/foundational_qa/finetune_normal_lm.sh multiturn_qa_blend_commercial_v5 43b 64 3e-7 1 gpt_1e-8_conv_quiet_cockatoo_pp1_addmultiturn  /lustre/fsw/adlr/adlr-nlp/boxinw/sft-megatron-lm/checkpoints/applications/sft_pp1_same_format_ctx1_43b_128_5e-6
bash examples/foundational_qa/finetune_normal_lm.sh multiturn_qa_blend_commercial_v5 43b 64 3e-7 1 gpt_1e-8_conv_quiet_cockatoo_pp1_addmultiturn_bak  /lustre/fsw/adlr/adlr-nlp/boxinw/sft-megatron-lm/checkpoints/applications/sft_pp1_same_format_ctx1_43b_128_5e-6_bak
bash examples/foundational_qa/finetune_normal_lm.sh multiturn_qa_blend_commercial_v5 43b 64 3e-7 1 nemo_gpt_1e-8_conv_quiet_cockatoo_pp1_addmultiturn  /lustre/fsw/adlr/adlr-nlp/pengx/shared_ckpts/megatron_sft_quiet_cockatoo_tp8_pp1/

bash examples/foundational_qa/finetune_normal_lm.sh multiturn_qa_blend_commercial_v5 43b 64 3e-7 1 gpt_fitting_1e-8_conv_full_quiet_cockatoo_pp1_addmultiturn  /lustre/fsw/adlr/adlr-nlp/boxinw/sft-megatron-lm/checkpoints/applications/sft_gpt-fitting-full-qc-pp1_same_format_ctx1_43b_128_5e-6
bash examples/foundational_qa/finetune_normal_lm.sh multiturn_qa_blend_commercial_v5 43b 64 3e-7 1 gpt_1e-8_conv_full_quiet_cockatoo_pp1_addmultiturn  /lustre/fsw/adlr/adlr-nlp/boxinw/sft-megatron-lm/checkpoints/applications/sft_full-qc-pp1_same_format_ctx1_43b_128_5e-6


bash examples/foundational_qa/finetune_normal_lm.sh multiturn_qa_blend_commercial_v5 43b 64 3e-7 1 gpt_1e-8_conv_full_quiet_cockatoo_pp1_addmultiturn  /lustre/fsw/adlr/adlr-nlp/boxinw/sft-megatron-lm/checkpoints/applications/sft_full-qc-pp1_same_format_ctx1_43b_128_5e-6
