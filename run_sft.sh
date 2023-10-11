## instructions
# qa blends: https://gitlab-master.nvidia.com/ADLR/megatron-lm/-/blob/main_sft/examples/foundational_qa/qa_blendv12.sh
# finetuning command: bash examples/foundational_qa/finetune_normal_lm.sh qa_blendv12 43b  64 3e-7 1 gpt_1e-8_conv_quiet_cockatoo_pp1
# all data under: /lustre/fsw/adlr/adlr-nlp/pengx/data/foundational_qa/s3_data/

# debug

python -m torch.distributed.launch --nproc_per_node 8 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000 /lustre/fsw/adlr/adlr-nlp/boxinw/sft-megatron-lm/tasks/foundational_QA/finetune_retro_with_pretrain.py --apply-layernorm-1p --untie-embeddings-and-output-weights --disable-bias-linear --no-position-embedding --use-rotary-position-embeddings --rotary-percent 0.5 --attention-dropout 0.0 --hidden-dropout 0.0 --pipeline-model-parallel-size 1 --tensor-model-parallel-size 4 --num-layers 32 --hidden-size 4096 --num-attention-heads 32 --seq-length 4096 --max-position-embeddings 4096 --lr-decay-style cosine --tokenizer-type GPTSentencePieceTokenizer --tokenizer-model /lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-1.1t/utils/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model --clip-grad 1.0 --weight-decay 0.01 --adam-beta1 0.9 --adam-beta2 0.98 --log-params-norm --log-num-zeros-in-grad --bf16 --DDP-impl local --use-distributed-optimizer --squared-relu --retro-workdir /lustre/fsw/adlr/adlr-nlp/boxinw/next-llm --retro-add-retriever --retro-num-neighbors 2 --retro-attention-gate 0 --data-path 1.0 quiet-cockatoo_commercial --data-folder /lustre/fsw/adlr/adlr-nlp/boxinw/instruction_tuning_data/ --recompute-activations --lr 5e-6 --micro-batch-size 1 --global-batch-size 128 --min-lr 5e-6 --train-iters 1000 --dataloader-type cyclic --save /lustre/fsw/adlr/adlr-nlp/boxinw/sft-megatron-lm/checkpoints/applications/retro-sft_pp1-3.5t_same_format_ctx1_8b_128_5e-6 --log-interval 10 --save-interval 500 --eval-interval 200 --tensorboard-dir /lustre/fsw/adlr/adlr-nlp/boxinw/sft-megatron-lm/tensorboard/retro-sft_pp1-3.5t_same_format_ctx1_8b_128_5e-6 --log-validation-ppl-to-tensorboard --eval-iters 100 --eod-mask-loss --answer-loss-only --ft_neighbours 1 --task none --load /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-8b-pretraining-retro-fitting-3.5t --finetune --no-load-rng --no-load-optim

python -m torch.distributed.launch --nproc_per_node 8 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000 /lustre/fsw/adlr/adlr-nlp/boxinw/sft-megatron-lm/tasks/foundational_QA/finetune_gpt_with_pretrain.py --apply-layernorm-1p --untie-embeddings-and-output-weights --disable-bias-linear --no-position-embedding --use-rotary-position-embeddings --rotary-percent 0.5 --attention-dropout 0.0 --hidden-dropout 0.0 --pipeline-model-parallel-size 1 --tensor-model-parallel-size 4 --num-layers 32 --hidden-size 4096 --num-attention-heads 32 --seq-length 4096 --max-position-embeddings 4096 --lr-decay-style cosine --tokenizer-type GPTSentencePieceTokenizer --tokenizer-model /lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-1.1t/utils/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model --clip-grad 1.0 --weight-decay 0.01 --adam-beta1 0.9 --adam-beta2 0.98 --log-params-norm --log-num-zeros-in-grad --bf16 --DDP-impl local --use-distributed-optimizer --squared-relu --data-path 1.0 quiet-cockatoo_commercial --data-folder /lustre/fsw/adlr/adlr-nlp/boxinw/instruction_tuning_data/ --sequence-parallel --recompute-activations --lr 5e-6 --micro-batch-size 1 --global-batch-size 128 --min-lr 5e-6 --train-iters 1000 --dataloader-type cyclic --save /lustre/fsw/adlr/adlr-nlp/boxinw/sft-megatron-lm/checkpoints/applications/sft_gpt-fitting-pp1-3.5t_same_format_ctx1_8b_128_5e-6 --log-interval 10 --save-interval 500 --eval-interval 200 --tensorboard-dir /lustre/fsw/adlr/adlr-nlp/boxinw/sft-megatron-lm/tensorboard/sft_gpt-fitting-pp1-3.5t_same_format_ctx1_8b_128_5e-6 --log-validation-ppl-to-tensorboard --eval-iters 100 --eod-mask-loss --answer-loss-only --ft_neighbours 1 --task none --load /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-8b-pretraining-gpt-fitting-3.5t --finetune --no-load-rng --no-load-optim

python -m torch.distributed.launch --nproc_per_node 8 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000 /lustre/fsw/adlr/adlr-nlp/boxinw/sft-megatron-lm/tasks/foundational_QA/finetune_gpt_with_pretrain.py --apply-layernorm-1p --untie-embeddings-and-output-weights --disable-bias-linear --no-position-embedding --use-rotary-position-embeddings --rotary-percent 0.5 --attention-dropout 0.0 --hidden-dropout 0.0 --pipeline-model-parallel-size 1 --tensor-model-parallel-size 4 --num-layers 32 --hidden-size 4096 --num-attention-heads 32 --seq-length 4096 --max-position-embeddings 4096 --lr-decay-style cosine --tokenizer-type GPTSentencePieceTokenizer --tokenizer-model /lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-1.1t/utils/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model --clip-grad 1.0 --weight-decay 0.01 --adam-beta1 0.9 --adam-beta2 0.98 --log-params-norm --log-num-zeros-in-grad --bf16 --DDP-impl local --use-distributed-optimizer --squared-relu --data-path 1.0 quiet-cockatoo_commercial --data-folder /lustre/fsw/adlr/adlr-nlp/boxinw/instruction_tuning_data/ --sequence-parallel --recompute-activations --lr 5e-6 --micro-batch-size 1 --global-batch-size 128 --min-lr 5e-6 --train-iters 1000 --dataloader-type cyclic --save /lustre/fsw/adlr/adlr-nlp/boxinw/sft-megatron-lm/checkpoints/applications/sft_pp1-3.5t_same_format_ctx1_8b_128_5e-6 --log-interval 10 --save-interval 500 --eval-interval 200 --tensorboard-dir /lustre/fsw/adlr/adlr-nlp/boxinw/sft-megatron-lm/tensorboard/sft_pp1-3.5t_same_format_ctx1_8b_128_5e-6 --log-validation-ppl-to-tensorboard --eval-iters 100 --eod-mask-loss --answer-loss-only --ft_neighbours 1 --task none --load /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-8b-multi-3.5t --finetune --no-load-rng --no-load-optim


bash examples/foundational_qa/finetune_normal_lm.sh qa_blendv12 2b  64 3e-7 1 pp1

# where is the quietcockatoo data?

# constant learning rate?

# how the training works?

bash examples/foundational_qa/sft_normal_lm.sh sft 2b   128 5e-6 1 pp1
bash examples/foundational_qa/sft_normal_lm.sh sft 8b   128 5e-6 1 pp1
bash examples/foundational_qa/sft_normal_lm.sh sft 43b  128 5e-6 1 pp1

bash examples/foundational_qa/sft_normal_lm.sh sft 43b  128 5e-6 1 gpt-fitting-pp1


bash examples/foundational_qa/sft_normal_lm.sh sft 8b   128 5e-6 1 pp1-3.5t  /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-8b-multi-3.5t
bash examples/foundational_qa/sft_normal_lm.sh sft 8b   128 5e-6 1 gpt-fitting-pp1-3.5t  /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-8b-pretraining-gpt-fitting-3.5t


## How about retro?

#1. no neighbors
#2. get the current smallest input len in the batch and set the chunk size = seq-len - smallest input
#3. (ablation) pad tokens to the left

bash examples/foundational_qa/sft_retro_lm.sh sft 2b  128 5e-6 1 pp1

bash examples/foundational_qa/sft_retro_lm.sh sft 8b  128 5e-6 1 pp1

bash examples/foundational_qa/sft_retro_lm.sh sft 43b  128 5e-6 1 pp1

bash examples/foundational_qa/sft_retro_lm.sh sft 8b  128 5e-6 1 pp1-3.5t /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-8b-pretraining-retro-fitting-3.5t


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
