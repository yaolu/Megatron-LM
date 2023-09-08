# interactive shell
#
#srun -p luna,interactive -A adlr -t 0:30:00 --job-name=adlr-nlp-largelm:cpu --container-mounts=/lustre/fsw/adlr:/lustre/fsw/adlr --container-image="/lustre/fsw/adlr/adlr-nlp/boxinw/images/retro.sqsh"   --export=ALL,PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python  --pty bash
#srun -p luna,interactive -A adlr -t 0:30:00 --job-name=adlr-nlp-largelm:cpu --container-mounts=/lustre/fsw/adlr:/lustre/fsw/adlr --container-image="/lustre/fsw/adlr/adlr-nlp/boxinw/images/retrov2.sqsh" --export=ALL,PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python  --pty bash
#
#
#sbatch scripts/pretrain-nextlm-800m-gpt.sh
#sbatch scripts/pretrain-nextlm-800m-gpt-lr-2e-6.sh
#sbatch scripts/pretrain-nextlm-800m-retro.sh
#
#sbatch scripts/pretrain-nextlm-8b-retro.sh
#
## eval ppl
#
#/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm

#sbatch scripts/evalppl-nextlm-800m-gpt.sh gpt3-800m-pretraining-gpt-fitting
#sbatch scripts/evalppl-nextlm-800m-gpt.sh gpt3-843m-multi-1.1t-gtc-llr
#sbatch scripts/evalppl-nextlm-800m-retro.sh gpt3-800m-pretraining-retro-fitting

#sbatch scripts/evalppl-nextlm-2b-gpt.sh gpt3-2b-multi-1.1t-gtc
#sbatch scripts/evalppl-nextlm-2b-gpt.sh gpt3-2b-pretraining-gpt-fitting
#sbatch scripts/evalppl-nextlm-2b-retro.sh gpt3-2b-pretraining-retro-fitting
#sbatch scripts/evalppl-nextlm-2b-retro-gate-0.sh gpt3-2b-pretraining-retro-fitting
#sbatch scripts/evalppl-nextlm-2b-retro-gate-1.sh gpt3-2b-pretraining-retro-fitting
#
#sbatch scripts/evalppl-nextlm-8b-gpt.sh gpt3-8b-multi-1.1t-gtc
#sbatch scripts/evalppl-nextlm-8b-gpt.sh gpt3-8b-pretraining-gpt-fitting
#sbatch scripts/evalppl-nextlm-8b-retro.sh gpt3-8b-pretraining-retro-fitting-noseqpar
#sbatch scripts/evalppl-nextlm-8b-retrotro-gate-0.sh gpt3-8b-pretraining-retro-fitting-noseqpar
#
#sbatch scripts/evalppl-nextlm-22b-gpt.sh gpt3-22b-multi-1.1t-gtc
#sbatch scripts/evalppl-nextlm-22b-gpt.sh gpt3-22b-pretraining-gpt-fitting
#sbatch scripts/evalppl-nextlm-22b-retro.sh gpt3-22b-pretraining-retro-fitting-noseqpar
#sbatch scripts/evalppl-nextlm-22b-retro-gate-0.sh gpt3-22b-pretraining-retro-fitting-noseqpar
#
#sbatch scripts/evalppl-nextlm-43b-gpt.sh gpt3-43b-multi-1.1t-gtc-tp8pp4vp1
#sbatch scripts/evalppl-nextlm-43b-gpt.sh gpt3-43b-pretraining-gpt-fitting
#sbatch scripts/evalppl-nextlm-43b-retro.sh gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed
#sbatch scripts/evalppl-nextlm-43b-retro-gate-0.sh gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed

#/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm
#
#sbatch scripts/evalppl-nextlm-800m-retro.sh gpt3-800m-pretraining-retro-fitting
#
#
#sbatch scripts/evalppl-nextlm-8b-retro.sh gpt3-8b-pretraining-retro-fitting
#
#python tools/checkpoint_util.py \
#        --model-type GPT \
#        --load-dir /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-gpt-fitting \
#        --save-dir /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-43b-pretraining-gpt-fitting-tp8pp1 \
#        --target-tensor-parallel-size 8 \
#        --target-pipeline-parallel-size 1

## preprocess dataset

python tools/preprocess_data.py \
       --input  /lustre/fsw/adlr/adlr-nlp/boxinw/coco/coco_text_train.json \
       --output-prefix text_doc \
       --dataset-impl mmap \
       --tokenizer-model  /lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-1.1t/utils/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
       --tokenizer-type GPTSentencePieceTokenizer \
       --append-eod  --workers 20 --chunk-size 25



python -m torch.distributed.launch --nproc_per_node 8 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000 pretrain_gpt.py --sequence-parallel --recompute-activations --use-flash-attn --apply-layernorm-1p --untie-embeddings-and-output-weights --disable-bias-linear --no-position-embedding --use-rotary-position-embeddings --rotary-percent 0.5 --swiglu --attention-dropout 0.0 --hidden-dropout 0.0 --exit-duration-in-mins 220 --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --save-interval 2000 --save /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/eval-retro-nvllm/gpt3-2b-multi-1.1t-gtc --load /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/eval-retro-nvllm/gpt3-2b-multi-1.1t-gtc \
                  --no-load-optim --finetune --tensorboard-dir /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/eval-retro-nvllm/gpt3-2b-multi-1.1t-gtc/tensorboard --log-validation-ppl-to-tensorboard --num-layers 24 --hidden-size 2048 --num-attention-heads 16 --seq-length 4096 --max-position-embeddings 4096 --micro-batch-size 1 --global-batch-size 768 --train-samples 25000000 --lr-decay-samples 23750000 --lr-warmup-samples 16667 --lr 2e-5 --min-lr 2e-6 --lr-decay-style cosine --log-interval 100 --eval-iters 32 --eval-interval 1260 --tokenizer-type GPTSentencePieceTokenizer --tokenizer-model /lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-1.1t/utils/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
                  --data-path 0.01920 /lustre/fsw/adlr/adlr-nlp/boxinw/retro/data/english/MTNLG/Books3_shuf_text_document 0.01602 /lustre/fsw/adlr/adlr-nlp/boxinw/retro/data/english/MTNLG/OpenWebText2_shuf_text_document 0.00751 /lustre/fsw/adlr/adlr-nlp/boxinw/retro/data/english/MTNLG/StackExchange_shuf_text_document 0.00324 /lustre/fsw/adlr/adlr-nlp/boxinw/retro/data/english/MTNLG/PubMedAbs_shuf_text_document 0.00653 /lustre/fsw/adlr/adlr-nlp/boxinw/retro/data/english/MTNLG/Wikipedia_shuf_text_document 0.00193 /lustre/fsw/adlr/adlr-nlp/boxinw/retro/data/english/MTNLG/Gutenberg_shuf_text_document 0.00117 /lustre/fsw/adlr/adlr-nlp/boxinw/retro/data/english/MTNLG/BookCorpus2_shuf_text_document 0.00023 /lustre/fsw/adlr/adlr-nlp/boxinw/retro/data/english/MTNLG/NIHExporter_shuf_text_document 0.01143 /lustre/fsw/adlr/adlr-nlp/boxinw/retro/data/english/MTNLG/ArXiv_shuf_text_document 0.00366 /lustre/fsw/adlr/adlr-nlp/boxinw/retro/data/english/MTNLG/Stories_shuf_text_document 0.03992 /lustre/fsw/adlr/adlr-nlp/boxinw/retro/data/english/BigScience/BigScience_shuf_text_document 0.04768 /lustre/fsw/adlr/adlr-nlp/boxinw/retro/data/english/Reddit-Plus/Reddit_all_dialogue_shuf_text_document 0.07199 /lustre/fsw/adlr/adlr-nlp/boxinw/retro/data/english/CC-NEWS/CC-NEWS_shuf_text_document 0.02180 /lustre/fsw/adlr/adlr-nlp/boxinw/retro/data/english/MTNLG/Pile-CC_shuf_text_document 0.07633 /lustre/fsw/adlr/adlr-nlp/boxinw/retro/data/english/CC-MAIN-2020-50/CC-MAIN-2020-50_shuf_text_document 0.07644 /lustre/fsw/adlr/adlr-nlp/boxinw/retro/data/english/CC-MAIN-2022-40/CC-MAIN-2022-40_00_shuf_text_document 0.07644 /lustre/fsw/adlr/adlr-nlp/boxinw/retro/data/english/CC-MAIN-2022-40/CC-MAIN-2022-40_01_shuf_text_document 0.09414 /lustre/fsw/adlr/adlr-nlp/boxinw/retro/data/english/CC-MAIN-2019-35/CC-MAIN-2019-35_shuf_text_document 0.03890 /lustre/fsw/adlr/adlr-nlp/boxinw/retro/data/english/MTNLG/CC-2021-04_shuf_text_document 0.08544 /lustre/fsw/adlr/adlr-nlp/boxinw/retro/data/english/mc4-en_1T-url/mc4-en_shuf_text_document \
                  --split 98,2,0 --split-constraint 99,1,0 --split-constraint 98,2,0 --clip-grad 1.0 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.95 --init-method-std 0.007 --log-params-norm --log-num-zeros-in-grad --bf16 --DDP-impl local --eval-ppl

# 2b
python -m torch.distributed.launch --nproc_per_node 8 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000 pretrain_gpt.py --sequence-parallel --recompute-activations --use-flash-attn --apply-layernorm-1p --untie-embeddings-and-output-weights --disable-bias-linear --no-position-embedding --use-rotary-position-embeddings --rotary-percent 0.5 --swiglu --attention-dropout 0.0 --hidden-dropout 0.0 --exit-duration-in-mins 220 --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --save-interval 2000 --save /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/eval-retro-nvllm/gpt3-2b-multi-1.1t-gtc --load /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/eval-retro-nvllm/gpt3-2b-multi-1.1t-gtc \
                  --no-load-optim --finetune --tensorboard-dir /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/eval-retro-nvllm/gpt3-2b-multi-1.1t-gtc/tensorboard --log-validation-ppl-to-tensorboard --num-layers 24 --hidden-size 2048 --num-attention-heads 16 --seq-length 256 --max-position-embeddings 4096 --micro-batch-size 1 --global-batch-size 768 --train-samples 25000000 --lr-decay-samples 23750000 --lr-warmup-samples 16667 --lr 2e-5 --min-lr 2e-6 --lr-decay-style cosine --log-interval 100 --eval-iters 32 --eval-interval 1260 --tokenizer-type GPTSentencePieceTokenizer --tokenizer-model /lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-1.1t/utils/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
                  --data-path 1 /lustre/fsw/adlr/adlr-nlp/boxinw/coco/coco_text_train_text_document \
                  --split 0,100,0  --split-constraint 0,100,0 --clip-grad 1.0 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.95 --init-method-std 0.007 --log-params-norm --log-num-zeros-in-grad --bf16 --DDP-impl local --eval-ppl
#256 * 768 * 32 = 6291456

#
#------------------------------------------------------------------------------------------------------------------------
# validation loss at the beginning of training for val data | lm loss value: 4.221286E+00 | lm loss PPL: 6.812102E+01 |
#------------------------------------------------------------------------------------------------------------------------



# seq-len=32
python -m torch.distributed.launch --nproc_per_node 8 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000 pretrain_gpt.py --sequence-parallel --recompute-activations --use-flash-attn --apply-layernorm-1p --untie-embeddings-and-output-weights --disable-bias-linear --no-position-embedding --use-rotary-position-embeddings --rotary-percent 0.5 --swiglu --attention-dropout 0.0 --hidden-dropout 0.0 --exit-duration-in-mins 220 --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --save-interval 2000 --save /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/eval-retro-nvllm/gpt3-2b-multi-1.1t-gtc --load /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/eval-retro-nvllm/gpt3-2b-multi-1.1t-gtc \
                  --no-load-optim --finetune --tensorboard-dir /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/eval-retro-nvllm/gpt3-2b-multi-1.1t-gtc/tensorboard --log-validation-ppl-to-tensorboard --num-layers 24 --hidden-size 2048 --num-attention-heads 16 --seq-length 32 --max-position-embeddings 4096 --micro-batch-size 16 --global-batch-size 768 --train-samples 25000000 --lr-decay-samples 23750000 --lr-warmup-samples 16667 --lr 2e-5 --min-lr 2e-6 --lr-decay-style cosine --log-interval 100 --eval-iters 256 --eval-interval 1260 --tokenizer-type GPTSentencePieceTokenizer --tokenizer-model /lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-1.1t/utils/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
                  --data-path 1 /lustre/fsw/adlr/adlr-nlp/boxinw/coco/coco_text_train_text_document \
                  --split 0,100,0  --split-constraint 0,100,0 --clip-grad 1.0 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.95 --init-method-std 0.007 --log-params-norm --log-num-zeros-in-grad --bf16 --DDP-impl local --eval-ppl

#32 * 768 * 256 = 6291456

#------------------------------------------------------------------------------------------------------------------------
# validation loss at the beginning of training for val data | lm loss value: 4.473447E+00 | lm loss PPL: 8.765839E+01 |
#------------------------------------------------------------------------------------------------------------------------

# seq-len=4096
python -m torch.distributed.launch --nproc_per_node 8 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000 pretrain_gpt.py --sequence-parallel --recompute-activations --use-flash-attn --apply-layernorm-1p --untie-embeddings-and-output-weights --disable-bias-linear --no-position-embedding --use-rotary-position-embeddings --rotary-percent 0.5 --swiglu --attention-dropout 0.0 --hidden-dropout 0.0 --exit-duration-in-mins 220 --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --save-interval 2000 --save /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/eval-retro-nvllm/gpt3-2b-multi-1.1t-gtc --load /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/eval-retro-nvllm/gpt3-2b-multi-1.1t-gtc \
                  --no-load-optim --finetune --tensorboard-dir /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/eval-retro-nvllm/gpt3-2b-multi-1.1t-gtc/tensorboard --log-validation-ppl-to-tensorboard --num-layers 24 --hidden-size 2048 --num-attention-heads 16 --seq-length 4096 --max-position-embeddings 4096 --micro-batch-size 1 --global-batch-size 768 --train-samples 25000000 --lr-decay-samples 23750000 --lr-warmup-samples 16667 --lr 2e-5 --min-lr 2e-6 --lr-decay-style cosine --log-interval 100 --eval-iters 2 --eval-interval 1260 --tokenizer-type GPTSentencePieceTokenizer --tokenizer-model /lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-1.1t/utils/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
                  --data-path 1 /lustre/fsw/adlr/adlr-nlp/boxinw/coco/coco_text_train_text_document \
                  --split 0,100,0  --split-constraint 0,100,0 --clip-grad 1.0 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.95 --init-method-std 0.007 --log-params-norm --log-num-zeros-in-grad --bf16 --DDP-impl local --eval-ppl

#4096 * 768 * 2 = 6291456

#------------------------------------------------------------------------------------------------------------------------
# validation loss at the beginning of training for val data | lm loss value: 4.040486E+00 | lm loss PPL: 5.685396E+01 |
#------------------------------------------------------------------------------------------------------------------------

# 8b
python -m torch.distributed.launch --nproc_per_node 8 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000 pretrain_gpt.py --sequence-parallel --recompute-activations --use-flash-attn --apply-layernorm-1p --untie-embeddings-and-output-weights --disable-bias-linear --no-position-embedding --use-rotary-position-embeddings --rotary-percent 0.5 --swiglu --attention-dropout 0.0 --hidden-dropout 0.0 --exit-duration-in-mins 220 --tensor-model-parallel-size 4 --pipeline-model-parallel-size 1 --save-interval 2000 --save /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/eval-retro-nvllm/gpt3-8b-multi-1.1t-gtc --load /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/eval-retro-nvllm/gpt3-8b-multi-1.1t-gtc \
                  --no-load-optim --finetune --tensorboard-dir /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/eval-retro-nvllm/gpt3-8b-multi-1.1t-gtc/tensorboard --log-validation-ppl-to-tensorboard --num-layers 32 --hidden-size 4096 --num-attention-heads 32 --seq-length 256 --max-position-embeddings 4096 --micro-batch-size 1 --global-batch-size 768 --train-samples 25000000 --lr-decay-samples 23750000 --lr-warmup-samples 16667 --lr 2e-5 --min-lr 2e-6 --lr-decay-style cosine --log-interval 100 --eval-iters 32 --eval-interval 1260 --tokenizer-type GPTSentencePieceTokenizer --tokenizer-model /lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-1.1t/utils/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
                  --data-path 1 /lustre/fsw/adlr/adlr-nlp/boxinw/coco/coco_text_train_text_document \
                  --split 0,100,0  --split-constraint 0,100,0 --clip-grad 1.0 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.95 --init-method-std 0.007 --log-params-norm --log-num-zeros-in-grad --bf16 --DDP-impl local --eval-ppl
#------------------------------------------------------------------------------------------------------------------------
# validation loss at the beginning of training for val data | lm loss value: 4.300717E+00 | lm loss PPL: 7.375268E+01 |
#------------------------------------------------------------------------------------------------------------------------


# seq-len=32
python -m torch.distributed.launch --nproc_per_node 8 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000 pretrain_gpt.py --sequence-parallel --recompute-activations --use-flash-attn --apply-layernorm-1p --untie-embeddings-and-output-weights --disable-bias-linear --no-position-embedding --use-rotary-position-embeddings --rotary-percent 0.5 --swiglu --attention-dropout 0.0 --hidden-dropout 0.0 --exit-duration-in-mins 220 --tensor-model-parallel-size 4 --pipeline-model-parallel-size 1 --save-interval 2000 --save /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/eval-retro-nvllm/gpt3-8b-multi-1.1t-gtc --load /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/eval-retro-nvllm/gpt3-8b-multi-1.1t-gtc \
                  --no-load-optim --finetune --tensorboard-dir /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/eval-retro-nvllm/gpt3-8b-multi-1.1t-gtc/tensorboard --log-validation-ppl-to-tensorboard --num-layers 32 --hidden-size 4096 --num-attention-heads 32 --seq-length 32 --max-position-embeddings 4096 --micro-batch-size 32 --global-batch-size 768 --train-samples 25000000 --lr-decay-samples 23750000 --lr-warmup-samples 16667 --lr 2e-5 --min-lr 2e-6 --lr-decay-style cosine --log-interval 100 --eval-iters 256 --eval-interval 1260 --tokenizer-type GPTSentencePieceTokenizer --tokenizer-model /lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-1.1t/utils/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
                  --data-path 1 /lustre/fsw/adlr/adlr-nlp/boxinw/coco/coco_text_train_text_document \
                  --split 0,100,0  --split-constraint 0,100,0 --clip-grad 1.0 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.95 --init-method-std 0.007 --log-params-norm --log-num-zeros-in-grad --bf16 --DDP-impl local --eval-ppl


#------------------------------------------------------------------------------------------------------------------------
# validation loss at the beginning of training for val data | lm loss value: 4.376585E+00 | lm loss PPL: 7.956589E+01 |
#------------------------------------------------------------------------------------------------------------------------


# seq-len=4096
python -m torch.distributed.launch --nproc_per_node 8 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000 pretrain_gpt.py --sequence-parallel --recompute-activations --use-flash-attn --apply-layernorm-1p --untie-embeddings-and-output-weights --disable-bias-linear --no-position-embedding --use-rotary-position-embeddings --rotary-percent 0.5 --swiglu --attention-dropout 0.0 --hidden-dropout 0.0 --exit-duration-in-mins 220 --tensor-model-parallel-size 4 --pipeline-model-parallel-size 1 --save-interval 2000 --save /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/eval-retro-nvllm/gpt3-8b-multi-1.1t-gtc --load /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/eval-retro-nvllm/gpt3-8b-multi-1.1t-gtc \
                  --no-load-optim --finetune --tensorboard-dir /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/eval-retro-nvllm/gpt3-8b-multi-1.1t-gtc/tensorboard --log-validation-ppl-to-tensorboard --num-layers 32 --hidden-size 4096 --num-attention-heads 32 --seq-length 4096 --max-position-embeddings 4096 --micro-batch-size 1 --global-batch-size 768 --train-samples 25000000 --lr-decay-samples 23750000 --lr-warmup-samples 16667 --lr 2e-5 --min-lr 2e-6 --lr-decay-style cosine --log-interval 100 --eval-iters 2 --eval-interval 1260 --tokenizer-type GPTSentencePieceTokenizer --tokenizer-model /lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-1.1t/utils/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
                  --data-path 1 /lustre/fsw/adlr/adlr-nlp/boxinw/coco/coco_text_train_text_document \
                  --split 0,100,0  --split-constraint 0,100,0 --clip-grad 1.0 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.95 --init-method-std 0.007 --log-params-norm --log-num-zeros-in-grad --bf16 --DDP-impl local --eval-ppl

#------------------------------------------------------------------------------------------------------------------------
# validation loss at the beginning of training for val data | lm loss value: 4.236993E+00 | lm loss PPL: 6.919944E+01 |
#------------------------------------------------------------------------------------------------------------------------

# 22b
python -m torch.distributed.launch --nproc_per_node 8 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000 pretrain_gpt.py --sequence-parallel --recompute-activations --use-flash-attn --apply-layernorm-1p --untie-embeddings-and-output-weights --disable-bias-linear --no-position-embedding --use-rotary-position-embeddings --rotary-percent 0.5 --swiglu --attention-dropout 0.0 --hidden-dropout 0.0 --exit-duration-in-mins 220 --tensor-model-parallel-size 8 --pipeline-model-parallel-size 1 --save-interval 2000 --save /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/eval-retro-nvllm/gpt3-8b-multi-1.1t-gtc --load /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/eval-retro-nvllm/gpt3-22b-multi-1.1t-gtc \
                  --no-load-optim --finetune --tensorboard-dir /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/eval-retro-nvllm/gpt3-22b-multi-1.1t-gtc/tensorboard --log-validation-ppl-to-tensorboard --num-layers 40 --hidden-size 6144 --num-attention-heads 48 --seq-length 256 --max-position-embeddings 4096 --micro-batch-size 1 --global-batch-size 768 --train-samples 25000000 --lr-decay-samples 23750000 --lr-warmup-samples 16667 --lr 2e-5 --min-lr 2e-6 --lr-decay-style cosine --log-interval 100 --eval-iters 32 --eval-interval 1260 --tokenizer-type GPTSentencePieceTokenizer --tokenizer-model /lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-1.1t/utils/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
                  --data-path 1 /lustre/fsw/adlr/adlr-nlp/boxinw/coco/coco_text_train_text_document \
                  --split 0,100,0  --split-constraint 0,100,0 --clip-grad 1.0 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.95 --init-method-std 0.007 --log-params-norm --log-num-zeros-in-grad --bf16 --DDP-impl local --eval-ppl

#------------------------------------------------------------------------------------------------------------------------
# validation loss at the beginning of training for val data | lm loss value: 4.039996E+00 | lm loss PPL: 5.682612E+01 |
#------------------------------------------------------------------------------------------------------------------------

# seq-len=32
python -m torch.distributed.launch --nproc_per_node 8 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000 pretrain_gpt.py --sequence-parallel --recompute-activations --use-flash-attn --apply-layernorm-1p --untie-embeddings-and-output-weights --disable-bias-linear --no-position-embedding --use-rotary-position-embeddings --rotary-percent 0.5 --swiglu --attention-dropout 0.0 --hidden-dropout 0.0 --exit-duration-in-mins 220 --tensor-model-parallel-size 8 --pipeline-model-parallel-size 1 --save-interval 2000 --save /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/eval-retro-nvllm/gpt3-8b-multi-1.1t-gtc --load /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/eval-retro-nvllm/gpt3-22b-multi-1.1t-gtc \
                  --no-load-optim --finetune --tensorboard-dir /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/eval-retro-nvllm/gpt3-22b-multi-1.1t-gtc/tensorboard --log-validation-ppl-to-tensorboard --num-layers 40 --hidden-size 6144 --num-attention-heads 48 --seq-length 32 --max-position-embeddings 4096 --micro-batch-size 32 --global-batch-size 768 --train-samples 25000000 --lr-decay-samples 23750000 --lr-warmup-samples 16667 --lr 2e-5 --min-lr 2e-6 --lr-decay-style cosine --log-interval 100 --eval-iters 256 --eval-interval 1260 --tokenizer-type GPTSentencePieceTokenizer --tokenizer-model /lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-1.1t/utils/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
                  --data-path 1 /lustre/fsw/adlr/adlr-nlp/boxinw/coco/coco_text_train_text_document \
                  --split 0,100,0  --split-constraint 0,100,0 --clip-grad 1.0 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.95 --init-method-std 0.007 --log-params-norm --log-num-zeros-in-grad --bf16 --DDP-impl local --eval-ppl

#------------------------------------------------------------------------------------------------------------------------
# validation loss at the beginning of training for val data | lm loss value: 4.320646E+00 | lm loss PPL: 7.523720E+01 |
#------------------------------------------------------------------------------------------------------------------------

# seq-len=4096
python -m torch.distributed.launch --nproc_per_node 8 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000 pretrain_gpt.py --sequence-parallel --recompute-activations --use-flash-attn --apply-layernorm-1p --untie-embeddings-and-output-weights --disable-bias-linear --no-position-embedding --use-rotary-position-embeddings --rotary-percent 0.5 --swiglu --attention-dropout 0.0 --hidden-dropout 0.0 --exit-duration-in-mins 220 --tensor-model-parallel-size 8 --pipeline-model-parallel-size 1 --save-interval 2000 --save /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/eval-retro-nvllm/gpt3-8b-multi-1.1t-gtc --load /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/eval-retro-nvllm/gpt3-22b-multi-1.1t-gtc \
                  --no-load-optim --finetune --tensorboard-dir /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/eval-retro-nvllm/gpt3-22b-multi-1.1t-gtc/tensorboard --log-validation-ppl-to-tensorboard --num-layers 40 --hidden-size 6144 --num-attention-heads 48 --seq-length 4096 --max-position-embeddings 4096 --micro-batch-size 1 --global-batch-size 768 --train-samples 25000000 --lr-decay-samples 23750000 --lr-warmup-samples 16667 --lr 2e-5 --min-lr 2e-6 --lr-decay-style cosine --log-interval 100 --eval-iters 2 --eval-interval 1260 --tokenizer-type GPTSentencePieceTokenizer --tokenizer-model /lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-1.1t/utils/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
                  --data-path 1 /lustre/fsw/adlr/adlr-nlp/boxinw/coco/coco_text_train_text_document \
                  --split 0,100,0  --split-constraint 0,100,0 --clip-grad 1.0 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.95 --init-method-std 0.007 --log-params-norm --log-num-zeros-in-grad --bf16 --DDP-impl local --eval-ppl

#------------------------------------------------------------------------------------------------------------------------
# validation loss at the beginning of training for val data | lm loss value: 3.957693E+00 | lm loss PPL: 5.233643E+01 |
#------------------------------------------------------------------------------------------------------------------------


