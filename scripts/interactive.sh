#!/bin/bash

set -u

######## Arguments. ########

# . args.sh
. /home/lmcafee/src/megatrons/megatron-lm-main/scripts/args.sh

######## Command. ########

NPROCS=1 # 8
CMD="\
    cd ${REPO_DIR} && \
    export PYTHONPATH=$PYTHONPATH:${REPO_DIR}:/home/lmcafee/src && \
    python -m torch.distributed.run \
    --nproc_per_node ${NPROCS} \
    --nnodes 1 \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port 6000 \
    ${SCRIPT} ${ARGS} \
"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "CMD = '$CMD'."
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"
eval $CMD

# eof.