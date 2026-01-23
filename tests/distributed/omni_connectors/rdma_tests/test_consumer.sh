#!/bin/bash
# Consumer script for cross-node RDMA test
# Usage: bash test_consumer.sh [copy|zerocopy|gpu] [--benchmark]

export RDMA_DEVICE_NAME='mlx5_0'
export HOST_PRODUCER='hk01dgx032'
export HOST_CONSUMER='hk01dgx006'
export DATA_SIZE_MB=1024

# Default mode is 'copy'
MODE=${1:-copy}
BENCHMARK=""

# Check for --benchmark flag
for arg in "$@"; do
    if [ "$arg" = "--benchmark" ]; then
        BENCHMARK="--benchmark"
    fi
done

echo "============================================================"
echo " Running Consumer in ${MODE} mode ${BENCHMARK}"
echo "============================================================"

python test_cross_node.py \
    --role consumer \
    --local-host $HOST_CONSUMER \
    --remote-host $HOST_PRODUCER \
    --tensor-size-mb $DATA_SIZE_MB \
    --num-transfers 20 \
    --mode $MODE \
    --gpu-id 0 \
    $BENCHMARK
