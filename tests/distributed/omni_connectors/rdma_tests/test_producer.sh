#!/bin/bash
# Producer script for cross-node RDMA test
# Usage: bash test_producer.sh [copy|zerocopy|gpu] [--benchmark]

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
echo " Running Producer in ${MODE} mode ${BENCHMARK}"
echo "============================================================"

python test_cross_node.py \
    --role producer \
    --local-host $HOST_PRODUCER \
    --remote-host $HOST_CONSUMER \
    --tensor-size-mb $DATA_SIZE_MB \
    --num-transfers 20 \
    --mode $MODE \
    --gpu-id 0 \
    $BENCHMARK
