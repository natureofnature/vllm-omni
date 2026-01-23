export RDMA_DEVICE_NAME='mlx5_0'
export MC_IB_PCI_RELAXED_ORDERING=1
ray start --address=10.248.12.106:6399 --num-gpus=1
