 #vllm serve /home/fq9hpsacuser01/models/Qwen-Image-2512 --omni -tp "4" --port "40021" --enable_cpu_offload
 export RDMA_DEVICE_NAME='mlx5_0'
 vllm serve /home/fq9hpsacuser01/models/BAGEL-7B-MoT/ --omni --port "40021"   --stage-configs /home/fq9hpsacuser01/vllm-omni/vllm_omni/model_executor/stage_configs/bagel_rdma.yaml
