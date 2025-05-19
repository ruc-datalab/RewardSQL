CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
--model ./checkpoints/grpo_model \
--served-model-name qwen2.5-sql \
--host localhost --port 8190 \
--gpu-memory-utilization 0.6