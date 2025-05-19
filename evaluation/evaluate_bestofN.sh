
# 7B full
python evaluate_bird_prm.py \
    --model_port 8190 \
    --model_tag qwen2.5-sql \
    --request_api http://localhost:5050/compute_score \
    --num_rollouts 32 \
    --temperature 1.0 \
    --save_path /your/path/save/results \
    --use_ex_filter \
    --add_ex_prm \
    --data_path ./data/bird/dev.json 
    # --data_path ./data/spider/test.json
