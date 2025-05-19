# Filename: app.py

from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from peft import PeftModel, PeftConfig
import os
import json
import re
from openai import OpenAI
from tqdm import tqdm
import argparse

app = Flask(__name__)

# 全局变量以持久化模型和分词器
tokenizer = None
model = None
device = None
candidate_tokens = None
step_tag_id = None
step_words = "\n\n"
client = None


parser = argparse.ArgumentParser(description="PRM API")
parser.add_argument('--model_path', type=str, default="./checkpoints/prm_model")
parser.add_argument('--port', type=int, default=5050, help='Port to run the API server on')
args = parser.parse_args()
print(f"Loading model from {args.model_path}")

def load_model():
    global tokenizer, model, device, candidate_tokens, step_tag_id, client

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=True,
        padding_side="right",
        add_eos_token=False
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        # 如果模型配置中没有pad_token_id，设置它
        if hasattr(model.config, 'pad_token_id'):
            model.config.pad_token_id = tokenizer.pad_token_id

    # 定义特殊token
    good_token = '+'
    bad_token = '-'
    step_tag = ' и'  # 根据您的需求定义
    step_tag2 = '\n\n'

    # 编码token
    candidate_tokens = tokenizer.encode(f" {good_token} {bad_token}")  # 例如 [488, 481]
    print("Candidate tokens:", candidate_tokens)

    step_tag_id = tokenizer.encode(f" {step_tag}")[-1]  # 例如 76325
    print('step_tag_id:', step_tag_id)
    print('step_tag_id2:', tokenizer.encode(f"{step_tag2}"))

    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",   # 自动分配模型到可用的GPU/CPU
        torch_dtype=torch.bfloat16,  # 混合精度
        trust_remote_code=True,  # 如果使用自定义模型，需要启用
        low_cpu_mem_usage=True,  # 减少CPU内存使用
    ).eval()  # 设置为评估模式

    print(model.device)

def get_score(input_text):
    input_for_prm = input_text
    input_id = torch.tensor([tokenizer.encode(input_for_prm)]).to(device)

    with torch.no_grad():
        logits = model(input_id).logits[:, :, candidate_tokens]
        scores = logits.softmax(dim=-1)[:, :, 0] 
        step_scores = scores[input_id == step_tag_id]
        if step_scores.numel() == 0:
            score = 0.0
        else:
            score = step_scores.tolist()
    return score

load_model()

@app.route('/compute_score', methods=['POST'])
def compute_score():
    try:
        data = request.get_json()
        if not data or 'input' not in data:
            return jsonify({"error": "Invalid input. 'input' field is required."}), 400
        
        input_text = data['input']
        
        # 计算得分
        score = get_score(input_text)

        return jsonify({"score": score})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # 运行Flask应用
    app.run(host='0.0.0.0', port=args.port)
