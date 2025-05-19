
from datasets import load_dataset
import argparse
import os
import json
import argparse
from collections import Counter
from multiprocessing import Pool
import numpy as np
import sys
from openai import OpenAI
from tqdm import tqdm

sys.path.append("./evaluation/")
from utils.execute_sql import execute_single_sql, compare_sql, process_single_item
from utils.request_api import get_score, uniform_cte_sql, calculate_scores, get_chat_completion

good_token = '+'
bad_token = '-'
step_tag = ' и '
step_words = "\n\n"

SCORING_STRATEGIES = ["product", "min", "last_step", "average"]

def get_args():
    parser = argparse.ArgumentParser(description="配置训练参数")
    parser.add_argument("--request_api", type=str, default="http://localhost:5058/compute_score")
    parser.add_argument("--data_path", type=str, default="./data/bird/dev.json")
    parser.add_argument("--save_path", type=str, default="./results_rlhf/bird_test_prm_reasoning_rlhf_520k_full_allsteps_codes_o1mini_results_bestof16_tp07/")
    parser.add_argument("--model_tag", type=str, default="qwen2.5-sql")
    parser.add_argument("--model_port", type=str, default="8190")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--num_rollouts", type=int, default=16)
    parser.add_argument("--use_ex_filter", action="store_true", default=False)
    parser.add_argument("--add_ex_prm", action="store_true", default=False)
    args = parser.parse_args()
    return args

def get_top_score_sql(question, sqls, item, args):
    candidate_details = []
    pass_in_n = False
    first_print = True
    for index, sql in enumerate(sqls):
        if ";" not in sql:
            sql = sql + ";" 
        can_execute = True
        # item["id"] = index
        if args.use_ex_filter:
            can_execute, _ = execute_single_sql(sql, item["db_id"], item["id"], 60)
        if not can_execute:
            continue
        concanate_sql = sql.replace(step_words, step_tag) + step_tag
        concanate_sql = uniform_cte_sql(concanate_sql, "weak", False)
        if args.add_ex_prm and len(sqls) > 1:
            sql_with_result = process_single_item({"sql": concanate_sql, "question_id": item["id"], "db_id": item["db_id"]})
        else:
            sql_with_result = concanate_sql
        if first_print:
            print("The first candidate SQL is: ", repr(f"{sql_with_result}"))
            first_print = False
        if len(sqls) > 1:
            step_scores = get_score(args.request_api, question, sql_with_result)
            strategy_scores = {
                strategy: calculate_scores(step_scores, strategy)
                for strategy in SCORING_STRATEGIES
            }
        else:
            step_scores = None
            strategy_scores = None
        is_correct, _ = compare_sql(
            predicted_sql=sql,
            ground_truth=item["output"],
            db_id=item["db_id"],
            query_id=item["id"]
        )
        if can_execute:
            candidate_detail = {
                "sql": sql,
                "step_scores": step_scores,
                "strategy_scores": strategy_scores,
                "is_correct": bool(is_correct)
            }
            if args.add_ex_prm:
                candidate_detail["sql_with_result"] = sql_with_result
            candidate_details.append(candidate_detail)
        if is_correct:
            pass_in_n = True
    # 选择各策略最佳答案
    best_answers = {}
    for strategy in SCORING_STRATEGIES:
        sorted_candidates = sorted(candidate_details, 
                                 key=lambda x: x["strategy_scores"][strategy], 
                                 reverse=True)
        best_answers[strategy] = {
            "sql": sorted_candidates[0]["sql"] if len(sorted_candidates) > 0 else "",
            "score": sorted_candidates[0]["strategy_scores"][strategy] if len(sorted_candidates) > 0 else 0.0,
            "is_correct": sorted_candidates[0]["is_correct"] if len(sorted_candidates) > 0 else False
        }
    return {
        "id": item["id"],
        "ground truth": item["output"],
        "db_id": item["db_id"],
        "candidates": candidate_details,
        "best_answers": best_answers,
        "pass@n": pass_in_n
    }

def save_statistics(results, save_path, dbs, save_predict=False):
    stats = {
        "pass@n": sum(item["pass@n"] for item in results) / len(results),
        "accuracy": {
            strategy: sum(item["best_answers"][strategy]["is_correct"] for item in results) / len(results)
            for strategy in SCORING_STRATEGIES
        }
    }
    # 保存统计结果
    with open(os.path.join(save_path, "statistics.txt"), "w") as f:
        f.write("Evaluation Results:\n")
        f.write(f"Pass@n: {stats['pass@n']:.4f}\n")
        for strategy in SCORING_STRATEGIES:
            f.write(f"{strategy} Accuracy: {stats['accuracy'][strategy]:.4f}\n")
    # 保存详细预测结果
    with open(os.path.join(save_path, "detailed_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    if save_predict:
        with open(os.path.join(save_path, "predict_dev.json"), "w") as f:
            json.dump(
                {str(item["id"]): item["best_answers"]["min"]["sql"] + "\t----- bird -----\t" + dbs[index] for item in results},
                f, indent=2
            )


args = get_args()
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

DATA_PATH = {
    "test": os.path.join(args.data_path),
}
dataset = load_dataset('json', data_files=DATA_PATH)
results = []

client = OpenAI(
    api_key="EMPTY",
    base_url=f"http://localhost:{args.model_port}/v1",
)

dbs = [item["db_id"] for item in dataset["test"]]
final_results = []
for index, item in enumerate(tqdm(dataset["test"])):
    sqls = get_chat_completion(client, item["instruction"], model=args.model_tag, temperature=args.temperature, n=args.num_rollouts)
    result = get_top_score_sql(item["instruction"], sqls, item, args)
    result["db_id"] = item["db_id"]
    result["instruction"] = item["instruction"]
    result["output"] = item["output"]
    final_results.append(result)
    
    # 实时保存进度
    with open(os.path.join(args.save_path, "detailed_results.json"), "w") as f:
        json.dump(final_results, f, indent=2)

# 最终统计保存
save_statistics(final_results, args.save_path, dbs, save_predict=True)
