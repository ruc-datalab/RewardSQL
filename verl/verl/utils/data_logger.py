import os
import json
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
import pandas as pd
import re

class TrainingDataLogger:
    """训练数据记录器，用于记录训练过程中的中间信息"""
    
    def __init__(self, log_dir, tokenizer=None, log_frequency=1, max_samples_per_step=100, max_rollouts_per_prompt=16):
        """
        初始化训练数据记录器
        
        Args:
            log_dir: 日志存储目录
            tokenizer: 用于解码token ID的分词器
            log_frequency: 记录频率，每隔多少步记录一次
            max_samples_per_step: 每个步骤最多记录的样本数
            max_rollouts_per_prompt: 每个提示最多记录的rollout数
        """
        self.log_dir = Path(log_dir)
        self.tokenizer = tokenizer
        self.log_frequency = log_frequency
        self.max_samples_per_step = max_samples_per_step
        self.max_rollouts_per_prompt = max_rollouts_per_prompt
        
        # 提取实验名称
        self.experiment_name = self.log_dir.name
        
        # 创建日志目录
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.log_dir / "steps", exist_ok=True)
        
        # 创建索引文件
        self.index_file = self.log_dir / "index.jsonl"
        if not os.path.exists(self.index_file):
            with open(self.index_file, "w") as f:
                pass
        
        # 创建汇总统计信息文件
        self.stats_file = self.log_dir / "stats.csv"
    
    def should_log(self, step):
        """判断当前步骤是否应该记录"""
        return step % self.log_frequency == 0
    
    def _convert_tensor_to_json(self, tensor):
        """将PyTorch张量转换为可JSON序列化的形式"""
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().tolist()
        return tensor
    
    def _clean_text(self, text):
        """清理文本，移除特殊标记并格式化"""
        if not text:
            return ""
            
        # 移除特殊标记
        text = text.replace("<|endoftext|>", "").strip()
        
        return text
    
    def _process_prompt(self, prompt_text):
        """处理提示文本，提取结构化信息"""
        if not prompt_text:
            return {"original": "", "processed": "", "structured": {}}
        
        # 保存原始提示
        result = {
            "original": prompt_text,
            "processed": prompt_text,
            "structured": {}
        }
        
        # 尝试提取指令、上下文等结构化信息
        # 这里只是一个示例，需要根据实际提示格式调整
        instruction_match = re.search(r"指令[:：](.*?)(?=上下文[:：]|$)", prompt_text, re.DOTALL)
        context_match = re.search(r"上下文[:：](.*?)(?=输入[:：]|$)", prompt_text, re.DOTALL)
        input_match = re.search(r"输入[:：](.*?)(?=输出[:：]|$)", prompt_text, re.DOTALL)
        
        if instruction_match:
            result["structured"]["instruction"] = instruction_match.group(1).strip()
        if context_match:
            result["structured"]["context"] = context_match.group(1).strip()
        if input_match:
            result["structured"]["input"] = input_match.group(1).strip()
            
        return result
    
    def _extract_tokens_from_text(self, text, tokenizer=None):
        """从文本中提取token"""
        if not text or not tokenizer:
            return []
            
        # 使用tokenizer对文本进行编码，然后解码每个token
        input_ids = tokenizer.encode(text)
        tokens = []
        
        for i in range(len(input_ids)):
            token = tokenizer.decode([input_ids[i]])
            tokens.append(token)
            
        return tokens
    
    def log_step(self, step, batch, metrics):
        """记录当前步骤的信息"""
        if not self.should_log(step):
            return
        
        step_data = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "batch_size": batch.batch.batch_size[0],
            "metrics": {k: self._convert_tensor_to_json(v) for k, v in metrics.items()},
            "samples": [],
            "experiment_name": self.experiment_name
        }
        
        # 提取批次中的样本信息
        try:
            batch_size = batch.batch.batch_size[0]
            responses = batch.batch.get('responses')
            prompts = batch.batch.get('prompts')
            token_level_scores = batch.batch.get('token_level_scores')
            token_level_rewards = batch.batch.get('token_level_rewards')
            advantages = batch.batch.get('advantages', None)
            
            # 获取所有可能的奖励/分数源
            possible_score_sources = {
                'rm_scores': 'prm_scores',  # PRM奖励
                'gt_scores': 'orm_scores',  # GT奖励
                'acc': 'orm_scores',        # 准确性奖励
                'kl_div': 'kl_scores',      # KL散度
                'length': 'length_scores'   # 长度惩罚
            }
            
            score_sources = {}
            for key, mapped_name in possible_score_sources.items():
                if key in batch.batch:
                    score_sources[key] = mapped_name
            
            # 记录每个样本
            # 对于GRPO，通常每n个样本是同一个提示的不同rollout
            # 我们需要按照uid分组，相同uid的是同一个提示的不同rollout
            uids = batch.non_tensor_batch.get('uid', [f"unknown_{i}" for i in range(batch_size)])
            
            # 创建uid到索引的映射
            uid_to_indices = {}
            for i, uid in enumerate(uids):
                if uid not in uid_to_indices:
                    uid_to_indices[uid] = []
                uid_to_indices[uid].append(i)
            
            # 对每个提示的多个rollout进行记录
            for uid, indices in uid_to_indices.items():
                # 限制记录的提示组数量
                if len(step_data["samples"]) >= self.max_samples_per_step:
                    break
                    
                prompt_group = {
                    "uid": str(uid),
                    "db_id": batch.non_tensor_batch.get('db_id', ['unknown'])[indices[0]] if 'db_id' in batch.non_tensor_batch else 'unknown',
                    "question_id": batch.non_tensor_batch.get('index', [indices[0]])[indices[0]] if 'index' in batch.non_tensor_batch else indices[0],
                    "rollouts": []
                }
                
                # 解码并处理prompt（对于同一组rollout，prompt是相同的）
                if prompts is not None and self.tokenizer is not None:
                    raw_prompt_text = self.tokenizer.decode(prompts[indices[0]])
                    # 清理prompt文本，移除<|endoftext|>
                    cleaned_prompt_text = self._clean_text(raw_prompt_text)
                    prompt_data = self._process_prompt(cleaned_prompt_text)
                    prompt_group["prompt"] = prompt_data["processed"]
                    prompt_group["prompt_original"] = prompt_data["original"]
                    prompt_group["prompt_structured"] = prompt_data["structured"]
                    
                    # 提取prompt的tokens，移除<|endoftext|>
                    try:
                        prompt_ids = prompts[indices[0]].tolist()
                        # 找到第一个<|endoftext|>的位置
                        eos_pos = -1
                        for pos, token_id in enumerate(prompt_ids):
                            if self.tokenizer.decode([token_id]) == "<|endoftext|>":
                                eos_pos = pos
                                break
                        
                        # 截取有效长度
                        valid_prompt_length = eos_pos if eos_pos != -1 else len(prompt_ids)
                        valid_prompt_ids = prompt_ids[:valid_prompt_length]
                        
                        # 获取token文本，过滤掉<|endoftext|>
                        prompt_tokens = []
                        for token_id in valid_prompt_ids:
                            token_text = self.tokenizer.decode([token_id])
                            if token_text != "<|endoftext|>":
                                prompt_tokens.append(token_text)
                        
                        prompt_group["prompt_tokens"] = prompt_tokens
                    except Exception as e:
                        prompt_group["prompt_tokens_error"] = str(e)
                
                # 获取每个rollout的响应和分数
                for idx, i in enumerate(indices):
                    # 限制每个提示的rollout数量
                    if idx >= self.max_rollouts_per_prompt:
                        break
                        
                    rollout = {
                        "rollout_id": i - indices[0],  # 相对于这个组的第几个rollout
                    }
                    
                    # 解码response
                    if responses is not None and self.tokenizer is not None:
                        raw_response_text = self.tokenizer.decode(responses[i])
                        response_text = self._clean_text(raw_response_text)
                        rollout["response"] = response_text
                        
                        # 提取response的tokens
                        try:
                            # 找到第一个<|endoftext|>的位置
                            eos_pos = -1
                            response_ids = responses[i].tolist()
                            for pos, token_id in enumerate(response_ids):
                                if self.tokenizer.decode([token_id]) == "<|endoftext|>":
                                    eos_pos = pos
                                    break
                            
                            valid_length = eos_pos if eos_pos != -1 else len(response_ids)
                            # 截取有效长度的response_ids
                            valid_response_ids = response_ids[:valid_length]
                            
                            response_tokens = []
                            for token_id in valid_response_ids:
                                token_text = self.tokenizer.decode([token_id])
                                response_tokens.append(token_text)
                            rollout["response_tokens"] = response_tokens
                        except Exception as e:
                            rollout["response_tokens_error"] = str(e)
                    
                    # 记录详细的token级别奖励信息
                    if token_level_scores is not None:
                        # 找到第一个<|endoftext|>的位置
                        valid_length = eos_pos if eos_pos != -1 else len(response_ids)
                        
                        # 总体奖励 - 只计算有效长度内的奖励
                        valid_scores = token_level_scores[i][:valid_length]
                        reward = torch.sum(valid_scores).item()
                        rollout["reward"] = reward
                        
                        # 记录token级别的分数 - 只记录有效长度内的分数
                        token_scores = token_level_scores[i][:valid_length].tolist()
                        rollout["token_level_scores"] = token_scores
                        
                        # 创建处理后的分数字典
                        processed_scores = {}
                        
                        # 同步截断其他token级别的信息
                        for source_key in score_sources.keys():
                            if source_key in batch.batch:
                                source_tensor = batch.batch[source_key][i]
                                if len(source_tensor) > valid_length:
                                    processed_scores[source_key] = source_tensor[:valid_length]
                                else:
                                    processed_scores[source_key] = source_tensor
                        
                        # 处理优势值
                        processed_advantages = None
                        if advantages is not None:
                            if len(advantages[i]) > valid_length:
                                processed_advantages = advantages[i][:valid_length]
                            else:
                                processed_advantages = advantages[i]
                        
                        # 创建详细的token级别信息列表
                        detailed_token_info = []
                        
                        # 记录有效的token信息
                        for t_idx in range(valid_length):
                            token_id = response_ids[t_idx]
                            token_text = self.tokenizer.decode([token_id])
                            
                            # 跳过<|endoftext|>标记
                            if token_text == "<|endoftext|>":
                                continue
                                
                            token_info = {
                                "token_id": token_id,
                                "token_text": token_text,
                                "score": token_scores[t_idx] if token_scores is not None else None
                            }
                            
                            # 添加其他分数源
                            for source_key, mapped_name in score_sources.items():
                                if source_key in processed_scores:
                                    scores = processed_scores[source_key]
                                    if t_idx < len(scores):
                                        token_info[mapped_name] = scores[t_idx].item()
                            
                            # 添加优势值
                            if processed_advantages is not None and t_idx < len(processed_advantages):
                                token_info["advantage"] = processed_advantages[t_idx].item()
                                
                            detailed_token_info.append(token_info)
                        
                        # 将详细的token级别信息添加到rollout中
                        rollout["detailed_token_info"] = detailed_token_info
                        
                        if token_level_rewards is not None:
                            # 截取有效长度的token_level_rewards
                            valid_rewards = token_level_rewards[i][:valid_length] if valid_length < len(token_level_rewards[i]) else token_level_rewards[i]
                            rollout["token_level_rewards"] = valid_rewards.tolist()
                        
                        # 记录各种奖励分数
                        reward_sources = {}
                        
                        # PRM奖励
                        if 'rm_scores' in batch.batch:
                            prm_scores = batch.batch['rm_scores'][i]
                            # 截取有效长度的prm_scores
                            valid_prm_scores = prm_scores[:valid_length] if valid_length < len(prm_scores) else prm_scores
                            rollout["prm_scores"] = valid_prm_scores.tolist()
                            rollout["prm_reward"] = torch.sum(valid_prm_scores).item()
                            reward_sources["PRM"] = rollout["prm_reward"]
                        
                        # ORM/准确性奖励
                        orm_source_found = False
                        
                        if 'gt_scores' in batch.batch:
                            orm_scores = batch.batch['gt_scores'][i]
                            # 截取有效长度的orm_scores
                            valid_orm_scores = orm_scores[:valid_length] if valid_length < len(orm_scores) else orm_scores
                            rollout["orm_scores"] = valid_orm_scores.tolist()
                            rollout["orm_reward"] = torch.sum(valid_orm_scores).item()
                            reward_sources["ORM"] = rollout["orm_reward"]
                            orm_source_found = True
                        
                        if 'acc' in batch.batch and not orm_source_found:
                            acc_scores = batch.batch['acc'][i]
                            # 截取有效长度的acc_scores
                            valid_acc_scores = acc_scores[:valid_length] if valid_length < len(acc_scores) else acc_scores
                            rollout["orm_scores"] = valid_acc_scores.tolist()
                            rollout["orm_reward"] = torch.sum(valid_acc_scores).item()
                            reward_sources["ACC"] = rollout["orm_reward"]
                            orm_source_found = True
                        
                        # KL散度惩罚
                        if 'kl_div' in batch.batch:
                            kl_scores = batch.batch['kl_div'][i]
                            # 截取有效长度的kl_scores
                            valid_kl_scores = kl_scores[:valid_length] if valid_length < len(kl_scores) else kl_scores
                            rollout["kl_scores"] = valid_kl_scores.tolist()
                            rollout["kl_penalty"] = torch.sum(valid_kl_scores).item()
                            reward_sources["KL"] = -rollout["kl_penalty"]  # 通常KL是负贡献
                        
                        # 长度惩罚
                        if 'length' in batch.batch:
                            length_scores = batch.batch['length'][i]
                            # 截取有效长度的length_scores
                            valid_length_scores = length_scores[:valid_length] if valid_length < len(length_scores) else length_scores
                            rollout["length_scores"] = valid_length_scores.tolist()
                            rollout["length_penalty"] = torch.sum(valid_length_scores).item()
                            reward_sources["Length"] = rollout["length_penalty"]
                        
                        if reward_sources:
                            rollout["reward_sources"] = reward_sources
                            # 根据主要来源确定奖励类型
                            reward_types = []
                            if "PRM" in reward_sources:
                                reward_types.append("PRM")
                            if "ORM" in reward_sources or "ACC" in reward_sources:
                                reward_types.append("ORM")
                            if "KL" in reward_sources:
                                reward_types.append("KL")
                                
                            rollout["reward_type"] = "+".join(reward_types) if reward_types else "Unknown"
                        else:
                            rollout["reward_type"] = "Combined"
                    
                    # 记录优势值
                    if advantages is not None:
                        # 截取有效长度的优势值
                        valid_advantages = advantages[i][:valid_length] if valid_length < len(advantages[i]) else advantages[i]
                        rollout["advantages"] = valid_advantages.tolist()
                        rollout["advantage"] = torch.sum(valid_advantages).item()
                    
                    prompt_group["rollouts"].append(rollout)
                
                # 按reward降序排序rollouts
                prompt_group["rollouts"].sort(key=lambda x: x.get("reward", 0), reverse=True)
                step_data["samples"].append(prompt_group)
                
        except Exception as e:
            step_data["error"] = f"记录样本时出错: {str(e)}"
            import traceback
            step_data["error_traceback"] = traceback.format_exc()
        
        # 计算汇总统计
        all_rewards = []
        all_prm_rewards = []
        all_orm_rewards = []
        all_advantages = []
        all_kl_penalties = []
        all_length_penalties = []
        
        for prompt_group in step_data["samples"]:
            for rollout in prompt_group.get("rollouts", []):
                all_rewards.append(rollout.get("reward", 0))
                if "prm_reward" in rollout:
                    all_prm_rewards.append(rollout["prm_reward"])
                if "orm_reward" in rollout:
                    all_orm_rewards.append(rollout["orm_reward"])
                if "advantage" in rollout:
                    all_advantages.append(rollout["advantage"])
                if "kl_penalty" in rollout:
                    all_kl_penalties.append(rollout["kl_penalty"])
                if "length_penalty" in rollout:
                    all_length_penalties.append(rollout["length_penalty"])
        
        # 计算并记录统计指标
        if all_rewards:
            step_data["mean_reward"] = np.mean(all_rewards)
            step_data["max_reward"] = np.max(all_rewards)
            step_data["min_reward"] = np.min(all_rewards)
            step_data["std_reward"] = np.std(all_rewards)
        else:
            step_data["mean_reward"] = 0
            step_data["max_reward"] = 0
            step_data["min_reward"] = 0
            step_data["std_reward"] = 0
        
        # 记录PRM和ORM奖励统计（如果存在）
        if all_prm_rewards:
            step_data["mean_prm_reward"] = np.mean(all_prm_rewards)
            step_data["max_prm_reward"] = np.max(all_prm_rewards)
            step_data["min_prm_reward"] = np.min(all_prm_rewards)
            step_data["std_prm_reward"] = np.std(all_prm_rewards)
            
        if all_orm_rewards:
            step_data["mean_orm_reward"] = np.mean(all_orm_rewards)
            step_data["max_orm_reward"] = np.max(all_orm_rewards)
            step_data["min_orm_reward"] = np.min(all_orm_rewards)
            step_data["std_orm_reward"] = np.std(all_orm_rewards)
        
        # 记录优势值统计（如果存在）
        if all_advantages:
            step_data["mean_advantage"] = np.mean(all_advantages)
            step_data["max_advantage"] = np.max(all_advantages)
            step_data["min_advantage"] = np.min(all_advantages)
            step_data["std_advantage"] = np.std(all_advantages)
        
        # 记录KL惩罚统计（如果存在）
        if all_kl_penalties:
            step_data["mean_kl_penalty"] = np.mean(all_kl_penalties)
            step_data["max_kl_penalty"] = np.max(all_kl_penalties)
            step_data["min_kl_penalty"] = np.min(all_kl_penalties)
        
        # 记录长度惩罚统计（如果存在）
        if all_length_penalties:
            step_data["mean_length_penalty"] = np.mean(all_length_penalties)
        
        # 从指标中获取KL惩罚（如果存在）
        if "metrics" in step_data and "critic/kl" in step_data["metrics"]:
            step_data["kl_penalty"] = step_data["metrics"]["critic/kl"]
        
        # 保存步骤数据
        with open(self.log_dir / f"steps/step_{step}.json", "w", encoding='utf-8') as f:
            json.dump(step_data, f, ensure_ascii=False, indent=2)
        
        # 更新索引
        with open(self.index_file, "a", encoding='utf-8') as f:
            index_entry = {
                "step": step,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "file": f"steps/step_{step}.json",
                "mean_reward": float(step_data["mean_reward"]),
                "sample_count": len(step_data["samples"])
            }
            f.write(json.dumps(index_entry, ensure_ascii=False) + "\n")
        
        # 更新统计数据CSV
        self.update_stats_csv(step, step_data)
        
        print(f"已记录步骤 {step} 的训练数据，共 {len(step_data['samples'])} 个样本组")
        
    def update_stats_csv(self, step, step_data):
        """更新统计数据CSV文件"""
        stats = {
            "step": step,
            "timestamp": step_data.get("timestamp", datetime.now().isoformat()),
            "mean_reward": step_data.get("mean_reward", 0),
            "max_reward": step_data.get("max_reward", 0),
            "min_reward": step_data.get("min_reward", 0),
            "std_reward": step_data.get("std_reward", 0),
            "batch_size": step_data.get("batch_size", 0),
            "sample_count": len(step_data.get("samples", [])),
            "experiment_name": step_data.get("experiment_name", "Unknown")
        }
        
        # 添加PRM和ORM奖励（如果存在）
        if "mean_prm_reward" in step_data:
            stats["mean_prm_reward"] = step_data["mean_prm_reward"]
            stats["max_prm_reward"] = step_data.get("max_prm_reward", 0)
            stats["min_prm_reward"] = step_data.get("min_prm_reward", 0)
            stats["std_prm_reward"] = step_data.get("std_prm_reward", 0)
            
        if "mean_orm_reward" in step_data:
            stats["mean_orm_reward"] = step_data["mean_orm_reward"]
            stats["max_orm_reward"] = step_data.get("max_orm_reward", 0)
            stats["min_orm_reward"] = step_data.get("min_orm_reward", 0)
            stats["std_orm_reward"] = step_data.get("std_orm_reward", 0)
        
        # 添加优势值均值（如果存在）
        if "mean_advantage" in step_data:
            stats["mean_advantage"] = step_data["mean_advantage"]
            stats["max_advantage"] = step_data.get("max_advantage", 0)
            stats["min_advantage"] = step_data.get("min_advantage", 0)
            stats["std_advantage"] = step_data.get("std_advantage", 0)
        
        # 添加KL惩罚均值（如果存在）
        if "mean_kl_penalty" in step_data:
            stats["mean_kl_penalty"] = step_data["mean_kl_penalty"]
            
        # 添加长度惩罚均值（如果存在）
        if "mean_length_penalty" in step_data:
            stats["mean_length_penalty"] = step_data["mean_length_penalty"]
        
        # 从指标中获取KL惩罚（如果存在）
        if "metrics" in step_data and "critic/kl" in step_data["metrics"]:
            stats["kl_penalty"] = step_data["metrics"]["critic/kl"]
        
        # 转换为DataFrame
        stats_df = pd.DataFrame([stats])
        
        # 如果文件存在，追加；否则创建新文件
        if os.path.exists(self.stats_file):
            stats_df.to_csv(self.stats_file, mode='a', header=False, index=False)
        else:
            stats_df.to_csv(self.stats_file, index=False) 