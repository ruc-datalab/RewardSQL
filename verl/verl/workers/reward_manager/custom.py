from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch
import statistics
import requests

def reward_func(port, db_id, question_id, solution_str, ground_truth, extra_info):
    try:
        api_url = f"http://localhost:{port}/compare_sql"
        payload = {
            "db_id": db_id,
            "query_id": question_id,
            "predicted_sql": solution_str,
            "ground_truth": ground_truth
        }
        
        # 发送HTTP请求到Flask API
        api_response = requests.post(api_url, json=payload, timeout=120)
        if api_response.status_code == 200:
            response_data = api_response.json()
            if response_data['status'] == 'success':
                return 1
            else:
                return 0
        else:
            return 0
    except Exception as e:
        return 0

class CustomRewardManager:
    """The custom reward manager.
    """

    def __init__(self, tokenizer, num_examine, config, compute_score=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or reward_func
        self.config = config

    def verify(self, data, eval_mode=False):
        """验证响应的准确性并返回分数"""
        scores = []
        already_print_data_sources = {}
        if 'outcome_reward' not in data.batch:
            data.batch['outcome_reward'] = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
    
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # 解码
            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            response_str = self.tokenizer.decode(valid_response_ids).replace("<|im_end|>", "")

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']
            db_id = data_item.non_tensor_batch['db_id']
            question_id = data_item.non_tensor_batch['index']
            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            # 计算分数
            score = self.compute_score(
                port=self.config.verifier.port,
                db_id=db_id,
                question_id=question_id,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            
            scores.append(score)
            
            # 打印示例
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                # print("[prompt]", prompt_str)
                print("[db_id]", db_id)
                print("[question_id]", question_id)
                print("[response length]", len(response_str))
                print("[response]")
                print(f"'''\n{repr(response_str)}\n'''")  # 使用三引号包围，保留所有格式
                print("[ground_truth]")
                print(f"'''\n{repr(ground_truth)}\n'''")  # 使用三引号包围，保留所有格式
                print("[score]", score)
        
        # 将分数转换为张量并存储到batch中
        score_tensor = torch.tensor(scores, dtype=torch.float32, device=data.batch['responses'].device)
        verifier_reward = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        for i in range(len(data)):
                data_item = data[i]
                valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
                # 将所有响应位置的token都设置为相同的分数
                if eval_mode or self.config.reward_model.rm_coef == 0:
                    verifier_reward[i, valid_response_length-1] = score_tensor[i]
                else:
                    verifier_reward[i, :valid_response_length] = score_tensor[i]
                data.batch['outcome_reward'][i] = score_tensor[i]
        data.batch['acc'] = verifier_reward
        
        return verifier_reward, ""

    def __call__(self, data: DataProto):
        """计算并返回奖励张量"""
        reward_tensor_dict = {}
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        # 如果已经有rm_scores，直接使用
        if 'rm_scores' in data.batch.keys():
            reward_tensor_dict['rm_scores'] = data.batch['rm_scores']
            
            # 如果rm_coef存在且不为0，将其加入总奖励
            if hasattr(self.config, 'reward_model') and hasattr(self.config.reward_model, 'rm_coef') and self.config.reward_model.rm_coef != 0:
                reward_tensor += self.config.reward_model.rm_coef * reward_tensor_dict['rm_scores']
                            # 获取当前reward_tensor的副本
            filled_reward_tensor = reward_tensor.clone()
            
            # 对每个样本进行处理
            batch_size, seq_length = filled_reward_tensor.shape
            for i in range(batch_size):
                # 找出所有非零位置
                non_zero_positions = torch.nonzero(filled_reward_tensor[i], as_tuple=True)[0]
                if len(non_zero_positions) > 0:
                    # 从前向后遍历非零位置
                    start_pos = 0
                    for pos in non_zero_positions:
                        # 获取当前非零位置的值
                        value = filled_reward_tensor[i, pos].item()
                        # 向前填充，从上一个位置到当前位置
                        for j in range(start_pos, pos + 1):
                            filled_reward_tensor[i, j] = value
                        # 更新下一个开始位置
                        start_pos = pos + 1
            # 更新reward_tensor
            reward_tensor = filled_reward_tensor
        # 验证器奖励        
        # 如果verifier_coef存在且不为0，将其加入总奖励
        if hasattr(self.config, 'verifier') and hasattr(self.config.verifier, 'reward_coef') and self.config.verifier.reward_coef != 0:
            # 获取响应的有效长度
            prompt_ids = data.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            
            verifier_reward, verifier_metrics = self.verify(data)
            reward_tensor_dict['gt_scores'] = verifier_reward
            reward_tensor += self.config.verifier.reward_coef * reward_tensor_dict['gt_scores']
        return reward_tensor
