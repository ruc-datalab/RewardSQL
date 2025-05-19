import numpy as np
import requests
import resource
step_tag = ' и'
from collections import Counter
import re

def get_chat_completion(client, input_text, model="qwen2.5-sql", temperature=1.0, n=16):
    input_message = input_text
    response = client.completions.create(
        model=model,
        prompt=input_message,
        max_tokens=1024,
        temperature=temperature,
        n=n,
        stream=False
    )
    SQLs = []
    # 输出结果
    for choice in response.choices:
        # if choice.text not in SQLs:
            SQLs.append(choice.text)
    return SQLs

def uniform_cte_sql(sql, kind="weak", replace_cte=False):
    def replace_cte_names(sql):
        # 查找所有的 CTE 名称
        cte_pattern = r'(?:WITH|, и)\s*([\w]+)\s+AS'
        cte_names = re.findall(cte_pattern, sql)
        # 创建名称映射字典
        cte_mapping = {}
        for idx, name in enumerate(cte_names):
            cte_mapping[name] = f'CTE_{chr(ord("A") + idx)}'
        # 替换 SQL 中的 CTE 名称
        try:
            for orig_name, new_name in cte_mapping.items():
                # 使用 \b 确保只匹配完整的单词
                sql = re.sub(r'\b' + re.escape(orig_name) + r'\b', new_name, sql)
                # sql = sql.replace(orig_name, new_name)
        except Exception as e:
            print(repr(sql))
        return sql
    # sql = sql.lower()
    if replace_cte:
        sql = replace_cte_names(sql)
    sql = sql.replace(step_tag, '<<DOUBLE_NEWLINE>>')
    if kind == "weak":
        sql = re.sub(r'\s+', ' ', sql)
    elif kind == "strong":
        sql = re.sub(r'\s+', '', sql)
    sql = sql.replace('<<DOUBLE_NEWLINE>>', step_tag)
    return sql

def get_score(request_api, question, process):
    input_for_prm = f"{question}{process}"
    url = request_api
    data = {
    "input": input_for_prm
    }
    response = requests.post(url, json=data)

    if response.status_code == 200:
        score = response.json()["score"]
    else:
        score = [0.0]
        print(f"Error: {response.status_code}")
    return score

def calculate_scores(step_scores, strategy):
    if strategy == "product":
        return np.prod(step_scores)
    elif strategy == "min":
        return np.min(step_scores)
    elif strategy == "last_step":
        return step_scores[-1]
    elif strategy == "average":
        return np.mean(step_scores)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    