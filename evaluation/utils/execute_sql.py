import sqlite3
import os
from multiprocessing import Pool
from collections import Counter
import re
import time
import sys
db_prefix = "./data/bird/database"
# sp1_test
db_prefix_spider = "./data/spider/test_database"


def execute_sql_in_subprocess(sql: str, db_path: str, db_id: str, query_id: int):
    """
    Execute a SQL query with caching in a subprocess.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        conn.close()
        return result
    except Exception as e:
        conn.close()
        return f"Execution error: {e}"

def execute_single_sql(predicted_sql: str, db_id: str, query_id: int = 0, timeout: int = 60):
    db_path = os.path.join(db_prefix, f"./{db_id}/{db_id}.sqlite")
    if not os.path.exists(db_path):
        db_path = os.path.join(db_prefix_spider, f"./{db_id}/{db_id}.sqlite")
    sql_query = (predicted_sql, db_path, db_id, query_id)
    with Pool(processes=1) as pool:
        result = pool.starmap_async(execute_sql_in_subprocess, [sql_query])
        try:
            res = result.get(timeout=timeout)[0]
            if isinstance(res, str):
                return 0, res
            else:
                return 1, res
        except Exception as e:
            pool.terminate()
            return 0, f"Execution error: {e}"
        finally:
            pool.close()
            pool.join()

def compare_sql(predicted_sql: str, ground_truth: str, db_id: str, query_id: int=0, timeout: int = 60):
    """
    Execute predicted and ground truth SQL queries with caching, 
    comparing results for correctness.
    """
    # Adjust the path as needed for your environment:
    db_path = os.path.join(db_prefix, f"{db_id}/{db_id}.sqlite")

    sql_queries = [
        (predicted_sql, db_path, db_id, query_id),
        (ground_truth, db_path, db_id, query_id)
    ]

    with Pool(processes=2) as pool:
        result = pool.starmap_async(execute_sql_in_subprocess, sql_queries)

        try:
            predicted_res, ground_truth_res = result.get(timeout=timeout)
            # If either returns a string, it likely indicates an error
            if isinstance(predicted_res, str) or isinstance(ground_truth_res, str):
                error_msg = predicted_res if isinstance(predicted_res, str) else ground_truth_res
                return 0, error_msg

            # Compare the sets of tuples
            if set(map(tuple, predicted_res)) == set(map(tuple, ground_truth_res)):
                return 1, f"predicted_res:\n {str(predicted_res)[:100]} \n\nground_truth_res:\n {str(ground_truth_res)[:100]}"
            else:
                return 0, (
                    "Results mismatch:\n"
                    f"predicted_res:\n {str(predicted_res)[:100]} \n\n"
                    f"ground_truth_res:\n {str(ground_truth_res)[:100]}"
                )
    

        except Exception as e:
            pool.terminate()
            return 0, f"Execution error: {e}"
        finally:
            pool.close()
            pool.join()

def match_last_cte(sql):
    pattern = re.compile(r"(\w+)\s+AS\s*\(.*?\)", re.DOTALL | re.IGNORECASE)
    matches = pattern.findall(sql)
    if matches:
        last_cte = matches[-1]
        return last_cte
    else:
        return None

def compact_result_with_counter(result: list) -> Counter:
    hashable_result = []
    for row in result:
        if isinstance(row, list):
            hashable_row = tuple(row)
        elif isinstance(row, tuple):
            hashable_row = row
        else:
            hashable_row = row  # 其他类型保持不变
        hashable_result.append(hashable_row)
    counter = Counter(hashable_result)
    return counter

def process_single_item(item, step_id=" и "):
    db_id = item["db_id"]
    total_sql = item['sql']
    sqls = total_sql.split(step_id)[:-1]
    accumulated_sql = ""
    sql_with_ex = ""

    for sql in sqls:
        # time.sleep(0.1)  # 可选：模拟延迟或限速
        sql_with_ex = f"{sql_with_ex}{step_id}{sql}".strip() if sql_with_ex else sql.strip()
        accumulated_sql = f"{accumulated_sql}{sql}".strip() if accumulated_sql else sql.strip()

        if ";" in accumulated_sql:
            ex_sql = accumulated_sql
        else:
            last_cte = match_last_cte(accumulated_sql)
            if last_cte:
                ex_sql = (accumulated_sql[:-1] if accumulated_sql.endswith(",") else accumulated_sql) + f"SELECT * FROM {last_cte}"
            else:
                ex_sql = accumulated_sql + ";"  # 如果未找到CTE，则追加分号
        res, info = execute_single_sql(ex_sql, db_id, item['question_id'], 30)
        if res:
            execution_results = str(dict(compact_result_with_counter(info)))
            if len(execution_results) > 500:
                execution_results = execution_results[:500] + "...}"
        else:
            execution_results = info  # 错误信息

        execution_results = f"Total SQL result:{res}, {info}" if not res else execution_results
        sql_with_ex = f"{sql_with_ex}<result>{execution_results}</result>"
    sql_with_ex = sql_with_ex + step_id

    return sql_with_ex