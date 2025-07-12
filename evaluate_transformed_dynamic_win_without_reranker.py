import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
import numpy as np  # 导入 numpy 用于 mean 函数
import matplotlib.pyplot as plt  # 导入 matplotlib 用于绘图
from tqdm import tqdm
import random

# 绝对导入
from embedding import *
from model.embeddding_model import DenseRetrievalModel
from evaluate import *

# hierarchical_atten_embedding
def hierarchical_atten_embedding(chunked_data, h_lengh):
    # 遍历chunked_data，滚动计算当前chunk的embedding和前5个chunk的embedding，然后加权将embedding合并
    # for i in range(len(chunked_data)):
    #     # chunked_data[i]["hierarchical_content"] = "[CLS] " +" | ".join([chunked_data[j]["content"] for j in range(max(0, i - h_lengh), i)]) + " | [context] | " + chunked_data[i]["content"]  
    #     chunked_data[i]["hierarchical_content"] = " ".join([chunked_data[j]["content"] for j in range(max(0, i - h_lengh), i)]) + " | [context] | " + chunked_data[i]["content"]  
    chunked_data[0]["hierarchical_content"] = chunked_data[0]["hierarchical_content"]
    return chunked_data


def embedding_data(args):
    # 集中配置参数
    config = {
        "content_key": "hierarchical_content",
        "chunk_id_key": "chunk_id",
        "vector_dim": 1024,
        "db_index_fields": ["bm25_sparse_vector", "dense_vector"],
        "batch_size": 10,
        "max_length": 512,
        "h_lengh": args.win_size,
        "milvus_host": "localhost",
        "milvus_port": "19530",
        "data_path": args.conv_data_path,
        "model_path": args.model_path
    }

    # model
    model = DenseRetrievalModel(config)
    client = MilvusClientWrapper(config)

    data_base_path = args.conv_data_path
    for path in os.listdir(data_base_path):
        if path.endswith(".json"):
            conv_data_name = os.path.splitext(os.path.basename(path))[0]
            with open(os.path.join(data_base_path, path), "r", encoding="utf-8") as f:
                conv_data = json.load(f)
            collection_name = f"bge_m3_win_{conv_data_name}_without_reranker"
            if client.has_collection(collection_name):            
                print(f"Collection {collection_name} already exists. Skipping...")
                continue
            else:
                try:
                    conv_data = hierarchical_atten_embedding(conv_data, args.win_size)
                    client.create_collection(collection_name, conv_data[0])
                    client.insert_data(conv_data, collection_name, model)
                except Exception as e:
                    print(f"Error creating collection {collection_name}: {e}")

# day2eval_data 每天需要评测的数据
def day2eval_data(qa_data, sample_ratio=0.5):
    """
    根据evidence的day信息组织数据，实现累积抽样
    
    Args:
        qa_data: 原始qa数据列表
        sample_ratio: 抽样比例，默认0.5
    
    Returns:
        day2eval_data: 字典，key为D1,D2,D3...，value为累积抽样后的数据列表
    """
    # 首先按day分组原始数据
    original_day_data = {}
    for data in qa_data:
        evidence_days = sorted([evidence.split(":")[0] for evidence in data["evidence"]])
        if evidence_days:
            day_list = original_day_data.get(evidence_days[-1], [])
            day_list.append(data)
            original_day_data[evidence_days[-1]] = day_list
    
    # 按day排序
    sorted_days = sorted(original_day_data.keys())
    
    # 实现累积抽样逻辑
    day2eval_data = {}
    accumulated_data = []
    
    for i, day in enumerate(sorted_days):
        current_day_data = original_day_data[day]
        
        if i == 0:
            # D1: 只抽取原D1数据的一定比例
            sample_size = max(1, int(len(current_day_data) * sample_ratio))
            sampled_data = random.sample(current_day_data, sample_size)
            accumulated_data = sampled_data.copy()
        else:
            # Dn (n>1): 抽取原D1到Dn的一定比例内容
            # 将当前day的数据加入到候选池中
            all_candidate_data = accumulated_data + current_day_data
            
            # 计算抽样大小，确保Dn的list比Dn-1更大
            prev_size = len(day2eval_data[sorted_days[i-1]])
            # 增长因子确保每个后续的day包含更多数据
            growth_factor = 1.2 + (i * 0.1)  # 递增的增长因子
            new_size = max(prev_size + 1, int(len(all_candidate_data) * sample_ratio * growth_factor))
            new_size = min(new_size, len(all_candidate_data))  # 不能超过总数据量
            
            sampled_data = random.sample(all_candidate_data, new_size)
            accumulated_data = sampled_data.copy()
        
        day2eval_data[day] = accumulated_data.copy()
    
    return day2eval_data

def evaulate_tranformed_data(args):
    config = {
        "milvus_host": "localhost",
        "milvus_port": "19530",
        "topk": 5,
        "model_path": args.model_path,
        "h_lengh": args.win_size,
        "max_length": 512,
        "search_field": "dense_vector",
        "search_metric_type": "IP",
        "search_nprobe": 10,
        "query_prefix": ""
    }

    model = DenseRetrievalModel(config)
    client = MilvusClientWrapper(config)

    eval_data_base_path = args.qa_data_path
    diaId_content_map_base_path = args.diaId_content_map_path
    total_recall_ratio = []

    eval_data_list = os.listdir(eval_data_base_path)
    eval_data_list.sort()
    map_data_list = os.listdir(diaId_content_map_base_path)
    map_data_list.sort()
    
    for i, (eval_data_path, map_data_path) in enumerate(zip(eval_data_list, map_data_list)):
        colleaction_name = f"bge_m3_win_dynamic_conv_data_{i}_without_reranker"
        if os.path.exists(os.path.join(args.results_path, "retri_"+colleaction_name)):
            with open(os.path.join(args.results_path,"retri_"+colleaction_name) , "r", encoding="utf-8") as f:
                retr_results = json.loads(f.read())
        else:
            with open(os.path.join(eval_data_base_path, eval_data_path), "r", encoding="utf-8") as f:
                qa_data = []
                for line in f:
                    qa_raw_data_dic = json.loads(line)
                    qa_data.append(qa_raw_data_dic)
            with open(os.path.join(diaId_content_map_base_path, map_data_path), "r", encoding="utf-8") as f:
                diaId_content_map = json.load(f)

            # loading data
            client.load_data(collection_name=colleaction_name)
            
            # 获取按天分组的评估数据
            eval_data_by_day = day2eval_data(qa_data)
            
            retr_results = []
            
            # 修复：正确获取排序后的天列表
            day_list = sorted(eval_data_by_day.keys())
            eval_ids = []
            for day in day_list:
                eval_day_list = eval_data_by_day[day]
                
                # 构建当前天的搜索ID范围
                eval_ids.extend([f"{day}:{j}" for j in range(50)])  # 根据实际需要调整范围
                
                for eval_item in tqdm(eval_day_list, desc=f"Processing {colleaction_name} Day {day}", unit="item"):
                    # 修复：检查数据结构并构建查询
                    if "conversation" in eval_item and isinstance(eval_item["conversation"], list):
                        query_parts = []
                        for conv in eval_item["conversation"]:
                            if "speaker" in conv and "content" in conv:
                                query_parts.append(f"{conv['speaker']}:{conv['content']}")
                        query = config["query_prefix"] + " ".join(query_parts)
                    else:
                        # 如果没有conversation字段，使用其他字段构建查询
                        query = config["query_prefix"] + str(eval_item.get("content", ""))
                    
                    query_embedding = model.embedding(query)
                    
                    # 修复：正确的Milvus filter语法
                    filter_expr = 'dia_id in {eval_ids}'
                    
                    res_list = client.search(
                        collection_name=colleaction_name,
                        query_embedding=query_embedding,
                        filter=filter_expr,
                        filter_params={"eval_ids": eval_ids},
                        output_fields=["content", "dia_id"],
                        search_field=config["search_field"],
                        limit=config["topk"],
                        search_params={"metric_type": config["search_metric_type"], "params": {"nprobe": config["search_nprobe"]}},
                    )

                    retr_result = {
                        "id": eval_item.get("id", ""),
                        "conv": eval_item.get("conversation", []),
                        "evidence": eval_item.get("evidence", []),
                        "evidence_text": [diaId_content_map[dia_id] for dia_id in eval_item.get("evidence", []) if dia_id in diaId_content_map],
                        "retrieve_text": [res_item["entity"]["content"] for res_item in res_list],
                    }
                    retr_results.append(retr_result)

        with open(os.path.join(args.results_path, "retri_"+colleaction_name), "w", encoding="utf-8") as f:
            json.dump(retr_results, f, ensure_ascii=False, indent=4)
        
        # evaluate recall ratio
        recall_ratio = evaluate_recall_ratio(retr_results, reranked=False)
        total_recall_ratio.append(recall_ratio)

    return total_recall_ratio


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conv_data_path", type=str, default=r"D:\project\myproject\ChatSeeker\data\dynamic_win_conv_data_without_reranker", help="chunked data path")
    parser.add_argument("--qa_data_path", type=str, default=r"D:\project\myproject\ChatSeeker\data\transformed_eval_data", help="qa data path")
    parser.add_argument("--diaId_content_map_path", type=str, default=r"D:\project\myproject\ChatSeeker\data\diaID_map_data", help="diaId content map path")
    parser.add_argument("--results_path", type=str, default=r"D:\project\myproject\ChatSeeker\transformed_results_without_reranker", help="results path")
    parser.add_argument("--win_size", type=int, default=0, help="length of hierarchical attention")
    parser.add_argument("--model_path", type=str, default=r"C:\Users\鉴\.cache\huggingface\hub\models--BAAI--bge-m3\snapshots\5617a9f61b028005a4858fdac845db406aefb181", help="model path")
    base_args = parser.parse_args()

    
    new_args = argparse.Namespace(**vars(base_args))
    embedding_data(new_args)

    results = []
    new_args = argparse.Namespace(**vars(base_args))
    total_recall_ratio = evaulate_tranformed_data(new_args)
    # results.append(np.mean(total_recall_ratio))
    # print
    # results.append(total_recall_ratio)
    
    print("Dynamic Win Recall Ratio: ", total_recall_ratio)
    print("Mean Recall Ratio: ", np.mean(total_recall_ratio))