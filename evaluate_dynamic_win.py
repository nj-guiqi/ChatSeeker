import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
import numpy as np  # 导入 numpy 用于 mean 函数
import matplotlib.pyplot as plt  # 导入 matplotlib 用于绘图
from tqdm import tqdm

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
            collection_name = f"bge_m3_win_{conv_data_name}"
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
                


def evaulate_qa(args):
    config = {
        "milvus_host": "localhost",
        "milvus_port": "19530",
        "topk": 5,
        "model_path": args.model_path,
        "h_lengh": args.win_size,
        # "model_path":r"C:\Users\鉴\.cache\huggingface\hub\models--BAAI--bge-m3\snapshots\5617a9f61b028005a4858fdac845db406aefb181",
        "max_length": 512,
        "search_field": "dense_vector",
        "search_metric_type": "IP",
        "search_nprobe": 10,
        # "query_prefix": "[CLS] | [context] | "
        "query_prefix": ""
    }

    # with open(config["qa_data_path"], "r") as f:
    #     qa_raw_data = json.load(f)
    # with open(config["diaId_content_map_path"], "r") as f:
    #     diaId_content_map = json.load(f)

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
        colleaction_name = f"bge_m3_win_dynamic_conv_data_{i}"
        if os.path.exists(os.path.join(args.results_path, "retri_"+colleaction_name)):
            with open(os.path.join(args.results_path,"retri_"+colleaction_name) , "r", encoding="utf-8") as f:
                retr_results = json.loads(f.read())
        else:
            with open(os.path.join(eval_data_base_path, eval_data_path), "r", encoding="utf-8") as f:
                qa_raw_data = json.load(f)
            with open(os.path.join(diaId_content_map_base_path, map_data_path), "r", encoding="utf-8") as f:
                diaId_content_map = json.load(f)

            # 过滤数据，忽略 category 为 2 和 evidence 为空的情况
            qa_data = [
                item for item in qa_raw_data[0]
                if "category" in item and "evidence" in item and "question" in item and "answer" in item
                and item["category"] != 2 and 0 < len(item["evidence"]) < 3
            ]

            
            # loading data
            client.load_data(collection_name=colleaction_name)

            retr_results = []
            # 添加 tqdm 进度条
            for qa in tqdm(qa_data, desc=f"Processing {colleaction_name} Pairs", unit="item"):
                query = config["query_prefix"] + qa["question"]
                query_embedding = model.embedding(query)

                res_list = client.search(
                    collection_name=colleaction_name,
                    query_embedding=query_embedding,
                    output_fields=["content", "dia_id"],
                    search_field=config["search_field"],
                    limit=config["topk"],
                    search_params={"metric_type": config["search_metric_type"], "params": {"nprobe": config["search_nprobe"]}},
                )

                retr_result = {
                    "qa": qa["question"],
                    "answer": qa["answer"],
                    "evidence": qa["evidence"],
                    "evidence_text": [diaId_content_map[dia_id] for dia_id in qa["evidence"] if dia_id in diaId_content_map],
                    "retrieve_text": [item["entity"]["content"] for item in res_list],
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
    parser.add_argument("--conv_data_path", type=str, default=r"D:\project\myproject\ChatSeeker\data\dynamic_win_conv_data", help="chunked data path")
    parser.add_argument("--qa_data_path", type=str, default=r"D:\project\myproject\ChatSeeker\data\eval_data", help="qa data path")
    parser.add_argument("--diaId_content_map_path", type=str, default=r"D:\project\myproject\ChatSeeker\data\diaID_map_data", help="diaId content map path")
    parser.add_argument("--results_path", type=str, default=r"D:\project\myproject\ChatSeeker\results", help="results path")
    parser.add_argument("--win_size", type=int, default=0, help="length of hierarchical attention")
    parser.add_argument("--model_path", type=str, default=r"C:\Users\鉴\.cache\huggingface\hub\models--BAAI--bge-m3\snapshots\5617a9f61b028005a4858fdac845db406aefb181", help="model path")
    base_args = parser.parse_args()

    
    new_args = argparse.Namespace(**vars(base_args))
    embedding_data(new_args)

    results = []
    new_args = argparse.Namespace(**vars(base_args))
    total_recall_ratio = evaulate_qa(new_args)
    # results.append(np.mean(total_recall_ratio))
    # print
    # results.append(total_recall_ratio)
    
    print("Dynamic Win Recall Ratio: ", total_recall_ratio)
    print("Mean Recall Ratio: ", np.mean(total_recaall_ratio))