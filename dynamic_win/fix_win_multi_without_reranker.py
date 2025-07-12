import json
from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig
from tqdm import tqdm
import numpy
import random
import os
from typing import List
import numpy as np
import ast

import sys
sys.path.append("/share/nijian")
from FlagEmbedding import FlagReranker

# ✅ 配置参数
config = {
    "data_path": "/share/nijian/project/myproject/dynamic_win/data/conv_data_0.json",
    "out_put_path": "/share/nijian/project/myproject/dynamic_win/result/dynamic_conv_data_0.json",
    "data_dir":"/share/nijian/project/myproject/dynamic_win/data",
    "output_dir":"/share/nijian/project/myproject/dynamic_win/result_fix_win_without_reranker",
    "prompt_file": "/share/nijian/project/myproject/dynamic_win/data/prompt.txt",
    "prompt_file_refine": "/share/nijian/project/myproject/dynamic_win/data/prompt_2.txt",
    "rerank_model": "/share/shared_models/models--BAAI--bge-reranker-v2-m3/snapshots/12e974610ba9083ed95f3edf08d7e899581f4de4",
    "max_win": 15
}

# ✅ llama model
model = "/share/shared_models/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/945c8663693130f8be2ee66210e062158b2a9693"
# model = "/share/shared_models/Qwen2.5-72B-Instruct"
pipe = pipeline(model, backend_config=TurbomindEngineConfig(tp=4, session_len=4096))
gen_config = GenerationConfig(top_p=0.9,
                              top_k=40,
                              temperature=0.2,
                              max_new_tokens=512)

# BGE rerank model
model = FlagReranker(config["rerank_model"], devices="cuda:4")

# ✅ 读取 Prompt
with open(config["prompt_file"], "r", encoding="utf-8") as f:
    prompt_template = f.read().strip()
with open(config["prompt_file_refine"], "r", encoding="utf-8") as f:
    prompt_template_refine = f.read().strip()


def dynamic_win_llama(dialogue_history, current_query, batch_size=10):
    """llama for select win"""
    formatted_history = [f"[{idx}] {turn}" for idx, turn in enumerate(dialogue_history)]
    conv_indo = f"<dialogue_history>:{formatted_history}\n<current_query>:{current_query},please find the index"
    input_msgs = []
    message = [
        {"role": "system", "content": prompt_template},
        {"role": "user", "content": conv_indo},
    ]
    input_msgs.append(message)
    response = pipe(input_msgs, gen_config)

    return eval(response[0].text)

def filter_win_bge_rerank(query: str, selected_win_content: List[str]) -> List[str]:
    # 对每个query的候选文档进行重排
    pairs = [(query, dialog) for dialog in selected_win_content]
    scores = model.compute_score(pairs, normalize=True)
    if isinstance(scores, np.ndarray):
        scores = scores.tolist()
    scored_dialogs = list(zip(selected_win_content, scores))
    # scored_docs.sort(key=lambda x: x[1], reverse=True)
    filterd_win = [dialog for dialog, score in scored_dialogs if score > 0.35]
    return filterd_win

def refine_select_win(current_query, selected_win_content):
    # llama,对filterd_win生成hype Query,将深层的语义关系显示化
    """llama for select win"""
    try:
        selected_win_content = ["\n".join(dialog) for dialog in selected_win_content]
    except:
        selected_win_content = selected_win_content
    conv_indo = f"<dialogue_history>:{selected_win_content}\n<current_query>:{current_query},please give the instructions"
    input_msgs = []
    message = [
        {"role": "system", "content": prompt_template_refine},
        {"role": "user", "content": conv_indo},
    ]
    input_msgs.append(message)
    for retry_count in range(3):
        try:
            response = pipe(input_msgs, gen_config)
            result = eval(response[0].text)
            return result
        except Exception as e:
            print(f"Error: {e}")
            print(f"Error: Failed to parse response. Retry {retry_count+1}/3")

    print("Max retries exceeded.")
    return [-1]

def hierarchical_atten_embedding(chunks_list, out_put_path, start_idx=0, processed_data=[]):
    """从指定索引开始处理，支持断点续存"""
    for i in tqdm(range(start_idx, len(chunks_list)), desc="Processing chunks", unit="chunk"):
        dialogue_history_list = [chunks_list[j]["content"] for j in range(max(0, i - config["max_win"]), i)]
        current_query = chunks_list[i]["content"]
        # win_index = dynamic_win_llama(dialogue_history_list, current_query)
        win_index = [2]

        if win_index[0] == -1 or len(win_index) == 0:
            h_lengh = 0
            selected_win_content = [""]
        else:
            # selected_win_index = win_index[0]
            # h_lengh = len(dialogue_history_list) - selected_win_index
            h_lengh = 2
            selected_win_content = [chunks_list[j]["content"] for j in range(max(0, i - h_lengh), i)]
            # selected_win_content = filter_win_bge_rerank(current_query, selected_win_content)
            selected_win_content = refine_select_win(current_query, selected_win_content)
        
        if selected_win_content[0] != -1:
            chunks_list[i]["hierarchical_content"] = "\n".join(selected_win_content) + " | [context] | " + chunks_list[i]["content"]
        else:
            chunks_list[i]["hierarchical_content"] = chunks_list[i]["content"]

        chunks_list[i]["h_length"] = h_lengh

        # 定期保存已处理的数据
        if i % 10 == 0:
            with open(out_put_path, "w", encoding="utf-8") as f:
                result_data = processed_data + chunks_list[start_idx:i + 1]
                json.dump(result_data, f, ensure_ascii=True, indent=4)

    return chunks_list


if __name__ == "__main__":

    for file in os.listdir(config["data_dir"]):
        if file.endswith(".json"):
            data_path = os.path.join(config["data_dir"], file)
            output_data_path = os.path.join(config["output_dir"], "dynamic_"+file)
        
            # ✅ 断点续存：检查是否已有部分处理数据
            if os.path.exists(output_data_path):
                with open(output_data_path, "r", encoding="utf-8") as f:
                    try:
                        processed_data = json.load(f)
                        last_processed_idx = len(processed_data)  # 计算已处理的索引
                    except json.JSONDecodeError:
                        processed_data = []
                        last_processed_idx = 0
            else:
                processed_data = []
                last_processed_idx = 0

            with open(data_path, "r", encoding="utf-8") as f:
                chunks_list = json.load(f)
            
            print(f"🔄 断点续存：从索引 {last_processed_idx} 继续处理...")
            hia_data = hierarchical_atten_embedding(chunks_list, output_data_path ,start_idx=last_processed_idx, processed_data=processed_data)

            result_data = processed_data + hia_data[last_processed_idx:]

            # ✅ 处理完成后最终保存
            with open(output_data_path, "w", encoding="utf-8") as f:
                json.dump(result_data, f, ensure_ascii=True, indent=4)

            print("✅ 处理完成，数据已保存！")
        