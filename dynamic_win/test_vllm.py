import json
import requests
import os
import sys
from typing import List, Dict, Any, Tuple
import numpy as np
from tqdm import tqdm
from collections import Counter  # 添加这个导入

# sys.path.append("/share/nijian")
# from FlagEmbedding import FlagReranker


class DynamicWindowProcessor:
    def __init__(self, config: Dict[str, Any]):
        """
        初始化动态窗口处理器
        
        Args:
            config: 配置字典，包含所有必要的路径和参数
        """
        self.config = config
        # self.reranker = FlagReranker(config["rerank_model"], devices="cuda:4")
        self.reranker = None
        
        # 读取prompt模板
        with open(config["prompt_file"], "r", encoding="utf-8") as f:
            self.prompt_template = f.read().strip()
        with open(config["prompt_file_refine"], "r", encoding="utf-8") as f:
            self.prompt_template_refine = f.read().strip()
    
    def _call_vllm_api(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
        """
        调用vLLM API进行推理
        
        Args:
            prompt: 输入的提示词
            max_tokens: 最大生成token数
            temperature: 采样温度
            
        Returns:
            生成的文本响应
        """
        url = "http://localhost:8000/v1/completions"
        payload = {
            "model": "/usr1/project/models/llama3.1-70B",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["text"].strip()
        except Exception as e:
            print(f"API调用失败: {e}")
            return ""
    
    def dynamic_win_llm(self, dialogue_history: List[str], current_query: str, 
                       vote_times: int = 5, conf_threshold: float = 0.6) -> Tuple[List[int], float]:
        """
        🔥 多次 LLM 推理 + 投票集成，减少不确定性
        
        Args:
            dialogue_history: 对话历史列表
            current_query: 当前查询
            vote_times: 投票次数
            conf_threshold: 置信度阈值
            
        Returns:
            选择的窗口索引列表和置信度
        """
        formatted_history = [f"[{idx}] {turn}" for idx, turn in enumerate(dialogue_history)]
        conv_info = f"<dialogue_history>:{formatted_history}\n<current_query>:{current_query}, please find the index"
        
        # 构建完整的prompt
        full_prompt = f"{self.prompt_template}\n\nUser: {conv_info}\n\nAssistant:"
        
        predictions = []
        for _ in range(vote_times):
            response = self._call_vllm_api(full_prompt, temperature=0.3)  # 稍微增加温度以获得多样性
            try:
                result = eval(response)
                predictions.append(tuple(result))
            except Exception as e:
                print(f"解析LLM响应失败: {e}")
                continue
        
        if not predictions:
            return [-1], 0.0  # 全部失败时返回
        
        # 投票统计
        counts = Counter(predictions)
        best_win, freq = counts.most_common(1)[0]
        confidence = freq / vote_times
        
        # 判断置信度
        if confidence < conf_threshold:
            print(f"⚠️ 低置信度 ({confidence:.2f})，回退固定窗口")
            return [-1], confidence
        
        print(f"✅ 高置信度 ({confidence:.2f})，选择窗口: {best_win}")
        return list(best_win), confidence
    
    def filter_win_rerank(self, query: str, selected_win_content: List[str]) -> List[str]:
        """
        使用BGE reranker过滤窗口内容
        
        Args:
            query: 查询文本
            selected_win_content: 选择的窗口内容列表
            
        Returns:
            过滤后的窗口内容列表
        """
        if not selected_win_content:
            return []
            
        pairs = [(query, dialog) for dialog in selected_win_content]
        scores = self.reranker.compute_score(pairs, normalize=True)
        
        if isinstance(scores, np.ndarray):
            scores = scores.tolist()
        
        scored_dialogs = list(zip(selected_win_content, scores))
        filtered_win = [dialog for dialog, score in scored_dialogs if score > 0.35]
        return filtered_win
    
    def refine_select_win(self, current_query: str, selected_win_content: List[str]) -> List[str]:
        """
        使用LLM refine选择的窗口内容
        
        Args:
            current_query: 当前查询
            selected_win_content: 选择的窗口内容列表
            
        Returns:
            精炼后的窗口内容列表
        """
        if not selected_win_content:
            return [-1]
            
        try:
            # 尝试处理嵌套列表
            processed_content = []
            for item in selected_win_content:
                if isinstance(item, list):
                    processed_content.append("\n".join(item))
                else:
                    processed_content.append(str(item))
            selected_win_content = processed_content
        except:
            selected_win_content = [str(item) for item in selected_win_content]
        
        conv_info = f"<dialogue_history>:{selected_win_content}\n<current_query>:{current_query},please give the instructions"
        
        # 构建完整的prompt
        full_prompt = f"{self.prompt_template_refine}\n\nUser: {conv_info}\n\nAssistant:"
        
        for retry_count in range(3):
            try:
                response = self._call_vllm_api(full_prompt)
                result = eval(response)
                return result
            except Exception as e:
                print(f"Error: {e}")
                print(f"Error: Failed to parse response. Retry {retry_count+1}/3")
        
        print("Max retries exceeded.")
        return [-1]
    
    def hierarchical_atten_embedding(self, chunks_list: List[Dict[str, Any]], 
                                   output_path: str, start_idx: int = 0, 
                                   processed_data: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        分层注意力嵌入处理核心函数
        
        Args:
            chunks_list: 数据块列表
            output_path: 输出路径
            start_idx: 开始索引
            processed_data: 已处理的数据
            
        Returns:
            处理后的数据块列表
        """
        if processed_data is None:
            processed_data = []
            
        for i in tqdm(range(start_idx, len(chunks_list)), desc="Processing chunks", unit="chunk"):
            # 获取对话历史
            dialogue_history_list = [chunks_list[j]["content"] for j in range(max(0, i - self.config["max_win"]), i)]
            current_query = chunks_list[i]["content"]
            
            # 使用LLM选择窗口（多次推理投票）
            win_index, confidence = self.dynamic_win_llm(dialogue_history_list, current_query)
            
            if win_index[0] == -1 or len(win_index) == 0:
                h_length = 0
                selected_win_content = [""]
            else:
                selected_win_index = win_index[0]
                h_length = len(dialogue_history_list) - selected_win_index
                selected_win_content = [chunks_list[j]["content"] for j in range(max(0, i - h_length), i)]
                
                # 使用reranker过滤
                # selected_win_content = self.filter_win_rerank(current_query, selected_win_content)
                
                # 使用LLM refine
                # selected_win_content = self.refine_select_win(current_query, selected_win_content)
            
            # 构建分层内容
            if selected_win_content and selected_win_content[0] != -1:
                chunks_list[i]["hierarchical_content"] = "\n".join(selected_win_content) + " | [context] | " + chunks_list[i]["content"]
            else:
                chunks_list[i]["hierarchical_content"] = chunks_list[i]["content"]
            
            chunks_list[i]["h_length"] = h_length
            chunks_list[i]["confidence"] = confidence  # 保存置信度信息
            
            # 定期保存已处理的数据
            if i % 10 == 0:
                with open(output_path, "w", encoding="utf-8") as f:
                    result_data = processed_data + chunks_list[start_idx:i + 1]
                    json.dump(result_data, f, ensure_ascii=True, indent=4)
        
        return chunks_list
    
    def test_vllm_determinism(self, test_prompt: str, test_times: int = 10):
        """测试vLLM的确定性"""
        results = []
        for i in range(test_times):
            response = self._call_vllm_api(test_prompt, temperature=0.0)
            results.append(response)
            print(f"第{i+1}次结果: {response}")
        
        # 检查结果是否一致
        unique_results = set(results)
        print(f"共产生了 {len(unique_results)} 种不同的结果")
        if len(unique_results) == 1:
            print("✅ 结果完全一致")
        else:
            print("⚠️ 结果存在差异，建议保留投票机制")
    
    def process_files(self):
        """
        处理所有文件的主函数
        """
        for file in os.listdir(self.config["data_dir"]):
            if file.endswith(".json"):
                data_path = os.path.join(self.config["data_dir"], file)
                output_data_path = os.path.join(self.config["output_dir"], "dynamic_" + file)
                
                # 断点续存：检查是否已有部分处理数据
                if os.path.exists(output_data_path):
                    with open(output_data_path, "r", encoding="utf-8") as f:
                        try:
                            processed_data = json.load(f)
                            last_processed_idx = len(processed_data)
                        except json.JSONDecodeError:
                            processed_data = []
                            last_processed_idx = 0
                else:
                    processed_data = []
                    last_processed_idx = 0
                
                with open(data_path, "r", encoding="utf-8") as f:
                    chunks_list = json.load(f)
                
                print(f"🔄 断点续存：从索引 {last_processed_idx} 继续处理...")
                hia_data = self.hierarchical_atten_embedding(chunks_list, output_data_path, 
                                                           start_idx=last_processed_idx, 
                                                           processed_data=processed_data)
                
                result_data = processed_data + hia_data[last_processed_idx:]
                
                # 处理完成后最终保存
                with open(output_data_path, "w", encoding="utf-8") as f:
                    json.dump(result_data, f, ensure_ascii=True, indent=4)
                
                print("✅ 处理完成，数据已保存！")
    

if __name__ == "__main__":
    config = {
        "data_dir": "./data",
        "output_dir": "./result",
        "prompt_file": "./data/prompt.txt",
        "prompt_file_refine": "./data/prompt_2.txt",
        "rerank_model": "/share/shared_models/models--BAAI--bge-reranker-v2-m3/snapshots/12e974610ba9083ed95f3edf08d7e899581f4de4",
        "max_win": 15
    }
    
    processor = DynamicWindowProcessor(config)
    processor.test_vllm_determinism("请问你是谁？", test_times=5)