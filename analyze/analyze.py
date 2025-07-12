import json
import os

path_dir = r"D:\\project\\myproject\\ChatSeeker\\data\\dynamic_win_conv_data"

all_data = []
for file in os.listdir(path_dir):
    if file.endswith(".json"):
        file_path = os.path.join(path_dir, file)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)  # 解析 JSON 数据
            if isinstance(data, list):
                all_data.extend(data)  # 如果是列表，扩展到 all_data
            else:
                all_data.append(data)  # 如果是单个对象，追加到 all_data
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading {file_path}: {e}")
            
print(len(all_data))  # 打印 all_data 的长度

# 统计 all_data 中每个 dict 中 "h_length"
h_length_stats = {}
for item in all_data:
    if isinstance(item, dict) and "h_length" in item:
        h_length = item["h_length"]
        h_length_stats[h_length] = h_length_stats.get(h_length, 0) + 1

# 将统计结果保存到文件
output_file = "h_length_stats.json"
try:
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(h_length_stats, f, indent=4, ensure_ascii=False)  # 确保非 ASCII 字符正确保存
except IOError as e:
    print(f"Error writing to {output_file}: {e}")