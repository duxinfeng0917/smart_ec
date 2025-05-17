import pandas as pd
import json
import time
import os
import datetime

# 获取当前日期和时间
current_datetime = datetime.datetime.now().strftime("%Y-%m-%d")

# 读取、存储路径
cur_path = os.path.abspath(os.path.dirname(__file__))
origin_file_path = r"model_sft0514.txt"



# Chatglm微调数据格式
def generate_fine_tune_json(origin_file_path, save_path):
    """
    根据提取的摘要数据生成微调数据
    :param origin_file_path:
    :param save_path:
    :return:
    """

    tune_list = []
    conversations = []
    with open(origin_file_path, mode='r', encoding='utf-8') as f:
        fr_data = f.readlines()

    # 循环一致列，如果一致列中值不等于0，提取对话列中的对话
    for index, dialog in enumerate(fr_data):
        print(index)
        usr_input = dialog.split('\t')[0].replace("原始：","")
        model_output = dialog.split('\t')[1].replace("模型：", "")
        prompt = f"""作为专业老年人餐饮助手，请严格遵循以下规则处理需求：
        【处理规则】
        1. 输入解析原则：
           - 当语句包含菜品特征词（如：养胃/少油/易消化/热乎/软糯/现熬/家常等）且包含具体菜品名称时：
             ✔️ 必须提取所有菜品（保持原有表述）
             ✔️ 必须提取所有菜品特征词(如:菜/汤/清蒸/清炒等)
           - 当语句只有菜品特征没有具体菜品词时：
             ✔️ 必须根据菜品特征(同义扩展词不超过5个)推荐1个具体菜品
           - 当语句只有菜品名称时：
             ✔️ 直接输出原始菜品名称
        2. 输出格式规范：
           → 仅有菜品：菜品名
           → 有菜品和特征：特征词,菜品名
           → 无菜品时："特征词,生成菜品名"""
        conversations = {"instruction": prompt, "input": usr_input, "output":model_output}
        tune_list.append(conversations)

    # 将tune_list写入json文件
    with open(save_path, mode="w", encoding='utf-8') as f:
        f.write(json.dumps(tune_list, ensure_ascii=False, indent=4) + "\n")


if __name__ == "__main__":
    save_path = os.path.join(cur_path,"fine_tune_json",current_datetime + "_tune.json")
    print(save_path)
    generate_fine_tune_json(origin_file_path, save_path)
