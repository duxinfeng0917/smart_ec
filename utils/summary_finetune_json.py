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


# 百川微调数据格式
def generate_Baichuan2_fine_tune_json(origin_file_path, save_path):
    """
    根据提取的摘要数据生成微调数据
    :param origin_file_path:
    :param save_path:
    :return:
    """

    tune_list = []
    conversations = []

    # 读取电销对话意向识别文件
    df = pd.read_excel(origin_file_path, sheet_name='Sheet1', engine='openpyxl')
    # 获取对话列和一致列以及意向列
    dialog_class = df['子目录'].tolist()
    dialogs = df['对话内容'].tolist()
    problemDescs = df['对话内容总结 - 标注'].tolist()

    # 循环一致列，如果一致列中值不等于0，提取对话列中的对话
    for index, dialog in enumerate(dialogs):
        print(index, dialogs[index])
        # if index == 101: break
        if dialog_class[index] == "打款报备":
            prompt = f"""你是优秀坐席，你的任务是与用户通话完毕后写一个会话总结，要求最多120字。以下三个反引号之间的是通话记录，请对这通对话进行总结，总结需要重点关注会话内容：用户诉求，用户还款是否成功，不成功原因，用户还款期数、还款日期、还款金额、还款方式，坐席是否帮用户认领或勾稽，坐席的建议等。
            通话记录: ```
            {dialogs[index]}
            '''
            """
        elif dialog_class[index] == "查询":
            prompt = f"""你是优秀坐席，你的任务是与用户通话完毕后写一个会话总结，要求最多120字。以下三个反引号之间的是通话记录，请对这通对话进行概括，需要关注会话内容中以下要点：用户来电原因，是否申请提前还款？是否咨询结清金额、邮寄进度、解押流程、解押时效？是否核对金额、账户信息？是否咨询违约金问题？坐席回复内容，用户对坐席回复是否满意，将时间、金额等细节也在概括中写明。
            通话记录: ```
            {dialogs[index]}
            '''
            """
        elif dialog_class[index] == "还款方式":
            prompt = f"""你是优秀坐席，你的任务是与用户通话完毕后写一个会话总结，要求最多120字。以下三个反引号之间的是通话记录，请对这通对话进行概括，需要关注会话内容中是否涉及以下要点，如果不涉及则无需关注：用户来电原因，是否存在扣款问题？是否咨询还款期数、还款金额、对公还款、逾期罚息、逾期宽限？是否申请内部账户还款？坐席是否提供对公或者电子账户？坐席回复内容，用户对坐席回复是否满意等在概括中写明。
            通话记录: ```
            {dialogs[index]}
            '''
            """
        elif dialog_class[index] == "扣款时间":
            prompt = f"""你是优秀坐席，你的任务是与用户通话完毕后写一个会话总结，要求最多120字。以下三个反引号之间的是通话记录，请对这通对话进行概括，需要关注会话内容中以下要点，如果未涉及则忽略：用户卡资金状态，用户还款是否成功，用户是否咨询扣款卡号、扣款时间、解押时效、地址修改、还款日修改，用户对坐席回复是否满意等在概括中写明。
            通话记录: ```
            {dialogs[index]}
            '''
            """
        else:
            prompt = f"""你是金牌坐席，你的任务是与用户通话完毕后写一个会话摘要，要求最多120字。
            以下三个反引号之间的是通话记录，请对这通对话进行概括，重点关注会话内容、客户诉求，如通话中涉及具体金额、期数、日期等数据在概括中写明。
            通话记录: ```
            {dialogs[index]}
            '''
            """
        conversations = [{"from": "human", "value": prompt}, {"from": "gpt", "value": problemDescs[index]}]
        tune_list.append({"id": index, "conversations": conversations})

    # 将tune_list写入json文件
    with open(save_path, mode="w", encoding='utf-8') as f:
        f.write(json.dumps(tune_list, ensure_ascii=False, indent=4))


# Chatglm微调数据格式
def generate_Chatglm3_fine_tune_json(origin_file_path, save_path):
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

    # for idx, line_data in enumerate(data):
    #     usr_input = line_data.split('\t')[0].replace("原始：", "")
    #     model_result = line_data.split('\t')[1].replace("模型：", "")
    #     print(f"Idx:{idx}\t{usr_input}\t{model_result}")

    # 循环一致列，如果一致列中值不等于0，提取对话列中的对话
    for index, dialog in enumerate(fr_data):
        print(index)
        usr_input = dialog.split('\t')[0].replace("原始：","")
        model_output = dialog.split('\t')[1].replace("模型：", "")
        # usr_input = dialog.split('\t')[0]
        # model_output = dialog.split('\t')[0]
     #    prompt1 = f"""作为专业餐饮助手，请严格遵循以下规则处理需求：
     # 【处理规则】
     # 1. 输入解析原则：
     #    - 当语句包含菜品特征词（如：养胃/少油/易消化/热乎/软糯/现熬/家常等）且包含具体菜品名称时：
     #      ✔️ 必须提取所有菜品（保持原有表述）
     #      ✔️ 必须提取所有菜品特征词
     #    - 当语句只有菜品特征没有具体菜品词时：
     #      ✔️ 必须根据提取扩展菜品特征词推荐1个具体菜品
     #    - 当语句只有菜品名称时：
     #      ✔️ 直接输出原始菜品名称
     # 2. 输出格式规范：
     #    → 仅有菜品：菜品名
     #    → 有菜品和特征：特征词,菜品名
     #    → 无菜品时："特征词,生成菜品名"""
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
    Baichuan = False
    save_path = os.path.join(cur_path, "fine_tune_json",
                             current_datetime + "_Baichuan2_tune.json") if Baichuan else os.path.join(cur_path,
                                                                                                      "fine_tune_json",
                                                                                                      current_datetime + "_Chatglm3_tune.json")
    print(save_path)
    if Baichuan:
        generate_Baichuan2_fine_tune_json(origin_file_path, save_path)
    else:
        generate_Chatglm3_fine_tune_json(origin_file_path, save_path)
