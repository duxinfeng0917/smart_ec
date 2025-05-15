import time

from transformers import AutoModelForCausalLM, AutoTokenizer
import re

# model_name = "/home/ander/workspace/Qwen3-8B"
model_name = "/home/ander/workspace/LLaMA-Factory/output2/Qwen3-8B_lora_sft"
# 初始化模型和分词器
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def parse_response(response):
    """解析模型输出结果"""
    # 使用正则表达式匹配结构化输出
    item_match = re.search(r'菜品名称：([^\n]+)', response)
    query_match = re.search(r'推荐菜品：([^\n]+)', response)

    if item_match:
        items = [item.strip() for item in item_match.group(1).split(',')]
        return {"type": "items", "data": items}
    elif query_match:
        keywords = query_match.group(1).split()
        return {"type": "query", "data": keywords}
    else:
        # 兜底策略：直接返回原始响应
        return {"type": "fallback", "data": [response.strip()]}


def correct_asr_text(text):

    """ASR文本纠错函数"""
    system_prompt = """作为音频文本纠错专家，请严格遵循以下规则修正ASR识别结果：

【背景知识库】
需优先保障以下关键实体识别准确：
1. 音乐作品：狼戈的苹果香/还珠格格/周柏豪的宏愿/赵雷的少年锦时/桑甜的想你的时候好想告诉你/
2. 艺人名称：彭沛绮/罗云熙/章伟杰/徐炳超/一只舟/雪二
3. 菜式名称：荞麦凉面/鳕鱼炖豆腐/香椿苗炒蛋/西芹腰果虾/萝卜丝花生汤
4. 生活指令：设置为零档/关机尝试/关闭后/爬楼梯/巧克力一号电梯
5. 文化内容：豫剧春回杏花开/回杯记/六祖的故事/响堂山石窟/凤求凰

【修正规则】
1. 只做近似音修正及错别字修正
2. 格式规范不填加任何符号

请直接输出修正后的文本，不回答，不添加任何解释。"""
    # system_prompt = """作为音频文本纠错专家，请严格遵循以下规则修正ASR识别结果：
    #
    # 【背景知识库】
    # ['狼戈的苹果香', '爬楼梯', '按摩床设置为零档', '给我去过人了过来啦过人啦', '荞麦凉面', '公务员', '播放还珠格格歌曲', '彭沛绮', '播放戏剧手心手背都是肉', '罗云熙', '又到梧桐吐绿时', '卷心菜', '备胎最多', '桑甜的想你的时候好想告诉你', '关机尝试', '张杰的看呐看', '上海风味', '软糯的餐食', '黎明的翅膀', '凉拌菜', '豫剧春回杏花开', '章伟杰的有始无终', '回杯记', '巧克力一号电梯', '徐炳超的人生一回合', '一只舟', '赵雷的少年锦时', '串烧坏女孩', '十万毫升泪水', '落日亲吻云朵', '周柏豪的宏愿', '六祖的故事', '荷叶消暑粥', '鳕鱼炖豆腐', '关闭后', '香椿苗炒蛋', '雪二的太过遗憾', '枸杞小米粥', '时鲜小炒时蔬', '艾莎公主', '萝卜丝花生汤', '响堂山石窟', '郎的诱惑', '凤求凰', '你浅浅的微笑就像乌梅子酱', '西芹腰果虾']
    #
    # 【修正规则】
    # 1. 只做近似音和错别字进行纠正
    # 2. 参考背景知识对音乐、地点、人名、指定、菜品错误的进行纠正
    # 3. 直接输出修正后的文本，不回答，不添加任何解释。"""
    system_prompt = """作为音频文本纠错专家，请严格遵循以下规则修正ASR识别结果：

    背景知识库
    狼戈的苹果香 爬楼梯 按摩床设置为零档 心上的罗加 花僮的处处人间 给我去过人了过来啦过人啦 荞麦凉面 公务员 播放还珠格格歌曲 彭沛绮 播放戏剧手心手背都是肉 罗云熙 又到梧桐吐绿时 卷心菜 备胎最多 桑甜的想你的时候好想告诉你 关机尝试 张杰的看呐看 上海风味 软糯的餐食 周深唱的黎明的翅膀 凉拌菜 豫剧春回杏花开 章伟杰的有始无终 回杯记 巧克力一号电梯 徐炳超的人生一回合 一只舟 赵雷的少年锦时 串烧坏女孩 十万毫升泪水 落日亲吻云朵 周柏豪的宏愿 六祖的故事 荷叶消暑粥 鳕鱼炖豆腐 关闭后 香椿苗炒蛋 雪二的太过遗憾 枸杞小米粥 时鲜小炒时蔬 艾莎公主 萝卜丝花生汤 响堂山石窟 郎的诱惑 凤求凰 你浅浅的微笑就像乌梅子酱 西芹腰果虾

    修正规则
    1. 仅修正近似音和错别字
    2. 优先匹配背景知识库中的音乐、地点、人名、指令、菜品等专有名词
    3. 输出纯文本格式，不加任何标点符号或格式标记
    4. 直接输出最终修正结果，禁止任何解释说明回答
    
    【示例】
    输入：天猫精灵奇数是双数还是单数
    输出：天猫精灵奇数是双数还是单数
    
    输入：播放罗莱斯贝约谈
    输出：播放罗莱斯贝约谈
    """


    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text}
    ]

    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking = False
    )
    model_inputs = tokenizer([chat_text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=100,
        temperature=0.1,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    return tokenizer.batch_decode(
        generated_ids[:, model_inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )[0].strip()


def process_user_input(user_input):
    """处理用户输入并返回结构化结果"""
    # system_prompt = """作为专业餐饮助手，请严格遵循以下规则处理需求：
    #  【处理规则】
    #  1. 输入解析原则：
    #     - 当语句包含菜品特征词（如：养胃/少油/易消化/热乎(现熬)/软糯/家常等）且包含具体菜品名称时：
    #       ✔️ 必须提取所有菜品（保持原有表述）
    #       ✔️ 必须提取所有菜品特征词
    #     - 当语句只有菜品特征没有具体菜品词时：
    #       ✔️ 必须根据菜品特征词推荐1个具体菜品(如:软烂点,推荐:南瓜蒸百合/汤汤水水,推荐:传统清汤面/好嚼的,推荐:肉沫炖豆腐等)
    #     - 当语句只有菜品名称时：
    #       ✔️ 直接输出原始菜品名称（禁止添加特征词）
    #  2. 输出格式规范：
    #     → 仅有菜品：[菜品名]
    #     → 有菜品和特征：[特征词],[菜品名]
    #     → 无菜品时："[特征词],[生成菜品名]"""
    # system_prompt = """作为专业餐饮助手，请严格遵循以下规则处理需求：
    #  【处理规则】
    #  1. 输入解析原则：
    #     - 当语句包含菜品特征词（如：养胃/少油/易消化/热乎(现熬)/软糯/家常等）且包含具体菜品名称时：
    #       ✔️ 必须提取所有菜品（保持原有表述）
    #       ✔️ 必须提取所有菜品特征词
    #     - 当语句只有菜品特征没有具体菜品词时：
    #       ✔️ 必须根据菜品特征词推荐1个具体菜品(如:软烂点,推荐:南瓜蒸百合/汤汤水水,推荐:传统清汤面/好嚼的,推荐:肉沫炖豆腐等)
    #     - 当语句只有菜品名称时：
    #       ✔️ 直接输出原始菜品名称（禁止添加特征词）
    #  2. 输出格式规范：
    #     → 仅有菜品：菜品名
    #     → 有菜品和特征：特征词,菜品名
    #     → 无菜品时："特征词,生成菜品名
    #     → 禁止输出为空"""
    # system_prompt = f"""作为专业餐饮助手，请严格遵循以下规则处理需求：
    #      【处理规则】
    #      1. 输入解析原则：
    #         - 当语句包含菜品特征词（如：养胃/少油/易消化/热乎/软糯/现熬/家常等）且包含具体菜品名称时：
    #           ✔️ 必须提取所有菜品（保持原有表述）
    #           ✔️ 必须提取所有菜品特征词
    #         - 当语句只有菜品特征没有具体菜品词时：
    #           ✔️ 必须根据菜品特征词推荐1个具体菜品
    #         - 当语句只有菜品名称时：
    #           ✔️ 直接输出原始菜品名称
    #      2. 输出格式规范：
    #         → 仅有菜品（必须提取所有菜品）：菜品名
    #         → 有菜品和特征：特征词,菜品名
    #         → 无菜品时："特征词,生成菜品名"""
    system_prompt = f"""作为专业老年人餐饮助手，请严格遵循以下规则处理需求：
            【处理规则】
            1. 输入解析原则：
               - 当语句包含菜品特征词（如：养胃/少油/易消化/热乎(现做)/软糯/现熬/家常等）且包含具体菜品名称时：
                 ✔️ 必须提取所有菜品（保持原有表述）
                 ✔️ 必须提取所有菜品特征词(如:菜/汤/清蒸/清炒等)
               - 当语句只有菜品特征没有具体菜品词时：
                 ✔️ 必须根据菜品特征(不超过5个)推荐1个具体菜品
               - 当语句只有菜品名称时：
                 ✔️ 直接输出原始菜品名称
            2. 输出格式规范：
               → 仅有菜品：菜品名
               → 有菜品和特征：特征词,菜品名
               → 无菜品时："特征词(不超过5个),生成菜品名"""


    #system_prompt = f"""作为专业餐饮助手，请严格按规则处理需求：
    #1. 当语句包含具体要求及菜品名称时，提取语句中的要求菜品特征(例如:养胃/少油/易消化/热乎/软糯/现熬/家常等）和所有菜品（保持原始表述）
    #2. 没有具体菜品时，根据具体要求菜品特征，给出1个具体菜品, (如:少油少盐,推荐南瓜蒸百合/汤汤水水,推荐:传统清汤面/好嚼的,推荐:肉沫炖豆腐等)
    #3. 严格使用以下格式：
    #    有菜品和具体要求时 → 具体要求,菜品名
    #    无菜品时 → 具体要求,菜品名"""



    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]

    # 生成模型输入
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # 生成输出
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=100,
        # temperature=0.1,
        # repetition_penalty=1.2,
        # do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    # 解码并解析结果
    response = tokenizer.batch_decode(
        generated_ids[:, model_inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )[0].strip()
    final_ret = response.split('：')[-1]
    # final_ret = response.split('：')
    # print(final_ret)
    # correct_asr_text = response.split('：')[1].split('\n')[0]
    # dishes_text = response.split('：')[-1]
    # return correct_asr_text, dishes_text
    return final_ret



if __name__ == "__main__":
    test_cases = [
        "要个番茄鸡蛋盖饭吧",
        "我要一份大头菜和番茄汤",
        "有没有适合糖尿病人的套餐",
        "来点下饭的川菜"
    ]
    # with open('./prompt_optimizer05051513.txt', mode='r', encoding='utf-8') as f:
    #     test_cases = f.readlines()
    # print(len(test_cases),test_cases[:3])

    for case in test_cases:
        start_time = time.time()
        usr_case = case.strip('\n')
        print(f"用户输入：{usr_case}")
        model_input = process_user_input(usr_case)
        model_input = model_input.replace("天猫精灵", "")
        print(f"模型输出：{model_input}")
        # ret_text = correct_asr_text(usr_case)
        # print(ret_text)
        # correct_asr, result = process_user_input(case)
        # result = process_user_input(case)
    #
    #     print(f" 菜单：{result}，耗时：{time.time()- start_time}")
    #     # if result["type"] == "items":
    #     #     print(f"提取菜品：{result['data']}")
    #     # elif result["type"] == "query":
    #     #     print(f"改写查询：{' '.join(result['data'])}")
    #     # else:
    #     #     print(f"原始响应：{result['data'][0]}")
    #     print("-" * 50)


