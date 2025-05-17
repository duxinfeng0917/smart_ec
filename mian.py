import os
import asyncio
import time
import json,random
import torch
import pandas as pd
import numpy as np
import jieba
import faiss

from typing import List, Dict, Any, Optional, Union
from sec_config import system_logger, AUDIO_DIR, AUDIO_File, asr_model_path,hot_words_path
import related_models.bert_class.bert as bert_model
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

from transformers import AutoModelForCausalLM, AutoTokenizer
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from rank_bm25 import BM25Okapi

class Model_Predict:
    # def __init__(self, asr_model_path, class_model_path, llm_model_path, embedding_model_path, rerank_model_path, hot_words_path, category_text_path):
    def __init__(self, asr_model_path,
                     hot_words_path):

        # 判断GPU是否可用
        if torch.cuda.is_available():
            print("GPU is available!")
        else:
            print("GPU is not available.")

        # 加载ASR模型
        self.asr_model = AutoModel(model=asr_model_path, disable_update=True)
        self.hot_worlds = hot_words_path

        # 加载分类模型
        self.bert_config = bert_model.Config()
        self.bert_model = bert_model.Model(self.bert_config).to(self.bert_config.device)

        self.bert_model.load_state_dict(torch.load("/home/ander/workspace/smart_ec/related_models/bert_class/bert250516.ckpt",
                       map_location='cuda:0'))
        self.bert_model.eval()

        # Qwen8B模型
        self.llm_model_path = "/home/ander/workspace/LLaMA-Factory/output0516_3/Qwen3-8B_lora_sft"
        # 初始化模型和分词器
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            self.llm_model_path,
            torch_dtype="auto",
            device_map="auto"
        )
        self.llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_model_path)

        self.system_prompt = f"""作为专业老年人餐饮助手，请严格遵循以下规则处理需求：
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

        # 召回相关配置
        self.stopwords = {'来', '玩', '吧', '的', '一份', '来份', "要碗", '要个', "想吃", '一个', '份', '人份', "天猫", "精灵", '天猫精灵'}
        # 构建语料库
        category_text_path = "/home/ander/workspace/smart_ec/sec_config/dim_ai_exam_food_category_filter_out.txt"
        self.df = pd.read_csv(category_text_path, sep="\t")
        self.corpus = self.df.apply(self.preprocess, axis=1).tolist()
        self.bm25 = BM25Okapi(self.corpus)

        embedding_model_path = '/home/ander/yx_projects/yx_web_search/cpu/func/bge_base'
        # 向量及精排模型
        self.emb_model = BGEM3FlagModel(embedding_model_path, use_fp16=False)
        self.reranker = FlagReranker("/home/ander/yx_projects/yx_rerank_bge/gpu/bge_rerank_process/rerank_infer/bge_m3_reranker", use_fp16=True)

        # 预计算所有文本的Embedding
        with open(category_text_path, mode='r', encoding='utf-8') as f:
            fr_data = f.readlines()
        self.texts_emb = self.emb_model.encode(fr_data[1:], batch_size=12)["dense_vecs"]
        self.texts_emb = self.texts_emb.astype('float32')  # 转换为float32格式

        # 构建FAISS索引
        self.faiss_index = faiss.IndexFlatIP(self.texts_emb.shape[1])
        self.faiss_index.add(self.texts_emb)


    def asr_transcribe(self, audio_path):
        """
        ASR文本转写
        :param audio_path: 传入音频路径
        :param hotwords_path: 热词路径
        :return: 转写结果
        """

        start = time.time()
        asr_result = self.asr_model.generate(input=audio_path, language='zh-cn', hotword=str(self.hot_worlds))
        system_logger.info(f"asr处理时间:{str(round(time.time()-start,4))}")
        return str(asr_result[0]['key']), str(asr_result[0]['text'])

    def bert_classify(self, query):
        """
        分类
        :param query: 文本
        :return: 分类标签
        """
        PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号

        # 1、emb操作
        contents = []
        pad_size = 32
        token = self.bert_config.tokenizer.tokenize(query)
        token = [CLS] + token
        seq_len = len(token)
        mask = []
        token_ids = self.bert_config.tokenizer.convert_tokens_to_ids(token)  # token操作
        if pad_size:
            # 填充操作，用[0]填充
            if len(token) < pad_size:
                mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                token_ids += ([0] * (pad_size - len(token)))
            else:
                mask = [1] * pad_size
                token_ids = token_ids[:pad_size]
                seq_len = pad_size
        contents.append((token_ids, -1, seq_len, mask))

        # 2、转换成tensor
        def _to_tensor(datas):
            x = torch.LongTensor([_[0] for _ in datas]).to(self.bert_config.device)  # 对应上文的，token_ids
            y = [_[1] for _ in datas]  # 对应上文的，int(label)
            # pad前的长度(超过pad_size的设为pad_size)
            seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.bert_config.device)  # 对应上文的，seq_len
            mask = torch.LongTensor([_[3] for _ in datas]).to(self.bert_config.device)  # 对应上文的，mask
            return (x, seq_len, mask), y

        texts, labels = _to_tensor(contents)

        with torch.no_grad():
            outputs = self.bert_model(texts)
            predict = torch.max(outputs.data, 1)[1].cpu().numpy().tolist()
            scores = torch.softmax(outputs, dim=1).cpu().numpy().tolist()  # 获取预测得分
            # print(predict, type(predict[0]))
            # print(scores[0][predict[0]], len(scores[0]))

        return predict[0]

    def llm_process_input(self, user_input):
        """
        文本信息提取
        :param user_input:
        :return:
        """

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input}
        ]

        # 生成模型输入
        text = self.llm_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        model_inputs = self.llm_tokenizer([text], return_tensors="pt").to(self.llm_model.device)

        # 生成输出
        generated_ids = self.llm_model.generate(
            **model_inputs,
            max_new_tokens=100,
            # temperature=0.1,
            # repetition_penalty=1.2,
            # do_sample=False,
            pad_token_id=self.llm_tokenizer.eos_token_id
        )

        # 解码并解析结果
        response = self.llm_tokenizer.batch_decode(
            generated_ids[:, model_inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0].strip()
        final_ret = response.split('：')[-1]
        return final_ret

    # 预处理函数
    def preprocess(self, row):
        categories = ' '.join([
            str(row['category_name']),
            str(row['cate_1_name']),
            str(row['cate_2_name'])
        ])
        full_text = f"{str(row['item_name'])} {categories}"
        words_all = [w for w in jieba.lcut(full_text) if w not in self.stopwords]

        # 使用搜索引擎模式（细粒度）
        words_search = jieba.cut_for_search(full_text)

        # 优化点6：合并结果并保留原始词
        words_all = list(words_all) + list(words_search) + list(str(row['item_name']))

        merge_words = list(words_all)
        merge_words.append(str(row['item_name']))

        return merge_words

    def hybrid_search(self, query, top_n=10):
        """混合检索函数"""
        # 生成预处理query
        dummy_row = pd.Series({
            'item_name': query,
            'category_name': '',
            'cate_1_name': '',
            'cate_2_name': ''
        })
        query_words = self.preprocess(dummy_row)

        # 计算BM25得分
        bm25_scores = self.bm25.get_scores(query_words)

        # 计算BGE得分
        query_emb = self.emb_model.encode(query)["dense_vecs"]
        query_emb = query_emb.astype('float32')
        if len(query_emb.shape) == 1:
            query_emb = query_emb.reshape(1, -1)
        bge_scores = np.dot(query_emb, self.texts_emb.T).flatten()

        # 归一化处理
        bm25_normalized = (bm25_scores - bm25_scores.min()) / (bm25_scores.ptp() + 1e-8)
        bge_normalized = (bge_scores - bge_scores.min()) / (bge_scores.ptp() + 1e-8)

        # 混合得分（可根据需要调整权重）
        combined_scores = 0.2 * bm25_normalized + 0.8 * bge_normalized

        # 获取最终结果
        self.df['combined_score'] = combined_scores
        return self.df.sort_values('combined_score', ascending=False).head(top_n)

    def recall_result(self, org_input, model_input):

        results = self.hybrid_search(model_input)

        rerank_list = []
        for idx, item in results.iterrows():
            # print(idx, item['item_name'])
            rerank_list.append([model_input, item['item_name']])

        rerank_score = self.reranker.compute_score(rerank_list, normalize=True)
        max_idx = rerank_score.index(max(rerank_score))

        return results.iloc[max_idx]['item_name']



def find_audio_files(folder_path: str, audio_path: str) -> tuple:
    """
    查找指定文件夹中的音频文件

    参数:
        folder_path: text文件

    返回:
       音频文件路径列表
    """
    system_logger.info(f"正在查找音频文件: {folder_path}")
    system_logger.info(f"正在查找音频路径: {audio_path}")

    audio_list = os.listdir(audio_path)

    filename = []
    taskname = []
    with open(folder_path, mode='r', encoding='utf-8') as f:
        data = f.readlines()

    for idx, audio_name in enumerate(data):
        if idx == 0: continue
        audio_name = audio_name.split('\t')[0]+'.wav'
        if audio_name in audio_list:
            filename.append(os.path.join(audio_path,audio_name))

    system_logger.info(f"找到 {len(filename)} 个音频文件")
    return filename

def main_process() -> None:
    """主处理流程"""

    system_logger.info("===== 开始处理流程 =====")

    # 保存最终结果
    fw_save = open("./B_final.txt", mode='w', encoding='utf-8')

    # 1. 查找音频文件
    system_logger.info(AUDIO_File)
    filenames = find_audio_files(AUDIO_File, AUDIO_DIR)

    # 2. 加载所有模型
    run_model = Model_Predict(asr_model_path=asr_model_path, hot_words_path=hot_words_path)

    # 3. 处理流程
    for index, audio_item in enumerate(filenames):
        # 4. ASR转写文本
        asr_id, asr_text = run_model.asr_transcribe(audio_item)
        system_logger.info(f"4.ASR结果: {asr_text}")

        # 5. 文本分类处理
        text_label = run_model.bert_classify(asr_text)
        system_logger.info(f"5.分类结果: {text_label}")

        if text_label == 1:
            # 6. 模型改写
            model_asr_text = run_model.llm_process_input(asr_text)
            system_logger.info(f"6.大模型改写结果: {model_asr_text}")
            # 7. 召回答案
            call_result = run_model.recall_result(asr_text, model_asr_text)
            system_logger.info(f"7.召回结果: {call_result}")
            # 8. 保存文件
            fw_save.writelines(f"{asr_id}\t{asr_text}\t{text_label}\t{call_result}\n")
        else:
            fw_save.writelines(f"{asr_id}\t{asr_text}\t{text_label}\n")
    fw_save.close()
    system_logger.info("===== 文件保存完成 =====")

if __name__ == "__main__":
    main_process()




