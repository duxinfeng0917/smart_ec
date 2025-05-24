# smart_ec
天池AI大模型赛——ELE AI算法大赛“赛道一：智慧养老—助老年群体智慧点餐”

语音识别(ASR)。接受音频输入，完成音频到文字的识别过程。例如，用户输入“wo zai jian fei, bang wo dian ge bu zhang pang de wai mai”，语音识别模块需输出“我在减肥，帮我点个不长胖的外卖”
领域分类(DOM)。接受输入的文本或语音“我在减肥，帮我点个不长胖的外卖”，判断是否与外卖相关“1/0”。
查询改写(QUE)。接受表意含糊的文本，按句式输出具体的点餐的品类或餐品。文本输入“我在减肥，帮我点个不长胖的外卖”，输出“帮我点沙拉”。由于天猫精灵在调用链路上需要判断调起那个服务接口，因此借用生产链路已有的DIS系统，需要将其改写为固定句式“我要点XXX”“帮我点XXX”等。

语音识别准确率ASR，领域分类准确率DOM，查询改写接受准确率QUE，三部分得分加权求总分得分：0.9612

## datasets
音频数据

## Finetune:微调

ASR参考链接: https://github.com/modelscope/FunASR

BERT参考链接: https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch

llm参考链接: https://github.com/hiyouga/LLaMA-Factory


## related_models:模型文件
向量模型: https://huggingface.co/BAAI/bge-base-zh
Rerank: https://huggingface.co/BAAI/bge-reranker-base
