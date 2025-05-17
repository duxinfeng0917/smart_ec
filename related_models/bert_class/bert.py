# coding: UTF-8
import os.path
import torch
import torch.nn as nn
from related_models.bert_class.pytorch_pretrained import BertModel, BertTokenizer


class Config(object):

    """配置参数"""
    # def __init__(self, dataset):
    def __init__(self):
        self.model_name = 'bert'

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.class_list = ['No', 'Yes']
        self.require_improvement = 100                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        print(f"num_classes：{len(self.class_list) }")
        self.num_epochs = 6
        # epoch数
        self.batch_size = 32
        # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 2e-5                                       # 学习率
        self.bert_path = '/home/ander/workspace/smart_ec/related_models/bert_class/bert_pretrain'

        # 验证数据时采用
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()

        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = self.fc(pooled)
        return out
