import os
import asyncio
import time
import json,random
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union


from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess


# LLM API相关库
from openai import AsyncOpenAI
from dotenv import load_dotenv

# 加载环境变量(.env文件)
load_dotenv()


# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


root_path = os.path.dirname(os.path.abspath(__file__))
asr_model_path = os.path.join(root_path, "related_models", "asr_model", "finetune_context")
audios_path = os.path.join(root_path, "datasets", "Audios")
A_asr_turbo = os.path.join(root_path, "datasets", "asr_results", "A_funasr_context_default.txt")

def asr_transcribe(audio_list: List[str], save_path: List[str]):
    """
    音频转录
    :param audio_list:
    :param save_path:
    :return:
    """
    fw = open(save_path, mode='w', encoding='utf-8')
    time_consume = 0
    asr_model = AutoModel(model=asr_model_path)

    for idx, audio in enumerate(audio_list):
        # if idx > 500: break
        audio_path = os.path.join(audios_path, audio)

        start = time.time()
        # result = transcribe_audio(audio_path)
        result = asr_model.generate(input=audio_path, language='zh-cn',
                                       hotword="/home/ander/workspace/sec/smart_elderly_care/dataset/Hotword.txt")

        fw.writelines(audio_path + '\t' + str(result[0]['text']) + '\n')
        audio_time = time.time() - start
        time_consume += audio_time
    fw.close()
    print(f"平均处理时间: {time_consume / audio_sum:.2f}秒/音频")


if __name__ == "__main__":
    audios_list = os.listdir(audios_path)

    audio_sum = len(audios_list)
    print(len(audios_list), audios_list[:3])
    asr_transcribe(audios_list, A_asr_turbo)




