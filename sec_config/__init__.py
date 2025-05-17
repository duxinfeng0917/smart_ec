import os
import json
from pathlib import Path

# 根据 APP_ENV 选择加载的 .env 文件
APP_ENV = os.getenv('APP_ENV', 'test')

# 加载相关配置文件
config_path = os.path.join(os.path.dirname(__file__), "config_env.json")
with open(config_path, mode="r", encoding='utf-8') as f:
    config_env = json.load(f)

LOG_NAME = config_env[APP_ENV]['log_name']
LOG_DIR = Path(os.path.join(os.path.dirname(__file__), "..", config_env[APP_ENV]['log_dir'])).resolve()

AUDIO_File = os.path.join(os.path.dirname(__file__), config_env[APP_ENV]['audio_file'])
AUDIO_DIR = Path(os.path.join(os.path.dirname(__file__), "..", config_env[APP_ENV]['audio_dir'])).resolve()
AUDIO_DIR = Path(os.path.join(os.path.dirname(__file__), "..", config_env[APP_ENV]['audio_dir'])).resolve()

asr_model_path = Path(os.path.join(os.path.dirname(__file__), "..", config_env[APP_ENV]['asr_model'])).resolve()
hot_words_path = Path(os.path.join(os.path.dirname(__file__), config_env[APP_ENV]['hot_words'])).resolve()

class_model_path = Path(os.path.join(os.path.dirname(__file__), "..", config_env[APP_ENV]['class_model'])).resolve()
llm_model_path = Path(os.path.join(os.path.dirname(__file__), "..", config_env[APP_ENV]['llm_model'])).resolve()
dom_text_path = Path(os.path.join(os.path.dirname(__file__), config_env[APP_ENV]['dom_text'])).resolve()
embedding_model_path = Path(os.path.join(os.path.dirname(__file__), "..", config_env[APP_ENV]['embedding_model'])).resolve()
rerank_model_path = Path(os.path.join(os.path.dirname(__file__), "..", config_env[APP_ENV]['rerank_model'])).resolve()

from utils.logger import LocalLogger

local_logger = LocalLogger(LOG_DIR, LOG_NAME)
system_logger = local_logger.logger

system_logger.info(f"当前环境APP_ENV={APP_ENV}")
system_logger.info(f"环境配置路径：{config_path}")
system_logger.info(f"环境配置详情：{config_env}")
system_logger.info(f"音频文件路径：{AUDIO_File}")
system_logger.info(f"音频路径：{AUDIO_DIR}")
system_logger.info(f"ASR模型路径：{asr_model_path}")
system_logger.info(f"热词路径：{hot_words_path}")
system_logger.info(f"分类模型路径：{class_model_path}")
system_logger.info(f"LLM模型路径：{llm_model_path}")
