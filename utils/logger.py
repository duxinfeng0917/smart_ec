import os
import logging

from logging import handlers

level_dict = {
    0: logging.DEBUG,
    1: logging.INFO,
    2: logging.WARNING,
    3: logging.ERROR,
    4: logging.CRITICAL,

    10: logging.DEBUG,
    20: logging.INFO,
    30: logging.WARNING,
    40: logging.ERROR,
    50: logging.CRITICAL
}


LOG_DIR = os.environ["LOG_DIR"] = "./logs/"
LOG_NAME = os.environ["LOG_NAME"] = "log.log"


class CustomFormatter(logging.Formatter):
    """
    项目版本信息
    """

    def __init__(self, fmt, version_field):
        super().__init__(fmt)
        self.version_field = version_field

    def format(self, record):
        record.version_field = self.version_field
        return super().format(record)


class LocalLogger:
    def __init__(self, log_dir, log_name, version_field: str = "parsefile", level=1, when='MIDNIGHT', backupCount=7,
                 fmt='[%(asctime)s] %(levelname)s [%(version_field)s] %(module)s:%(lineno)d: %(message)s',
                 buffer_capacity=20):
        self.log_dir = log_dir
        self.log_name = log_name
        self.level = level
        self.version_field = version_field

        self.filename = os.path.join(log_dir, log_name)
        self.logger = logging.getLogger("{}".format(log_name))
        self.logger.setLevel(level_dict.get(level))
        self.logger.propagate = False
        os.makedirs(log_dir, exist_ok=True)

        format_str = CustomFormatter(fmt, version_field=self.version_field)
        self.sh = logging.StreamHandler()
        self.sh.setFormatter(format_str)
        self.th = handlers.TimedRotatingFileHandler(filename=self.filename, when=when,
                                                    backupCount=backupCount, encoding='utf-8')
        self.th.setFormatter(format_str)
        self.logger.addHandler(self.sh)
        self.logger.addHandler(self.th)

    def setLevel(self, level):
        self.logger.setLevel(level)

    def info(self, prefix='', string=''):
        self.logger.info(str(prefix) + str(string))

    def warning(self, prefix='', string=''):
        self.logger.warning(str(prefix) + str(string))

    def debug(self, prefix='', string=''):
        self.logger.debug(str(prefix) + str(string))

    def error(self, prefix='', string=''):
        self.logger.error(str(prefix) + str(string))

    def exception(self, prefix='', string=''):
        self.logger.exception(str(prefix) + str(string))



if __name__ == "__main__":
    local_logger = LocalLogger('.', 'project_logs')
