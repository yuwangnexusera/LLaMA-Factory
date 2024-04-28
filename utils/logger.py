# logger_config.py
import logging

logger = logging.getLogger('root_logger')
logger.setLevel(logging.INFO)

# 创建一个处理程序
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 创建一个格式化程序
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# 将处理程序添加到logger
logger.addHandler(console_handler)
