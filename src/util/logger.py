"""
@project: machine_game
@File   : logger.py
@Desc   :
@Author : gql
@Date   : 2024/6/20 14:39
"""
import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

# ANSI escape sequences for colors
RESET_COLOR = "\033[0m"
COLORS = {
    'DEBUG': "\033[34m",  # 蓝色
    'INFO': "\033[32m",  # 绿色
    'WARNING': "\033[33m",  # 黄色
    'ERROR': "\033[31m",  # 红色
    'CRITICAL': "\033[35m",  # 紫红色
}
WHITE_COLOR = "\033[97m"
GRAY_COLOR = "\033[37m"


class ColoredFormatter(logging.Formatter):
    def format(self, record):
        """
        格式化日志输出的颜色
        """
        # 方案1：全部着色
        '''
            log_color = COLORS.get(record.levelname, RESET_COLOR)
            # 包裹整个消息体以应用颜色
            record.msg = f"{log_color}{record.msg}{RESET_COLOR}"
            # 格式化日志信息，包括时间戳、名称、等级和消息内容
            formatted_record = super().format(record)
            # 包裹整个格式化的记录以应用颜色
            return f"{log_color}{formatted_record}{RESET_COLOR}"
        '''
        # 方案2：仅对日志等级、文件名、行号和函数名着色
        log_color = COLORS.get(record.levelname, RESET_COLOR)
        record.msg = f"{WHITE_COLOR}{record.msg}{RESET_COLOR}"
        record.threadName = f'{GRAY_COLOR}{record.threadName}{RESET_COLOR}'
        formatted_record = super().format(record)
        if record.asctime:  # asctime是通过logging.Formatter的formatTime方法生成的
            formatted_record = formatted_record.replace(f'{record.asctime}',
                                                        f'{GRAY_COLOR}{record.asctime}{RESET_COLOR}')
        formatted_record = formatted_record.replace(f'[{record.levelname}]',
                                                    f'{log_color}[{record.levelname}]{RESET_COLOR}')
        formatted_record = formatted_record.replace(f'{record.filename}:{record.lineno} {record.funcName}()',
                                f'{log_color}{record.filename}:{record.lineno} {record.funcName}(){RESET_COLOR}')
        formatted_record = formatted_record
        return formatted_record


def get_logger(log_to_console=True, log_to_file=False, log_file_path='app.log',
               use_rotation=False, use_timed_rotation=False, level=logging.INFO,
               max_bytes=10 * 1024 * 1024, backup_count=5, rotation_interval='D'):
    """
    创建和配置logger对象，可以根据需要将日志输出到控制台和/或文件，并支持文件大小轮转和时间轮转
    :param log_to_console: 控制是否输出到控制台
    :param log_to_file: 控制是否输出到文件
    :param log_file_path: 指定日志文件的路径
    :param use_rotation: 是否启用基于文件大小的日志轮转(用于RotatingFileHandler)
    :param use_timed_rotation: 是否启用基于时间的日志轮转，和use_rotation参数只能二选一
    :param level: 设置日志的最低输出级别
    :param max_bytes: 设置日志文件的最大大小(用于RotatingFileHandler)
    :param backup_count: 设置保留的备份文件数量
    :param rotation_interval: 设置时间轮转的时间间隔(用于TimedRotatingFileHandler)，例如S, M, H, D, W0~W6)
    :return: 自定义的logger
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    # 定义日志格式
    log_format_str = '%(asctime)s %(threadName)s [%(levelname)s] %(filename)s:%(lineno)d %(funcName)s()  %(message)s'
    formatter = ColoredFormatter(log_format_str)
    # 清除默认的处理器
    if logger.hasHandlers():
        logger.handlers.clear()
    # 控制台日志处理器
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    # 文件日志处理器
    if log_to_file:
        if use_rotation:
            # 使用RotatingFileHandler，根据文件大小进行日志轮转
            file_handler = RotatingFileHandler(log_file_path, maxBytes=max_bytes, backupCount=backup_count)
        elif use_timed_rotation:
            # 使用TimedRotatingFileHandler，根据时间间隔进行日志轮转
            file_handler = TimedRotatingFileHandler(log_file_path, when=rotation_interval, backupCount=backup_count)
        else:
            # 普通的FileHandler
            file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(level)
        # 使用普通格式，文件中不需要彩色
        file_formatter = logging.Formatter(log_format_str)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    return logger


def test():
    # 示例用法
    logger = get_logger(
        log_to_console=True,
        log_to_file=False,
        log_file_path='application.log',
        use_rotation=True,  # 设置为 True 使用大小轮转
        max_bytes=1 * 1024 * 1024,  # 每个日志文件最大 1 MB
        backup_count=3,  # 保留3个备份
        level=logging.DEBUG
    )
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")


if __name__ == '__main__':
    test()
    pass
