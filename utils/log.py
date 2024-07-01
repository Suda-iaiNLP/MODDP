#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   log.py
@Time    :   2021/11/17 21:15:35
@Author  :   sakurakdx
@Contact :   sakurakdx@163.com
from https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/utils/logging.py
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


logger = logging.getLogger()

def init_logger(
    log_file=None,
    log_file_level=logging.NOTSET,
    rotate=False,
    log_level=logging.INFO,
):
    """初始化logger

    Args:
        log_file (str, optional): 保存log的路径. Defaults to None.
        log_file_level ([type], optional): TODO 做啥的. Defaults to logging.NOTSET.
        rotate (bool, optional): TODO 做啥的. Defaults to False.
        log_level ([type], optional): 设置输出log的等级. Defaults to logging.INFO.

    Returns:
        [type]: [description]
    """
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(log_level)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != "":
        if rotate:
            file_handler = RotatingFileHandler(
                log_file, maxBytes=1000000, backupCount=10
            )
        else:
            file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger
