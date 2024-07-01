#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   config.py
@Time    :   2021/11/17 21:14:23
@Author  :   sakurakdx
@Contact :   sakurakdx@163.com
"""

from configparser import ConfigParser
import os
import re
from .log import init_logger, logger


class Configurable:

    def __init__(self, path, extra_args=None) -> None:
        """读取config文件中的配置 并将额外的参数添加到类属性中

        Args:
            path (str): config文件路径
            extra_args (list, optional): 额外的参数. Defaults to None.
        """
        self.config = ConfigParser()
        self.config.read(path, encoding="utf-8")

        if extra_args:
            self._add_extra_args(extra_args)

        # 加载文件中的参数
        sections = self.config.sections()
        for section in sections:
            items = self.config.items(section)
            self._add_attr(items)

        self._prepare()
        self.print()

    def print(self):
        logger.info("-"*27 + "  Process ID {}, Process Parent ID {}  ".format(os.getpid(), os.getppid()) + "-"*28)
        
        self.config.write(open(self.config_file,'w'))
        logger.info("-" * 42 + "  Config Info  " + "-" * 43)
        for key, value in self.__dict__.items():
            logger.info(f"{key}: {str(value)}")

    def _add_extra_args(self, extra_args):
        """添加额外的参数"""
        extra_args = dict([
            (k[2:], v) for k, v in zip(extra_args[0::2], extra_args[1::2])
        ])
        for section in self.config.sections():
            for k, v in self.config.items(section):
                if k in extra_args:
                    v = extra_args.pop(k)
                    self.config.set(section, k, v)

    def _add_attr(self, items):
        """将参数添加到类属性中

        Args:
            items (dict): {k: 属性名 v: 属性值}
        """
        num_value = re.compile(r"^[-+]?[0-9]+\.[0-9]+$")
        for k, v in items:
            try:
                v = int(v)
            except:
                try:
                    v = float(v)
                except:
                    if "None" in v:
                        v = None
                    elif "True" == v or "False" == v:
                        v = True if "True" == v else False

            self.__setattr__(k, v)

    def _prepare(self):
        """
        build save dir and file
        """
        import time

        save_dir = self.config.get("Save", "save_dir")
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        
        try:
            save_dev_dir = self.config.get("Save", "save_dev_dir")
        except:
            save_dev_dir = save_dir + "/dev"
            self.save_dev_dir = save_dev_dir
        if not os.path.isdir(save_dev_dir):
            os.makedirs(save_dev_dir)

        try:
            tensorboard_log_dir = self.config.get("Save", "tensorboard_log_dir")
        except:
            tensorboard_log_dir = save_dir + "/tensorboard"
            self.tensorboard_log_dir = tensorboard_log_dir
        if not os.path.isdir(tensorboard_log_dir):
            os.makedirs(tensorboard_log_dir)
            
        time_str = time.strftime('%Y-%m-%d-%H-%M')
        self.log_file = os.path.join(save_dir, time_str + ".log")
        init_logger(self.log_file)
