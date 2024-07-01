#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   utils.py
@Time    :   2021/10/20 19:29:30
@Author  :   sakurakdx
@Contact :   sakurakdx@163.com
"""

import sys

sys.path.extend(["./", "../"])
import torch
import numpy as np
import random
import pickle
import json
from utils.log import logger


def sort_dict(d, mode="k", reverse=False):
    """对字典按照key或者value排序

    Args:
        d (dict): 待排序的字典对象
        mode (str, optional): 'k'-->键排序, 'v'-->值排序 . Defaults to 'k'.
        reverse (bool, optional): True为降序排列. Defaults to False.

    Returns:
        list(tuple): 返回一个list, 里边tuple第一个为key, 第二个为value
    """
    # assert type(d) == dict, 'sort_dict仅支持对dict排序, 当前对象为:{}'.format(type(d))
    if mode == "k":
        return [(i, d[i]) for i in sorted(d, reverse=reverse)]
    elif mode == "v":
        return sorted(d.items(), key=lambda kv: kv[1], reverse=reverse)
    else:
        print("排序失败")
        return d


def load_pkl(file_path):
    """加载pkl

    Args:
        file_path (str): 加载文件路径

    Returns:
        dict: 加载之后的dict
    """
    with open(file_path, "rb") as f:
        return pickle.load(f)


def dump_pkl(obj, file_path):
    """将obj保存为pkl文件

    Args:
        obj (dict): 保存的数据
        file_path (str): 保存的路径一般是.pkl
    """
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def load_json(file_path):
    """加载json

    Args:
        file_path (str): 加载的路径

    Returns:
        list: 数据的列表
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj, file_path):
    """保存json

    Args:
        obj ([type]): data
        file_path (str): 保存路径
    """
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4)


def same_seed(seed):
    """设置随机种子，使得结果可复现

    Args:
        seed (int): 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
        
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 固定随机状态
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_seed(seed=None):
    """初始化随机种子大小，并进行打印"""
    if seed is None:
        seed = random.randint(0, 10000)
    same_seed(seed)
    logger.info(f"The random seed is set to {seed}")


def is_whitespace(c):
    """判断字符串是否是空白的

    Args:
        c (str): 要判断的字符串

    Returns:
        bool: 空白返回True，否则返回False
    """
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def is_(c):
    """判断乱码

    Args:
        c (str): 字符串
    """
    if (c == "\u202c" 
        or c == "\ufeff" 
        or c == "\x80" 
        or c == "\x95"
        or c == "\x97" 
        or c == "\x96" 
        or c == "\x98" 
        or c == "\u200b"
        or c == "\u202a" 
        or c == "\ufffd" 
        or c == "\u202a"):
        return True
    return False


def whitespace_tokenize(text):
    """清除空格并返回按空格划分的token

    Args:
        text (str): 需要分割的文本

    Returns:
        list: 文本按空格划分的token
    """
    text = text.strip()
    if not text:
        return []

    token = text.split()
    return token


def to_list(tensor):
    """将Tensor转为list

    Args:
        tensor (torch.Tensor): Tensor

    Returns:
        list: Tensor转为list
    """
    return tensor.detach().cpu().tolist()


def batch_padding(inputs, padding='longest', max_length=None, padding_id=0, padding_side='right'):
    """将一个batch的数据padding到同样长度, 返回 (real_length, padded)

    Args:
        inputs (List): 待padding的数据
		padding (str, optional): padding的策略， ['longest', 'max_length']. Defaults to 'longest'.
        max_length(int, optional): 如果 padding=='max_length', 则padding到指定的长度。
        padding_id (int, optional): padding的id. Defaults to 0.
        padding_side (str, optional): padding的方向， ['left','right']. Defaults to 'left'.
    """
    real_length = [len(i) for i in inputs]
    if padding == 'longest':
        max_length = max(real_length)
    padding_ids = [[padding_id]*(max_length-i) for i in real_length]
    if padding_side == 'right':
        padded = [inputs[idx]+padding_ids[idx] for idx in range(len(inputs))]
    else:
        padded = [padding_ids[idx]+inputs[idx] for idx in range(len(inputs))]
        
    return real_length, padded