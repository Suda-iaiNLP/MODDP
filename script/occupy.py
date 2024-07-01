#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   occupy.py
@Time    :   2023/06/14 18:06:37
@Author  :   sakurakdx
@Contact :   sakurakdx@163.com
@Intro   :   占位程序
'''

import torch
import time 

def occupy():
    tmp = torch.tensor(1)
    tmp.cuda()
    # start_time = time.time()
    time.sleep(100)

    tmp.cpu()
    del tmp
    torch.cuda.empty_cache()
    time.sleep(100)
    # print(time.time() - start_time)

if __name__ == "__main__":
    occupy()