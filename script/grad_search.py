import sys

sys.path.extend(["./", "../"])
import os, re
import pynvml
import time
import random
import psutil
import subprocess
from utils.log import init_logger
import torch
from occupy import occupy

logger = init_logger()

LEARNIN_RATES = [1e-5, 1e-3, 3e-5]
NUM_HEADS = [1, 3, 4, 6, 12]
NUM_HIDDEN_LAYERS = [1, 2, 4, 6]
INTER_SIZES = [250, 300, 768]
CMD_FILE = "runs/search.sh"


def get_gpu_info(search_user):
    """获取gpu的info 主要是user目前使用的卡数以及空卡号

    Args:
        search_user (str): 搜索的用户

    Returns:
        dict: {used: user正在使用的卡数, empty: 空卡号}
    """
    used = 0
    empty = []

    pynvml.nvmlInit()
    cnt = pynvml.nvmlDeviceGetCount()
    for idx in range(cnt):
        handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
        useInfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pidInfo = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)  # 获取所有GPU上正在运行的进程信息

        if useInfo.used < 1000 * 1000000:  # 小于1G表示没人用
            empty.append(idx)

        for pro in pidInfo:
            name = psutil.Process(pro.pid).username()
            if name == search_user:
                used += 1

    return used, empty


def grid_search(search_user, max_use=4):
    for lr in LEARNIN_RATES:
        for num_head in NUM_HEADS:
            for num_hidden_layer in NUM_HIDDEN_LAYERS:
                for inter_size in INTER_SIZES:
                    used, empty = get_gpu_info(search_user)

                    if used >= max_use or not empty:  # 使用数超过max_use 或者目前没有空卡
                        while used >= max_use or not empty:
                            logger.info(f"{search_user} use {used} GPU. empty GPUs are {empty}. wait for 300s")
                            time.sleep(300)
                            used, empty = get_gpu_info(search_user)

                    else:
                        postfix = "-".join([str(i) for i in [lr, num_head, num_hidden_layer, inter_size]])
                        os.environ['CUDA_VISIBLE_DEVICES'] = str(empty[0])
                        logger.info(f"learning_rate = {lr}, num_heads = {num_head}, num_hidden_layers = {num_hidden_layer}, inter_size = {inter_size}")
                        logger.info(f"Start train on GPU{empty[0]}")
                        shell = f"""nohup python -u main.py --config_file config.cfg --postfix grad_search/{postfix} \
                        --inter_learning_rate {lr} --num_global_heads {num_head} --num_global_layers {num_hidden_layer} \
                        --global_interaction_size {inter_size} --inter_size {inter_size} \
                        --intra_interaction False --inter_interaction False --global_interaction True \
                        --local_interaction False --pre_interaction False \
                        > logs/grad_search/{postfix}.log 2>&1 &"""

                        tmp = os.system(shell)
                        if tmp == 0:
                            logger.info(f"save log to logs/grad_search/{postfix}.log")
                        else:
                            logger.error(f"Startup failed")
                        time.sleep(300)


def run_cmd(cmd, search_user="kdx", max_use=4):
    used, empty = get_gpu_info(search_user)

    if used >= max_use or not empty:  # 使用数超过max_use 或者目前没有空卡
        while used >= max_use or not empty:
            logger.info(f"{search_user} use {used} GPU. empty GPUs are {empty}. wait for 300s")
            time.sleep(300)
            used, empty = get_gpu_info(search_user)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(empty[0])
    # postfix = re.search(r'postfix\s*(\S+)', cmd).group(1)
    logger.info(f"cmd: {cmd}")
    logger.info(f"Start train on GPU{empty[0]}")
    tmp = os.system(cmd)
    if tmp == 0:
        logger.info(f"Startup sucessful")
    else:
        logger.error(f"Startup failed")
    os.system("python script/occupy.py >/dev/null 2>&1")  # 只需要占位 不需要输出


if __name__ == "__main__":
    logger.info("-" * 27 + "  Process ID {}, Process Parent ID {}  ".format(os.getpid(), os.getppid()) + "-" * 28)
    # grid_search("kdx")
    with open(CMD_FILE, "r", encoding="utf8") as f:
        for line in f.readlines():
            if line.strip():
                cmd = line.strip()
                flag = 1
                while flag:
                    try:
                        run_cmd(cmd, "kdx", 4)
                        flag = 0
                    except BaseException as e:
                        logger.error(e)
                        time.sleep(100)
                        logger.error(f"error wait 100s. run again")

    logger.info("-" * 45 + "  FINISH  " + "-" * 45)
