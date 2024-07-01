import sys

sys.path.extend(["./", "../"])
import os
import pynvml
import time
import psutil
from loguru import logger
import argparse


argparser = argparse.ArgumentParser()
argparser.add_argument("--cmd_file", default="../run.sh")
argparser.add_argument("--max_use", type=int, default=4)
args, extra_args = argparser.parse_known_args()


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
    time.sleep(300)
    # os.system("python script/occupy.py > /dev/null 2>&1")  # 只需要占位 不需要输出


if __name__ == "__main__":
    logger.info("-" * 27 + "  Process ID {}, Process Parent ID {}  ".format(os.getpid(), os.getppid()) + "-" * 28)
    # grid_search("kdx")
    with open(args.cmd_file, "r", encoding="utf8") as f:
        for line in f.readlines():
            if line.strip() and not line.startswith("#"):  # 非空
                cmd = line.strip()
                flag = 1
                while flag:
                    try:
                        run_cmd(cmd, "kdx", args.max_use)
                        time.sleep(100)
                        flag = 0
                    except BaseException as e:
                        logger.error(e)
                        time.sleep(100)
                        logger.error(f"error wait 100s. run again")

    logger.info("-" * 45 + "  FINISH  " + "-" * 45)
