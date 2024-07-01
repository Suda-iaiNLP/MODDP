import os
import logging

logger = logging.getLogger(__name__.replace('_', ''))

def check_empty_gpu(find_ours=11.5, threshold_mem=5000*1000000, to_used=True):
    """检查GPU的空闲状态，设定判断阈值，默认为5G。先找没人使用的GPU，如果全部GPU都有人用，则随机等待1~5分钟后找小于阈值的可用的GPU。
    这个设定主要是为了避免两个程序同时争抢一个GPU。  
    Args:
        find_ours (float, optional): 寻找多久. Defaults to 11.5.
        100*1000000 = 100 M
    Returns:
        GPU序号
    """
    try:
        import pynvml
        import time
        import random
        logger.warning(f"------------  Process ID {os.getpid()}, Process Parent ID {os.getppid()}  --------------------\n")

        start = time.time()
        find_times = 0
        pynvml.nvmlInit()
        cnt = pynvml.nvmlDeviceGetCount()
        available_gpus = os.getenv('VISIBLE_DEVICES','0,1,2,3,4,5,6,7').split(',') # 从哪些GPU中寻找可用的
        logger.warning(f'Find one available gpus from {available_gpus}...')
        while True:
            # 1~3分钟后开始下一次寻找可用的GPU
            for i in range(cnt):
                if str(i) not in available_gpus:
                    continue
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                if info.used < threshold_mem:  # 100*1000000:  # 2M    # 小于2M表示没人用
                    logger.warning(f'GPU-{i} used {info.used/1000000} M, so the program will use GPU-{i}.') 
                    return i
            cur_time = time.time()
            during = int(cur_time-start)+1
            if  during % 1800 == 0:
                find_times+=0.5
                logger.warning(f'已经经过{find_times}小时，还未找到可用的GPU。')
            
            if find_times > find_ours: # 如果超过 find_ours 个小时还没有分配到GPU，则停止程序
                logger.warning(f'已经经过{find_times}小时，还未找到可用的GPU，终止程序。')
                exit()
            # 随机停止 1~5分钟后开始下一次寻找可用的GPU
            random_time = random.randint(1,5)
            logger.warning(f'当前无可用的GPU，{random_time}分钟后开始下一次寻找可用的GPU。')
            time.sleep(random_time*60)
            if to_used: # 
                for i in range(cnt):
                    if str(i) not in available_gpus:
                        continue
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    if info.used < threshold_mem:  # 5G  # 小于5G表示虽然有人用，但是用的不多，我也能用
                        logger.warning(f'GPU-{i} used {info.used/1000000} M, so the program will use GPU-{i}.') 
                        return i
    except Exception as e:
        logger.warning(e)
        return 0
   
# if 'OMP_NUM_THREADS' not in os.environ.keys():
#     max_thread = 1
#     os.environ['OMP_NUM_THREADS'] = str(max_thread) # 该代码最多可建立多少个进程， 要放在最前边才能生效。防止占用过多CPU资源
#     logger.warning(f' set OMP_NUM_THREADS={max_thread}')

if 'CUDA_VISIBLE_DEVICES' not in os.environ.keys(): 
    gpu_number = check_empty_gpu()
    logger.warning(f' 未指定使用的GPU，如果存在{gpu_number}卡，则将使用 {gpu_number} 卡。不存在就使用CPU模式')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_number)
    # logger.warning(f' Set CUDA_VISIBLE_DEVICES={gpu_number}')
else:
    logger.warning(f' CUDA_VISIBLE_DEVICES={os.environ["CUDA_VISIBLE_DEVICES"]}，将使用 {os.environ["CUDA_VISIBLE_DEVICES"]} 卡。')

def run_init():
    """形式函数，作为此文件说明。由于一台机器可能存在多张GPU，但是大多数时候只想使用其中的一张GPU，而hf的Trainer默认使用的是多卡DP的模式，因此
    在不想命令行指定 CUDA_VISIBLE_DEVICES 变量时，可导入该文件帮助选择合适的一张GPU。
    该文件主要作用是帮助选择GPU。先判断是否已经存在环境变量 CUDA_VISIBLE_DEVICES， 如果不存在则调用 scripts.utils 中的 check_empty_gpu 函数从环境变量'VISIBLE_DEVICES'（默认为8)中选择使用的GPU。具体逻辑可见函数说明。
    """
    pass


