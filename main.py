import sys, os

sys.path.extend(["../../", "../", "./"])

import argparse
from utils import *
from data.data import Data
from module import Model
from script.trainer import Trainer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


if __name__ == '__main__':
    # args
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='config.cfg')
    argparser.add_argument('--thread', default=4, type=int, help='thread num')
    argparser.add_argument('--use-cuda', action='store_true', default=True)
    args, extra_args = argparser.parse_known_args()

    config = Configurable(args.config_file, extra_args)

    set_seed(config.seed)
    torch.set_num_threads(args.thread)

    logger.info("-" * 43 + "  GPU State  " + "-" * 44)
    # torch version
    logger.info(f"Torch Version: {torch.__version__}")

    # gpu state
    gpu = torch.cuda.is_available()
    logger.info(
        f"GPU available: {gpu}",
    )
    logger.info(f"CuDNN: {torch.backends.cudnn.enabled}")
    config.use_cuda = False
    if gpu and args.use_cuda:
        config.use_cuda = True
    logger.info(f"GPU using status: {config.use_cuda}")
    # logger.info(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

    data = Data(config)
    model = Model(config, data)
    trainer = Trainer(config, model, data)

    if config.train:
        trainer.train()

    if config.predict:
        logger.info("Start predict" + "=" * 30 + ">")
        dev_out_file = config.dev_file + ".pred"
        dev_uas_metric, dev_las_metric = trainer.predict(config.dev_file, dev_out_file, stage="test", ckpt_path=config.ckpt_path)
        dev_uas_f1 = dev_uas_metric["f1_score"] * 100
        dev_las_f1 = dev_las_metric["f1_score"] * 100
        logger.info(f"DEV:")
        logger.info(f"UAS F1 score: {dev_uas_f1: 5.4f}, LAS F1 score: {dev_las_f1: 5.4f}")

        test_out_file = config.test_file + ".pred"
        test_uas_metric, test_las_metric = trainer.predict(config.test_file, test_out_file, stage="test", ckpt_path=config.ckpt_path)
        test_uas_f1 = test_uas_metric["f1_score"] * 100
        test_las_f1 = test_las_metric["f1_score"] * 100
        logger.info(f"TEST:")
        logger.info(f"UAS F1 score: {test_uas_f1: 5.4f}, LAS F1 score: {test_las_f1: 5.4f}")

        logger.info(f"{dev_uas_f1:5.2f} {dev_las_f1:5.2f} {test_uas_f1:5.2f} {test_las_f1:5.2f}")

    logger.info("-" * 45 + "  FINISH  " + "-" * 45)
