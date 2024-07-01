import os, sys
from .dataloader import *
from utils import logger
from .vocab import create_vocab
from torch.utils.data import DataLoader
from time import time
import pickle
from utils import load_pkl


class Data(object):
    def __init__(self, config):
        self.config = config

        # create instances
        self.train_instances = read_corpus(config.train_file, config.max_insts_num)

        self.dev_instances = read_corpus(config.dev_file, config.max_insts_num)
        dev_file_name = self.config.dev_file.split("/")[-1]
        config.dev_file = self.config.save_dev_dir + '/' + dev_file_name
        self.write_instances(config.dev_file, self.dev_instances)

        self.test_instances = read_corpus(config.test_file, config.max_insts_num)
        test_file_name = self.config.test_file.split("/")[-1]
        config.test_file = self.config.save_dev_dir + '/' + test_file_name
        self.write_instances(config.test_file, self.test_instances)

        self.vocab = create_vocab(config)

        self.collector = DataCollator(config)
        self.show_data_summary()
        self.set_up()

    @staticmethod
    def write_instances(out_file, instances):
        with open(out_file, "w", encoding="utf-8") as outf:
            for i, inst in enumerate(instances):
                for index, relation in enumerate(inst.pred_relations):
                    arc = str(relation["x"]) + "," + str(relation["y"])
                    rel = relation["type"]
                    inst.info[index][6] = arc
                    inst.info[index][7] = rel
                    outf.write("\t".join(inst.info[index]) + "\n")

    def set_up(self):
        logger.info("-" * 42 + "  Data Set Up  " + "-" * 43)
        logger.info("Start build dataset...")
        self.train_set = MyDataset(self.config, self.train_instances)
        self.train_set.label2id(self.vocab)
        self.dev_set = MyDataset(self.config, self.dev_instances)
        self.dev_set.label2id(self.vocab)
        self.test_set = MyDataset(self.config, self.test_instances)
        self.test_set.label2id(self.vocab)
        logger.info("Sucessfully build dataset.")
        start_time = time()

        logger.info("Convert insts to fetures")
        video_features = load_pkl(self.config.video_features)
        audio_features = load_pkl(self.config.audio_features)
        DataCollator.convert_insts_to_features(
            self.train_instances, self.collector.text_processor, self.collector.audio_processor, video_features, audio_features
        )
        DataCollator.convert_insts_to_features(
            self.dev_instances, self.collector.text_processor, self.collector.audio_processor, video_features, audio_features
        )
        DataCollator.convert_insts_to_features(
            self.test_instances, self.collector.text_processor, self.collector.audio_processor, video_features, audio_features
        )
        logger.info(f"Sucessfully convert dataset. Cost {time() - start_time}s")
        logger.info("-" * 100)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            collate_fn=self.collector.train_wrapper,
            num_workers=self.config.num_works,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dev_set,
            batch_size=self.config.dev_batch_size,
            shuffle=False,
            collate_fn=self.collector.train_wrapper,
            num_workers=self.config.num_works,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.config.dev_batch_size,
            shuffle=False,
            collate_fn=self.collector.train_wrapper,
            num_workers=self.config.num_works,
        )

    def show_data_summary(self):
        logger.info("-" * 43 + "  Data INFO  " + "-" * 44)
        logger.info(f"Train Instance Number: {(len(self.train_instances))}")
        logger.info(f"Valid Instance Number: {(len(self.dev_instances))}")
        logger.info(f"Test  Instance Number: {(len(self.test_instances))}")

        logger.info(f"relation: {self.vocab._id2rel}")
        logger.info(f"relation size: {len(self.vocab._rel2id)}")
