import json
from collections import Counter
import numpy as np
from utils import logger


class Vocab(object):
    ROOT, PAD, UNK = 0, 1, 2

    def __init__(self, config):
        label_file = config.label_file
        infile = open(label_file, "r", encoding="utf8")
        labels = json.load(infile)
        labels = sorted(labels.items(), key=lambda x: x[1])
        infile.close()

        self._id2rel = [i[0] for i in labels]

        reverse = lambda x: dict(zip(x, range(len(x))))
        self._rel2id = reverse(self._id2rel)
        self._rel2id['<root>'] = -1
        if len(self._rel2id) != len(self._id2rel):
            logger.error("serious bug: relations dumplicated, please check!")

    def rel2id(self, xs):
        if isinstance(xs, list):
            return [self._rel2id.get(x) for x in xs]
        return self._rel2id.get(xs)

    def id2rel(self, xs):
        if isinstance(xs, list):
            return [self._id2rel[x] for x in xs]
        return self._id2rel[xs]

    @property
    def rel_size(self):
        return len(self._id2rel)


def create_vocab(config):
    return Vocab(config)
