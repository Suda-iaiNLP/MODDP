from torch.utils.data import Dataset
import torch
import numpy as np


class MyDataset(Dataset):

    def __init__(self, config, instances):
        super().__init__()
        self.config = config
        self.instances = instances

    def __getitem__(self, index):
        return self.instances[index]

    def __len__(self):
        return len(self.instances)

    def label2id(self, vocab):
        for inst in self.instances:
            gold_arc_labels = np.zeros([len(inst.gold_arcs)])
            gold_rel_labels = np.zeros([len(inst.gold_arcs)])
            for idy, _ in enumerate(inst.gold_arcs):
                gold_arc_labels[idy] = inst.gold_arcs[idy][0]

            for idy, _ in enumerate(inst.gold_rels):
                rel = inst.gold_rels[idy][0]
                # if idy == 0:
                #     gold_rel_labels[idy] = -1
                # else:
                gold_rel_labels[idy] = vocab.rel2id(rel)

            inst.gold_arc_labels = gold_arc_labels
            inst.gold_rel_labels = gold_rel_labels

class Instance:
    def __init__(self):
        self.id = ""
        self.edus = []
        self.relations = []
        self.image_indexs = []

        self.speakers = []
        