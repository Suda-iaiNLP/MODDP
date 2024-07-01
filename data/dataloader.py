#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   Dataloader.py
@Time    :   2021/11/17 21:54:03
@Author  :   sakurakdx
@Contact :   sakurakdx@163.com
"""

from .dataset import *
import torch
import numpy as np
from transformers import AutoProcessor
import re, os
import librosa
import av
from utils import load_pkl
from utils import logger


def read_info(inf):
    info = []
    ids = set()
    for line in inf:
        line = line.strip().split("\t")
        # 处理(x,y) 因为有的不是以","为分割符
        index = line[5]
        if line[6].replace(index, "", 1).replace(",", "", 1):
            head = line[6].replace(index, "", 1).replace(",", "", 1)
        else:
            head = 0
        line[6] = (head, index)
        id = line[0]
        if len(ids) == 0:
            ids.add(id)

        if id not in ids:
            yield info
            ids.add(id)
            info = []
            info.append(line)
        else:
            info.append(line)
    if len(info) > 0:
        yield info


def info2inst(info):
    inst = Instance()

    inst.info = info
    inst.id = info[0][2]
    inst.original_edus = [{"text": data[4], "speaker": data[3]} for data in info]

    root_edu = dict()
    root_edu["text"] = "<root>"
    root_edu["speaker"] = "<root>"
    root_edu["tokens"] = ["<root>"]

    inst.edus.append(root_edu)
    inst.edus += [{"text": data[4], "speaker": data[3]} for data in info]

    inst.relations = [{"y": int(data[6][-1]), "x": int(data[6][0]), "type": data[7].strip()} for data in info]
    inst.pred_relations = inst.relations.copy()

    for rel in inst.relations:
        if rel["y"] <= rel["x"]:
            rel["x"] = -1
            rel["type"] = "<root>"
        if rel["y"] == 1:  # 第1个话语不参与训练 因为总是指向虚根  TODO：或者参与训练但是不参与测评？
            rel["x"] = -1
            rel["type"] = "<root>"

        # if "( reverse )" in rel["type"] == '<root>':
        #     rel["x"] = -1

    inst.real_relations = [[] for _ in range(len(inst.edus))]

    rel_matrix = np.zeros([len(inst.original_edus) + 1, len(inst.original_edus) + 1])  # arc flag
    for relation in inst.relations:
        index = relation["y"]
        head = relation["x"]
        if rel_matrix[index, head] >= 1:
            continue

        if head > index:
            continue
        if index >= len(inst.real_relations):
            continue
        if head >= len(inst.real_relations):
            continue

        rel_matrix[index, head] += 1
        inst.real_relations[index].append(relation)  # x是head 指向y

    inst.sorted_real_relations = []
    for idx, rel_relation in enumerate(inst.real_relations):
        r = sorted(rel_relation, key=lambda rel_relation: rel_relation["x"], reverse=False)
        inst.sorted_real_relations.append(r)

    inst.gold_arcs = [[] for idx in range(len(inst.edus))]
    inst.gold_rels = [[] for idx in range(len(inst.edus))]

    inst.gold_arcs[0].append(-1)
    inst.gold_rels[0].append("<root>")
    for idx, relation_list in enumerate(inst.sorted_real_relations):
        if len(relation_list) > 0:
            relation = relation_list[0]
            rel = relation["type"]
            index = relation["y"]
            head = relation["x"]
            if head >= index:
                inst.gold_arcs[index].append(-1)
                inst.gold_rels[index].append("<root>")
            else:
                inst.gold_arcs[index].append(head)
                inst.gold_rels[index].append(rel)
    for idx, arc in enumerate(inst.gold_arcs):
        if len(arc) == 0:
            inst.gold_arcs[idx].append(-1)
            inst.gold_rels[idx].append("<root>")

    for idx, arc in enumerate(inst.gold_arcs):
        assert len(arc) == 1
        assert arc[0] < idx
    for rel in inst.gold_rels:
        assert len(rel) == 1

    for idx, cur_EDU in enumerate(inst.edus):
        if idx == 0:
            turn = 0
        else:
            last_EDU = inst.edus[idx - 1]
            if last_EDU["speaker"] != cur_EDU["speaker"]:
                turn += 1
        cur_EDU["turn"] = turn

        if re.match(r"([a-z]*\d*)*\.jpg", cur_EDU["text"]):
            inst.image_indexs.append(idx)
            cur_EDU["modality"] = "Visual"
        else:
            cur_EDU["modality"] = "Text"

    inst.speakers = [edu["speaker"] for edu in inst.edus]
    inst.speakers[0] = "3"  # root

    inst.edu_num = len(inst.edus)

    # 处理topic分割
    topic_seg = []
    for idx, i in enumerate(info):
        if i[-1] == "1":
            topic_seg.append(idx)

    inst.topic_seg = topic_seg + [len(inst.speakers) - 1]

    audio_paths = ["root.wav"]
    video_paths = ["root.mp4"]

    for idx, i in enumerate(inst.info):
        audio_paths.append(f"{i[0]}_{idx}.wav")
        video_paths.append(f"{i[0]}_{idx}.mp4")

    inst.audio_paths = audio_paths
    inst.video_paths = video_paths

    return inst


def read_corpus(file_path, max_insts=100000):
    """读取file

    Args:
        filename (str/os.path): 文件路径

    Returns:
        book(list): inst的列表
        word_list(list): inst中词的列表
    """
    insts = []
    with open(file_path, mode="r") as inf:
        for info in read_info(inf):
            inst = info2inst(info)
            info = info[:50]  # 最大长度 50
            insts.append(inst)
            if len(insts) >= max_insts:
                break

    return insts


def input_variable(onebatch, config, mode="train"):
    batch_size = len(onebatch)

    text_shapes = [inst.inputs["text_inputs"]["input_ids"].shape for inst in onebatch]
    text_max_edu_num, text_max_token_num = [max(i) for i in zip(*text_shapes)]
    if text_max_token_num > config.max_token_num:
        text_max_token_num = config.max_token_num

    video_shapes = [inst.inputs["video_inputs"]["pixel_values"].shape for inst in onebatch]
    video_max_edu_num, N, C, H, W = [max(i) for i in zip(*video_shapes)]

    audio_shapes = [inst.inputs["audio_inputs"]["input_values"].shape for inst in onebatch]
    audio_max_edu_num, audio_max_token_num = [max(i) for i in zip(*audio_shapes)]
    assert text_max_edu_num == audio_max_edu_num

    text_input_ids = np.zeros([batch_size, text_max_edu_num, text_max_token_num], dtype=np.int64)
    text_attention_mask = np.zeros([batch_size, text_max_edu_num, text_max_token_num], dtype=np.int64)
    # token_type_ids = np.zeros([batch_size, max_edu_num, max_token_num], dtype=np.int64)

    video_pixel_values = np.zeros([batch_size, video_max_edu_num, N, C, H, W], dtype=np.float32)

    audio_input_values = np.zeros([batch_size, text_max_edu_num, audio_max_token_num], dtype=np.float32)
    audio_attention_mask = np.zeros([batch_size, text_max_edu_num, audio_max_token_num], dtype=np.int64)

    for idx, inst in enumerate(onebatch):
        # text features
        text_edu_num = len(inst.inputs["text_inputs"]["input_ids"])
        text_token_num = (
            len(inst.inputs["text_inputs"]["input_ids"][0]) if len(inst.inputs["text_inputs"]["input_ids"][0]) < text_max_token_num else text_max_token_num
        )

        text_input_ids[idx, :text_edu_num, :text_token_num] = inst.inputs["text_inputs"]["input_ids"][:text_edu_num, :text_token_num]
        text_attention_mask[idx, :text_edu_num, :text_token_num] = inst.inputs["text_inputs"]["attention_mask"][:text_edu_num, :text_token_num]

        # video features
        video_edu_num = len(inst.inputs["video_inputs"]["pixel_values"])
        video_pixel_values[idx, :video_edu_num] = inst.inputs["video_inputs"]["pixel_values"]

        # audio features
        audio_edu_num = len(inst.inputs["audio_inputs"]["input_values"])
        audio_token_num = (
            len(inst.inputs["audio_inputs"]["input_values"][0])
            if len(inst.inputs["audio_inputs"]["input_values"][0]) < audio_max_token_num
            else audio_max_token_num
        )

        audio_input_values[idx, :audio_edu_num, :audio_token_num] = inst.inputs["audio_inputs"]["input_values"][:audio_edu_num, :audio_token_num]
        audio_attention_mask[idx, :audio_edu_num, :audio_token_num] = inst.inputs["audio_inputs"]["attention_mask"][:audio_edu_num, :audio_token_num]

    text_input_ids = torch.as_tensor(text_input_ids)
    text_attention_mask = torch.as_tensor(text_attention_mask)
    video_pixel_values = torch.as_tensor(video_pixel_values)
    audio_input_values = torch.as_tensor(audio_input_values)
    audio_attention_mask = torch.as_tensor(audio_attention_mask)

    text_inputs = {
        "input_ids": text_input_ids,
        "attention_mask": text_attention_mask,
    }
    image_inputs = {"pixel_values": video_pixel_values}
    audio_inputs = {
        "input_values": audio_input_values,
        "attention_mask": audio_attention_mask,
    }

    return text_inputs, image_inputs, audio_inputs


def mask_variable(onebatch, window_size):
    batch_size = len(onebatch)

    max_edu_num = max([inst.edu_num for inst in onebatch])

    intra_speaker_masks = np.zeros([batch_size, max_edu_num, max_edu_num])  # 看到相同的speaker
    inter_speaker_masks = np.zeros([batch_size, max_edu_num, max_edu_num])  # 看到不同的speaker
    global_masks = np.zeros([batch_size, max_edu_num, max_edu_num])  # 全局都可以看到
    local_masks = np.zeros([batch_size, max_edu_num, max_edu_num])  # 只能看到windows中的
    pre_masks = np.zeros([batch_size, max_edu_num, max_edu_num])  # 看到pre
    topic_masks = np.zeros([batch_size, max_edu_num, max_edu_num])  # 按照照片切分topic进行mask

    for idx, inst in enumerate(onebatch):
        speakers = np.array(inst.speakers)

        for idy in range(len(inst.speakers)):
            cur_speaker = inst.speakers[idy]
            intra_speaker_masks[idx, idy, : len(speakers)][speakers == cur_speaker] = 1
            inter_speaker_masks[idx, idy, : len(speakers)][speakers != cur_speaker] = 1
            global_masks[idx, idy, : len(speakers)] = 1

            local_masks[
                idx,
                idy,
                max(0, idy - window_size) : min(idy + window_size + 1, len(inst.speakers)),
            ] = 1
            pre_masks[idx, idy, : idy + 1] = 1

        for start, end in zip(inst.topic_seg[:-1], inst.topic_seg[1:]):
            topic_masks[
                idx,
                start:end,
                max(0, start - window_size) : min(end + window_size, len(speakers)),
            ] = 1

    intra_speaker_masks = torch.from_numpy(intra_speaker_masks)
    inter_speaker_masks = torch.from_numpy(inter_speaker_masks)
    global_masks = torch.from_numpy(global_masks)
    local_masks = torch.from_numpy(local_masks)
    pre_masks = torch.from_numpy(pre_masks)
    topic_masks = torch.from_numpy(topic_masks)

    masks = {
        "inter_masks": inter_speaker_masks,
        "intra_masks": intra_speaker_masks,
        "global_masks": global_masks,
        "local_masks": local_masks,
        "pre_masks": pre_masks,
        "topic_masks": topic_masks,
    }
    return masks


def offset_variable(onebatch):
    batch_size = len(onebatch)
    edu_lengths = [len(instance.edus) for instance in onebatch]
    max_edu_len = max(edu_lengths)
    arc_masks = np.zeros([batch_size, max_edu_len, max_edu_len])

    for idx, instance in enumerate(onebatch):
        edu_len = len(instance.edus)
        for idy in range(edu_len):
            for idz in range(idy):
                arc_masks[idx, idy, idz] = 1.0

    arc_masks = torch.tensor(arc_masks)
    image_indexs = [inst.image_indexs for inst in onebatch]

    return edu_lengths, arc_masks, image_indexs


def label_variable(onebatch):
    gold_arc_labels = []
    gold_rel_labels = []
    for inst in onebatch:
        gold_arc_labels.append(inst.gold_arc_labels)
        gold_rel_labels.append(inst.gold_rel_labels)

    gold_labels = {
        "gold_arc_labels": gold_arc_labels,
        "gold_rel_labels": gold_rel_labels,
    }

    return gold_labels


def modal_index(instance):
    instance.modal_indexs = list()
    for edu in instance.edus:
        if re.match(r"([a-z]*\d*)*\.jpg", edu["text"]):
            instance.modal_indexs.append(1)  # 1 means vision modal
        else:
            instance.modal_indexs.append(0)  # 0 means text modal


class DataCollator:
    def __init__(self, config) -> None:
        self.config = config
        self.text_processor = AutoProcessor.from_pretrained(config.text_plm_name_or_path)
        self.video_processor = AutoProcessor.from_pretrained(config.vision_plm_name_or_path)
        # self.video_processor = VideoProcessor(config.video_dir)
        self.audio_processor = AutoProcessor.from_pretrained(config.audio_plm_name_or_path)

    def train_wrapper(self, batch):
        text_inputs, image_inputs, audio_inputs = input_variable(batch, self.config)
        masks = mask_variable(batch, self.config.window_size)
        edu_lengths, arc_masks, image_indexs = offset_variable(batch)

        gold_labels = label_variable(batch)
        inputs = {
            "text_inputs": text_inputs,
            "image_inputs": image_inputs,
            "audio_inputs": audio_inputs,
            "edu_lengths": edu_lengths,
            "arc_masks": arc_masks,
            "image_indexs": image_indexs,
            "masks": masks,
        }

        return batch, inputs, gold_labels

    def test_wrapper(self, batch, mode="dev"):
        text_inputs, image_inputs, audio_inputs = input_variable(batch, self.config, mode)
        edu_lengths, arc_masks, image_indexs = offset_variable(batch)
        masks = mask_variable(batch, self.config.window_size)

        inputs = {
            "text_inputs": text_inputs,
            "image_inputs": image_inputs,
            "audio_inputs": audio_inputs,
            "edu_lengths": edu_lengths,
            "arc_masks": arc_masks,
            "image_indexs": image_indexs,
            "masks": masks,
        }

        return batch, inputs

    @staticmethod
    def convert_insts_to_features(insts, text_processor, audio_processor, video_features, audio_features):
        """将insts实例转换成特征 主要负责tokenizer和窗口化等
        Args:
            insts (data.dataset.Instance): 实例
            tokenizer (transformers.Tokenizer): Tokenizer
        """
        all_path = []
        for inst in insts:
            sent_lists = []
            audio_np = []
            video_np = []
            for idx, edu in enumerate(inst.edus):
                sent_lists.append(edu["text"])

                if idx == 0:
                    video_np.append(video_features["root"]["pixel_values"].reshape(1, 3, 224, 224))  # N C H W
                    audio_np.append(audio_features["root"])
                else:
                    video_np.append(video_features[f"{inst.info[idx-1][0]}_{inst.info[idx-1][5]}"]["pixel_values"].reshape(1, 3, 224, 224))  # 编号从1开始
                    audio_np.append(audio_features[f"{inst.info[idx-1][0]}_{inst.info[idx-1][5]}"])

            inst.inputs = {}
            inst.inputs["text_inputs"] = text_processor(sent_lists, add_special_tokens=True, padding=True, return_tensors="np")
            inst.inputs["video_inputs"] = {"pixel_values": np.array(video_np)}  # edu_num, N, C, H, W
            inst.inputs["audio_inputs"] = audio_processor(audio_np, sampling_rate=16000, padding=True, return_tensors="np")

            a, b = inst.inputs["audio_inputs"]["input_values"].shape
            if a * b > 4_500_000:
                while a * b > 4_500_000:
                    inst.inputs["audio_inputs"]["input_values"] = inst.inputs["audio_inputs"]["input_values"][:, : b - 10000]
                    inst.inputs["audio_inputs"]["attention_mask"] = inst.inputs["audio_inputs"]["input_values"][:, : b - 10000]
                    a, b = inst.inputs["audio_inputs"]["input_values"].shape
                logger.warning(f"{inst.info[idx-1][0]}_{inst.info[idx-1][5]} is to long")


def get_video_features(video_np, video_processor):
    N, H, W, C = video_np[1].shape
    edu_num = len(video_np)
    video_np[0] = np.zeros((N, H, W, C), dtype=np.int8)
    video_np = np.array(video_np).reshape(edu_num * N, H, W, C)
    video_features = video_processor(video_np, padding=True, return_tensors="np")
    _, C, W, H = video_features["pixel_values"].shape
    video_features["pixel_values"] = video_features["pixel_values"].reshape(edu_num, N, C, H, W)
    return video_features


def video_file_to_array_fn(path):
    container = av.open(path)

    # sample 3 frames
    indices = sample_frame_indices(clip_len=1, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
    # indices = [5, 10]
    video = read_video_pyav(container=container, indices=indices)
    return video


def speech_file_to_array_fn(path):
    speech_array, sampling_rate = librosa.load(path, sr=16_000)

    return speech_array


def read_video_pyav(container, indices):
    """
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    """
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)

    while len(frames) != len(indices):  # 兜底策略
        frames.append(frame)

    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    """
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices
    """
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices
