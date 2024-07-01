import math
import torch


def chuncking(inputs, slide_size=200, window_size=500):
    token_len = len(inputs)
    pad_token_len = 0
    if token_len <= window_size:
        max_window_num = 1
        pad_token_len = token_len
    else:
        max_window_num = math.ceil(
            (len(inputs) - window_size) / slide_size) + 1
        pad_token_len = window_size + (max_window_num - 1) * slide_size

    chuncked_inputs = []

    for idx in range(max_window_num):
        start = idx * slide_size
        end = start + window_size
        chuncked_inputs.append(inputs[start:end])

    return chuncked_inputs, pad_token_len


def merge_chuncked_reps(chuncked_reps, slide_size=200, window_size=500):
    b, t, w, h = chuncked_reps.size()

    splited_reps = torch.split(chuncked_reps, 1, dim=1)
    assert t == len(splited_reps)

    x_embeds = splited_reps[0].squeeze(1)
    zero_pad = torch.zeros([b, slide_size, h]).type(chuncked_reps.type())

    for idx in range(1, t):
        x_embeds = torch.cat([x_embeds, zero_pad], dim=1)
        slide_embeds = torch.cat(
            [zero_pad for i in range(idx)] + [splited_reps[idx].squeeze(1)], dim=1)
        x_embeds = x_embeds + slide_embeds

    return x_embeds
