import torch.nn as nn
import torch
from .plm_tune import PLMExtractor
from .decoder import Decoder
from .encoder import GlobalEncoder
from .state_encoder import StateEncoder
from .layers import _model_var, pad_sequence
import torch.nn.functional as F
import numpy as np
from .interaction import InteractionModule
from prettytable import PrettyTable
from utils import logger


class Model(nn.Module):
    def __init__(self, config, data) -> None:
        super(Model, self).__init__()
        self.config = config
        self.vocab = data.vocab
        self.init_model(data)
        self.use_cuda = False
    
    def cuda(self, device=None):
        self.use_cuda = True
        return self._apply(lambda t: t.cuda(device))
        
    def forward(self, inputs):
        self.encode(**inputs)

    def init_model(self, data):
        enc_model = PLMExtractor(self.config)
        self.encoder = GlobalEncoder(self.config, enc_model)
        self.inter_module = InteractionModule(self.config)
        self.state_encoder = StateEncoder(self.config, enc_model)
        self.decoder = Decoder(self.vocab, self.config)
    
    def encode(self, text_inputs, image_inputs, audio_inputs, edu_lengths, arc_masks, image_indexs, masks):

        global_outputs = self.encoder(text_inputs, image_inputs, audio_inputs)

        hidden = self.inter_module(global_outputs, **masks)

        state_hidden = self.state_encoder(hidden)
        self.arc_logits, self.rel_logits = self.decoder(state_hidden, arc_masks)

    def is_finished(self, cur_step, edu_lengths):
        finished_flag = True
        for edu_length in edu_lengths:
            if cur_step < edu_length:
                finished_flag = False
        return finished_flag

    def decode(self, edu_lengths):
        cur_step = 0
        pred_arcs = None
        pred_rels = None
        while not self.is_finished(cur_step, edu_lengths):
            arc_logit, rel_logit = self.arc_logits[:, cur_step, :], self.rel_logits[:, cur_step, :]

            pred_arc = arc_logit.detach().max(-1)[1].cpu().numpy()
            batch_size, max_edu_size, label_size = rel_logit.size()
            rel_probs = _model_var(self.decoder, torch.zeros(batch_size, label_size))
            for batch_index, (logits, arc) in enumerate(zip(rel_logit, pred_arc)):
                rel_probs[batch_index] = logits[arc]
            pred_rel = rel_probs.detach().max(-1)[1].cpu().numpy()

            pred_arc = pred_arc.reshape(batch_size, 1)
            pred_rel = pred_rel.reshape(batch_size, 1)

            if cur_step == 0:
                pred_arcs = pred_arc
                pred_rels = pred_rel
            else:
                pred_arcs = np.concatenate((pred_arcs, pred_arc), axis=-1)
                pred_rels = np.concatenate((pred_rels, pred_rel), axis=-1)
            cur_step += 1

        return pred_arcs, pred_rels

    def compute_loss(self, gold_arc_labels, gold_rel_labels):
        batch_size, max_edu_size, _ = self.arc_logits.size()
        gold_arc_labels = _model_var(self.decoder, pad_sequence(gold_arc_labels,
                                                          length=max_edu_size, padding=-1, dtype=np.int64))
        batch_size, max_edu_size, _ = self.arc_logits.size()

        arc_loss = F.cross_entropy(self.arc_logits.view(batch_size * max_edu_size, -1),
                                   gold_arc_labels.view(-1), ignore_index=-1)

        _, _, _, label_size = self.rel_logits.size()
        rel_logits = _model_var(self.decoder, torch.zeros(batch_size, max_edu_size, label_size))
        for batch_index, (logits, arcs) in enumerate(zip(self.rel_logits, gold_arc_labels)):
            rel_probs = []
            for i in range(max_edu_size):
                rel_probs.append(logits[i][int(arcs[i])])
            rel_probs = torch.stack(rel_probs, dim=0)
            rel_logits[batch_index] = torch.squeeze(rel_probs, dim=1)

        gold_rel_labels = _model_var(self.decoder, pad_sequence(gold_rel_labels,
                                                          length=max_edu_size, padding=-1, dtype=np.int64))

        rel_loss = F.cross_entropy(rel_logits.view(batch_size * max_edu_size, -1),
                                   gold_rel_labels.view(-1), ignore_index=-1)

        return arc_loss + rel_loss

    def compute_accuracy(self, gold_arc_labels, gold_rel_labels):
        arc_correct, arc_total, rel_correct = 0, 0, 0
        pred_arcs = self.arc_logits.detach().max(2)[1].cpu().numpy()
        assert len(pred_arcs) == len(gold_arc_labels)

        batch_idx = 0
        for p_arcs, g_arcs in zip(pred_arcs, gold_arc_labels):
            edu_len = len(g_arcs)
            for idx in range(edu_len):
                if idx == 0: continue
                if p_arcs[idx] == g_arcs[idx]:
                    arc_correct += 1
                arc_total += 1
            batch_idx += 1

        batch_size, max_edu_size, _, label_size = self.rel_logits.size()

        gold_arcs_index = _model_var(self.decoder, pad_sequence(gold_arc_labels,
                                                                length=max_edu_size,
                                                                padding=-1, dtype=np.int64))
        rel_logits = _model_var(self.decoder, torch.zeros(batch_size, max_edu_size, label_size))
        for batch_index, (logits, arcs) in enumerate(zip(self.rel_logits, gold_arcs_index)):
            rel_probs = []
            for i in range(max_edu_size):
                rel_probs.append(logits[i][int(arcs[i])])
            rel_probs = torch.stack(rel_probs, dim=0)
            rel_logits[batch_index] = torch.squeeze(rel_probs, dim=1)
        pred_rels = rel_logits.detach().max(2)[1].cpu().numpy()

        assert len(pred_rels) == len(gold_rel_labels)
        batch_idx = 0
        for p_rels, g_rels in zip(pred_rels, gold_rel_labels):
            edu_len = len(g_rels)
            for idx in range(edu_len):
                if idx == 0: continue
                if p_rels[idx] == g_rels[idx]:
                    rel_correct += 1
            batch_idx += 1

        return arc_correct, arc_total, rel_correct

    @staticmethod
    def loss_fun(y_true, y_pred):
        """
        y_true:(batch_size, seq_len, seq_len)
        y_pred:(batch_size, seq_len, seq_len)
        """
        batch_size = y_pred.shape[0]
        y_true = y_true.reshape(batch_size, -1)
        y_pred = y_pred.reshape(batch_size, -1)
        loss = Model.multilabel_categorical_crossentropy(y_pred, y_true)

        return loss

    @staticmethod
    def multilabel_categorical_crossentropy(y_pred, y_true):
        """
        https://kexue.fm/archives/7359
        """
        y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
        y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
        y_pred_pos = (y_pred - (1 - y_true) * 1e12)  # mask the pred outputs of neg classes
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

        return (neg_loss + pos_loss).mean()
    
    def log_model_weights(self):
        named_parameters = list(self.named_parameters())
        total_weights = sum(p.numel() for _, p in named_parameters) / (1024 * 1024)
        train_weights = sum(p.numel() for _, p in named_parameters if p.requires_grad) / (1024 * 1024)

        model_weights_summary = f"Total model weights: {total_weights :>8.2f}M, Trainable model weights: {train_weights :>8.2f}M"
        param_table = PrettyTable()
        param_table.field_names = ["name", "device", "shape", "dtype", "trainable"]
        for name, param in named_parameters:
            param_content = (
                name,
                param.device,
                str(param.shape),
                str(param.dtype),
                param.requires_grad,
            )
            param_table.add_row(param_content)
        param_table.align["shape"] = "l"
        param_table.align["name"] = "l"
        
        logger.info(model_weights_summary + "\n    " + str(param_table).replace("\n", "\n    "))