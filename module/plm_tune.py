from numpy import pad
import torch.nn as nn
import torch
from utils import logger
from transformers import ChineseCLIPModel, AutoModel


class PLMExtractor(nn.Module):
    def __init__(self, config):
        super(PLMExtractor, self).__init__()
        self.config = config

        self.text_plm = AutoModel.from_pretrained(config.text_plm_name_or_path)
        config.text_plm_config = self.text_plm.config
        self.vision_plm = AutoModel.from_pretrained(config.vision_plm_name_or_path)
        config.vision_plm_config = self.vision_plm.config
        self.audio_plm = AutoModel.from_pretrained(config.audio_plm_name_or_path)
        config.audio_plm_config = self.audio_plm.config

        if config.tie:
            self._tie_weights()

        logger.info("Load PLM model finished.")

    def get_text_features(self, input_ids, attention_mask):
        return self.text_plm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=self.config.output_hidden_states)

    def get_image_features(self, pixel_values):
        return self.vision_plm(pixel_values=pixel_values, output_hidden_states=self.config.output_hidden_states)

    def get_audio_features(self, input_values, attention_mask):
        return self.audio_plm(input_values=input_values, attention_mask=attention_mask, output_hidden_states=self.config.output_hidden_states)

    def _tie_weights(self):
        for n, p in self.audio_plm.named_parameters():
            if "embed" in n:
                continue
            else:
                p.requires_grad = False
        for n, p in self.text_plm.named_parameters():
            if "embed" in n:
                continue
            else:
                p.requires_grad = False
        for n, p in self.vision_plm.named_parameters():
            if "embed" in n:
                continue
            else:
                p.requires_grad = False
