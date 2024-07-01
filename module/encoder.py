import torch.nn as nn
import math
from .layers import *
from .scalarmix import *
from .slide_window import merge_chuncked_reps
from .modeling_bert import BertModel, BertConfig, BertSelfAttention, BertAttention


class GlobalEncoder(nn.Module):
    def __init__(self, config, plm_extractor):
        super(GlobalEncoder, self).__init__()
        self.plm_extractor = plm_extractor
        self.drop_emb = nn.Dropout(config.dropout_emb)

        if config.with_ac:
            self.text_projection = NonLinear(config.text_plm_config.hidden_size, config.modal_proj_dim, activation=nn.LeakyReLU())
            self.image_projection = NonLinear(config.vision_plm_config.hidden_size, config.modal_proj_dim, activation=nn.LeakyReLU())
            self.audio_projection = NonLinear(config.audio_plm_config.hidden_size, config.modal_proj_dim, activation=nn.LeakyReLU())
        else:
            self.text_projection = nn.Linear(config.text_plm_config.hidden_size, config.modal_proj_dim, bias=False)
            self.image_projection = nn.Linear(config.vision_plm_config.hidden_size, config.modal_proj_dim, bias=False)
            self.audio_projection = nn.Linear(config.audio_plm_config.hidden_size, config.modal_proj_dim, bias=False)

        # self.hidden_drop = nn.Dropout(config.dropout_gru_hidden)
        interaction_config = BertConfig()
        interaction_config.num_hidden_layers = config.modal_inter_layers
        interaction_config.hidden_size = config.modal_proj_dim
        interaction_config.num_attention_heads = config.modal_inter_heads
        interaction_config.is_decoder = True
        # self.fusion = BertModel(interaction_config)
        self.fusion = nn.ModuleList([BertSelfAttention(interaction_config) for _ in range(config.modal_inter_layers)])

        hidden_size = 1 * config.modal_proj_dim
        self.cross_projection = NonLinear(hidden_size, config.modal_proj_dim, activation=nn.LeakyReLU())

        self.config = config

    def forward(self, text_inputs, image_inputs, audio_inputs):
        batch_size, max_edu_num, max_tok_len = text_inputs["input_ids"].size()

        text_features = None
        if "t" in self.config.modal:
            input_ids = text_inputs["input_ids"].view(-1, max_tok_len)
            attention_mask = text_inputs["attention_mask"].view(-1, max_tok_len)
            text_features = self.plm_extractor.get_text_features(input_ids, attention_mask)[0]
            text_features = self.drop_emb(text_features)
            text_features = self.text_projection(text_features)

        image_features = None
        if "v" in self.config.modal:
            _, _, N, C, H, W = image_inputs["pixel_values"].size()
            pixel_values = image_inputs["pixel_values"].view(-1, C, H, W)
            image_features = self.plm_extractor.get_image_features(pixel_values)[0]
            image_features = self.drop_emb(image_features)
            image_features = self.image_projection(image_features)

        audio_features = None
        if "a" in self.config.modal:
            _, _, audio_max_tok_len = audio_inputs["input_values"].size()
            input_values = audio_inputs["input_values"].view(-1, audio_max_tok_len)
            attention_mask = audio_inputs["attention_mask"].view(-1, audio_max_tok_len)
            audio_features = self.plm_extractor.get_audio_features(input_values, attention_mask)[0]
            audio_features = self.drop_emb(audio_features)
            audio_features = self.audio_projection(audio_features)

        # x_embed = text_features + image_features
        text_atts = text_inputs["attention_mask"].view(-1, max_tok_len)
        image_atts = torch.ones(image_features.size()[:-1], dtype=torch.long).to(image_features.device)
        audio_atts = torch.ones(audio_features.size()[:-1], dtype=torch.long).to(audio_features.device)

        video_features = torch.concat([image_features, audio_features], dim=1)
        video_atts = torch.concat([image_atts, audio_atts], dim=1)

        text_atts = get_extended_attention_mask(text_atts, text_features.shape, dtype=text_features.dtype)
        video_atts = get_extended_attention_mask(video_atts, video_features.shape, dtype=image_features.dtype)
        # audio_atts = get_extended_attention_mask(audio_atts, audio_features.shape, dtype=audio_features.dtype)

        # bs * edu_num, text_seq_len, hidden
        hidden_states = video_features
        for layer_module in self.fusion:
            layer_outputs = layer_module(
                hidden_states,
                video_atts,
                encoder_hidden_states=text_features,
                encoder_attention_mask=text_atts,
            )

            hidden_states = layer_outputs[0]

        # h_ti = self.fusion(
        #     inputs_embeds=text_features,
        #     attention_mask=text_atts,
        #     encoder_hidden_states=image_features,
        #     encoder_attention_mask=image_atts,
        # ).last_hidden_state

        # h_ta = self.fusion(
        #     inputs_embeds=text_features,  # bs * edu_num, seq_len, hidden
        #     attention_mask=text_atts,
        #     encoder_hidden_states=audio_features,  # bs * edu_num, image_seq_len, hidden
        #     encoder_attention_mask=audio_atts,
        # ).last_hidden_state

        # hyper_text = torch.concat([text_features[:, 0, :], h_ta[:, 0, :], h_ti[:, 0, :]], dim=-1)
        hyper_text = torch.concat([text_features[:, 0, :], hidden_states[:, 0, :]], dim=-1)
        # utt_embed = self.cross_projection(hidden_states[:, 0, :])  # bs * edu_num, hidden
        utt_embed = hidden_states[:, 0, :]  # bs * edu_num, hidden
        utt_embed = utt_embed.view(batch_size, max_edu_num, -1)

        return utt_embed

    def modal_interaction(self, all_features_dict):
        if self.config.modal_strategy == "sum":
            all_features = list(all_features_dict.values())
            return torch.sum(all_features, dim=-1)
        elif self.config.modal_strategy == "cat":
            all_features = list(all_features_dict.values())
            features = torch.concat(all_features, dim=-1)
            features = self.projection(features)
            return self.drop_emb(features)

        elif self.config.modal_strategy == "gate":
            modals = self.config.modal
            t, v, a = all_features_dict["text_features"], all_features_dict["image_features"], all_features_dict["audio_features"]
            ht = torch.tanh(self.transform_t(t)) if t is not None else t
            hv = torch.tanh(self.transform_t(v)) if v is not None else v
            ha = torch.tanh(self.transform_t(a)) if a is not None else a

            if 'a' in modals and 'v' in modals:
                z_av = torch.sigmoid(self.transform_av(torch.cat([a, v, a * v], dim=-1)))
                h_av = z_av * ha + (1 - z_av) * hv
                if 't' not in modals:
                    return h_av
            if 'a' in modals and 't' in modals:
                z_al = torch.sigmoid(self.transform_at(torch.cat([a, t, a * t], dim=-1)))
                h_al = z_al * ha + (1 - z_al) * ht
                if 'v' not in modals:
                    return h_al
            if 'v' in modals and 't' in modals:
                z_vl = torch.sigmoid(self.transform_vt(torch.cat([v, t, v * t], dim=-1)))
                h_vl = z_vl * hv + (1 - z_vl) * ht
                if 'a' not in modals:
                    return h_vl
            features = torch.cat([h_av, h_al, h_vl], dim=-1)
            features = self.projection(features)
            return features
