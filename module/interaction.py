from .layers import *
from .scalarmix import *
import torch.nn as nn
from .modeling_bert import BertEncoder, BertConfig, BertModel, BertEmbeddings, BertModelForInteraction
from .scalarmix import ScalarMix


class InteractionModule(nn.Module):
    def __init__(self, config):
        super(InteractionModule, self).__init__()
        mix_num = 0
        inter_config = BertConfig.from_pretrained(config.bert_path)
        inter_config.num_hidden_layers = config.num_utt_layers
        inter_config.num_attention_heads = config.num_utt_heads
        inter_config.hidden_size = config.utt_interaction_size
        
        if config.intra_interaction:
            self.intra_linear = NonLinear(config.modal_proj_dim, config.utt_interaction_size)
            self.intra_att = BertModel(inter_config, add_pooling_layer=False)
            mix_num += 1
           
        if config.inter_interaction:
            self.inter_linear = NonLinear(config.modal_proj_dim, config.utt_interaction_size)
            self.inter_att = BertModel(inter_config, add_pooling_layer=False)
            mix_num += 1
        
        if config.global_interaction:
            self.global_linear = NonLinear(config.modal_proj_dim, config.utt_interaction_size) 
            self.global_att = BertModel(inter_config, add_pooling_layer=False)
            mix_num += 1

        self.mix_num = mix_num
        self.scalar = ScalarMix(self.mix_num)
        self.config = config
        self.drop = nn.Dropout(config.dropout_emb)

    def forward(self, utt_rep, inter_masks, intra_masks, global_masks, local_masks, pre_masks, topic_masks):
        """
        包含四种交互
        global
        local
        inter_speaker
        intra_speaker
        """
        outputs = []
        if self.config.intra_interaction:
            hidden = self.intra_linear(utt_rep)
            intra_output = self.intra_att(inputs_embeds=hidden, attention_mask=intra_masks)[0]
            outputs.append(intra_output)
        
        if self.config.inter_interaction:
            hidden = self.inter_linear(utt_rep)
            inter_output = self.inter_att(inputs_embeds=hidden, attention_mask=inter_masks)[0]
            outputs.append(inter_output)

        if self.config.global_interaction:
            hidden = self.global_linear(utt_rep)
            global_output = self.global_att(inputs_embeds=hidden, attention_mask=global_masks)[0]
            outputs.append(global_output)

        if self.mix_num > 1:
            outputs = self.scalar(outputs)
        else:
            outputs = outputs[0]
        return outputs
    