from .layers import *
from .scalarmix import *
import torch.nn as nn
import math


class StateEncoder(nn.Module):
    def __init__(self, config, plm_model):
        super(StateEncoder, self).__init__()
        self.utt_nonlinear = NonLinear(input_size=config.inter_size * 2,
                                    hidden_size=config.hidden_size,
                                    activation=nn.Tanh())

        self.rescale = ScalarMix(mixture_size=2)

    def forward(self, global_outputs):
        batch_size, max_edu_len, _ = global_outputs.size()
        global_outputs = global_outputs.unsqueeze(1).repeat(1, max_edu_len, 1, 1)
        utt_state_input = torch.cat([global_outputs, global_outputs.transpose(1, 2)], dim=-1)
        state_hidden = self.utt_nonlinear(utt_state_input)

        return state_hidden