import math
import random

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class RmixAgent(nn.Module):
    def __init__(self, input_shape, args):
        super().__init__()
        self.args = args
        self.input_shape = input_shape
        self.risk_level_range = self.args.risk_level_range
        # static risk level including 0.1-1
        if self.args.alpha_risk_static:
            self.current_risk_level = self.args.alpha_risk
        if self.args.use_cuda:
            self.arrange = th.arange(self.args.num_atoms).int().cuda()
        else:
            self.arrange = th.arange(self.args.num_atoms).int()
        self._build_risk_level_controller_net()

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions * args.num_atoms)

    def _forward(self, inputs, hidden_state, reshape_logits=True):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        logits = self.fc2(h)

        if reshape_logits:
            logits = logits.view(-1, self.args.n_actions, self.args.num_atoms)
            return logits, h
        return logits, h

    def forward(self, inputs, r_inputs, hidden_state, r_hidden_state):
        logits, h = self._forward(inputs, hidden_state)
        a_inputs = logits.view(-1, self.args.n_actions*self.args.num_atoms).detach()  # stop gradient
        r_logits, r_h, masks = None, None, th.zeros_like(logits, device=logits.device)
        risk_level_logits, r_logits, r_h = self._forward_risk_net(r_inputs, a_inputs, r_hidden_state)
        risk_weights = th.softmax(risk_level_logits, dim=-1)
        self.current_risk_level = th.div((th.argmax(risk_weights, dim=-1)+1).float(), float(risk_weights.shape[-1]))
        risk_atoms_num = th.ceil(self.current_risk_level * self.args.num_atoms-1).int()
        sorted_logits, indices = th.sort(logits, dim=2)
        masks = (self.arrange <= risk_atoms_num[..., None]).float()
        masks = masks[:, None, :]

        sum_cvar_q = th.sum(sorted_logits * masks, dim=2)
        cvar_q = sum_cvar_q / risk_atoms_num.float()[:, None]
        masked_logits = (sorted_logits * masks).view(-1, self.args.n_actions, self.args.num_atoms)

        q = th.mean(logits, dim=2)
        return cvar_q, q, logits, r_logits, h, r_h, masked_logits, masks

    def _build_risk_level_controller_net(self):
        self.r_fc1 = nn.Linear(self.input_shape, self.args.rnn_hidden_dim)
        self.r_rnn = nn.GRUCell(self.args.rnn_hidden_dim, self.args.rnn_hidden_dim)
        # risk controller
        # history
        self.history = nn.Linear(self.args.n_actions*self.args.num_atoms, self.risk_level_range*self.args.n_agents, bias=False)
        # current
        self.current = nn.Linear(self.args.n_actions*self.args.num_atoms, self.args.n_agents, bias=False)

    def _forward_risk_net(self, inputs, a_inputs, hidden_state):
        x = F.relu(self.r_fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.r_rnn(x, h_in)
        logits = self.fc2(h).detach()  # share the weights without updating the weights in this part
        history = self.history(logits)
        current = self.current(a_inputs)
        _logits = th.matmul(current.view(-1, 1, 1, self.args.n_agents),
                            history.view(-1, 1, self.args.n_agents, self.risk_level_range))
        logits = th.squeeze(_logits / np.sqrt(self.args.rnn_hidden_dim))
        return logits, logits, h

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_(), \
            self.r_fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()
