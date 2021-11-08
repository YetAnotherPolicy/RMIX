import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from itertools import chain

from controllers.basic_controller import BasicMAC


class RmixMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)
        self.logits = None
        self.q = None

        self.input_shape = self._get_input_shape(scheme)
        self.r_hidden_states = None  # GRU of risk controller
        self.current_risk_level = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs, _ = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs],
                                                            t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        risk_agent_inputs = self._build_inputs(ep_batch, 0 if t == 0 else t-1)
        avail_actions = ep_batch["avail_actions"][:, t]

        cvar_q, q, self.logits, r_logits, self.hidden_states, self.r_hidden_states, masked_logits, mask = \
            self.agent(agent_inputs, risk_agent_inputs, self.hidden_states, self.r_hidden_states)

        self.q = q.view(ep_batch.batch_size, self.n_agents, -1)
        self.mask = mask
        agent_outs = cvar_q
        self.current_risk_level = self.agent.current_risk_level

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                              + th.ones_like(agent_outs) * self.action_selector.epsilon/epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1), \
               self.logits.view(ep_batch.batch_size, self.n_agents, self.args.n_actions, self.args.num_atoms)

    def init_hidden(self, batch_size):
        hidden_states, r_hidden_states = self.agent.init_hidden()
        hidden_states = hidden_states.unsqueeze(0).expand(batch_size, self.n_agents, -1)
        if r_hidden_states is not None:
            r_hidden_states = r_hidden_states.unsqueeze(0).expand(batch_size, self.n_agents, -1)

        self.hidden_states, self.r_hidden_states = hidden_states, r_hidden_states   # bav
        
        self._init_other_hidden(batch_size)

    def _init_other_hidden(self, batch_size):
        pass
