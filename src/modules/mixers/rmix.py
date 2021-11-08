import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RMixer(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim

        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim))
            self.hyper_moq = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.n_agents))
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, agent_dists, states, masks=None, rewards=None):
        """
        agent_qs: agent_cvar
        agent_dists: agent distribution
        states: the whole state
        rewards: rewards are used for credit assignment
        """
        z_tot = None

        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)

        if agent_dists is not None:
            # Mixtures of Quantile layer # rmix v2_2
            agent_dists = agent_dists.view(-1, self.n_agents, self.args.num_atoms)
            agent_dists, indices = th.sort(agent_dists, dim=2)  # for each quantile
            
            if masks is not None and self.args.use_masked_out_distribution:                
                if agent_dists is None:
                    z_tot = None
                else:
                    if self.args.use_masked_out_distribution_real:  # rmix v2_4
                        _masks = masks.reshape(-1, self.n_agents, self.args.num_atoms)
                        _agent_dists = agent_dists.clone().detach()
                        agent_dists = agent_dists * _masks + _agent_dists * (1 - _masks)
                    else:  # rmix v2_3, v2_7
                        # maybe use the mask caused the big number issue
                        agent_dists = agent_dists  # * masks.reshape(-1, self.n_agents, self.args.num_atoms) 

            w_moq = th.abs(self.hyper_moq(states))
            w_moq = w_moq.view(-1, 1, self.n_agents)
            weights = w_moq / th.sum(w_moq, dim=2, keepdim=True)
            z_tot = th.bmm(weights, agent_dists)
            z_tot = z_tot.view(bs, -1, self.args.num_atoms)
    
        if self.args.cvar_vdn: # VDN
            q_tot = th.sum(agent_qs, dim=2, keepdim=True).view(bs, -1, 1)
        else: # Monotonic
            # First layer
            w1 = th.abs(self.hyper_w_1(states))
            b1 = self.hyper_b_1(states)
            w1 = w1.view(-1, self.n_agents, self.embed_dim)
            b1 = b1.view(-1, 1, self.embed_dim)
            hidden = F.elu(th.bmm(agent_qs, w1) + b1)
            
            # Second layer
            w_final = th.abs(self.hyper_w_final(states))
            w_final = w_final.view(-1, self.embed_dim, 1)
            
            # State-dependent bias
            v = self.V(states).view(-1, 1, 1)
            
            # Compute final output
            y = th.bmm(hidden, w_final) + v
            
            # Reshape and return
            q_tot = y.view(bs, -1, 1)
        return q_tot, z_tot
