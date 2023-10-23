import torch
import copy
import torch.nn as nn
import copy
import torch.nn.functional as F
import torch.distributions as D
import importlib
import wandb
from torch import Tensor
from .functions import *
from .common import *

class ActorCritic(nn.Module):

    def __init__(self,
                 in_dim,
                 out_actions,
                 normalizer,
                 device="cuda:0",
                 hidden_dim=256,
                 min_std=0.1,
                 fixed_std=False,
                 decay_std_steps=500000,
                 init_std=0.5,
                 hidden_layers=2,
                 layer_norm=True,
                 gamma=0.99,
                 ema=0.995,
                 lambda_gae=0.9,
                 entropy_weight=1e-3,
                 entropy_target=-1,
                 tune_entropy=True,
                 target_interval=100,
                 lr_actor=1e-4,
                 lr_critic=3e-4,
                 lr_alpha=1e-2,
                 actor_optim='AdamW',
                 actor_grad='reinforce',
                 actor_dist='normal_tanh',
                 normalize_adv=True,
                 grad_clip=None,
                 clip_logprob=True,
                 min_logprob=-10.0,
                 learned_std=True,
                 ac_use_normed_inputs=False,
                 target_update=0.02,
                 tune_actor_lr=3e-4,
                 lr_schedule='constant',
                 lr_decay_steps=1000000,
                 log_interval=20000,
                 **kwargs
                 ):
        super().__init__()
        self.in_dim = in_dim
        self.action_dim = out_actions
        self.gamma = gamma
        self.lambda_ = lambda_gae
        self.target_interval = target_interval
        self.actor_grad = actor_grad
        self.actor_dist = actor_dist
        self.min_std = min_std
        self.clip_logprob = clip_logprob
        self.normalizer = normalizer
        self.min_logprob = min_logprob * self.action_dim
        self.learned_std = learned_std
        self.fixed_std = fixed_std
        self.decay_std_steps = decay_std_steps
        self.init_std = init_std
        self.current_std = init_std
        self.use_normed_inputs = ac_use_normed_inputs
        self.lr_decay_steps = lr_decay_steps
        self.log_interval = log_interval
        self.last_log = -float('inf')

        if not self.fixed_std and not self.learned_std:
            actor_out_dim = 2 * out_actions
        else:
            actor_out_dim = out_actions

        self.actor = MLP(in_dim, actor_out_dim, hidden_dim, hidden_layers, layer_norm).to(device)
        self.critic = MLP(in_dim, 1, hidden_dim, hidden_layers, layer_norm).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.requires_grad_(False)
        self.device = device
        self.train_steps = 0

        optim = importlib.import_module('torch.optim')
        actor_optim = getattr(optim, actor_optim)

        if self.learned_std:
            self.logstd = AddBias((torch.ones(actor_out_dim)*np.log(self.init_std-self.min_std)).to(self.device))
            self._optimizer_actor = actor_optim(list(self.actor.parameters()) + list(self.logstd.parameters()), lr=lr_actor)
        else:
            self._optimizer_actor = torch.optim.AdamW(self.actor.parameters(), lr=lr_actor)
        self._optimizer_critic = actor_optim(self.critic.parameters(), lr=lr_critic)
        self.grad_clip = grad_clip
        self.normalize_adv = normalize_adv
        self.tune_entropy = tune_entropy
        self.entropy_target = entropy_target
        self.log_alpha = torch.log(torch.tensor(entropy_weight)).to(device)
        if self.tune_entropy:
            self.log_alpha.requires_grad_(True)
            self._optimizer_alpha = torch.optim.AdamW([self.log_alpha], lr=lr_alpha)
        
        self.lr_schedule = lr_schedule
        self.tune_actor_lr = tune_actor_lr
        self.target_update = target_update
        self.max_lr = lr_actor
        if self.lr_schedule == "target":
            self.log_actor_lr = torch.log(torch.tensor(lr_actor)).to(device)
            self.log_actor_lr.requires_grad_(True)
            self._optimizer_actor_lr = torch.optim.AdamW([self.log_actor_lr], lr=tune_actor_lr)

    def forward_actor(self, features: Tensor, normed_input=True) -> D.Distribution:
        """Takes as input either normalized or unnnormalized features. Outputs
        unnormalized action distribution. """

        if not normed_input and self.use_normed_inputs:
            features = self.normalizer.normalize(features, "observations")
        elif normed_input and not self.use_normed_inputs:
            features = self.normalizer.unnormalize(features, "observations")
            
        y = self.actor.forward(features).float()

        if self.actor_dist == 'normal_tanh':
            if not self.fixed_std and not self.learned_std:
                return normal_tanh(y, min_std=self.min_std)
            else:
                if len(y.shape) == 0 or y.shape[-1] != self.action_dim:
                    # TODO: Fix this.
                    y = y.unsqueeze(-1)

                if self.fixed_std:
                    std = self.current_std
                elif self.learned_std:
                    std = self.logstd(torch.zeros_like(y)).exp() + self.min_std
                else:
                    raise NotImplementedError
                return normal_tanh(y, fixed_std=std)
        else:
            raise NotImplementedError

    
