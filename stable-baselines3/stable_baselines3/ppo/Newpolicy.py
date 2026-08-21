import torch as th
from torch import nn
import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import DiagGaussianDistribution

class SineParamActorCriticPolicy(ActorCriticPolicy):
    def _build(self, lr_schedule):
        print("✅ Using SineParamActorCriticPolicy")

        self._build_mlp_extractor()
        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        # 原 action_dim 是输出目标动作，现在扩展为正弦参数
        self.param_dim = self.action_space.shape[0]   # A, ω, ϕ, offset

        # 输出正弦参数向量的均值和 log_std
        self.action_net = nn.Linear(latent_dim_pi, self.param_dim)
        self.log_std = nn.Parameter(th.ones(self.param_dim) * self.log_std_init, requires_grad=True)

        # value network
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        # init weights
        if self.ortho_init:
            self.action_net.apply(lambda m: self.init_weights(m, gain=0.01))
            self.value_net.apply(lambda m: self.init_weights(m, gain=1.0))

        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

        # 使用 DiagGaussianDistribution 构建参数分布
        self.action_dist = DiagGaussianDistribution(self.param_dim)

    def decode_param_output(self, raw_output):
        """
        将网络输出的 raw logits 映射到合理范围的正弦函数参数上。
        """
        action_dim = self.action_space.shape[0]
        A = 0.5 * th.sigmoid(raw_output[:, 0*action_dim:1*action_dim])                     # A ∈ (0, 0.5)
        omega = 2 * np.pi * th.sigmoid(raw_output[:, 1*action_dim:2*action_dim])           # ω ∈ (0, 2π)
        phi = 2 * np.pi * th.sigmoid(raw_output[:, 2*action_dim:3*action_dim])             # ϕ ∈ (0, 2π)
        offset = 0.5 * th.tanh(raw_output[:, 3*action_dim:4*action_dim])                   # offset ∈ [-0.5, 0.5]
        return th.cat([A, omega, phi, offset], dim=1)

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor):
        raw_mean = self.action_net(latent_pi)
        mean = self.decode_param_output(raw_mean)
        return self.action_dist.proba_distribution(mean, self.log_std)
