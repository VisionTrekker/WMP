# MIT License

# Copyright (c) 2023 NM512

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
# All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates.

import copy
import torch
from torch import nn

from . import tools
from . import networks

to_np = lambda x: x.detach().cpu().numpy()

class WorldModel(nn.Module):
    def __init__(self, config, obs_shape, use_camera):
        super(WorldModel, self).__init__()
        # self._step = step
        self._use_amp = True if config.precision == 16 else False   # 32，因此 False
        self._config = config
        self.device = self._config.device

        # 1. encoder: 特征输出维度 (5120,)
        self.encoder = networks.MultiEncoder(obs_shape, **config.encoder, use_camera=use_camera)

        # 2. 循环状态空间模型 RSSM
        self.embed_size = self.encoder.outdim   # 5120
        self.dynamics = networks.RSSM(
            config.dyn_stoch,   # 32
            config.dyn_deter,   # 512
            config.dyn_hidden,  # 512
            config.dyn_rec_depth,   # 1
            config.dyn_discrete,    # 32
            config.act,     # 'SiLU'
            config.norm,    # True
            config.dyn_mean_act,    # 'none'
            config.dyn_std_act, # 'sigmoid2'
            config.dyn_min_std, # 0.1
            config.unimix_ratio,    # 0.01
            config.initial,     # 'learned'
            config.num_actions, # action历史维度 5*12
            self.embed_size,    # 5120
            config.device,
        )

        # 3. Decoder 及 奖励预测器
        self.heads = nn.ModuleDict()
        # 计算从 RSSM 输出的特征维度：离散模式下：随机部分(32类别×32维度) + 确定性部分(512维) = 1536维
        if config.dyn_discrete: # True
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter

        # 3.1 Decoder: 使用MSE损失重建 深度图像 (64, 64, 1) ，适用symbol_mse处理 本体感知数据 (33,)
        #   输出 {"images": MSEDist(out_cnn), "prop": SymlogDist(out_mlp)}
        self.heads["decoder"] = networks.MultiDecoder(
            feat_size, obs_shape, **config.decoder, use_camera=use_camera
        )

        # 3.2 Reward: 使用离散分布预测奖励值(255个分桶), 并经过symlog变换处理
        #   输出 DiscDist(out)
        self.heads["reward"] = networks.MLP(
            feat_size,
            (255,) if config.reward_head["dist"] == "symlog_disc" else (),
            config.reward_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist=config.reward_head["dist"],
            outscale=config.reward_head["outscale"],
            device=config.device,
            name="Reward",
        )
        # self.heads["cont"] = networks.MLP(
        #     feat_size,
        #     (),
        #     config.cont_head["layers"],
        #     config.units,
        #     config.act,
        #     config.norm,
        #     dist="binary",
        #     outscale=config.cont_head["outscale"],
        #     device=config.device,
        #     name="Cont",
        # )
        for name in config.grad_heads:
            assert name in self.heads, name

        # 4. 配置优化器
        self._model_opt = tools.Optimizer(
            "model",
            self.parameters(),
            config.model_lr,    # 1e-4
            config.opt_eps,     # 1e-8
            config.grad_clip,   # 1000
            config.weight_decay,    # 0.0
            opt=config.opt,     # 'adam'
            use_amp=self._use_amp,
        )
        print(
            f"Optimizer model_opt has {sum(param.numel() for param in self.parameters())} variables."
        )

        # 5. 设置 损失权重 （奖励 0.0, image 1.0）
        # other losses are scaled by 1.0.
        # can set different scale for terms in decoder here
        self._scales = dict(
            reward=config.reward_head["loss_scale"],
            image = 1.0,
            # clean_prop = 0,
            # cont=config.cont_head["loss_scale"],
        )

    def _train(self, data):
        # action (batch_size, batch_length, act_dim)
        # image (batch_size, batch_length, h, w, ch)
        # reward (batch_size, batch_length)
        # discount (batch_size, batch_length)
        data = self.preprocess(data)

        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                embed = self.encoder(data)
                post, prior = self.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )
                kl_free = self._config.kl_free
                dyn_scale = self._config.dyn_scale
                rep_scale = self._config.rep_scale
                kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )
                assert kl_loss.shape == embed.shape[:2], kl_loss.shape
                preds = {}
                for name, head in self.heads.items():
                    grad_head = name in self._config.grad_heads
                    feat = self.dynamics.get_feat(post)
                    feat = feat if grad_head else feat.detach()
                    pred = head(feat)
                    if type(pred) is dict:
                        preds.update(pred)
                    else:
                        preds[name] = pred
                losses = {}
                for name, pred in preds.items():
                    loss = -pred.log_prob(data[name])
                    assert loss.shape == embed.shape[:2], (name, loss.shape)
                    losses[name] = loss
                scaled = {
                    key: value * self._scales.get(key, 1.0)
                    for key, value in losses.items()
                }
                model_loss = sum(scaled.values()) + kl_loss
            metrics = self._model_opt(torch.mean(model_loss), self.parameters())

        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        metrics["kl_free"] = kl_free
        metrics["dyn_scale"] = dyn_scale
        metrics["rep_scale"] = rep_scale
        metrics["dyn_loss"] = to_np(dyn_loss)
        metrics["rep_loss"] = to_np(rep_loss)
        metrics["kl"] = to_np(torch.mean(kl_value))
        with torch.cuda.amp.autocast(self._use_amp):
            metrics["prior_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(prior).entropy())
            )
            metrics["post_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(post).entropy())
            )
            context = dict(
                embed=embed,
                feat=self.dynamics.get_feat(post),
                kl=kl_value,
                postent=self.dynamics.get_dist(post).entropy(),
            )
        post = {k: v.detach() for k, v in post.items()}
        return post, context, metrics

    # this function is called during both rollout and training
    def preprocess(self, obs):
        # obs = obs.copy()
        # obs["image"] = torch.Tensor(obs["image"]) / 255.0

        # discount in obs seems useless
        # if "discount" in obs:
        #     obs["discount"] *= self._config.discount
            # (batch_size, batch_length) -> (batch_size, batch_length, 1)
            # obs["discount"] = torch.Tensor(obs["discount"]).unsqueeze(-1)
        # 'is_first' is necesarry to initialize hidden state at training
        assert "is_first" in obs
        # 'is_terminal' is necesarry to train cont_head
        # assert "is_terminal" in obs
        # obs["cont"] = torch.Tensor(1.0 - obs["is_terminal"]).unsqueeze(-1)
        obs = {k: torch.Tensor(v).to(self._config.device) for k, v in obs.items()}
        return obs

    def video_pred(self, data):
        data = self.preprocess(data)
        embed = self.encoder(data)

        states, _ = self.dynamics.observe(
            embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
        )
        recon = self.heads["decoder"](self.dynamics.get_feat(states))["image"].mode()[
            :6
        ]
        reward_post = self.heads["reward"](self.dynamics.get_feat(states)).mode()[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.dynamics.imagine_with_action(data["action"][:6, 5:], init)
        openl = self.heads["decoder"](self.dynamics.get_feat(prior))["image"].mode()
        reward_prior = self.heads["reward"](self.dynamics.get_feat(prior)).mode()
        # observed image is given until 5 steps
        model = torch.cat([recon[:, :5], openl], 1)
        truth = data["image"][:6]
        model = model
        error = (model - truth + 1.0) / 2.0

        return torch.cat([truth, model], 2)
