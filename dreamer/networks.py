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

import math
import numpy as np
import re

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as torchd

from . import tools

class RSSM(nn.Module):
    """
    循环状态空间模型
    """
    def __init__(
        self,
        stoch=30,   # 随机状态维度，32
        deter=200,  # 确定性状态维度，512
        hidden=200, # 隐藏层维度，512
        rec_depth=1,    # 1
        discrete=False, # 离散状态数，32
        act="SiLU",
        norm=True,
        mean_act="none",
        std_act="softplus", # 'sigmoid2'
        min_std=0.1,
        unimix_ratio=0.01,
        initial="learned",
        num_actions=None,   # 动作空间维度，12
        embed=None,     # 编码器输出维度，5120
        device=None,
    ):
        super(RSSM, self).__init__()
        self._stoch = stoch
        self._deter = deter
        self._hidden = hidden
        self._min_std = min_std
        self._rec_depth = rec_depth
        self._discrete = discrete
        act = getattr(torch.nn, act)
        self._mean_act = mean_act
        self._std_act = std_act
        self._unimix_ratio = unimix_ratio
        self._initial = initial
        self._num_actions = num_actions
        self._embed = embed
        self._device = device

        # 1. _img_in_layers: Linear(1036/44 -> 512) --> LayerNorm --> SiLU
        inp_layers = []
        if self._discrete:  # 离散情况: 32 * 32 + 12 = 1036
            inp_dim = self._stoch * self._discrete + num_actions
        else:    # 连续情况: 32 + 12 = 44
            inp_dim = self._stoch + num_actions
        inp_layers.append(nn.Linear(inp_dim, self._hidden, bias=False)) # (1036, 512)
        if norm:
            inp_layers.append(nn.LayerNorm(self._hidden, eps=1e-03))
        inp_layers.append(act())
        self._img_in_layers = nn.Sequential(*inp_layers)
        self._img_in_layers.apply(tools.weight_init)

        # 2. _cell: GRU单元, Linear(512 + 512 -> 512 * 3) --> LayerNorm
        self._cell = GRUCell(self._hidden, self._deter, norm=norm)
        self._cell.apply(tools.weight_init)

        # 3. _img_out_layers: Linear(512 -> 512) --> LayerNorm --> SiLU
        img_out_layers = []
        inp_dim = self._deter
        img_out_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
        if norm:
            img_out_layers.append(nn.LayerNorm(self._hidden, eps=1e-03))
        img_out_layers.append(act())
        self._img_out_layers = nn.Sequential(*img_out_layers)
        self._img_out_layers.apply(tools.weight_init)

        # 4. _obs_out_layers: Linear(512 + 5120 -> 512) --> LayerNorm ->- SiLU
        obs_out_layers = []
        inp_dim = self._deter + self._embed
        obs_out_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
        if norm:
            obs_out_layers.append(nn.LayerNorm(self._hidden, eps=1e-03))
        obs_out_layers.append(act())
        self._obs_out_layers = nn.Sequential(*obs_out_layers)
        self._obs_out_layers.apply(tools.weight_init)


        if self._discrete:  # 离散
            # 5. _imgs_stat_layer: Linear(512 -> 32 * 32 = 1024)
            self._imgs_stat_layer = nn.Linear(
                self._hidden, self._stoch * self._discrete
            )
            self._imgs_stat_layer.apply(tools.uniform_weight_init(1.0))
            # 6. _obs_stat_layer: Linear(512 -> 32 * 32 = 1024)
            self._obs_stat_layer = nn.Linear(self._hidden, self._stoch * self._discrete)
            self._obs_stat_layer.apply(tools.uniform_weight_init(1.0))
        else:   # 连续
            self._imgs_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)
            self._imgs_stat_layer.apply(tools.uniform_weight_init(1.0))
            self._obs_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)
            self._obs_stat_layer.apply(tools.uniform_weight_init(1.0))

        if self._initial == "learned":  # 默认
            # (1, 512)
            self.W = torch.nn.Parameter(
                torch.zeros((1, self._deter), device=torch.device(self._device)),
                requires_grad=True,
            )

    def initial(self, batch_size):
        deter = torch.zeros(batch_size, self._deter).to(self._device)
        if self._discrete:
            state = dict(
                logit=torch.zeros([batch_size, self._stoch, self._discrete]).to(
                    self._device
                ),
                stoch=torch.zeros([batch_size, self._stoch, self._discrete]).to(
                    self._device
                ),
                deter=deter,
            )
        else:
            state = dict(
                mean=torch.zeros([batch_size, self._stoch]).to(self._device),
                std=torch.zeros([batch_size, self._stoch]).to(self._device),
                stoch=torch.zeros([batch_size, self._stoch]).to(self._device),
                deter=deter,
            )
        if self._initial == "zeros":
            return state
        elif self._initial == "learned":
            state["deter"] = torch.tanh(self.W).repeat(batch_size, 1)
            state["stoch"] = self.get_stoch(state["deter"])
            return state
        else:
            raise NotImplementedError(self._initial)

    def observe(self, embed, action, is_first, state=None):
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        # (batch, time, ch) -> (time, batch, ch)
        embed, action, is_first = swap(embed), swap(action), swap(is_first)
        # prev_state[0] means selecting posterior of return(posterior, prior) from obs_step
        post, prior = tools.static_scan(
            lambda prev_state, prev_act, embed, is_first: self.obs_step(
                prev_state[0], prev_act, embed, is_first
            ),
            (action, embed, is_first),
            (state, state),
        )

        # (batch, time, stoch, discrete_num) -> (batch, time, stoch, discrete_num)
        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}
        return post, prior

    def imagine_with_action(self, action, state):
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        assert isinstance(state, dict), state
        action = action
        action = swap(action)
        prior = tools.static_scan(self.img_step, [action], state)
        prior = prior[0]
        prior = {k: swap(v) for k, v in prior.items()}
        return prior

    def get_feat(self, state):
        stoch = state["stoch"]
        if self._discrete:
            shape = list(stoch.shape[:-2]) + [self._stoch * self._discrete]
            stoch = stoch.reshape(shape)
        return torch.cat([stoch, state["deter"]], -1)

    def get_deter_feat(self, state):
        return state["deter"]

    def get_dist(self, state, dtype=None):
        if self._discrete:
            logit = state["logit"]
            dist = torchd.independent.Independent(
                tools.OneHotDist(logit, unimix_ratio=self._unimix_ratio), 1
            )
        else:
            mean, std = state["mean"], state["std"]
            dist = tools.ContDist(
                torchd.independent.Independent(torchd.normal.Normal(mean, std), 1)
            )
        return dist

    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True):
        # initialize all prev_state
        if prev_state == None or torch.sum(is_first) == len(is_first):
            prev_state = self.initial(len(is_first))
            prev_action = torch.zeros((len(is_first), self._num_actions)).to(
                self._device
            )
        # overwrite the prev_state only where is_first=True
        elif torch.sum(is_first) > 0:
            is_first = is_first[:, None]
            prev_action *= 1.0 - is_first
            init_state = self.initial(len(is_first))
            for key, val in prev_state.items():
                is_first_r = torch.reshape(
                    is_first,
                    is_first.shape + (1,) * (len(val.shape) - len(is_first.shape)),
                )
                prev_state[key] = (
                    val * (1.0 - is_first_r) + init_state[key] * is_first_r
                )

        prior = self.img_step(prev_state, prev_action)
        x = torch.cat([prior["deter"], embed], -1)
        # (batch_size, prior_deter + embed) -> (batch_size, hidden)
        x = self._obs_out_layers(x)
        # (batch_size, hidden) -> (batch_size, stoch, discrete_num)
        stats = self._suff_stats_layer("obs", x)
        if sample:
            stoch = self.get_dist(stats).sample()
        else:
            stoch = self.get_dist(stats).mode()
        post = {"stoch": stoch, "deter": prior["deter"], **stats}
        return post, prior

    def img_step(self, prev_state, prev_action, sample=True):
        # (batch, stoch, discrete_num)
        prev_stoch = prev_state["stoch"]
        if self._discrete:
            shape = list(prev_stoch.shape[:-2]) + [self._stoch * self._discrete]
            # (batch, stoch, discrete_num) -> (batch, stoch * discrete_num)
            prev_stoch = prev_stoch.reshape(shape)
        # (batch, stoch * discrete_num) -> (batch, stoch * discrete_num + action)
        x = torch.cat([prev_stoch, prev_action], -1)
        # (batch, stoch * discrete_num + action, embed) -> (batch, hidden)
        x = self._img_in_layers(x)
        for _ in range(self._rec_depth):  # rec depth is not correctly implemented
            deter = prev_state["deter"]
            # (batch, hidden), (batch, deter) -> (batch, deter), (batch, deter)
            x, deter = self._cell(x, [deter])
            deter = deter[0]  # Keras wraps the state in a list.
        # (batch, deter) -> (batch, hidden)
        x = self._img_out_layers(x)
        # (batch, hidden) -> (batch_size, stoch, discrete_num)
        stats = self._suff_stats_layer("ims", x)
        if sample:
            stoch = self.get_dist(stats).sample()
        else:
            stoch = self.get_dist(stats).mode()
        prior = {"stoch": stoch, "deter": deter, **stats}
        return prior

    def get_stoch(self, deter):
        x = self._img_out_layers(deter)
        stats = self._suff_stats_layer("ims", x)
        dist = self.get_dist(stats)
        return dist.mode()

    def _suff_stats_layer(self, name, x):
        if self._discrete:
            if name == "ims":
                x = self._imgs_stat_layer(x)
            elif name == "obs":
                x = self._obs_stat_layer(x)
            else:
                raise NotImplementedError
            logit = x.reshape(list(x.shape[:-1]) + [self._stoch, self._discrete])
            return {"logit": logit}
        else:
            if name == "ims":
                x = self._imgs_stat_layer(x)
            elif name == "obs":
                x = self._obs_stat_layer(x)
            else:
                raise NotImplementedError
            mean, std = torch.split(x, [self._stoch] * 2, -1)
            mean = {
                "none": lambda: mean,
                "tanh5": lambda: 5.0 * torch.tanh(mean / 5.0),
            }[self._mean_act]()
            std = {
                "softplus": lambda: torch.softplus(std),
                "abs": lambda: torch.abs(std + 1),
                "sigmoid": lambda: torch.sigmoid(std),
                "sigmoid2": lambda: 2 * torch.sigmoid(std / 2),
            }[self._std_act]()
            std = std + self._min_std
            return {"mean": mean, "std": std}

    def kl_loss(self, post, prior, free, dyn_scale, rep_scale):
        kld = torchd.kl.kl_divergence
        dist = lambda x: self.get_dist(x)
        sg = lambda x: {k: v.detach() for k, v in x.items()}

        rep_loss = value = kld(
            dist(post) if self._discrete else dist(post)._dist,
            dist(sg(prior)) if self._discrete else dist(sg(prior))._dist,
        )
        dyn_loss = kld(
            dist(sg(post)) if self._discrete else dist(sg(post))._dist,
            dist(prior) if self._discrete else dist(prior)._dist,
        )
        # this is implemented using maximum at the original repo as the gradients are not backpropagated for the out of limits.
        rep_loss = torch.clip(rep_loss, min=free)
        dyn_loss = torch.clip(dyn_loss, min=free)
        loss = dyn_scale * dyn_loss + rep_scale * rep_loss

        return loss, value, dyn_loss, rep_loss


class MultiEncoder(nn.Module):
    def __init__(
        self,
        shapes, # {"prop": (33,), "image": (64, 64, 1)}
        mlp_keys,   # '.*'
        cnn_keys,   # 'image'
        act,    # 'SiLU'
        norm,   # True
        cnn_depth,  # 32
        kernel_size,    # 4
        minres,     # 4
        mlp_layers, # 5
        mlp_units,  # 1024
        symlog_inputs,  # True
        use_camera = False,
    ):
        super(MultiEncoder, self).__init__()
        self.use_camera = use_camera
        # 过滤一些 keys
        excluded = ("is_first", "is_last", "is_terminal", "reward",  "height_map")
        # 过滤后的 shapes: {"prop": (33,), "image": (64, 64, 1)}
        shapes = {
            k: v
            for k, v in shapes.items()
            if k not in excluded and not k.startswith("log_")
        }
        # {"image": (64, 64, 1)}
        self.cnn_shapes = {
            k: v for k, v in shapes.items() if len(v) == 3 and re.match(cnn_keys, k)
        }
        # {"prop": (33,)}
        self.mlp_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) in (1, 2) and re.match(mlp_keys, k)
        }
        print("Encoder CNN shapes:", self.cnn_shapes)
        print("Encoder MLP shapes:", self.mlp_shapes)

        self.outdim = 0
        if self.cnn_shapes:
            input_ch = sum([v[-1] for v in self.cnn_shapes.values()])   # 所有输入图像的 ch（因只有深度图，所以为 1）
            input_shape = tuple(self.cnn_shapes.values())[0][:2] + (input_ch,)  # (64, 64, 1)
            # (batch, time, 64, 64, 1) ==> (batch, time, 4096)
            self._cnn = ConvEncoder(
                input_shape, # (64, 64, 1)
                cnn_depth,  # 32
                act,        # 'SiLU'
                norm,       # True
                kernel_size, # 4
                minres  # 4
            )
            self.outdim += self._cnn.outdim # 4096
            print('cnn outdim', self._cnn.outdim)
        if self.mlp_shapes:
            input_size = sum([sum(v) for v in self.mlp_shapes.values()])
            # (batch, time, 33) ==> (batch, time, 1024)
            self._mlp = MLP(
                input_size, # 33
                None,
                mlp_layers, # 5
                mlp_units,  # 1024
                act,    # 'SiLU'
                norm,   # True
                symlog_inputs=symlog_inputs,    # True
                name="Encoder",
            )
            self.outdim += mlp_units    # 4096 + 1024 = 5120
            print('mlp outdim', mlp_units)

        print('total outdim:', self.outdim)

    def forward(self, obs):
        outputs = []
        if self.cnn_shapes:
            if(self.use_camera):
                inputs = torch.cat([obs[k] for k in self.cnn_shapes], -1)
                outputs.append(self._cnn(inputs))
            else:
                outputs.append(torch.zeros((obs["is_first"].shape + (self._cnn.outdim,)), device=obs["is_first"].device))
        if self.mlp_shapes:
            inputs = torch.cat([obs[k] for k in self.mlp_shapes], -1)
            outputs.append(self._mlp(inputs))
        outputs = torch.cat(outputs, -1)    # (batch, time, 5120)
        return outputs


class MultiDecoder(nn.Module):
    def __init__(
        self,
        feat_size,  # 1536
        shapes,     # {"prop": (33,), "image": (64, 64, 1)}
        mlp_keys,   # '.*'
        cnn_keys,   # 'image'
        act,    # 'SiLU'
        norm,   # True
        cnn_depth,  # 32
        kernel_size,    # 4
        minres,     # 4
        mlp_layers, # 5
        mlp_units,  # 1024
        cnn_sigmoid,    # False
        image_dist,     # mse
        vector_dist,    # symlog_mse
        outscale,       # 1.0
        use_camera=False,
    ):
        super(MultiDecoder, self).__init__()
        self.use_camera = use_camera
        excluded = ("is_first", "is_last", "is_terminal", "height_map")
        shapes = {k: v for k, v in shapes.items() if k not in excluded}
        # {"image": (64, 64, 1)}
        self.cnn_shapes = {
            k: v for k, v in shapes.items() if len(v) == 3 and re.match(cnn_keys, k)
        }
        # {"prop": (33,)}
        self.mlp_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) in (1, 2) and re.match(mlp_keys, k)
        }
        print("Decoder CNN shapes:", self.cnn_shapes)
        print("Decoder MLP shapes:", self.mlp_shapes)

        if self.cnn_shapes:
            some_shape = list(self.cnn_shapes.values())[0]  # (64, 64, 1)
            shape = (sum(x[-1] for x in self.cnn_shapes.values()),) + some_shape[:-1]   # 所有输入图像的维度（因只有深度图，所以为 ch = 1, (1, 64, 64)）
            # (batch, time, 1536) ==> (batch, time, 64, 64, 1)
            self._cnn = ConvDecoder(
                feat_size,  # 1536
                shape,      # (1, 64, 64)
                cnn_depth,  # 32
                act,    # 'SiLU'
                norm,   # True
                kernel_size,    # 4
                minres,         # 4
                outscale=outscale,  # 1.0
                cnn_sigmoid=cnn_sigmoid,    # False
            )
        if self.mlp_shapes:
            # (batch, time, 1536) ==> {"prop": SymlogDist(out_mlp)}, 其中out_mlp的维度为 (33,)
            self._mlp = MLP(
                feat_size,  # 1536
                self.mlp_shapes,    # {"prop": (33,)}
                mlp_layers,     # 5
                mlp_units,      # 1024
                act,    # 'SiLU'
                norm,   # True
                vector_dist,    # symlog_mse
                outscale=outscale,  # 1.0
                name="Decoder",
            )
        self._image_dist = image_dist

    def forward(self, features):
        dists = {}
        if self.cnn_shapes and self.use_camera:
            feat = features
            outputs = self._cnn(feat)   # (batch, time, 64, 64, 1)
            split_sizes = [v[-1] for v in self.cnn_shapes.values()]
            outputs = torch.split(outputs, split_sizes, -1)
            # {"images": MSEDist(mean)}, 其中mean的维度为 (batch, time, 64, 64, 1)
            dists.update(
                {
                    key: self._make_image_dist(output)
                    for key, output in zip(self.cnn_shapes.keys(), outputs)
                }
            )
        if self.mlp_shapes:
            dists.update(self._mlp(features))
        # {"images": MSEDist(out_cnn), "prop": SymlogDist(out_mlp)}
        # out_cnn 维度为 (batch, time, 64, 64, 1) , out_mlp 维度为 (33,)
        return dists

    def _make_image_dist(self, mean):
        if self._image_dist == "normal":
            return tools.ContDist(
                torchd.independent.Independent(torchd.normal.Normal(mean, 1), 3)
            )
        if self._image_dist == "mse":
            return tools.MSEDist(mean)
        raise NotImplementedError(self._image_dist)


class ConvEncoder(nn.Module):
    def __init__(
        self,
        input_shape,    # (64, 64, 1)
        depth=32,   # 第一个卷积的输出通道数
        act="SiLU",
        norm=True,  # 是否使用层归一化
        kernel_size=4,
        minres=4,   # 最小分辨率
    ):
        super(ConvEncoder, self).__init__()
        act = getattr(torch.nn, act)
        h, w, input_ch = input_shape
        stages = int(np.log2(w) - np.log2(minres))  # 下采样卷积层的个数 = 6 - 2 = 4
        in_dim = input_ch
        out_dim = depth
        layers = []
        for i in range(stages):
            # 添加带padding的卷积层
            layers.append(
                Conv2dSamePad(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    kernel_size=kernel_size,
                    stride=2,
                    bias=False,
                )
            )
            if norm:    # 添加层归一化
                layers.append(ImgChLayerNorm(out_dim))
            layers.append(act())    # 添加激活函数
            in_dim = out_dim
            out_dim *= 2
            h, w = (h+1) // 2, (w+1) // 2
        # out_dim = 32 * 2^4 = 512
        # h, w = 4, 4

        self.outdim = out_dim // 2 * h * w  # 256 * 4 * 4 = 4096
        # Conv(1, 32, 32) --> LayerNorm --> SiLU
        # Conv(32, 64, 16) --> LayerNorm --> SiLU
        # Conv(64, 128, 8) --> LayerNorm --> SiLU
        # Conv(128, 256, 4) --> LayerNorm --> SiLU
        self.layers = nn.Sequential(*layers)
        self.layers.apply(tools.weight_init)

    def forward(self, obs):
        obs -= 0.5  # 归一化到 [-0.5, 0.5]
        # (batch, time, h, w, ch) -> (batch * time, h, w, ch)   ==> (batch * time, 64, 64, 1)
        x = obs.reshape((-1,) + tuple(obs.shape[-3:]))
        # (batch * time, h, w, ch) -> (batch * time, ch, h, w)  ==> (batch * time, 1, 64, 64)
        x = x.permute(0, 3, 1, 2)
        # print('init encoder shape:', x.shape)
        # for layer in self.layers:
        #     x = layer(x)
        #     print(x.shape)
        x = self.layers(x)  # (batch * time, ch, h, w)  ==> (batch * time, 256, 4, 4)
        # (batch * time, ...) -> (batch * time, -1)     ==> (batch * time, 256 * 4 * 4)
        x = x.reshape([x.shape[0], np.prod(x.shape[1:])])
        # (batch * time, -1) -> (batch, time, -1)       ==> (batch, time, 256 * 4 * 4)
        return x.reshape(list(obs.shape[:-3]) + [x.shape[-1]])


class ConvDecoder(nn.Module):
    def __init__(
        self,
        feat_size,  # 1536
        shape=(3, 64, 64),  # (1, 64, 64)
        depth=32,
        act=nn.ELU, # 'SiLU'
        norm=True,
        kernel_size=4,
        minres=4,
        outscale=1.0,
        cnn_sigmoid=False,
    ):
        # add this to fully recover the process of conv encoder
        input_ch, h, w = shape
        stages = int(np.log2(w) - np.log2(minres))  # 4

        self.h_list = []
        self.w_list = []
        for i in range(stages):
            h, w = (h+1) // 2, (w+1) // 2
            self.h_list.append(h)
            self.w_list.append(w)
        self.h_list = self.h_list[::-1] # [4, 8, 16, 32]
        self.w_list = self.w_list[::-1]
        self.h_list.append(shape[1])    # [4, 8, 16, 32, 64]
        self.w_list.append(shape[2])

        super(ConvDecoder, self).__init__()
        act = getattr(torch.nn, act)
        self._shape = shape
        self._cnn_sigmoid = cnn_sigmoid
        layer_num = len(self.h_list) - 1    # 4
        # layer_num = int(np.log2(shape[2]) - np.log2(minres))
        # self._minres = minres
        # out_ch = minres**2 * depth * 2 ** (layer_num - 1)
        out_ch = self.h_list[0] * self.w_list[0] * depth * 2 ** (len(self.h_list) - 2)  # 4 * 4 * 32 * 2^3 = 4096
        self._embed_size = out_ch   # 4096

        self._linear_layer = nn.Linear(feat_size, out_ch)   # Linear(1536 -> 4096)
        self._linear_layer.apply(tools.uniform_weight_init(outscale))
        in_dim = out_ch // (self.h_list[0] * self.w_list[0])    # 256
        out_dim = in_dim // 2   # 128

        layers = []
        # h, w = minres, minres
        for i in range(layer_num):
            bias = False
            if i == layer_num - 1:
                out_dim = self._shape[0]    # 1
                act = False
                bias = True
                norm = False

            if i != 0:
                in_dim = 2 ** (layer_num - (i - 1) - 2) * depth     # 32 * 2^[2,1,0] = [128, 64, 32]
            # pad_h, outpad_h = self.calc_same_pad(k=kernel_size, s=2, d=1)
            # pad_w, outpad_w = self.calc_same_pad(k=kernel_size, s=2, d=1)

            if(self.h_list[i] * 2 == self.h_list[i+1]):
                pad_h, outpad_h = 1, 0
            else:
                pad_h, outpad_h = 2, 1

            if(self.w_list[i] * 2 == self.w_list[i+1]):
                pad_w, outpad_w = 1, 0
            else:
                pad_w, outpad_w = 2, 1

            layers.append(
                nn.ConvTranspose2d(
                    in_dim,
                    out_dim,
                    kernel_size,
                    2,
                    padding=(pad_h, pad_w),
                    output_padding=(outpad_h, outpad_w),
                    bias=bias,
                )
            )
            if norm:
                layers.append(ImgChLayerNorm(out_dim))
            if act:
                layers.append(act())
            in_dim = out_dim
            out_dim //= 2
            # h, w = h * 2, w * 2
        [m.apply(tools.weight_init) for m in layers[:-1]]
        layers[-1].apply(tools.uniform_weight_init(outscale))   # 最后一层做特殊初始化
        # ConvT(256, 128, 8) --> LayerNorm --> SiLU
        # ConvT(128, 64, 16) --> LayerNorm --> SiLU
        # ConvT(64, 32, 32) --> LayerNorm --> SiLU
        # ConvT(32, 1, 64) --> LayerNorm --> SiLU
        self.layers = nn.Sequential(*layers)

    def calc_same_pad(self, k, s, d):
        val = d * (k - 1) - s + 1
        pad = math.ceil(val / 2)
        outpad = pad * 2 - val
        return pad, outpad

    def forward(self, features, dtype=None):
        x = self._linear_layer(features)    # (batch, time, 1536) ==> (batch, time, 4096)
        # (batch, time, -1) -> (batch * time, h, w, ch)     ==> (batch * time, 4, 4, 256)
        x = x.reshape(
            [-1, self.h_list[0], self.w_list[0], self._embed_size // (self.h_list[0] * self.w_list[0])]
        )
        # (batch, time, -1) -> (batch * time, ch, h, w)     ==> (batch * time, 256, 4, 4)
        x = x.permute(0, 3, 1, 2)
        # print('init decoder shape:', x.shape)
        # for layer in self.layers:
        #     x = layer(x)
        #     print(x.shape)
        x = self.layers(x)      # (batch * time, 256, 4, 4)  ==> (batch * time, 1, 64, 64)
        # (batch, time, -1) -> (batch, time, ch, h, w)          ==> (batch, time, 1, 64, 64)
        mean = x.reshape(features.shape[:-1] + self._shape)
        # (batch, time, ch, h, w) -> (batch, time, h, w, ch)    ==> (batch, time, 64, 64, 1)
        mean = mean.permute(0, 1, 3, 4, 2)
        if self._cnn_sigmoid:
            mean = F.sigmoid(mean)
        # else:
        #     mean += 0.5
        return mean


class MLP(nn.Module):
    def __init__(
        self,
        inp_dim,    # 输入维度: Encoder 33; Decoder 1536; Reward 1536
        shape,      # Encoder None,无特定输出形状要求; Decoder {"prop": (33,)}; Reward (255,)
        layers,     # 隐藏层数，Encoder Decoder 5; Reward 2
        units,      # 每层神经元个数，Encoder Decoder 1024; Reward 512
        act="SiLU",
        norm=True,  # 使用层归一化
        dist="normal",  # Encoder "normal"; Decoder "symlog_mse"; Reward  "symlog_disc"
        std=1.0,
        min_std=0.1,
        max_std=1.0,
        absmax=None,
        temp=0.1,
        unimix_ratio=0.01,
        outscale=1.0,   # Encoder Decoder 1.0; Reward 0.0
        symlog_inputs=False, # Encoder True, 输入使用symlog变换; Decoder False
        device="cuda",
        name="NoName",  # "Encoder"; "Decoder"; "Reward"
    ):
        super(MLP, self).__init__()
        self._shape = (shape,) if isinstance(shape, int) else shape
        if self._shape is not None and len(self._shape) == 0:
            self._shape = (1,)
        act = getattr(torch.nn, act)
        self._dist = dist
        self._std = std if isinstance(std, str) else torch.tensor((std,), device=device)
        self._min_std = min_std
        self._max_std = max_std
        self._absmax = absmax
        self._temp = temp
        self._unimix_ratio = unimix_ratio
        self._symlog_inputs = symlog_inputs
        self._device = device

        self.layers = nn.Sequential()
        for i in range(layers):
            self.layers.add_module(
                f"{name}_linear{i}", nn.Linear(inp_dim, units, bias=False)
            )
            if norm:
                self.layers.add_module(
                    f"{name}_norm{i}", nn.LayerNorm(units, eps=1e-03)
                )
            self.layers.add_module(f"{name}_act{i}", act())
            if i == 0:
                inp_dim = units
        self.layers.apply(tools.weight_init)


        if isinstance(self._shape, dict): # Decoder
            self.mean_layer = nn.ModuleDict()
            for name, shape in self._shape.items():
                # {"prop": Linear(1024, 33)}
                self.mean_layer[name] = nn.Linear(inp_dim, np.prod(shape))
            self.mean_layer.apply(tools.uniform_weight_init(outscale))
            if self._std == "learned":
                assert dist in ("tanh_normal", "normal", "trunc_normal", "huber"), dist
                self.std_layer = nn.ModuleDict()
                for name, shape in self._shape.items():
                    self.std_layer[name] = nn.Linear(inp_dim, np.prod(shape))
                self.std_layer.apply(tools.uniform_weight_init(outscale))
        elif self._shape is not None: # Reward
            # Linear(512, 255)
            self.mean_layer = nn.Linear(inp_dim, np.prod(self._shape))
            self.mean_layer.apply(tools.uniform_weight_init(outscale))
            if self._std == "learned":
                assert dist in ("tanh_normal", "normal", "trunc_normal", "huber"), dist
                self.std_layer = nn.Linear(units, np.prod(self._shape))
                self.std_layer.apply(tools.uniform_weight_init(outscale))

    def forward(self, features, dtype=None):
        x = features
        if self._symlog_inputs: # Encoder中的 对输入进行symlog变换
            x = tools.symlog(x)
        out = self.layers(x)    # Encoder：(33,) --> (1024,)； Decoder：(1536,) --> (1024,); Reward: (1536,) --> (512,)
        # Used for encoder output
        if self._shape is None: # Encoder 直接返回 (1024,)
            return out

        if isinstance(self._shape, dict): # Decoder
            dists = {}
            for name, shape in self._shape.items():
                mean = self.mean_layer[name](out)   # (1024,) --> (33,)
                if self._std == "learned":
                    std = self.std_layer[name](out)
                else:   # 使用固定标准差 1.0
                    std = self._std
                # {"prop": tools.SymlogDist(mean)}
                dists.update({name: self.dist(self._dist, mean, std, shape)})
            return dists
        else: # Reward
            mean = self.mean_layer(out) # (512) --> (255,)
            if self._std == "learned":
                std = self.std_layer(out)
            else:
                std = self._std
            # tools.DiscDist(mean)
            return self.dist(self._dist, mean, std, self._shape)

    def dist(self, dist, mean, std, shape):
        if self._dist == "tanh_normal":
            mean = torch.tanh(mean)
            std = F.softplus(std) + self._min_std
            dist = torchd.normal.Normal(mean, std)
            dist = torchd.transformed_distribution.TransformedDistribution(
                dist, tools.TanhBijector()
            )
            dist = torchd.independent.Independent(dist, 1)
            dist = tools.SampleDist(dist)
        elif self._dist == "normal":
            std = (self._max_std - self._min_std) * torch.sigmoid(
                std + 2.0
            ) + self._min_std
            dist = torchd.normal.Normal(torch.tanh(mean), std)
            dist = tools.ContDist(
                torchd.independent.Independent(dist, 1), absmax=self._absmax
            )
        elif self._dist == "normal_std_fixed":
            dist = torchd.normal.Normal(mean, self._std)
            dist = tools.ContDist(
                torchd.independent.Independent(dist, 1), absmax=self._absmax
            )
        elif self._dist == "trunc_normal":
            mean = torch.tanh(mean)
            std = 2 * torch.sigmoid(std / 2) + self._min_std
            dist = tools.SafeTruncatedNormal(mean, std, -1, 1)
            dist = tools.ContDist(
                torchd.independent.Independent(dist, 1), absmax=self._absmax
            )
        elif self._dist == "onehot":
            dist = tools.OneHotDist(mean, unimix_ratio=self._unimix_ratio)
        elif self._dist == "onehot_gumble":
            dist = tools.ContDist(
                torchd.gumbel.Gumbel(mean, 1 / self._temp), absmax=self._absmax
            )
        elif dist == "huber":
            dist = tools.ContDist(
                torchd.independent.Independent(
                    tools.UnnormalizedHuber(mean, std, 1.0),
                    len(shape),
                    absmax=self._absmax,
                )
            )
        elif dist == "binary":
            dist = tools.Bernoulli(
                torchd.independent.Independent(
                    torchd.bernoulli.Bernoulli(logits=mean), len(shape)
                )
            )
        elif dist == "symlog_disc":
            # symlog(x) = sign(x) * log(1 + |x|)
            # 基于分桶(buckets)的离散概率分布： 将连续值空间离散化为255个桶（-20到20区间），使用神经网络输出的logits预测每个桶的概率，通过加权最近邻桶的方式计算log_prob
            # 适用于：需要精确建模多峰分布的场景，处理有明确离散特性的数据（如控制指令），当需要避免连续分布的平滑假设时
            dist = tools.DiscDist(logits=mean, device=self._device)
        elif dist == "symlog_mse":
            # symlog(x) = sign(x) * log(1 + |x|)
            # 基于MSE的连续值分布： 直接在symlog空间计算均方误差，将大范围数值压缩到较小范围，保持零点的连续性，对接近零的值近似线性，对大值近似对数
            # 适用于：用于预测连续信号，需要保持数值相对大小的场景，处理动态范围大的传感器数据
            dist = tools.SymlogDist(mean)
        else:
            raise NotImplementedError(dist)
        return dist


class GRUCell(nn.Module):
    def __init__(self, inp_size, size, norm=True, act=torch.tanh, update_bias=-1):
        super(GRUCell, self).__init__()
        self._inp_size = inp_size
        self._size = size
        self._act = act
        self._update_bias = update_bias
        self.layers = nn.Sequential()
        self.layers.add_module(
            "GRU_linear", nn.Linear(inp_size + size, 3 * size, bias=False)
        )
        if norm:
            self.layers.add_module("GRU_norm", nn.LayerNorm(3 * size, eps=1e-03))

    @property
    def state_size(self):
        return self._size

    def forward(self, inputs, state):
        state = state[0]  # Keras wraps the state in a list.
        parts = self.layers(torch.cat([inputs, state], -1))
        reset, cand, update = torch.split(parts, [self._size] * 3, -1)
        reset = torch.sigmoid(reset)
        cand = self._act(reset * cand)
        update = torch.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * state
        return output, [output]


class Conv2dSamePad(torch.nn.Conv2d):
    def calc_same_pad(self, i, k, s, d):
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x):
        ih, iw = x.size()[-2:]
        pad_h = self.calc_same_pad(
            i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0]
        )
        pad_w = self.calc_same_pad(
            i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1]
        )

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )

        ret = F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return ret


class ImgChLayerNorm(nn.Module):
    def __init__(self, ch, eps=1e-03):
        super(ImgChLayerNorm, self).__init__()
        self.norm = torch.nn.LayerNorm(ch, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x
