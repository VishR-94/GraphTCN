import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


modern_tcn_root = Path(__file__).resolve().parents[2] / "external" / "ModernTCN" / "ModernTCN-Long-term-forecasting"
sys.path.insert(0, str(modern_tcn_root))
from models.ModernTCN import Model


class ModernTCNBackbone(nn.Module):
    def __init__(self, input_dim=5, target_index=3, context_length=60, num_outputs=5, d_model=32,
                 patch_size=8, patch_stride=4, large_kernel=15, small_kernel=5, num_blocks=1,
                 ffn_ratio=1, dropout=0.05, head_dropout=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.target_index = target_index
        self.context_length = context_length
        self.num_outputs = num_outputs
        self.d_model = d_model
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.num_patches = context_length // patch_stride

        config = SimpleNamespace(
            stem_ratio=6, downsample_ratio=2, ffn_ratio=ffn_ratio, num_blocks=[num_blocks],
            large_size=[large_kernel], small_size=[small_kernel], dims=[d_model] * 4,
            dw_dims=[d_model] * 4, enc_in=input_dim, small_kernel_merged=False, dropout=dropout,
            head_dropout=head_dropout, use_multi_scale=False, revin=0, affine=0, subtract_last=0,
            freq="t", seq_len=context_length, pred_len=num_outputs, individual=0, decomposition=0,
            kernel_size=25, patch_size=patch_size, patch_stride=patch_stride,
        )
        self.model = Model(config).model

    def forward(self, x):
        batch, steps, nodes, channels = x.shape
        x = x.permute(0, 2, 3, 1).reshape(batch * nodes, channels, steps)
        hidden = self.model.forward_feature(x)[:, self.target_index]
        return hidden.reshape(batch, nodes, self.d_model, self.num_patches).permute(0, 3, 1, 2).contiguous()

    def forecast(self, hidden):
        batch, patches, nodes, hidden_dim = hidden.shape
        hidden = hidden.permute(0, 2, 3, 1).reshape(batch * nodes, 1, hidden_dim, patches)
        predictions = self.model.head(hidden)[:, 0]
        return predictions.reshape(batch, nodes, self.num_outputs).permute(0, 2, 1).unsqueeze(-1).contiguous()


class GraphTCN(nn.Module):
    def __init__(self, prior, input_dim=5, target_index=3, context_length=60, num_outputs=5,
                 d_model=32, graph_dim=32, patch_size=8, patch_stride=4, large_kernel=15,
                 small_kernel=5, num_blocks=1, ffn_ratio=1, temporal_dropout=0.05,
                 head_dropout=0.0, spatial_dropout=0.0, feedforward_multiplier=2,
                 prior_scale=4.0, prior_jitter=0.02, alpha_init=0.5, beta_init=0.5, seed=42):
        super().__init__()
        self.temporal = ModernTCNBackbone(
            input_dim, target_index, context_length, num_outputs, d_model, patch_size, patch_stride,
            large_kernel, small_kernel, num_blocks, ffn_ratio, temporal_dropout, head_dropout,
        )
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.graph_dim = graph_dim

        self.state_projection = nn.Linear(input_dim, d_model)
        self.query = nn.Linear(2 * d_model, graph_dim)
        self.key = nn.Linear(2 * d_model, graph_dim)
        self.value = nn.Linear(2 * d_model, graph_dim)
        self.output = nn.Linear(graph_dim, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(spatial_dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, feedforward_multiplier * d_model), 
            nn.GELU(), 
            nn.Dropout(spatial_dropout),
            nn.Linear(feedforward_multiplier * d_model, d_model), 
            nn.Dropout(spatial_dropout),
        )

        prior = torch.as_tensor(prior).detach().cpu().float()
        prior = prior / prior.max().clamp_min(1e-6)
        static_logits = prior_scale * (prior - prior.mean())
        if prior_jitter:
            generator = torch.Generator().manual_seed(seed)
            static_logits += torch.randn(static_logits.shape, generator=generator) * prior_jitter

        self.static_logits = nn.Parameter(static_logits.unsqueeze(0))
        self.raw_alpha = nn.Parameter(torch.tensor(math.log(alpha_init / (1 - alpha_init))))
        self.raw_beta = nn.Parameter(torch.tensor(math.log(beta_init / (1 - beta_init))))

    @staticmethod
    def _normalise_graph(logits):
        logits = logits.float()
        mask = torch.eye(logits.shape[-1], device=logits.device, dtype=torch.bool)
        graph = logits.masked_fill(mask, -1e9).softmax(dim=-1).masked_fill(mask, 0.0)
        return graph / graph.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    def _direct_states(self, x):
        states = self.state_projection(x)
        padding = self.patch_size - self.patch_stride
        if padding:
            states = torch.cat([states, states[:, -1:].expand(-1, padding, -1, -1)], dim=1)
        return states.unfold(1, self.patch_size, self.patch_stride)[..., -1].contiguous()

    def alpha(self):
        return self.raw_alpha.sigmoid()

    def beta(self):
        return self.raw_beta.sigmoid()

    def forward(self, x, return_graphs=False):
        h_temp = self.temporal(x)
        h_direct = self._direct_states(x)

        graph_input = torch.cat([h_temp[:, -1], h_direct[:, -1]], dim=-1)
        q, k = self.query(graph_input), self.key(graph_input)
        dynamic_logits = q @ k.transpose(-1, -2) / math.sqrt(self.graph_dim)
        dynamic = self._normalise_graph(dynamic_logits.unsqueeze(1))
        static = self._normalise_graph(self.static_logits.unsqueeze(0))

        alpha = self.alpha()
        adjacency = (1 - alpha) * static + alpha * dynamic

        values = self.value(torch.cat([h_temp, h_direct], dim=-1))
        messages = torch.einsum("bij,bpjd->bpid", adjacency[:, 0].to(values), values)
        h_graph = self.norm1(h_temp + self.dropout(self.output(messages)))
        h_graph = self.norm2(h_graph + self.ffn(h_graph))

        beta = self.beta()
        hidden = (1 - beta.to(h_temp)) * h_temp + beta.to(h_graph) * h_graph
        predictions = self.temporal.forecast(hidden)

        if return_graphs:
            return predictions, {"static": static, "dynamic": dynamic, "mixed": adjacency,
                                 "alpha": alpha, "beta": beta}
        return predictions

    def graph_parameters(self):
        return [self.static_logits, self.raw_alpha, *self.query.parameters(), *self.key.parameters()]
