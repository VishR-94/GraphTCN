import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


modern_tcn_root = Path(__file__).resolve().parents[2] / "external" / "ModernTCN" / "ModernTCN-Long-term-forecasting"
sys.path.insert(0, str(modern_tcn_root))
from models.ModernTCN import Model


class FutureBlock(nn.Module):
    def __init__(self, d_model=32, num_heads=4, multiplier=2, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, multiplier * d_model), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(multiplier * d_model, d_model),
                                 nn.Dropout(dropout))

    def forward(self, future, memory):
        x = self.norm1(future)
        future = future + self.self_attention(x, x, x, need_weights=False)[0]
        x = self.norm2(future)
        future = future + self.cross_attention(x, memory, memory, need_weights=False)[0]
        return future + self.ffn(self.norm3(future))


class TokenGraphTCN(nn.Module):
    def __init__(self, prior, num_nodes=93, context_length=60, prediction_length=60, vocabulary_size=1024,
                 d_model=32, graph_dim=32, patch_size=8, patch_stride=4, large_kernel=15,
                 small_kernel=5, num_blocks=1, temporal_dropout=0.05, spatial_dropout=0.0,
                 prior_scale=4.0, prior_jitter=0.02, alpha_init=0.5, beta_init=0.5, seed=42):
        super().__init__()
        self.num_nodes = num_nodes
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.d_model = d_model
        self.graph_dim = graph_dim
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.num_patches = context_length // patch_stride

        self.token_embedding = nn.Embedding(vocabulary_size, d_model)
        self.node_embedding = nn.Embedding(num_nodes, d_model)
        self.position_embedding = nn.Embedding(context_length, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.state_projection = nn.Linear(d_model, d_model)
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.node_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

        config = SimpleNamespace(
            stem_ratio=6, downsample_ratio=2, ffn_ratio=1, num_blocks=[num_blocks],
            large_size=[large_kernel], small_size=[small_kernel], dims=[d_model] * 4,
            dw_dims=[d_model] * 4, enc_in=d_model, small_kernel_merged=False,
            dropout=temporal_dropout, head_dropout=0.0, use_multi_scale=False, revin=0,
            affine=0, subtract_last=0, freq="t", seq_len=context_length,
            pred_len=prediction_length, individual=0, decomposition=0, kernel_size=25,
            patch_size=patch_size, patch_stride=patch_stride,
        )
        self.temporal = Model(config).model
        self.variable_pool = nn.Linear(d_model, 1, bias=False)
        nn.init.constant_(self.variable_pool.weight, 1.0 / d_model)
        self.temporal_norm = nn.LayerNorm(d_model)

        self.query = nn.Linear(2 * d_model, graph_dim)
        self.key = nn.Linear(2 * d_model, graph_dim)
        self.value = nn.Linear(2 * d_model, graph_dim)
        self.output = nn.Linear(graph_dim, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(spatial_dropout)
        self.ffn = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Dropout(spatial_dropout),
                                 nn.Linear(2 * d_model, d_model), nn.Dropout(spatial_dropout))

        prior = torch.as_tensor(prior).detach().cpu().float()
        prior = prior / prior.max().clamp_min(1e-6)
        static_logits = prior_scale * (prior - prior.mean())
        if prior_jitter:
            generator = torch.Generator().manual_seed(seed)
            static_logits += torch.randn(static_logits.shape, generator=generator) * prior_jitter
        self.static_logits = nn.Parameter(static_logits.unsqueeze(0))
        self.raw_alpha = nn.Parameter(torch.tensor(math.log(alpha_init / (1 - alpha_init))))
        self.raw_beta = nn.Parameter(torch.tensor(math.log(beta_init / (1 - beta_init))))

        self.future_positions = nn.Embedding(prediction_length, d_model)
        nn.init.normal_(self.future_positions.weight, std=0.02)
        self.future_norm = nn.LayerNorm(d_model)
        self.future_block = FutureBlock(d_model)
        self.classifier = nn.Linear(d_model, vocabulary_size)

    @staticmethod
    def _normalise_graph(logits):
        logits = logits.float()
        mask = torch.eye(logits.shape[-1], device=logits.device, dtype=torch.bool)
        graph = logits.masked_fill(mask, -1e9).softmax(dim=-1).masked_fill(mask, 0.0)
        return graph / graph.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    def _encode(self, tokens):
        batch = tokens.shape[0]
        raw = self.state_projection(self.token_embedding(tokens))
        nodes = self.node_embedding(torch.arange(self.num_nodes, device=tokens.device))[None, None]
        positions = self.position_embedding(torch.arange(self.context_length, device=tokens.device))[None, :, None]
        embedded = self.input_norm(raw + nodes + positions)

        x = embedded.permute(0, 2, 3, 1).reshape(batch * self.num_nodes, self.d_model, self.context_length)
        features = self.temporal.forward_feature(x)
        pooled = self.variable_pool(features.permute(0, 2, 3, 1)).squeeze(-1)
        h_temp = pooled.reshape(batch, self.num_nodes, self.d_model, self.num_patches).permute(0, 3, 1, 2)
        return self.temporal_norm(h_temp), raw

    def _direct_states(self, states):
        padding = self.patch_size - self.patch_stride
        if padding:
            states = torch.cat([states, states[:, -1:].expand(-1, padding, -1, -1)], dim=1)
        return states.unfold(1, self.patch_size, self.patch_stride)[..., -1].contiguous()

    def forward(self, context_s1, return_graphs=False):
        batch = context_s1.shape[0]
        h_temp, raw = self._encode(context_s1)
        h_direct = self._direct_states(raw)

        graph_input = torch.cat([h_temp[:, -1], h_direct[:, -1]], dim=-1)
        q, k = self.query(graph_input), self.key(graph_input)
        dynamic = self._normalise_graph((q @ k.transpose(-1, -2) / math.sqrt(self.graph_dim)).unsqueeze(1))
        static = self._normalise_graph(self.static_logits.unsqueeze(0))
        alpha = self.raw_alpha.sigmoid()
        adjacency = (1 - alpha) * static + alpha * dynamic

        values = self.value(torch.cat([h_temp, h_direct], dim=-1))
        messages = torch.einsum("bij,bpjd->bpid", adjacency[:, 0].to(values), values)
        h_graph = self.norm1(h_temp + self.dropout(self.output(messages)))
        h_graph = self.norm2(h_graph + self.ffn(h_graph))
        beta = self.raw_beta.sigmoid()
        hidden = (1 - beta.to(h_temp)) * h_temp + beta.to(h_graph) * h_graph

        memory = hidden.permute(0, 2, 1, 3).reshape(batch * self.num_nodes, self.num_patches, self.d_model)
        summary = hidden[:, -1].reshape(batch * self.num_nodes, self.d_model)
        positions = self.future_positions(torch.arange(self.prediction_length, device=context_s1.device))
        future = self.future_norm(summary[:, None] + positions[None])
        future = self.future_block(future, memory)
        future = future.reshape(batch, self.num_nodes, self.prediction_length, self.d_model).permute(0, 2, 1, 3)
        logits = self.classifier(future)

        if return_graphs:
            return logits, {"static": static, "dynamic": dynamic, "mixed": adjacency,
                            "alpha": alpha, "beta": beta}
        return logits

    def graph_parameters(self):
        return [self.static_logits, self.raw_alpha, *self.query.parameters(), *self.key.parameters()]
