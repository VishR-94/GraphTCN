import math
import torch
from torch import nn
from .modern_tcn import ModernTCNBackbone


class GraphTCN(nn.Module):
    def __init__(self, prior, input_dim=5, d_model=32, graph_dim=32, patch_size=8, patch_stride=4,
                 prior_scale=4.0, prior_jitter=0.02, alpha_init=0.5, beta_init=0.5,
                 dropout=0.0, seed=42):
        super().__init__()

        self.temporal = ModernTCNBackbone()
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
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.Dropout(dropout),
        )

        prior = torch.as_tensor(prior, dtype=torch.float32).cpu()
        prior = prior / prior.max().clamp_min(1e-8)
        static_logits = prior_scale * (prior - prior.mean())

        if prior_jitter:
            generator = torch.Generator().manual_seed(seed)
            static_logits += torch.randn(static_logits.shape, generator=generator) * prior_jitter

        self.static_logits = nn.Parameter(static_logits)
        self.raw_alpha = nn.Parameter(torch.tensor(math.log(alpha_init / (1 - alpha_init))))
        self.raw_beta = nn.Parameter(torch.tensor(math.log(beta_init / (1 - beta_init))))

    @staticmethod
    def _normalise_graph(logits):
        mask = torch.eye(logits.shape[-1], device=logits.device, dtype=torch.bool)
        return logits.masked_fill(mask, -torch.inf).softmax(dim=-1)

    def _direct_states(self, x):
        states = self.state_projection(x)
        padding = self.patch_size - self.patch_stride

        if padding:
            final_state = states[:, -1:].expand(-1, padding, -1, -1)
            states = torch.cat([states, final_state], dim=1)

        return states.unfold(1, self.patch_size, self.patch_stride)[..., -1]

    def forward(self, x, return_graphs=False):
        h_temp = self.temporal(x)
        h_direct = self._direct_states(x)

        graph_input = torch.cat([h_temp[:, -1], h_direct[:, -1]], dim=-1)
        q = self.query(graph_input)
        k = self.key(graph_input)

        dynamic = self._normalise_graph(q @ k.transpose(-1, -2) / math.sqrt(self.graph_dim))
        static = self._normalise_graph(self.static_logits).unsqueeze(0)

        alpha = self.raw_alpha.sigmoid()
        adjacency = (1 - alpha) * static + alpha * dynamic

        values = self.value(torch.cat([h_temp, h_direct], dim=-1))

        # adjacency[target, source]
        messages = torch.einsum("bij,bpjd->bpid", adjacency, values)

        h_graph = self.norm1(h_temp + self.dropout(self.output(messages)))
        h_graph = self.norm2(h_graph + self.ffn(h_graph))

        beta = self.raw_beta.sigmoid()
        hidden = (1 - beta) * h_temp + beta * h_graph
        predictions = self.temporal.forecast(hidden)

        if not return_graphs:
            return predictions

        graphs = {
            "static": static[0],
            "dynamic": dynamic,
            "mixed": adjacency,
            "alpha": alpha,
            "beta": beta,
        }
        return predictions, graphs

    def graph_parameters(self):
        return [self.static_logits, self.raw_alpha, *self.query.parameters(), *self.key.parameters()]