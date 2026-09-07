import sys
from pathlib import Path
from types import SimpleNamespace
from torch import nn


modern_tcn_root = Path(__file__).resolve().parents[3] / "external" / "ModernTCN" / "ModernTCN-Long-term-forecasting"
sys.path.insert(0, str(modern_tcn_root))

from models.ModernTCN import Model


class ModernTCNBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        config = SimpleNamespace(
            stem_ratio=6,
            downsample_ratio=2,
            ffn_ratio=1,
            num_blocks=[1],
            large_size=[15],
            small_size=[5],
            dims=[32] * 4,
            dw_dims=[32] * 4,
            enc_in=5,
            small_kernel_merged=False,
            dropout=0.05,
            head_dropout=0.0,
            use_multi_scale=False,
            revin=0,
            affine=0,
            subtract_last=0,
            freq="t",
            seq_len=60,
            pred_len=5,
            individual=0,
            decomposition=0,
            kernel_size=25,
            patch_size=8,
            patch_stride=4,
        )

        self.model = Model(config).model

    def forward(self, x):
        batch_size, context_length, num_nodes, num_channels = x.shape
        x = x.permute(0, 2, 3, 1).reshape(batch_size * num_nodes, num_channels, context_length)
        hidden = self.model.forward_feature(x)[:, 3]

        return hidden.reshape(batch_size, num_nodes, 32, 15).permute(0, 3, 1, 2).contiguous()

    def forecast(self, hidden):
        batch_size, num_patches, num_nodes, hidden_dim = hidden.shape
        hidden = hidden.permute(0, 2, 3, 1).reshape(batch_size * num_nodes, 1, hidden_dim, num_patches)
        predictions = self.model.head(hidden)[:, 0]

        return predictions.reshape(batch_size, num_nodes, 5).permute(0, 2, 1).unsqueeze(-1).contiguous()