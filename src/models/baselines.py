import sys
from pathlib import Path
from types import SimpleNamespace
from torch import nn


modern_tcn_root = Path(__file__).resolve().parents[2] / "external" / "ModernTCN" / "ModernTCN-Long-term-forecasting"
sys.path.insert(0, str(modern_tcn_root))

from models.ModernTCN import Model


class PersistenceBaseline(nn.Module):
    def __init__(self, num_horizons=5):
        super().__init__()
        self.num_horizons = num_horizons

    def forward(self, last_close):
        return last_close.unsqueeze(1).expand(-1, self.num_horizons, -1, -1)


class ModernTCNBaseline(nn.Module):
    def __init__(self, num_nodes=93, input_dim=5, context_length=60, num_horizons=5,
                 d_model=32, patch_size=4, patch_stride=2, large_kernel=15,
                 small_kernel=5, num_blocks=1, dropout=0.05):
        super().__init__()

        config = SimpleNamespace(
            stem_ratio=6,
            downsample_ratio=2,
            ffn_ratio=1,
            num_blocks=[num_blocks],
            large_size=[large_kernel],
            small_size=[small_kernel],
            dims=[d_model] * 4,
            dw_dims=[d_model] * 4,
            enc_in=num_nodes * input_dim,
            small_kernel_merged=False,
            dropout=dropout,
            head_dropout=0.0,
            use_multi_scale=False,
            revin=0,
            affine=0,
            subtract_last=0,
            freq="t",
            seq_len=context_length,
            pred_len=num_horizons,
            individual=0,
            decomposition=0,
            kernel_size=25,
            patch_size=patch_size,
            patch_stride=patch_stride,
        )

        self.model = Model(config)
        self.num_horizons = num_horizons
        self.close_index = 3

    def forward(self, x):
        batch_size, context_length, num_nodes, num_channels = x.shape

        x = x.reshape(batch_size, context_length, num_nodes * num_channels)
        predictions = self.model(x)
        predictions = predictions.reshape(batch_size, self.num_horizons, num_nodes, num_channels)

        return predictions[..., self.close_index:self.close_index + 1]