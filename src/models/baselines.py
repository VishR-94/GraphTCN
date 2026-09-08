import random
import sys
import warnings
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from arch import arch_model
from statsmodels.tsa.api import VAR
from statsmodels.tsa.statespace.sarimax import SARIMAX
from torch import nn

from src.data.finance import CONTEXT_LENGTH, HORIZONS, INPUT_CHANNELS, STRIDE


ROOT = Path(__file__).resolve().parents[2]
MAX_HORIZON = max(HORIZONS)
HORIZON_INDEX = np.asarray(HORIZONS) - 1

modern_tcn_path = ROOT / "external" / "ModernTCN" / "ModernTCN-Long-term-forecasting"
sys.path.insert(0, str(modern_tcn_path))
from models.ModernTCN import Model as ModernTCNModel


def _windows(split):
    close_id = split["channels"].index("close")
    for values, _, day in split["samples"]:
        for origin in range(CONTEXT_LENGTH - 1, len(values) - MAX_HORIZON, STRIDE):
            start = origin - CONTEXT_LENGTH + 1
            context = values[start:origin + 1, :, close_id].double().clamp_min(1e-8).numpy()
            yield context, values, day, origin


def _training_returns(split):
    close_id = split["channels"].index("close")
    days = []
    for values, _, _ in split["samples"]:
        close = values[:, :, close_id].double().clamp_min(1e-8)
        days.append((close[1:].log() - close[:-1].log()).numpy())
    return np.concatenate(days)


def _predict_returns(split, forecast):
    predictions = []
    for context, _, _, _ in _windows(split):
        one_step = forecast(np.diff(np.log(context), axis=0))
        cumulative = np.cumsum(one_step, axis=0)[HORIZON_INDEX]
        predictions.append(context[-1][None] * np.exp(cumulative))
    return torch.from_numpy(np.stack(predictions)).float().unsqueeze(-1)


class PersistenceBaseline(nn.Module):
    def forward(self, last_close):
        return last_close.unsqueeze(1).expand(-1, len(HORIZONS), -1, -1)


class ModernTCNBaseline(nn.Module):
    def __init__(self, num_nodes=93, input_dim=5, context_length=60, num_horizons=5, d_model=32,
                 patch_size=4, patch_stride=2, large_kernel=51, small_kernel=5,
                 num_blocks=1, dropout=0.05):
        super().__init__()
        config = SimpleNamespace(
            stem_ratio=6, downsample_ratio=2, ffn_ratio=1, num_blocks=[num_blocks],
            large_size=[large_kernel], small_size=[small_kernel], dims=[d_model] * 4,
            dw_dims=[d_model] * 4, enc_in=num_nodes * input_dim, small_kernel_merged=False,
            dropout=dropout, head_dropout=0.0, use_multi_scale=False, revin=0, affine=0,
            subtract_last=0, freq="t", seq_len=context_length, pred_len=num_horizons,
            individual=0, decomposition=0, kernel_size=25, patch_size=patch_size,
            patch_stride=patch_stride,
        )
        self.model = ModernTCNModel(config)
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.num_horizons = num_horizons
        self.close_id = INPUT_CHANNELS.index("close")

    def forward(self, x):
        batch, length, nodes, channels = x.shape
        output = self.model(x.reshape(batch, length, nodes * channels))
        output = output.reshape(batch, self.num_horizons, self.num_nodes, self.input_dim)
        return output[..., self.close_id:self.close_id + 1]


class ArimaBaseline:
    def __init__(self, order=(1, 0, 1), trend="c", maxiter=50):
        self.order, self.trend, self.maxiter = order, trend, maxiter

    def fit(self, train_split):
        returns = _training_returns(train_split)
        self.means = returns.mean(0)
        self.models = []
        for series in returns.T:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = SARIMAX(series, order=self.order, trend=self.trend,
                                    enforce_stationarity=True, enforce_invertibility=True)
                    fitted = model.fit(method="powell", disp=False, maxiter=self.maxiter,
                                       cov_type="none", low_memory=True, full_output=False)
            except Exception:
                fitted = None
            self.models.append(fitted)
        return self

    def _forecast(self, context):
        output = np.empty((MAX_HORIZON, context.shape[1]))
        for i, model in enumerate(self.models):
            if model is None:
                output[:, i] = self.means[i]
            else:
                try:
                    output[:, i] = np.nan_to_num(
                        model.apply(context[:, i], refit=False).forecast(MAX_HORIZON), nan=self.means[i])
                except Exception:
                    output[:, i] = self.means[i]
        return output

    def predict(self, split):
        return _predict_returns(split, self._forecast)


class GarchBaseline:
    def __init__(self, return_scale=10000.0, maxiter=1000):
        self.return_scale, self.maxiter = return_scale, maxiter

    def fit(self, train_split):
        returns = _training_returns(train_split)
        self.params = []
        for series in returns.T:
            try:
                fitted = arch_model(series * self.return_scale, mean="AR", lags=1, vol="GARCH",
                                    p=1, q=1, dist="normal", rescale=False).fit(
                                    disp="off", update_freq=0, show_warning=False,
                                    options={"maxiter": self.maxiter})
                params = fitted.params
                self.params.append((float(params.get("Const", 0.0)), float(params.get("y[1]", 0.0))))
            except Exception:
                self.params.append((float(series.mean() * self.return_scale), 0.0))
        return self

    def _forecast(self, context):
        output = np.empty((MAX_HORIZON, context.shape[1]))
        for i, (const, ar) in enumerate(self.params):
            value = context[-1, i] * self.return_scale
            for step in range(MAX_HORIZON):
                value = const + ar * value
                output[step, i] = value / self.return_scale
        return output

    def predict(self, split):
        return _predict_returns(split, self._forecast)


class VarBaseline:
    def __init__(self, maxlags=15, ic="aic", trend="c"):
        self.maxlags, self.ic, self.trend = maxlags, ic, trend

    def fit(self, train_split):
        returns = _training_returns(train_split)
        self.mean = returns.mean(0)
        try:
            self.model = VAR(returns).fit(maxlags=self.maxlags, ic=self.ic, trend=self.trend)
        except Exception:
            self.model = None
        return self

    def _forecast(self, context):
        if self.model is None:
            return np.repeat(self.mean[None], MAX_HORIZON, axis=0)
        if self.model.k_ar == 0:
            return np.repeat(np.asarray(self.model.intercept)[None], MAX_HORIZON, axis=0)
        try:
            return self.model.forecast(context[-self.model.k_ar:], MAX_HORIZON)
        except Exception:
            return np.repeat(self.mean[None], MAX_HORIZON, axis=0)

    def predict(self, split):
        return _predict_returns(split, self._forecast)


class KronosBaseline:
    def __init__(self, device="auto", dtype="float32", temperature=0.6, top_p=0.9,
                 sample_count=10, series_batch_size=1, seed=42):
        self.device, self.dtype = device, dtype
        self.temperature, self.top_p = temperature, top_p
        self.sample_count, self.series_batch_size, self.seed = sample_count, series_batch_size, seed

    def fit(self, train_split=None):
        sys.path.insert(0, str(ROOT / "external" / "Kronos"))
        from model import Kronos, KronosPredictor, KronosTokenizer

        if self.device == "auto":
            has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            self.device = "cuda:0" if torch.cuda.is_available() else "mps" if has_mps else "cpu"

        tokenizer = KronosTokenizer.from_pretrained(
            "NeoQuasar/Kronos-Tokenizer-base", revision="9ef143b98ee3c2488eebd85404e0c215c112b46a")
        model = Kronos.from_pretrained(
            "NeoQuasar/Kronos-small", revision="ac5c409a313c4eadd1ca78201322f5cadb9c34ab")
        tokenizer.eval()
        model.eval()
        self.predictor = KronosPredictor(model=model, tokenizer=tokenizer, device=self.device, max_context=512, clip=5.0)
        return self

    @staticmethod
    def _timestamps(split, day, indices):
        market_open = pd.to_datetime(str(split["market_open"])).time()
        start = pd.Timestamp(day).normalize() + pd.Timedelta(
            hours=market_open.hour, minutes=market_open.minute, seconds=market_open.second)
        return pd.Series([start + pd.Timedelta(minutes=i + 1) for i in indices])

    def predict(self, split):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        input_ids = [split["channels"].index(channel) for channel in INPUT_CHANNELS]
        predictions = []

        with torch.inference_mode():
            for _, values, day, origin in _windows(split):
                start = origin - CONTEXT_LENGTH + 1
                x = values[start:origin + 1, :, input_ids].float().numpy()
                x_time = self._timestamps(split, day, range(start, origin + 1))
                y_time = self._timestamps(split, day, range(origin + 1, origin + MAX_HORIZON + 1))
                frames = [pd.DataFrame(x[:, i], columns=INPUT_CHANNELS) for i in range(x.shape[1])]
                result = []

                for i in range(0, len(frames), self.series_batch_size):
                    batch = frames[i:i + self.series_batch_size]
                    precision = (torch.autocast("cuda", dtype=torch.float16)
                                 if self.dtype == "float16" else nullcontext())
                    with precision:
                        result.extend(self.predictor.predict_batch(
                            df_list=batch, x_timestamp_list=[x_time] * len(batch),
                            y_timestamp_list=[y_time] * len(batch), pred_len=MAX_HORIZON,
                            T=self.temperature, top_k=0, top_p=self.top_p,
                            sample_count=self.sample_count, verbose=False))

                predictions.append(np.stack(
                    [frame["close"].to_numpy()[HORIZON_INDEX] for frame in result], axis=1))

        return torch.from_numpy(np.stack(predictions)).float().unsqueeze(-1)