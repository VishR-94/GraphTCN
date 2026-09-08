from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


WEATHER_NODES = ("C", "NW", "N", "NE", "W", "E", "SW", "S", "SE")
WEATHER_FEATURES = ("z500", "t850", "t2m", "u10", "v10")
WEATHER_SETTINGS = {
    4: {"context_length": 28, "patch_stride": 2, "large_kernel": 7},
    12: {"context_length": 28, "patch_stride": 4, "large_kernel": 7},
    28: {"context_length": 56, "patch_stride": 8, "large_kernel": 15},
    120: {"context_length": 240, "patch_stride": 8, "large_kernel": 119},
}


def _node_array(frame):
    arrays = []
    for node in WEATHER_NODES:
        if node == "C":
            columns = WEATHER_FEATURES
        else:
            columns = (f"z_{node}", f"t_{node}", f"t2m_{node}", f"u10_{node}", f"v10_{node}")
        arrays.append(frame.loc[:, columns].to_numpy(dtype=np.float64))
    return np.stack(arrays, axis=1)


def _positions(index, start, end):
    return np.flatnonzero((index >= start) & (index <= end))


def _split_positions(index, start, end, context_length, horizon):
    step = pd.Timedelta(hours=6)
    inputs = _positions(index, start - step * (context_length + horizon - 1), end - step * horizon)
    targets = _positions(index, start - step * (horizon - 1), end)
    starts = inputs[:len(inputs) - context_length + 1]
    return starts, inputs, targets


class WeatherDataset(Dataset):
    def __init__(self, x, y, y_raw, timestamps, starts, context_length, horizon, target_mean, target_std):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.y_raw = torch.from_numpy(y_raw)
        self.timestamps = torch.from_numpy(timestamps)
        self.starts = starts
        self.context_length = context_length
        self.horizon = horizon
        self.target_mean = torch.from_numpy(target_mean.astype(np.float32))
        self.target_std = torch.from_numpy(target_std.astype(np.float32))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        start = self.starts[index]
        origin = start + self.context_length - 1
        target_start = origin + 1
        target_end = target_start + self.horizon

        return {
            "x": self.x[start:origin + 1],
            "y": self.y[target_start:target_end].unsqueeze(-1),
            "y_raw": self.y_raw[target_start:target_end].unsqueeze(-1),
            "last_context_target": self.y_raw[origin].unsqueeze(-1),
            "sample_idx": index,
            "origin_idx": origin,
            "origin_time": self.timestamps[origin],
            "target_times": self.timestamps[target_start:target_end],
        }


def load_weather_data(data_path, test_year, horizon, start_year=1980):
    context_length = WEATHER_SETTINGS[horizon]["context_length"]
    frame = pd.read_csv(Path(data_path).expanduser(), index_col=0)
    frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index()

    raw = _node_array(frame)
    t850 = raw[:, :, WEATHER_FEATURES.index("t850")]
    timestamps = frame.index.to_numpy(dtype="datetime64[ns]").astype(np.int64)

    train_start = pd.Timestamp(f"{start_year}-01-01")
    train_end = pd.Timestamp(f"{test_year - 2}-12-31")
    val_start, val_end = pd.Timestamp(f"{test_year - 1}-01-01"), pd.Timestamp(f"{test_year - 1}-12-31")
    test_start, test_end = pd.Timestamp(f"{test_year}-01-01"), pd.Timestamp(f"{test_year}-12-31")

    train_starts, input_fit, target_fit = _split_positions(frame.index, train_start, train_end, context_length, horizon)
    val_starts, _, _ = _split_positions(frame.index, val_start, val_end, context_length, horizon)
    test_starts, _, _ = _split_positions(frame.index, test_start, test_end, context_length, horizon)

    input_mean, input_std = raw[input_fit].mean(0), raw[input_fit].std(0)
    target_mean, target_std = t850[target_fit].mean(0), t850[target_fit].std(0)
    input_std = np.where(input_std < 1e-12, 1.0, input_std)
    target_std = np.where(target_std < 1e-12, 1.0, target_std)

    x = ((raw - input_mean) / input_std).astype(np.float32)
    y = ((t850 - target_mean) / target_std).astype(np.float32)
    y_raw = t850.astype(np.float32)

    nominal_train = _positions(frame.index, train_start, train_end)
    correlation = np.corrcoef(np.diff(y_raw[nominal_train].astype(np.float64), axis=0), rowvar=False)
    prior = np.abs(np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)).astype(np.float32)
    np.fill_diagonal(prior, 0.0)
    prior /= np.maximum(prior.sum(axis=1, keepdims=True), 1e-12)

    train = WeatherDataset(x, y, y_raw, timestamps, train_starts, context_length, horizon, target_mean, target_std)
    val = WeatherDataset(x, y, y_raw, timestamps, val_starts, context_length, horizon, target_mean, target_std)
    test = WeatherDataset(x, y, y_raw, timestamps, test_starts, context_length, horizon, target_mean, target_std)

    return train, val, test, torch.from_numpy(prior)
