from datetime import date
from pathlib import Path
import torch
from torch.utils.data import Dataset

VAL_START = date(2024, 9, 1)
TEST_START = date(2024, 10, 1)
CONTEXT_LENGTH = 60
HORIZONS = (1, 5, 15, 30, 60)
STRIDE = 15
INPUT_CHANNELS = ("open", "high", "low", "close", "volume")


def _as_date(value):
    return date.fromisoformat(str(value)[:10])


def _build_split(template, samples, dropped_days):
    split = template.copy()
    split["samples"] = [(x[1:], aux, day) for x, aux, day in samples]
    split["dropped_days"] = dropped_days
    split["T"] = 390
    return split


def load_finance_splits(data_dir):
    data_dir = Path(data_dir).expanduser()
    filenames = ("train.pt", "val.pt", "test.pt")
    stored = [torch.load(data_dir / name, map_location="cpu", weights_only=False) for name in filenames]

    samples = sorted(
        (sample for split in stored for sample in split["samples"]),
        key=lambda sample: _as_date(sample[2]),
    )
    dropped_days = sorted(
        (item for split in stored for item in split.get("dropped_days", [])),
        key=lambda item: _as_date(item[0]),
    )

    train_samples = [sample for sample in samples if _as_date(sample[2]) < VAL_START]
    val_samples = [sample for sample in samples if VAL_START <= _as_date(sample[2]) < TEST_START]
    test_samples = [sample for sample in samples if _as_date(sample[2]) >= TEST_START]

    train_dropped = [item for item in dropped_days if _as_date(item[0]) < VAL_START]
    val_dropped = [item for item in dropped_days if VAL_START <= _as_date(item[0]) < TEST_START]
    test_dropped = [item for item in dropped_days if _as_date(item[0]) >= TEST_START]

    template = stored[0]
    train = _build_split(template, train_samples, train_dropped)
    val = _build_split(template, val_samples, val_dropped)
    test = _build_split(template, test_samples, test_dropped)

    return train, val, test

class FinanceDataset(Dataset):
    def __init__(self, split):
        self.samples = split["samples"]
        self.horizons = torch.tensor(HORIZONS)
        self.input_ids = [split["channels"].index(channel) for channel in INPUT_CHANNELS]
        self.close_id = split["channels"].index("close")
        self.close_pos = INPUT_CHANNELS.index("close")

        self.index = [
            (day_idx, origin)
            for day_idx, (values, _, _) in enumerate(self.samples)
            for origin in range(CONTEXT_LENGTH - 1, len(values) - max(HORIZONS), STRIDE)
        ]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        day_idx, origin = self.index[idx]
        values, _, day = self.samples[day_idx]
        start = origin - CONTEXT_LENGTH + 1

        x_raw = values[start:origin + 1, :, self.input_ids].float()
        mean = x_raw.mean(dim=0)
        std = x_raw.std(dim=0, unbiased=False).clamp_min(1e-8)
        x = (x_raw - mean) / std

        target_close = values[origin + self.horizons, :, self.close_id].float().unsqueeze(-1)
        last_close = values[origin, :, self.close_id].float().unsqueeze(-1)
        target_mean = mean[:, self.close_pos:self.close_pos + 1]
        target_std = std[:, self.close_pos:self.close_pos + 1]
        y = (target_close - target_mean) / target_std

        return {
            "x": x,
            "y": y,
            "target_close": target_close,
            "last_close": last_close,
            "target_mean": target_mean,
            "target_std": target_std,
            "day": day,
            "origin_idx": origin,
        }

def build_correlation_prior(train_split, eps=1e-12):
    close_id = train_split["channels"].index("close")
    returns = []

    for values, _, _ in train_split["samples"]:
        close = values[:, :, close_id].double().clamp_min(eps)
        returns.append(close[1:].log() - close[:-1].log())

    returns = torch.cat(returns)
    returns = returns - returns.mean(dim=0, keepdim=True)

    covariance = returns.T @ returns
    variance = returns.square().sum(dim=0)
    denominator = torch.sqrt(variance[:, None] * variance[None, :]).clamp_min(eps)

    prior = torch.nan_to_num((covariance / denominator).abs()).float()
    prior.fill_diagonal_(0)
    return prior / prior.sum(dim=1, keepdim=True).clamp_min(eps)

