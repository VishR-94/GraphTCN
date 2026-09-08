import torch
from torch.utils.data import DataLoader


def _device(device=None):
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def collect_targets(dataset, batch_size=64):
    result = {key: [] for key in ("targets", "last_close", "sample_idx", "origin_idx")}
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        result["targets"].append(batch["target_close"])
        result["last_close"].append(batch["last_close"])
        result["sample_idx"].append(batch["sample_idx"])
        result["origin_idx"].append(batch["origin_idx"])
    return {key: torch.cat(value) for key, value in result.items()}


def collect_predictions(model, dataset, batch_size=64, device=None):
    device = _device(device)
    model = model.to(device).eval()
    result = {key: [] for key in ("predictions", "targets", "last_close", "sample_idx", "origin_idx")}

    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            prediction = model(batch["x"].to(device)).float().cpu()
            prediction = prediction * batch["target_std"][:, None] + batch["target_mean"][:, None]
            result["predictions"].append(prediction)
            result["targets"].append(batch["target_close"])
            result["last_close"].append(batch["last_close"])
            result["sample_idx"].append(batch["sample_idx"])
            result["origin_idx"].append(batch["origin_idx"])

    return {key: torch.cat(value) for key, value in result.items()}


def cumulative_log_returns(prices, last_close):
    return prices.clamp_min(1e-8).log() - last_close[:, None].clamp_min(1e-8).log()


def _pearson(x, y, dim=0):
    x, y = x.double(), y.double()
    valid = torch.isfinite(x) & torch.isfinite(y)
    count = valid.sum(dim).double()
    x, y = torch.where(valid, x, 0), torch.where(valid, y, 0)
    sx, sy = x.sum(dim), y.sum(dim)
    sx2, sy2, sxy = x.square().sum(dim), y.square().sum(dim), (x * y).sum(dim)
    numerator = count * sxy - sx * sy
    denominator = ((count * sx2 - sx.square()) * (count * sy2 - sy.square())).clamp_min(0).sqrt()
    correlation = (numerator / denominator.clamp_min(1e-30)).clamp(-1, 1)
    return torch.where((count > 1) & (denominator > 0), correlation, torch.full_like(correlation, torch.nan))


def pooled_pearson(predicted_returns, target_returns):
    horizons = predicted_returns.shape[1]
    predicted = predicted_returns[..., 0].permute(0, 2, 1).reshape(-1, horizons)
    target = target_returns[..., 0].permute(0, 2, 1).reshape(-1, horizons)
    return _pearson(predicted, target)


def series_pearson(predictions, targets, sample_idx, origin_idx, stride=15):
    order = torch.argsort(sample_idx * (origin_idx.max() + 1) + origin_idx)
    predictions, targets = predictions[order, ..., 0].double(), targets[order, ..., 0].double()
    sample_idx, origin_idx = sample_idx[order], origin_idx[order]
    adjacent = (sample_idx[1:] == sample_idx[:-1]) & (origin_idx[1:] - origin_idx[:-1] == stride)

    current_pred, previous_pred = predictions[1:], predictions[:-1]
    current_true, previous_true = targets[1:], targets[:-1]
    valid = (current_pred > 0) & (previous_pred > 0) & (current_true > 0) & (previous_true > 0)
    predicted_returns = torch.where(valid, current_pred.log() - previous_pred.log(), torch.nan)[adjacent]
    target_returns = torch.where(valid, current_true.log() - previous_true.log(), torch.nan)[adjacent]
    return torch.nanmean(_pearson(predicted_returns, target_returns), dim=1)


def compute_metrics(predictions, targets, last_close, sample_idx, origin_idx, stride=15):
    predicted_returns = cumulative_log_returns(predictions, last_close)
    target_returns = cumulative_log_returns(targets, last_close)
    model_error = (predictions - targets).abs()
    persistence_error = (last_close[:, None] - targets).abs()
    ties = torch.isclose(model_error, persistence_error, rtol=1e-6, atol=1e-8)
    wins = (model_error < persistence_error) & ~ties

    return {
        "log_mae_bps": (predicted_returns - target_returns).abs().mean((0, 2, 3)) * 10000,
        "win_rate": (wins.float() + 0.5 * ties.float()).mean((0, 2, 3)),
        "sign_accuracy": (torch.sign(predicted_returns) == torch.sign(target_returns)).float().mean((0, 2, 3)),
        "pearson": pooled_pearson(predicted_returns, target_returns),
        "series_pearson": series_pearson(predictions, targets, sample_idx, origin_idx, stride),
    }


def bootstrap_pearson(predictions, targets, last_close, sample_idx, n_bootstrap=10000, seed=42):
    predicted = cumulative_log_returns(predictions, last_close)[..., 0].double().cpu()
    target = cumulative_log_returns(targets, last_close)[..., 0].double().cpu()
    sample_idx = sample_idx.cpu()
    days = torch.unique(sample_idx, sorted=True)
    stats = []

    for day in days:
        x = predicted[sample_idx == day].permute(0, 2, 1).reshape(-1, predicted.shape[1])
        y = target[sample_idx == day].permute(0, 2, 1).reshape(-1, target.shape[1])
        count = torch.full((x.shape[1],), x.shape[0], dtype=torch.float64)
        stats.append(torch.stack((count, x.sum(0), y.sum(0), x.square().sum(0), y.square().sum(0), (x * y).sum(0))))

    stats = torch.stack(stats)
    generator = torch.Generator().manual_seed(seed)
    draws = torch.randint(len(days), (n_bootstrap, len(days)), generator=generator)
    counts = torch.zeros(n_bootstrap, len(days), dtype=torch.float64)
    counts.scatter_add_(1, draws, torch.ones_like(draws, dtype=torch.float64))
    n, sx, sy, sx2, sy2, sxy = (counts @ stats[:, i] for i in range(6))
    numerator = n * sxy - sx * sy
    denominator = ((n * sx2 - sx.square()) * (n * sy2 - sy.square())).clamp_min(0).sqrt()
    samples = torch.where(denominator > 0, (numerator / denominator).clamp(-1, 1), torch.nan)
    quantiles = torch.tensor([0.025, 0.975], dtype=samples.dtype)
    lower, upper = torch.nanquantile(samples, quantiles, dim=0)
    return lower, upper


def evaluate_model(model, dataset, batch_size=64, device=None):
    result = collect_predictions(model, dataset, batch_size, device)
    return compute_metrics(**result), result
