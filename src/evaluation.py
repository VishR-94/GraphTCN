import torch
from torch.utils.data import DataLoader

# Financial Price metrics

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

# Financial Token metrics

def _sample_top_p(logits, num_paths=10, temperature=1.0, top_p=0.9):
    sorted_logits, sorted_ids = (logits / temperature).sort(dim=-1, descending=True)
    probs = sorted_logits.softmax(dim=-1)
    probs[(probs.cumsum(dim=-1) - probs) > top_p] = 0

    flat_probs = probs.reshape(-1, probs.shape[-1])
    flat_ids = sorted_ids.reshape(-1, sorted_ids.shape[-1])
    samples = torch.multinomial(flat_probs, num_paths, replacement=True)
    samples = flat_ids.gather(1, samples)

    return samples.T.reshape(num_paths, *logits.shape[:-1])


def token_topk_metrics(top_ids, targets, train_targets, horizons, vocabulary_size=1024, ks=(1, 5, 10)):
    frequent = torch.stack([
        torch.bincount(train_targets[:, h].reshape(-1), minlength=vocabulary_size).topk(max(ks)).indices
        for h in range(train_targets.shape[1])
    ])

    metrics = {"horizons": tuple(horizons)}
    for k in ks:
        accuracy = top_ids[..., :k].eq(targets.unsqueeze(-1)).any(-1).float().mean((0, 2)) * 100
        baseline = targets.unsqueeze(-1).eq(frequent[None, :, None, :k]).any(-1).float().mean((0, 2)) * 100
        metrics[f"top_{k}_accuracy_pct"] = accuracy
        metrics[f"top_{k}_excess_pct_points"] = accuracy - baseline

    return metrics


def evaluate_token_model(model, train_data, test_data, tokenizer, batch_size=2, num_paths=10,
                         temperature=1.0, top_p=0.9, device=None, seed=42):
    device = _device(device)
    model = model.to(device).eval()

    horizons = tuple(test_data.data["horizons"])
    indices = torch.tensor([h - 1 for h in horizons])
    results = {key: [] for key in ("predictions", "targets", "last_close", "sample_idx", "origin_idx")}
    top_ids, token_targets = [], []

    torch.manual_seed(seed)

    with torch.inference_mode():
        for batch in DataLoader(test_data, batch_size=batch_size, shuffle=False):
            logits = model(batch["context_tokens"][..., 0].to(device)).float()
            selected = logits.index_select(1, indices.to(device))

            top_ids.append(selected.topk(10, dim=-1).indices.cpu())
            token_targets.append(batch["target_s1"].index_select(1, indices))

            paths = _sample_top_p(logits, num_paths, temperature, top_p).cpu()
            decoded = torch.stack([
                tokenizer.decode_coarse(batch["context_tokens"], path, batch["context_mean"], batch["context_std"])
                for path in paths
            ]).mean(0)

            results["predictions"].append(decoded.index_select(1, indices)[..., 3:4])
            results["targets"].append(batch["evaluation_true"][..., 3:4])
            results["last_close"].append(batch["last_context_target"][..., 3:4])
            results["sample_idx"].append(batch["sample_idx"])
            results["origin_idx"].append(batch["origin_idx"])

    results = {key: torch.cat(value) for key, value in results.items()}
    top_ids = torch.cat(top_ids)
    token_targets = torch.cat(token_targets)
    train_targets = train_data.data["target_s1"].index_select(1, indices).long()
    token_metrics = token_topk_metrics(top_ids, token_targets, train_targets, horizons)

    return compute_metrics(**results), token_metrics, results

# Weather metrics

def weather_metrics(predictions, targets):
    prediction, target = predictions[:, -1, 0, 0].float(), targets[:, -1, 0, 0].float()
    prediction_anomaly, target_anomaly = prediction - prediction.mean(), target - target.mean()
    denominator = (prediction_anomaly.square().sum() * target_anomaly.square().sum()).sqrt().clamp_min(1e-12)
    offset = target.min() + 30
    return {
        "mae": float((prediction - target).abs().mean()),
        "r": float((prediction_anomaly * target_anomaly).sum() / denominator),
        "smape": float((2 * (prediction - target).abs() / ((prediction - offset).abs() + (target - offset).abs())).mean() * 100),
    }


def evaluate_weather(model, dataset, batch_size=64, device=None, return_graphs=False):
    device = _device(device)
    model = model.to(device).eval()
    mean = dataset.target_mean.view(1, 1, -1, 1)
    std = dataset.target_std.view(1, 1, -1, 1)
    keys = ("predictions", "targets", "last_context_target", "sample_idx", "origin_idx", "origin_time", "target_times")
    result = {key: [] for key in keys}
    dynamic, mixed, static, alpha, beta = [], [], None, None, None

    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            output = model(batch["x"].to(device), return_graphs=return_graphs)
            prediction, graphs = output if return_graphs else (output, None)
            prediction = prediction.float().cpu() * std + mean

            result["predictions"].append(prediction)
            result["targets"].append(batch["y_raw"])
            result["last_context_target"].append(batch["last_context_target"])
            for key in ("sample_idx", "origin_idx", "origin_time", "target_times"):
                result[key].append(batch[key])

            if return_graphs:
                dynamic.append(graphs["dynamic"].detach().cpu())
                mixed.append(graphs["mixed"].detach().cpu())
                static = graphs["static"].detach().cpu()[0]
                alpha, beta = graphs["alpha"].detach().cpu(), graphs["beta"].detach().cpu()

    result = {key: torch.cat(value) for key, value in result.items()}
    if return_graphs:
        result["graphs"] = {"static": static, "dynamic": torch.cat(dynamic), "mixed": torch.cat(mixed), "alpha": alpha, "beta": beta}
    return weather_metrics(result["predictions"], result["targets"]), result

