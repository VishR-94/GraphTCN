from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader


def _autocast(device, enabled):
    return torch.autocast("cuda", dtype=torch.float16) if enabled and device.type == "cuda" else nullcontext()


def _optimizer(model, lr, graph_lr):
    graph = list(model.graph_parameters()) if graph_lr is not None and hasattr(model, "graph_parameters") else []
    graph_ids = {id(parameter) for parameter in graph}
    main = [parameter for parameter in model.parameters() if parameter.requires_grad and id(parameter) not in graph_ids]
    groups = [{"params": main, "lr": lr, "base_lr": lr}]
    if graph:
        groups.append({"params": graph, "lr": graph_lr, "base_lr": graph_lr})
    return torch.optim.Adam(groups)


def _set_learning_rate(optimizer, epoch, decay_start, decay_factor):
    multiplier = 1.0 if epoch <= decay_start else decay_factor ** (epoch - decay_start)
    for group in optimizer.param_groups:
        group["lr"] = group["base_lr"] * multiplier


def continuous_loss(prediction, batch, device):
    mean = batch["target_mean"].to(device)[:, None]
    std = batch["target_std"].to(device)[:, None]
    target = batch["target_close"].to(device).clamp_min(1e-8)
    last = batch["last_close"].to(device)[:, None].clamp_min(1e-8)

    predicted_price = (prediction.float() * std + mean).clamp_min(1e-8)
    predicted_return = torch.log(predicted_price / last)
    target_return = torch.log(target / last)
    return 10000.0 * (predicted_return - target_return).abs().mean()


def _continuous_step(model, batch, device, use_amp):
    with _autocast(device, use_amp):
        prediction = model(batch["x"].to(device))
    return continuous_loss(prediction, batch, device)


def _token_step(model, batch, device, use_amp):
    context = batch["context_tokens"][..., 0].to(device).long()
    target = batch["target_s1"].to(device).long()
    with _autocast(device, use_amp):
        logits = model(context)
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), target.reshape(-1))

def _weather_train_step(model, batch, device, use_amp):
    with _autocast(device, use_amp):
        prediction = model(batch["x"].to(device))
    return F.mse_loss(prediction.float(), batch["y"].to(device).float())

def _weather_val_step(model, batch, device, use_amp):
    with _autocast(device, use_amp):
        prediction = model(batch["x"].to(device))
    target = batch["y"].to(device)
    return F.mse_loss(prediction[:, -1, 0, 0].float(), target[:, -1, 0, 0].float())


def _epoch(model, loader, step, device, optimizer=None, scaler=None, use_amp=False, clip=1.0):
    training = optimizer is not None
    model.train(training)
    total = 0.0

    with torch.set_grad_enabled(training):
        for batch in loader:
            if training:
                optimizer.zero_grad(set_to_none=True)

            loss = step(model, batch, device, use_amp)

            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                scaler.step(optimizer)
                scaler.update()

            total += loss.item() * len(next(value for value in batch.values() if torch.is_tensor(value)))

    return total / len(loader.dataset)


def _train(model, train_data, val_data, save_path, step, batch_size, lr, graph_lr, decay_start,
           decay_factor=0.9, max_epochs=100, patience=10, clip=1.0, seed=42, device=None):
    torch.manual_seed(seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
    model = model.to(device)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    optimizer = _optimizer(model, lr, graph_lr)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        _set_learning_rate(optimizer, epoch, decay_start, decay_factor)
        train_loss = _epoch(model, train_loader, step, device, optimizer, scaler, use_amp, clip)
        val_loss = _epoch(model, val_loader, step, device, use_amp=use_amp)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            break

    model.load_state_dict(best_state)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "epoch": best_epoch, "val_loss": best_loss, "history": history}, save_path)
    return history


def train_continuous(model, train_data, val_data, save_path, lr=2.5e-4, graph_lr=5e-4):
    return _train(model, train_data, val_data, save_path, _continuous_step, batch_size=16,
                  lr=lr, graph_lr=graph_lr, decay_start=15)


def train_tokens(model, train_data, val_data, save_path, lr=1e-4, graph_lr=5e-4):
    return _train(model, train_data, val_data, save_path, _token_step, batch_size=2,
                  lr=lr, graph_lr=graph_lr, decay_start=3)


def train_weather(model, train_data, val_data, save_path, lr=2.5e-4, graph_lr=5e-4):
    return _train(model, train_data, val_data, save_path, _weather_train_step, 16, lr, graph_lr, 15,
                  val_step=_weather_val_step, val_batch_size=32)