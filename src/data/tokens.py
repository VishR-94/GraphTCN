import math
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .finance import CONTEXT_LENGTH, HORIZONS, INPUT_CHANNELS, STRIDE


PREDICTION_LENGTH = max(HORIZONS)
TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-base"
TOKENIZER_REVISION = "9ef143b98ee3c2488eebd85404e0c215c112b46a"


class KronosTokenizer:
    def __init__(self, batch_size=93, clip=5.0, eps=1e-5):
        kronos_root = Path(__file__).resolve().parents[2] / "external" / "Kronos"
        sys.path.insert(0, str(kronos_root))
        from model import KronosTokenizer as OfficialTokenizer

        self.model = OfficialTokenizer.from_pretrained(TOKENIZER_ID, revision=TOKENIZER_REVISION).cpu().eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.batch_size = batch_size
        self.clip = clip
        self.eps = eps

    def encode(self, values):
        batch, length, nodes, _ = values.shape
        values = values.permute(0, 2, 1, 3).reshape(batch * nodes, length, 6).cpu().float()
        coarse, fine = [], []
        with torch.inference_mode():
            for chunk in values.split(self.batch_size):
                s1, s2 = self.model.encode(chunk, half=True)
                coarse.append(s1.cpu())
                fine.append(s2.cpu())
        coarse = torch.cat(coarse).reshape(batch, nodes, length).permute(0, 2, 1)
        fine = torch.cat(fine).reshape(batch, nodes, length).permute(0, 2, 1)
        return torch.stack((coarse, fine), dim=-1).to(torch.int16)

    def decode_coarse(self, context_tokens, future_s1, mean, std):
        batch, context_length, nodes, _ = context_tokens.shape
        full_s1 = torch.cat((context_tokens[..., 0], future_s1), dim=1)
        length = full_s1.shape[1]
        full_s1 = full_s1.permute(0, 2, 1).reshape(batch * nodes, length).long()

        parameter = next(self.model.parameters())
        device, dtype = parameter.device, parameter.dtype
        bit_mask = 2 ** torch.arange(self.model.s1_bits, device=device)
        decoded = []

        with torch.inference_mode():
            for chunk in full_s1.split(self.batch_size):
                bits = ((chunk.to(device).unsqueeze(-1) & bit_mask) != 0).to(dtype)
                x = (bits * 2 - 1) / math.sqrt(self.model.codebook_dim)
                x = self.model.post_quant_embed_pre(x)
                for layer in self.model.decoder:
                    x = layer(x)
                decoded.append(self.model.head(x).float().cpu())

        decoded = torch.cat(decoded)
        mean = mean.float().reshape(batch * nodes, 6)
        std = std.float().reshape(batch * nodes, 6)
        decoded = decoded * (std[:, None] + self.eps) + mean[:, None]
        decoded = decoded.reshape(batch, nodes, length, 6).permute(0, 2, 1, 3)
        return decoded[:, context_length:, :, :5].contiguous()


class TokenDataset(Dataset):
    def __init__(self, cache):
        self.data = load_token_cache(cache) if isinstance(cache, (str, Path)) else cache

    def __len__(self):
        return len(self.data["context_tokens"])

    def __getitem__(self, index):
        return {
            "context_tokens": self.data["context_tokens"][index].long(),
            "target_s1": self.data["target_s1"][index].long(),
            "target_s2": self.data["target_s2"][index].long(),
            "context_mean": self.data["context_mean"][index],
            "context_std": self.data["context_std"][index],
            "evaluation_true": self.data["evaluation_true"][index],
            "last_context_target": self.data["last_context_target"][index],
            "sample_idx": self.data["sample_idx"][index],
            "origin_idx": self.data["origin_idx"][index],
        }


def build_token_cache(split, tokenizer, window_batch_size=2):
    channel_ids = [split["channels"].index(channel) for channel in INPUT_CHANNELS]
    windows = [(day_idx, origin) for day_idx, (values, _, _) in enumerate(split["samples"])
               for origin in range(CONTEXT_LENGTH - 1, len(values) - PREDICTION_LENGTH, STRIDE)]
    evaluation_indices = torch.tensor([horizon - 1 for horizon in HORIZONS])
    output = {key: [] for key in ("context_tokens", "target_s1", "target_s2", "context_mean",
                                  "context_std", "evaluation_true", "last_context_target")}
    sample_indices, origin_indices = [], []

    for start in range(0, len(windows), window_batch_size):
        batch_windows = windows[start:start + window_batch_size]
        context, future = [], []
        for day_idx, origin in batch_windows:
            values = split["samples"][day_idx][0]
            context.append(values[origin - CONTEXT_LENGTH + 1:origin + 1, :, channel_ids])
            future.append(values[origin + 1:origin + PREDICTION_LENGTH + 1, :, channel_ids])
            sample_indices.append(day_idx)
            origin_indices.append(origin)

        context = torch.stack(context).float()
        future = torch.stack(future).float()
        context_ohlcva = torch.cat((context, torch.zeros_like(context[..., :1])), dim=-1)
        future_ohlcva = torch.cat((future, torch.zeros_like(future[..., :1])), dim=-1)
        mean = context_ohlcva.mean(dim=1)
        std = context_ohlcva.std(dim=1, unbiased=False)
        full_path = torch.cat((context_ohlcva, future_ohlcva), dim=1)
        normalised = ((full_path - mean[:, None]) / (std[:, None] + tokenizer.eps)).clamp(-tokenizer.clip, tokenizer.clip)
        tokens = tokenizer.encode(normalised)
        if start == 0 and not torch.equal(tokens[:, :CONTEXT_LENGTH], tokenizer.encode(normalised[:, :CONTEXT_LENGTH])):
            raise RuntimeError("Kronos context tokens changed when future labels were appended")

        output["context_tokens"].append(tokens[:, :CONTEXT_LENGTH])
        output["target_s1"].append(tokens[:, CONTEXT_LENGTH:, :, 0])
        output["target_s2"].append(tokens[:, CONTEXT_LENGTH:, :, 1])
        output["context_mean"].append(mean)
        output["context_std"].append(std)
        output["evaluation_true"].append(future.index_select(1, evaluation_indices))
        output["last_context_target"].append(context[:, -1])

    cache = {key: torch.cat(value) for key, value in output.items()}
    cache["sample_idx"] = torch.tensor(sample_indices)
    cache["origin_idx"] = torch.tensor(origin_indices)
    cache["asset_cols"] = split["asset_cols"]
    cache["horizons"] = HORIZONS
    return cache


def save_token_cache(cache, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)


def load_token_cache(path):
    return torch.load(Path(path), map_location="cpu", weights_only=False)
