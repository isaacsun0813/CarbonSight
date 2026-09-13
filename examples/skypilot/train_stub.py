"""Small Bitcoin price RNN smoke job for CarbonSight + SkyPilot cloud tests."""

from __future__ import annotations

import argparse
import csv
import io
import os
import signal
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

DEFAULT_BITCOIN_URL = (
    "https://raw.githubusercontent.com/RDeconomist/observatory/main/Bitcoin%20Price.csv"
)
PRICE_COLUMN = "Closing Price (USD)"


def load_csv_from_url(url: str) -> dict[str, list[str]]:
    """Download a CSV and return column name -> list of string values."""
    with urllib.request.urlopen(url, timeout=60) as response:
        csv_data = response.read().decode("utf-8")
    csv_file = io.StringIO(csv_data)
    csv_reader = csv.DictReader(csv_file)

    data_dict: dict[str, list[str]] = defaultdict(list)
    for row in csv_reader:
        for key, value in row.items():
            if key is not None:
                data_dict[key].append(value)
    return dict(data_dict)


def parse_prices(data: dict[str, list[str]], column: str = PRICE_COLUMN) -> list[float]:
    """Extract and min-max normalize closing prices."""
    if column not in data:
        raise KeyError(f"Column {column!r} not in CSV; got: {list(data.keys())}")
    raw: list[float] = []
    for value in data[column]:
        try:
            raw.append(float(value.replace(",", "").strip()))
        except ValueError:
            continue
    if len(raw) < 2:
        raise ValueError("Not enough numeric price rows in CSV")
    lo, hi = min(raw), max(raw)
    span = hi - lo
    if span <= 0:
        return [0.0] * len(raw)
    return [(p - lo) / span for p in raw]


class TimeSeriesDataset(Dataset):
    """Next-step prediction: x = window, y = value immediately after the window."""

    def __init__(self, time_series: list[float], sequence_length: int) -> None:
        if sequence_length < 1:
            raise ValueError("sequence_length must be >= 1")
        if len(time_series) <= sequence_length:
            raise ValueError("time_series too short for sequence_length")
        self.time_series = time_series
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return len(self.time_series) - self.sequence_length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        window = self.time_series[idx : idx + self.sequence_length]
        x = torch.tensor(window, dtype=torch.float32).unsqueeze(-1)  # (seq, 1)
        y = torch.tensor(self.time_series[idx + self.sequence_length], dtype=torch.float32)
        return x, y


class PriceRNN(nn.Module):
    """Stacked RNN with linear head for scalar next-step prediction."""

    def __init__(self, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity="tanh",
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)


def save_checkpoint(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"checkpoint saved: {path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a small RNN on Bitcoin closing prices.")
    p.add_argument("--url", default=DEFAULT_BITCOIN_URL, help="CSV URL")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--sequence-length", type=int, default=60)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-minutes", type=float, default=25.0, help="Wall-clock budget")
    p.add_argument(
        "--require-cuda",
        action="store_true",
        help="Exit with error if CUDA is unavailable (use on cloud GPU boxes)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    start = time.monotonic()
    deadline = start + args.max_minutes * 60.0

    if args.require_cuda and not torch.cuda.is_available():
        print("Error: --require-cuda set but CUDA is not available", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} epochs={args.epochs} max_minutes={args.max_minutes}")

    print(f"loading CSV from {args.url}")
    data = load_csv_from_url(args.url)
    prices = parse_prices(data)
    print(f"loaded {len(prices)} normalized price points")

    split = int(len(prices) * 0.9)
    train_prices = prices[:split]
    val_prices = prices[split:]

    train_ds = TimeSeriesDataset(train_prices, args.sequence_length)
    val_ds = TimeSeriesDataset(val_prices, args.sequence_length)

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=pin,
    )

    model = PriceRNN(args.hidden_size, args.num_layers).to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    ckpt_dir = os.environ.get("CARBONSIGHT_CKPT_DIR")
    ckpt_path = Path(ckpt_dir) / "model.pt" if ckpt_dir else None

    def _on_sigint(signum: int, frame: object) -> None:
        if ckpt_path is not None:
            save_checkpoint(model, ckpt_path)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, _on_sigint)

    for epoch in range(1, args.epochs + 1):
        if time.monotonic() >= deadline:
            print(f"time budget reached after {epoch - 1} epochs")
            break
        train_mae = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_mae = eval_epoch(model, val_loader, criterion, device)
        elapsed_min = (time.monotonic() - start) / 60.0
        print(
            f"epoch={epoch:03d} train_mae={train_mae:.6f} val_mae={val_mae:.6f} "
            f"elapsed_min={elapsed_min:.1f}"
        )
        if ckpt_path is not None and epoch % 10 == 0:
            save_checkpoint(model, ckpt_path)

    if ckpt_path is not None:
        save_checkpoint(model, ckpt_path)

    elapsed_min = (time.monotonic() - start) / 60.0
    print(f"done in {elapsed_min:.1f} min")


if __name__ == "__main__":
    main()
