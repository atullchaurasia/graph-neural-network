"""
train.py
Training loop for STGNN on METR-LA.

Usage:
  python train.py
  python train.py --epochs 100 --batch_size 32 --lr 0.001 --hidden 64
"""

import os
import argparse
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.stgnn import STGNN
from utils.data_loader import get_dataloaders
from utils.metrics import all_metrics


# ── CLI args ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--hidden",      type=int,   default=64)
    p.add_argument("--t_in",        type=int,   default=12)
    p.add_argument("--t_out",       type=int,   default=12)
    p.add_argument("--patience",    type=int,   default=15)
    p.add_argument("--checkpoint",  type=str,   default="checkpoints/best_model.pt")
    p.add_argument("--speeds_h5",   type=str,   default="data/raw/metr-la.h5")
    p.add_argument("--dist_csv",    type=str,   default="data/raw/distances_la_2012.csv")
    return p.parse_args()


# ── Training step ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, edge_index, edge_weight, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x = x.to(device)           # [B, T, N, F]
        y = y.to(device)           # [B, T_out, N]

        optimizer.zero_grad()
        pred  = model(x, edge_index, edge_weight)   # [B, T_out, N]
        loss  = criterion(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion, edge_index, edge_weight, device):
    model.eval()
    total_loss = 0.0
    all_pred, all_true = [], []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x, edge_index, edge_weight)
        loss = criterion(pred, y)
        total_loss += loss.item() * x.size(0)
        all_pred.append(pred.cpu())
        all_true.append(y.cpu())

    preds  = torch.cat(all_pred)
    truths = torch.cat(all_true)
    metrics = all_metrics(preds, truths)
    return total_loss / len(loader.dataset), metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs("checkpoints", exist_ok=True)

    # Data
    train_loader, val_loader, test_loader, edge_index, edge_weight, scaler = get_dataloaders(
        speeds_h5  = args.speeds_h5,
        dist_csv   = args.dist_csv,
        t_in       = args.t_in,
        t_out      = args.t_out,
        batch_size = args.batch_size,
    )
    edge_index  = edge_index.to(device)
    edge_weight = edge_weight.to(device)

    # Model
    # Infer num_nodes from first batch
    x_sample, _ = next(iter(train_loader))
    num_nodes    = x_sample.shape[2]
    in_features  = x_sample.shape[3]

    model = STGNN(
        num_nodes   = num_nodes,
        in_features = in_features,
        hidden_dim  = args.hidden,
        t_out       = args.t_out,
    ).to(device)

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer  = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler  = ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)
    criterion  = nn.HuberLoss(delta=1.0)   # robust to outlier speed spikes

    best_val_loss = float("inf")
    no_improve    = 0

    print(f"\n{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>10} {'MAE':>8} {'RMSE':>8} {'MAPE':>8}  Time")
    print("-" * 72)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, criterion, edge_index, edge_weight, device)
        val_loss, metrics = eval_epoch(model, val_loader, criterion, edge_index, edge_weight, device)

        scheduler.step(val_loss)
        elapsed = time.time() - t0

        print(
            f"{epoch:>6} {train_loss:>12.4f} {val_loss:>10.4f} "
            f"{metrics['MAE']:>8.3f} {metrics['RMSE']:>8.3f} {metrics['MAPE']:>7.2f}%  {elapsed:.1f}s"
        )

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve    = 0
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(), "val_loss": val_loss, "args": vars(args)},
                args.checkpoint,
            )
            print(f"  ✓ Saved checkpoint (val_loss={val_loss:.4f})")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    # Final test evaluation
    print("\n── Test set evaluation ──")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    _, test_metrics = eval_epoch(model, test_loader, criterion, edge_index, edge_weight, device)
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
