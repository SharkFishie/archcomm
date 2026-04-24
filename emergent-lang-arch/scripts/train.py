"""
Training entry point.

Usage:
    python scripts/train.py --config configs/base_config.yaml --arch lstm --epochs 100
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

import egg.core as core

from agents import get_agents
from games.referential_game import ReferentialDataset, build_game
from analysis import compute_topo_similarity, collect_messages, compute_all_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base_config.yaml")
    p.add_argument("--arch", default=None, help="Override config arch")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--use_wandb", action="store_true")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # Flatten nested config for convenience
    flat = {}
    for section in cfg.values():
        if isinstance(section, dict):
            flat.update(section)
    return flat


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # CLI overrides
    if args.arch:
        cfg["arch"] = args.arch
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.seed:
        cfg["seed"] = args.seed
    if args.lr:
        cfg["lr"] = args.lr
    if args.use_wandb:
        cfg["use_wandb"] = True

    set_seed(cfg["seed"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    arch = cfg["arch"]

    results_dir = Path(cfg["results_dir"]) / arch / f"seed_{cfg['seed']}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ data
    train_ds = ReferentialDataset(
        cfg["n_objects"], cfg["n_features"], cfg["n_distractors"], cfg["n_train"], seed=cfg["seed"]
    )
    val_ds = ReferentialDataset(
        cfg["n_objects"], cfg["n_features"], cfg["n_distractors"], cfg["n_val"], seed=cfg["seed"] + 1
    )
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False)

    # ------------------------------------------------------------------ model
    sender, receiver = get_agents(arch, cfg)
    game = build_game(sender, receiver, cfg)
    game = game.to(device)

    optimizer = torch.optim.Adam(game.parameters(), lr=cfg["lr"])

    # Optional wandb
    if cfg.get("use_wandb"):
        import wandb
        wandb.init(project=cfg.get("wandb_project", "emergent-lang-arch"), config=cfg, name=f"{arch}-{cfg['seed']}")

    # --------------------------------------------------------------- training
    n_batches = len(train_loader)
    print(f"Starting training | arch={arch} | epochs={cfg['epochs']} | batches/epoch={n_batches}", flush=True)
    best_val_acc = 0.0
    for epoch in range(1, cfg["epochs"] + 1):
        game.train()
        epoch_loss, epoch_acc = 0.0, 0.0
        for batch_idx, (sender_input, labels, receiver_input) in enumerate(train_loader):
            sender_input = sender_input.to(device)
            labels = labels.to(device)
            receiver_input = receiver_input.to(device)

            optimizer.zero_grad()
            loss, interaction = game(sender_input, labels, receiver_input)
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(game.parameters(), cfg.get("grad_clip", 1.0))
            optimizer.step()

            epoch_loss += loss.mean().item()
            epoch_acc += interaction.aux["acc"].mean().item()

            if batch_idx == 0:
                print(f"  Epoch {epoch} batch 1/{n_batches} running...", flush=True)

        epoch_loss /= len(train_loader)
        epoch_acc /= len(train_loader)

        print(f"Epoch {epoch}/{cfg['epochs']} | loss: {epoch_loss:.4f} | train_acc: {epoch_acc:.3f}", flush=True)

        # ----------------------------------------------------------- eval
        if epoch % cfg.get("eval_every", 5) == 0 or epoch == cfg["epochs"]:
            val_acc = evaluate(game, val_loader, device)
            meanings, messages = collect_messages(sender, val_loader, device)
            topo = compute_topo_similarity(meanings, messages)
            lang_stats = compute_all_metrics(messages, cfg["vocab_size"])

            print(
                f"[{arch}] epoch {epoch:4d} | loss {epoch_loss:.4f} | "
                f"train_acc {epoch_acc:.3f} | val_acc {val_acc:.3f} | "
                f"topo_rho {topo['rho']:.3f}"
            )

            if cfg.get("use_wandb"):
                import wandb
                wandb.log({"epoch": epoch, "train_acc": epoch_acc, "val_acc": val_acc,
                           "loss": epoch_loss, "topo_rho": topo["rho"], **lang_stats})

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(game.state_dict(), results_dir / "best_model.pt")

            if cfg.get("log_messages"):
                np.save(results_dir / f"messages_epoch{epoch}.npy", messages)
                np.save(results_dir / f"meanings_epoch{epoch}.npy", meanings)

    print(f"Done. Best val acc: {best_val_acc:.3f}. Results in {results_dir}")


@torch.no_grad()
def evaluate(game, loader, device):
    game.eval()
    total_acc = 0.0
    for sender_input, labels, receiver_input in loader:
        sender_input = sender_input.to(device)
        labels = labels.to(device)
        receiver_input = receiver_input.to(device)
        _, interaction = game(sender_input, labels, receiver_input)
        total_acc += interaction.aux["acc"].mean().item()
    game.train()
    return total_acc / len(loader)


if __name__ == "__main__":
    main()
