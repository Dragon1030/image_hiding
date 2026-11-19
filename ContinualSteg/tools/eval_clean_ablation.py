"""Evaluate clean-only checkpoints on fixed attack paths.

Both checkpoints always use task_id=0 for hiding and revealing. This isolates
the effect of the clean-stage adapter from the attack transformation itself.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from continual_steg.data import PairedImageDataset
from continual_steg.metrics import apd, psnr, ssim
from continual_steg.models import ContinualHidingModel
from continual_steg.tasks import TASKS, apply_attack


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with_adapter_checkpoint", required=True)
    parser.add_argument("--without_adapter_checkpoint", required=True)
    parser.add_argument("--val_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_model(checkpoint_path, device, disable_adapters):
    state = torch.load(checkpoint_path, map_location=device)
    weights = state["model"]
    task_count = sum(key.startswith("prompt_bank.prompts.") for key in weights)
    prompt_dim = weights["prompt_bank.prompts.0"].shape[0]
    adapter_weight = weights["hiding.adapters.0.down.weight"]
    bottleneck = adapter_weight.shape[0]
    model = ContinualHidingModel(task_count, bottleneck, prompt_dim).to(device)
    model.load_state_dict(weights)
    if disable_adapters:
        model.disable_adapters()
    model.eval()
    return model


@torch.no_grad()
def evaluate_model(model, loader, device, attack_tasks):
    rows = []
    samples = {task.name: None for task in attack_tasks}
    for batch_index, (cover, secret) in enumerate(loader):
        if args.max_batches and batch_index >= args.max_batches:
            break
        cover, secret = cover.to(device), secret.to(device)
        container, _ = model.hide_reveal(cover, secret, 0)
        for task in attack_tasks:
            attacked = apply_attack(container, task)
            recovered = model.reveal_container(attacked, 0)
            rows.append({
                "attack": task.name,
                "cover_psnr": psnr(container, cover),
                "cover_ssim": ssim(container, cover),
                "secret_psnr": psnr(recovered, secret),
                "secret_ssim": ssim(recovered, secret),
                "cover_apd": apd(container, cover),
            })
            if samples[task.name] is None and len(samples[task.name] or []) < args.max_samples:
                samples[task.name] = (cover[:1].cpu(), container[:1].cpu(), attacked[:1].cpu(),
                                      secret[:1].cpu(), recovered[:1].cpu())
    return rows, samples


def average_rows(rows, method):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["attack"], []).append(row)
    output = []
    for attack, values in grouped.items():
        output.append({
            "method": method,
            "attack": attack,
            **{key: sum(item[key] for item in values) / len(values)
               for key in ("cover_psnr", "cover_ssim", "secret_psnr", "secret_ssim", "cover_apd")},
        })
    return output


def save_samples(all_samples, out_dir):
    for method, samples in all_samples.items():
        (out_dir / "samples" / method).mkdir(parents=True, exist_ok=True)
        for attack, sample in samples.items():
            if sample is None:
                continue
            cover, container, attacked, secret, recovered = sample
            diff_cover = (container - cover).abs().mul(8).clamp(0, 1)
            diff_secret = (recovered - secret).abs().mul(8).clamp(0, 1)
            grid = make_grid(torch.cat([cover, container, attacked, diff_cover,
                                        secret, recovered, diff_secret]), nrow=7, padding=2)
            save_image(grid, out_dir / "samples" / method / f"{attack}.png")


def save_plot(summary, out_dir):
    attacks = list(dict.fromkeys(row["attack"] for row in summary))
    methods = ["with_adapter", "without_adapter"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for axis, metric, title in zip(axes, ("secret_psnr", "secret_ssim"),
                                   ("Recovered secret PSNR", "Recovered secret SSIM")):
        width = 0.36
        positions = list(range(len(attacks)))
        for offset, method in enumerate(methods):
            values = [next(row[metric] for row in summary
                           if row["method"] == method and row["attack"] == attack)
                      for attack in attacks]
            axis.bar([p + (offset - 0.5) * width for p in positions], values,
                     width=width, label=method)
        axis.set_xticks(positions)
        axis.set_xticklabels(attacks)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "clean_ablation_comparison.png", dpi=180)
    plt.close(fig)


def main():
    global args
    args = arguments()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda:0" or torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(PairedImageDataset(args.val_dir, args.image_size),
                        batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=True)
    attack_tasks = [TASKS[name] for name in ("clean", "jpeg", "gaussian", "resize", "blur")]
    with_adapter = build_model(args.with_adapter_checkpoint, device, False)
    without_adapter = build_model(args.without_adapter_checkpoint, device, True)
    rows_with, samples_with = evaluate_model(with_adapter, loader, device, attack_tasks)
    rows_without, samples_without = evaluate_model(without_adapter, loader, device, attack_tasks)
    summary = average_rows(rows_with, "with_adapter") + average_rows(rows_without, "without_adapter")
    with open(out_dir / "clean_ablation_metrics.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    save_plot(summary, out_dir)
    save_samples({"with_adapter": samples_with, "without_adapter": samples_without}, out_dir)
    print(f"saved metrics, plot, and samples under {out_dir}")


if __name__ == "__main__":
    main()
