from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from continual_steg.data import PairedImageDataset
from continual_steg.metrics import apd, psnr, ssim
from continual_steg.models import ContinualHidingModel
from continual_steg.replay import ReplayBuffer
from continual_steg.tasks import TASKS, Task, apply_attack


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Continual robust image hiding")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--train_dir", default="")
    parser.add_argument("--val_dir", default="")
    parser.add_argument("--checkpoint_dir", default="runs/continual")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--resume", default="")
    parser.add_argument("--tasks", default="clean,jpeg,gaussian,resize,blur")
    parser.add_argument("--strategy", choices=["continual", "sequential_finetune"], default="continual")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs_per_task", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--shared_lr_scale", type=float, default=0.1,
                        help="learning-rate multiplier for shared UDH weights after task 0")
    parser.add_argument("--secret_weight", type=float, default=0.75,
                        help="UDH reconstruction-loss weight")
    parser.add_argument("--loss", choices=["mse", "l1"], default="mse",
                        help="clean-task loss; UDH baseline uses mse")
    parser.add_argument("--adapter_bottleneck", type=int, default=32)
    parser.add_argument("--prompt_dim", type=int, default=64)
    parser.add_argument("--replay_capacity", type=int, default=256)
    parser.add_argument("--replay_per_batch", type=int, default=4)
    parser.add_argument("--replay_seed_per_task", type=int, default=64)
    parser.add_argument("--replay_weight", type=float, default=1.0)
    parser.add_argument("--lambda_distill", type=float, default=0.2)
    parser.add_argument("--no_shared_training", action="store_true",
                        help="freeze shared weights after clean stage for adapter-only ablation")
    parser.add_argument("--disable_adapters", action="store_true",
                        help="disable all task adapters and prompt biases")
    parser.add_argument("--stop_after_task", type=int, default=-1,
                        help="stop after this task id, useful for clean-only ablations")
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_val_batches", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def get_tasks(text: str):
    names = [item.strip() for item in text.split(",") if item.strip()]
    unknown = [name for name in names if name not in TASKS]
    if unknown:
        raise ValueError(f"unknown task(s): {unknown}; available={sorted(TASKS)}")
    if not names or names[0] != "clean":
        raise ValueError("the first task must be clean to preserve the UDH reference stage")
    return [TASKS[name] for name in names]


def make_loader(root, image_size, batch_size, workers, shuffle):
    dataset = PairedImageDataset(root, image_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, drop_last=shuffle, pin_memory=True)


def seed_replay_from_loader(replay, loader, task_id, target_count):
    if replay.capacity <= 0 or target_count <= 0:
        return
    collected = 0
    for cover, secret in loader:
        replay.add(cover, secret, task_id)
        collected += int(cover.shape[0])
        if collected >= target_count:
            break


def _task_loss(model, cover, secret, task: Task, loss_name="mse", secret_weight=0.75):
    container, _ = model.hide_reveal(cover, secret, task.task_id)
    attacked = apply_attack(container, task)
    recovered = model.reveal_container(attacked, task.task_id)
    criterion = F.mse_loss if loss_name == "mse" else F.l1_loss
    return (criterion(container, cover), secret_weight * criterion(recovered, secret), container, recovered)


def _distill_loss(model, old_model, cover, secret, task_ids, device):
    terms = []
    for old_task_id in task_ids.unique().tolist():
        mask = task_ids == old_task_id
        task = next(item for item in TASKS.values() if item.task_id == int(old_task_id))
        with torch.no_grad():
            old_container, _ = old_model.hide_reveal(cover[mask], secret[mask], int(old_task_id))
            noise = (torch.randn_like(old_container) * task.severity
                     if task.name == "gaussian" else None)
            old_recovered = old_model.reveal_container(
                apply_attack(old_container, task, noise), int(old_task_id))
        new_container, _ = model.hide_reveal(cover[mask], secret[mask], int(old_task_id))
        new_recovered = model.reveal_container(
            apply_attack(new_container, task, noise), int(old_task_id))
        terms.extend([F.mse_loss(new_container, old_container), F.mse_loss(new_recovered, old_recovered)])
    return torch.stack(terms).mean() if terms else torch.zeros((), device=device)


def train_one_task(model, old_model, task, loader, replay, args, device, optimizer,
                   start_epoch=0, checkpoint_callback=None):
    model.train()
    for epoch in range(start_epoch, args.epochs_per_task):
        cover_sum = secret_sum = replay_sum = distill_sum = 0.0
        replay_count = distill_count = empty_replay_batches = 0
        batches = 0
        for batch_index, (cover, secret) in enumerate(tqdm(loader, desc=f"task={task.name} epoch={epoch + 1}")):
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
            cover, secret = cover.to(device, non_blocking=True), secret.to(device, non_blocking=True)
            cover_loss, secret_loss, _, _ = _task_loss(
                model, cover, secret, task, args.loss, args.secret_weight)
            replay_loss = torch.zeros((), device=device)
            distill_loss = torch.zeros((), device=device)

            memory = replay.sample(args.replay_per_batch)
            if memory is not None:
                old_cover, old_secret, old_task = memory
                old_cover, old_secret = old_cover.to(device), old_secret.to(device)
                replay_count += int(old_task.numel())
                replay_terms = []
                for replay_task_id in old_task.unique().tolist():
                    mask = old_task == replay_task_id
                    replay_task = next(item for item in TASKS.values() if item.task_id == int(replay_task_id))
                    replay_cover_loss, replay_secret_loss, _, _ = _task_loss(
                        model, old_cover[mask], old_secret[mask], replay_task,
                        args.loss, args.secret_weight)
                    replay_terms.append(replay_cover_loss + replay_secret_loss)
                replay_loss = torch.stack(replay_terms).mean()
                if old_model is not None:
                    distill_loss = _distill_loss(model, old_model, old_cover, old_secret, old_task, device)
                    distill_count += int(old_task.numel())
            elif old_model is not None:
                empty_replay_batches += 1

            loss = cover_loss + secret_loss + args.replay_weight * replay_loss + args.lambda_distill * distill_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            cover_sum += cover_loss.item()
            secret_sum += secret_loss.item()
            replay_sum += replay_loss.item()
            distill_sum += distill_loss.item()
            batches += 1
        divisor = max(1, batches)
        print(f"task={task.name} epoch={epoch + 1}/{args.epochs_per_task} "
              f"cover={cover_sum / divisor:.6f} secret={secret_sum / divisor:.6f} "
              f"replay={replay_sum / divisor:.6f} distill={distill_sum / divisor:.6f} "
              f"replay_n={replay_count} distill_n={distill_count} "
              f"empty_replay_batches={empty_replay_batches}")
        if checkpoint_callback is not None:
            checkpoint_callback(epoch + 1)


@torch.no_grad()
def evaluate(model, task, loader, device, max_batches=0):
    model.eval()
    values = {"cover_psnr": [], "cover_ssim": [], "secret_psnr": [], "secret_ssim": [], "cover_apd": []}
    for batch_index, (cover, secret) in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        cover, secret = cover.to(device), secret.to(device)
        container, _ = model.hide_reveal(cover, secret, task.task_id)
        recovered = model.reveal_container(apply_attack(container, task), task.task_id)
        values["cover_psnr"].append(psnr(container, cover))
        values["cover_ssim"].append(ssim(container, cover))
        values["secret_psnr"].append(psnr(recovered, secret))
        values["secret_ssim"].append(ssim(recovered, secret))
        values["cover_apd"].append(apd(container, cover))
    return {key: sum(value) / max(1, len(value)) for key, value in values.items()}


def save_stage(path, model, optimizer, replay, task_id, epoch, strategy="continual"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "replay": replay.state_dict(), "task_id": task_id, "epoch": epoch,
                "strategy": strategy}, path)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    tasks = get_tasks(args.tasks)
    model = ContinualHidingModel(max(task.task_id for task in tasks) + 1,
                                 args.adapter_bottleneck, args.prompt_dim).to(device)
    if args.disable_adapters:
        model.disable_adapters()
    replay = ReplayBuffer(args.replay_capacity)
    start_task, start_epoch, resume_state = 0, 0, None
    checkpoint_path = args.checkpoint or args.resume
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location=device)
        if state.get("strategy", "continual") != args.strategy:
            raise ValueError("checkpoint strategy does not match requested strategy")
        resume_state = state
        model.load_state_dict(state["model"])
        if args.disable_adapters:
            model.disable_adapters()
        replay.load_state_dict(state["replay"], capacity=args.replay_capacity)
        if int(state.get("epoch", args.epochs_per_task)) < args.epochs_per_task:
            start_task, start_epoch = int(state["task_id"]), int(state.get("epoch", 0))
        else:
            start_task = int(state["task_id"]) + 1

    train_root = args.train_dir or os.path.join(args.data_root, "train")
    val_root = args.val_dir or os.path.join(args.data_root, "val")
    train_loader = make_loader(train_root, args.image_size, args.batch_size, args.workers, True)
    val_loader = make_loader(val_root, args.image_size, args.batch_size, args.workers, False)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    # A clean-only checkpoint intentionally contains no Replay items. Seed the
    # first old task before starting task 1 so Replay and Distillation are real.
    if start_task > 0 and replay.capacity > 0 and not replay.items:
        seed_replay_from_loader(replay, train_loader, 0, args.replay_seed_per_task)
        print(f"seeded replay task=0 items={len(replay.items)} capacity={replay.capacity}")
    if args.mode == "eval":
        for task in tasks:
            print(task.name, evaluate(model, task, val_loader, device, args.max_val_batches))
        return

    old_model = None
    run_tasks = tasks if args.stop_after_task < 0 else tasks[:args.stop_after_task + 1]
    for task in run_tasks[start_task:]:
        if args.strategy == "sequential_finetune":
            old_model = None
            for parameter in model.parameters():
                parameter.requires_grad = True
        elif task.task_id == 0 and start_task == 0:
            model.train_clean_baseline()
        else:
            old_model = model.clone_frozen()
            model.train_only_task(task.task_id, train_shared=not args.no_shared_training)

        if args.disable_adapters:
            model.disable_adapters()

        active, shared = [], []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if any(token in name for token in ("adapters.", "prompt_bias.", "prompt_bank.prompts", "prompt_bank.task_logits")):
                active.append(parameter)
            else:
                shared.append(parameter)
        groups = [{"params": active, "lr": args.learning_rate}]
        if shared:
            groups.append({"params": shared, "lr": args.learning_rate * args.shared_lr_scale})
        optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
        if resume_state is not None and task.task_id == start_task and start_epoch > 0:
            optimizer.load_state_dict(resume_state["optimizer"])
        latest_path = os.path.join(args.checkpoint_dir, "latest.pt")
        checkpoint_callback = lambda epoch: save_stage(
            latest_path, model, optimizer, replay, task.task_id, epoch, args.strategy)
        train_one_task(model, old_model, task, train_loader, replay, args, device, optimizer,
                       start_epoch=start_epoch if task.task_id == start_task else 0,
                       checkpoint_callback=checkpoint_callback)

        replay_cover, replay_secret = next(iter(train_loader))
        with torch.no_grad():
            prototype_container, _ = model.hide_reveal(replay_cover.to(device), replay_secret.to(device), task.task_id)
            model.update_prototype(prototype_container, task.task_id)
        seed_replay_from_loader(replay, train_loader, task.task_id, args.replay_seed_per_task)
        rows = []
        for seen_task in tasks[:tasks.index(task) + 1]:
            metrics = evaluate(model, seen_task, val_loader, device, args.max_val_batches)
            rows.append({"stage": task.task_id, "task": seen_task.name, **metrics})
            print(f"stage={task.name} eval={seen_task.name} {metrics}")
        save_stage(os.path.join(args.checkpoint_dir, f"stage_{task.task_id:02d}.pt"),
                   model, optimizer, replay, task.task_id, args.epochs_per_task, args.strategy)
        start_epoch = 0
        with open(os.path.join(args.checkpoint_dir, "metrics.csv"), "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
            if handle.tell() == 0:
                writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
