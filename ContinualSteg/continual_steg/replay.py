from dataclasses import dataclass
from typing import List

import torch


@dataclass
class ReplayItem:
    cover: torch.Tensor
    secret: torch.Tensor
    task_id: int


class ReplayBuffer:
    """CPU reservoir replay with balanced task sampling."""

    def __init__(self, capacity: int = 256):
        self.capacity = max(0, int(capacity))
        self.items: List[ReplayItem] = []
        self.seen = 0

    def add(self, cover: torch.Tensor, secret: torch.Tensor, task_id: int):
        if self.capacity == 0:
            return
        for c, s in zip(cover.detach().cpu(), secret.detach().cpu()):
            self.seen += 1
            item = ReplayItem(c.clone(), s.clone(), int(task_id))
            if len(self.items) < self.capacity:
                self.items.append(item)
                continue
            slot = torch.randint(self.seen, (1,)).item()
            if slot < self.capacity:
                self.items[slot] = item

    def sample(self, count: int):
        if not self.items or count <= 0:
            return None
        count = min(int(count), len(self.items))
        by_task = {}
        for index, item in enumerate(self.items):
            by_task.setdefault(int(item.task_id), []).append(index)
        task_ids = sorted(by_task)
        selected = []
        for offset in range(count):
            task_id = task_ids[offset % len(task_ids)]
            choices = by_task[task_id]
            selected.append(choices[torch.randint(len(choices), (1,)).item()])
        items = [self.items[i] for i in selected]
        return (
            torch.stack([x.cover.detach().cpu() for x in items]),
            torch.stack([x.secret.detach().cpu() for x in items]),
            torch.tensor([x.task_id for x in items], dtype=torch.long),
        )

    def state_dict(self):
        return {"capacity": self.capacity, "seen": self.seen, "items": self.items}

    def load_state_dict(self, state, capacity=None):
        # Keep the new run's requested capacity instead of restoring an old
        # clean-only checkpoint's capacity=0.
        self.capacity = max(0, int(state.get("capacity", 0) if capacity is None else capacity))
        self.seen = int(state.get("seen", 0))
        self.items = [
            ReplayItem(item.cover.detach().cpu(), item.secret.detach().cpu(), int(item.task_id))
            for item in state.get("items", [])
        ]
        if self.capacity == 0:
            self.items = []
        elif len(self.items) > self.capacity:
            self.items = self.items[-self.capacity:]
