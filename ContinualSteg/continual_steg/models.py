from __future__ import annotations

from copy import deepcopy
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class SpatialAdapter(nn.Module):
    """Zero-initialized bottleneck residual path, so task 0 starts as UDH."""

    def __init__(self, channels: int, bottleneck: int):
        super().__init__()
        hidden = max(1, min(channels, bottleneck))
        self.norm = nn.GroupNorm(1, channels)
        self.down = nn.Conv2d(channels, hidden, 1)
        self.up = nn.Conv2d(hidden, channels, 1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.up(F.gelu(self.down(self.norm(x))))


class PromptBank(nn.Module):
    def __init__(self, num_tasks: int, prompt_dim: int):
        super().__init__()
        self.prompts = nn.ParameterList([nn.Parameter(torch.zeros(prompt_dim)) for _ in range(num_tasks)])

    def forward(self, task_id: int, batch_size: int, device: torch.device) -> Tensor:
        return self.prompts[int(task_id)].view(1, -1).expand(batch_size, -1).to(device)


class HidingNet(nn.Module):
    def __init__(self, adapter_bottleneck: int, num_tasks: int, prompt_dim: int):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(6, 64, 3, 2, 1), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(inplace=True))
        self.enc3 = nn.Sequential(nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU(inplace=True))
        self.adapters = nn.ModuleList([SpatialAdapter(256, adapter_bottleneck) for _ in range(num_tasks)])
        self.prompt_bias = nn.ModuleList([nn.Linear(prompt_dim, 256) for _ in range(num_tasks)])
        for layer in self.prompt_bias:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, cover: Tensor, secret: Tensor, task_id: int, prompt: Tensor) -> Tensor:
        x = self.enc3(self.enc2(self.enc1(torch.cat([cover, secret], dim=1))))
        x = self.adapters[int(task_id)](x)
        x = x + self.prompt_bias[int(task_id)](prompt).unsqueeze(-1).unsqueeze(-1)
        return self.dec(x)


class RevealNet(nn.Module):
    def __init__(self, adapter_bottleneck: int, num_tasks: int, prompt_dim: int):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3, 64, 3, 2, 1), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(inplace=True))
        self.enc3 = nn.Sequential(nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU(inplace=True))
        self.adapters = nn.ModuleList([SpatialAdapter(256, adapter_bottleneck) for _ in range(num_tasks)])
        self.prompt_bias = nn.ModuleList([nn.Linear(prompt_dim, 256) for _ in range(num_tasks)])
        for layer in self.prompt_bias:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Sigmoid(),
        )

    def extract_feature(self, container: Tensor) -> Tensor:
        return self.enc3(self.enc2(self.enc1(container)))

    def forward(self, container: Tensor, task_id: int, prompt: Tensor) -> Tensor:
        x = self.extract_feature(container)
        x = self.adapters[int(task_id)](x)
        x = x + self.prompt_bias[int(task_id)](prompt).unsqueeze(-1).unsqueeze(-1)
        return self.dec(x)


class ContinualHidingModel(nn.Module):
    def __init__(self, num_tasks: int, adapter_bottleneck: int = 32, prompt_dim: int = 64):
        super().__init__()
        self.num_tasks = num_tasks
        self.hiding = HidingNet(adapter_bottleneck, num_tasks, prompt_dim)
        self.reveal = RevealNet(adapter_bottleneck, num_tasks, prompt_dim)
        self.prompt_bank = PromptBank(num_tasks, prompt_dim)
        self.register_buffer("prototypes", torch.zeros(num_tasks, 256))
        self.register_buffer("prototype_valid", torch.zeros(num_tasks, dtype=torch.bool))

    def hide_reveal(self, cover: Tensor, secret: Tensor, task_id: int):
        prompt = self.prompt_bank(task_id, cover.shape[0], cover.device)
        container = self.hiding(cover, secret, task_id, prompt)
        return container, self.reveal(container, task_id, prompt)

    def reveal_container(self, container: Tensor, task_id: int):
        prompt = self.prompt_bank(task_id, container.shape[0], container.device)
        return self.reveal(container, task_id, prompt)

    def train_only_task(self, task_id: int, train_shared: bool = True):
        """Train shared weights with replay protection plus only the active adapter."""
        for name, parameter in self.named_parameters():
            is_active = (f"adapters.{int(task_id)}." in name or
                         f"prompt_bias.{int(task_id)}." in name or
                         f"prompt_bank.prompts.{int(task_id)}" in name)
            is_shared = not any(token in name for token in ("adapters.", "prompt_bias.", "prompt_bank.prompts"))
            parameter.requires_grad = is_active or (train_shared and is_shared)

    def clone_frozen(self):
        old = deepcopy(self).eval()
        for parameter in old.parameters():
            parameter.requires_grad = False
        return old

    @torch.no_grad()
    def update_prototype(self, container: Tensor, task_id: int):
        feature = self.reveal.extract_feature(container).mean(dim=(2, 3))
        feature = F.normalize(feature.mean(dim=0, keepdim=True), dim=-1).squeeze(0)
        self.prototypes[int(task_id)].copy_(feature)
        self.prototype_valid[int(task_id)] = True

    @torch.no_grad()
    def route(self, container: Tensor):
        feature = F.normalize(self.reveal.extract_feature(container).mean(dim=(2, 3)), dim=-1)
        valid = self.prototype_valid
        if not bool(valid.any()):
            raise RuntimeError("no task prototypes have been collected")
        scores = feature @ F.normalize(self.prototypes[valid], dim=-1).t()
        valid_ids = valid.nonzero(as_tuple=False).flatten()
        return valid_ids[scores.argmax(dim=-1)]

    @torch.no_grad()
    def recover_unknown(self, container: Tensor):
        task_ids = self.route(container)
        outputs = []
        for index, task_id in enumerate(task_ids.tolist()):
            outputs.append(self.reveal_container(container[index:index + 1], int(task_id)))
        return torch.cat(outputs, dim=0), task_ids
