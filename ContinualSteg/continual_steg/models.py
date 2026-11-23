from __future__ import annotations

from copy import deepcopy

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class SpatialAdapter(nn.Module):
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


class _UnetBlock(nn.Module):
    """Recursive block matching the original UDH HidingUNet."""

    def __init__(self, outer_channels: int, inner_channels: int, submodule=None,
                 input_channels: int | None = None, outermost=False, innermost=False):
        super().__init__()
        self.outermost = outermost
        self.innermost = innermost
        input_channels = input_channels or outer_channels
        down = nn.Conv2d(input_channels, inner_channels, 4, 2, 1)
        if outermost:
            up = nn.ConvTranspose2d(inner_channels * 2, outer_channels, 4, 2, 1)
            self.model = nn.Sequential(down, submodule, nn.ReLU(True), up, nn.Tanh())
        elif innermost:
            up = nn.ConvTranspose2d(inner_channels, outer_channels, 4, 2, 1)
            self.model = nn.Sequential(nn.LeakyReLU(0.2, True), down, nn.ReLU(True), up)
        else:
            up = nn.ConvTranspose2d(inner_channels * 2, outer_channels, 4, 2, 1)
            self.model = nn.Sequential(nn.LeakyReLU(0.2, True), down, submodule, nn.ReLU(True), up)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        return torch.cat([x, self.model(x)], dim=1)


class HidingNet(nn.Module):
    """UDH cover-independent HidingUNet with a residual output."""

    def __init__(self, adapter_bottleneck: int, num_tasks: int, prompt_dim: int):
        super().__init__()
        block = _UnetBlock(512, 512, innermost=True)
        block = _UnetBlock(256, 512, block)
        block = _UnetBlock(128, 256, block)
        block = _UnetBlock(64, 128, block)
        self.model = _UnetBlock(3, 64, block, input_channels=3, outermost=True)
        self.adapters = nn.ModuleList([SpatialAdapter(3, adapter_bottleneck) for _ in range(num_tasks)])
        self.prompt_bias = nn.ModuleList([nn.Linear(prompt_dim, 3) for _ in range(num_tasks)])
        for layer in self.prompt_bias:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.adapters_enabled = True

    def forward_udh(self, secret: Tensor) -> Tensor:
        """Exact UDH hiding path with no task conditioning."""
        return self.model(secret) * (10.0 / 255.0)

    def forward(self, cover: Tensor, secret: Tensor, task_id: int, prompt: Tensor) -> Tensor:
        del cover
        residual = self.forward_udh(secret)
        if self.adapters_enabled and int(task_id) != 0:
            residual = self.adapters[int(task_id)](residual)
            residual = residual + self.prompt_bias[int(task_id)](prompt).unsqueeze(-1).unsqueeze(-1)
        return residual


class RevealNet(nn.Module):
    """The original UDH six-convolution RevealNet with task adapters."""

    def __init__(self, adapter_bottleneck: int, num_tasks: int, prompt_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv3 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv4 = nn.Conv2d(256, 128, 3, 1, 1)
        self.conv5 = nn.Conv2d(128, 64, 3, 1, 1)
        self.conv6 = nn.Conv2d(64, 3, 3, 1, 1)
        self.adapters = nn.ModuleList([SpatialAdapter(256, adapter_bottleneck) for _ in range(num_tasks)])
        self.prompt_bias = nn.ModuleList([nn.Linear(prompt_dim, 256) for _ in range(num_tasks)])
        for layer in self.prompt_bias:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.relu = nn.ReLU(True)
        self.adapters_enabled = True

    def extract_feature(self, container: Tensor) -> Tensor:
        return self.relu(self.conv3(self.relu(self.conv2(self.relu(self.conv1(container))))))

    def forward_udh(self, container: Tensor) -> Tensor:
        """Exact UDH reveal path with no task conditioning."""
        x = self.extract_feature(container)
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        return torch.sigmoid(self.conv6(x))

    def forward(self, container: Tensor, task_id: int, prompt: Tensor) -> Tensor:
        x = self.extract_feature(container)
        if self.adapters_enabled and int(task_id) != 0:
            x = self.adapters[int(task_id)](x)
            x = x + self.prompt_bias[int(task_id)](prompt).unsqueeze(-1).unsqueeze(-1)
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        return torch.sigmoid(self.conv6(x))


class ContinualHidingModel(nn.Module):
    def __init__(self, num_tasks: int, adapter_bottleneck: int = 32, prompt_dim: int = 64):
        super().__init__()
        self.num_tasks = num_tasks
        self.hiding = HidingNet(adapter_bottleneck, num_tasks, prompt_dim)
        self.reveal = RevealNet(adapter_bottleneck, num_tasks, prompt_dim)
        self.prompt_bank = PromptBank(num_tasks, prompt_dim)
        self.register_buffer("prototypes", torch.zeros(num_tasks, 256))
        self.register_buffer("prototype_valid", torch.zeros(num_tasks, dtype=torch.bool))

    def disable_adapters(self):
        self.hiding.adapters_enabled = False
        self.reveal.adapters_enabled = False
        for name, parameter in self.named_parameters():
            if any(token in name for token in ("adapters.", "prompt_bias.", "prompt_bank.prompts")):
                parameter.requires_grad = False

    def hide_reveal(self, cover: Tensor, secret: Tensor, task_id: int):
        if int(task_id) == 0:
            container = (cover + self.hiding.forward_udh(secret)).clamp(0, 1)
            return container, self.reveal.forward_udh(container)
        prompt = self.prompt_bank(task_id, cover.shape[0], cover.device)
        container = (cover + self.hiding(cover, secret, task_id, prompt)).clamp(0, 1)
        return container, self.reveal(container, task_id, prompt)

    def reveal_container(self, container: Tensor, task_id: int):
        if int(task_id) == 0:
            return self.reveal.forward_udh(container)
        prompt = self.prompt_bank(task_id, container.shape[0], container.device)
        return self.reveal(container, task_id, prompt)

    def train_clean_baseline(self):
        """Train only the original UDH parameters for task 0."""
        for name, parameter in self.named_parameters():
            parameter.requires_grad = not any(token in name for token in (
                "adapters.", "prompt_bias.", "prompt_bank.prompts"))

    def train_only_task(self, task_id: int, train_shared: bool = True):
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
        outputs = [self.reveal_container(container[i:i + 1], int(task_id))
                   for i, task_id in enumerate(task_ids.tolist())]
        return torch.cat(outputs, dim=0), task_ids
