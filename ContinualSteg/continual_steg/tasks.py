from dataclasses import dataclass
import io

import torch
import torch.nn.functional as F
from PIL import Image


@dataclass(frozen=True)
class Task:
    task_id: int
    name: str
    severity: float = 0.0


TASKS = {
    "clean": Task(0, "clean"),
    "jpeg": Task(1, "jpeg", 50.0),
    "gaussian": Task(2, "gaussian", 0.05),
    "resize": Task(3, "resize", 0.75),
    "blur": Task(4, "blur", 1.2),
}


def _jpeg_one(image: torch.Tensor, quality: int) -> torch.Tensor:
    """Encode/decode one tensor with the same JPEG operation used at test time."""
    array = (image.detach().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy())
    buffer = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(buffer, format="JPEG", quality=int(quality), subsampling=0)
    buffer.seek(0)
    decoded = Image.open(buffer).convert("RGB")
    result = torch.from_numpy(__import__("numpy").array(decoded)).permute(2, 0, 1).float().div(255)
    return result.to(device=image.device, dtype=image.dtype)


def apply_attack(x: torch.Tensor, task: Task, gaussian_noise: torch.Tensor | None = None) -> torch.Tensor:
    """Apply a training/evaluation attack. JPEG uses real encode/decode with STE."""
    if task.name == "clean":
        return x.clamp(0, 1)
    if task.name == "gaussian":
        noise = gaussian_noise if gaussian_noise is not None else torch.randn_like(x) * task.severity
        return (x + noise).clamp(0, 1)
    if task.name == "jpeg":
        quality = max(1, min(100, int(task.severity)))
        encoded = torch.stack([_jpeg_one(item, quality) for item in x])
        return x + (encoded - x).detach()
    if task.name == "resize":
        h, w = x.shape[-2:]
        small = F.interpolate(x, scale_factor=task.severity, mode="bilinear", align_corners=False)
        return F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False).clamp(0, 1)
    if task.name == "blur":
        radius = max(1, int(round(task.severity)))
        size = radius * 2 + 1
        coords = torch.arange(size, device=x.device, dtype=x.dtype) - radius
        kernel_1d = torch.exp(-(coords ** 2) / (2 * task.severity ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel = kernel_1d[:, None] * kernel_1d[None, :]
        kernel = kernel.expand(x.shape[1], 1, size, size)
        return F.conv2d(x, kernel, padding=radius, groups=x.shape[1]).clamp(0, 1)
    raise ValueError(f"unknown task {task.name}")
