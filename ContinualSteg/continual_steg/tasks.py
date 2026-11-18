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
    # Kept for ablations, but deliberately excluded from the default stream.
    "crop": Task(4, "crop", 0.75),
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


def apply_attack(x: torch.Tensor, task: Task) -> torch.Tensor:
    """Apply a training/evaluation attack. JPEG uses real encode/decode with STE."""
    if task.name == "clean":
        return x.clamp(0, 1)
    if task.name == "gaussian":
        return (x + torch.randn_like(x) * task.severity).clamp(0, 1)
    if task.name == "jpeg":
        quality = max(1, min(100, int(task.severity)))
        encoded = torch.stack([_jpeg_one(item, quality) for item in x])
        return x + (encoded - x).detach()
    if task.name == "resize":
        h, w = x.shape[-2:]
        small = F.interpolate(x, scale_factor=task.severity, mode="bilinear", align_corners=False)
        return F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False).clamp(0, 1)
    if task.name == "crop":
        h, w = x.shape[-2:]
        crop_h, crop_w = max(1, int(h * task.severity)), max(1, int(w * task.severity))
        top, left = (h - crop_h) // 2, (w - crop_w) // 2
        cropped = x[..., top:top + crop_h, left:left + crop_w]
        return F.interpolate(cropped, size=(h, w), mode="bilinear", align_corners=False).clamp(0, 1)
    raise ValueError(f"unknown task {task.name}")
