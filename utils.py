import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import List


def create_parent_directory(filepath: str):
    parent_dir = os.path.dirname(filepath)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def visualize(
        tensor: torch.Tensor,
        save_path: str,
        title: str = None,
        max_images: int = 8,
        img_size: int = 3
):
    """
    - visualize images of (c, h, w) or (bs, c, h, w)
      if bs > max_images, it will show the first max_images images.
    - the tensor will be considered within [-1, 1]
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu()
    else:
        raise TypeError("The input should be torch.Tensor")

    if tensor.dim() == 3:
        imgs = tensor.unsqueeze(0)
    elif tensor.ndim == 4:
        imgs = tensor
    else:
        raise ValueError(f"Unsupported dim: {tensor.shape}")

    create_parent_directory(save_path)
    bs, c, h, w = imgs.shape
    num_show = min(bs, max_images)

    # convert image from [-1, 1] to [0, 1]
    imgs = (imgs + 1) / 2
    imgs = imgs.clamp(0, 1)

    # create a row of images of num_show columns,
    # with eatch image of size img_size
    fig, axes = plt.subplots(1, num_show, figsize=(img_size * num_show, img_size))

    if num_show == 1:
        axes = [axes]

    for i in range(num_show):
        img = imgs[i]
        if c == 1:
            img = img[0]
            axes[i].imshow(img, cmap='gray')
        else:
            img = img.permute(1, 2, 0)  # (c, h, w) -> (h, w, c)
            axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f"{title or ''} [{i}]")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def draw_loss_curve(
        loss_history: List[float],
        save_path: str,
        label: str,
        img_title: str,
        step_info: List[int] = None
):
    create_parent_directory(save_path)
    plt.figure()

    if step_info is not None:
        plt.plot(step_info, loss_history, label=label)
    else:
        plt.plot(loss_history, label=label)

    plt.yscale('log')
    plt.title(img_title)
    plt.xlabel('steps')
    plt.ylabel('loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


MiB = 1024 ** 2


def model_size(model: nn.Module) -> float:
    size = 0
    for param in model.parameters():
        size += param.nelement() * param.element_size()
    for buf in model.buffers():
        size += buf.nelement() * buf.element_size()
    return size / MiB
