import torch
import torch.nn as nn
from typing import Tuple, Optional, Callable
import torch.nn.functional as F


def mse_loss(v_cond: torch.Tensor, v_theta: torch.Tensor) -> torch.Tensor:
    # calulate the mse loss for two tensors
    return F.mse_loss(v_cond, v_theta, reduction='mean')


def euler_simulator(
        model: nn.Module,
        num_timesteps: int,
        x_init: torch.Tensor,
        labels: Optional[torch.Tensor],
        guidance: Optional[float],
        num_classes: Optional[int]
) -> torch.Tensor:
    """
    Euler Simulator to simulate the ODE
    Args:
    - model: v_theta(x_t, t | y), where y is optional
    - num_timesteps: number of timesteps to simulate ODE
    - x_init: (bs, c, h, w), samples from initial distribution
    - labels: (bs,), of type torch.int
    - guidance: cfg guidance scale
    Returns:
    - x: (bs, c, h, w), the final image
    """
    x = x_init.clone()
    timesteps = torch.linspace(1.0, 0.0, num_timesteps, device=x_init.device)
    if labels is not None:
        if not isinstance(num_classes, int):
            raise TypeError("The num_classes should be int.")
        labels_uncond = torch.ones_like(labels) * num_classes

    for idx in range(num_timesteps - 1):
        t = timesteps[idx].view(-1, 1, 1, 1)
        h = timesteps[idx + 1] - timesteps[idx]

        if labels is None:
            x = x + model(x, t) * h
        else:
            # use cfg
            if guidance is None:
                guidance = 1.0

            x = x + (
                guidance * model(x, t, labels) + (1-guidance) * model(x, t, labels_uncond)
            ) * h

    return x


def heun_simulator(
        model: nn.Module,
        num_timesteps: int,
        x_init: torch.Tensor,
        labels: Optional[torch.Tensor],
        guidance: Optional[float],
        num_classes: Optional[int]
) -> torch.Tensor:
    """
    Euler Simulator to simulate the ODE
    Args:
    - model: v_theta(x_t, t | y), where y is optional
    - num_timesteps: number of timesteps to simulate ODE
    - x_init: (bs, c, h, w), samples from initial distribution
    - labels: (bs,), of type torch.int
    - guidance: cfg guidance scale
    Returns:
    - x: (bs, c, h, w), the final image
    """
    x = x_init.clone()
    timesteps = torch.linspace(1.0, 0.0, num_timesteps, device=x_init.device)
    if labels is not None:
        if not isinstance(num_classes, int):
            raise TypeError("The num_classes should be int.")
        if num_classes:
            labels_uncond = torch.ones_like(labels) * num_classes

    for idx in range(num_timesteps - 1):
        t_curr = timesteps[idx].view(-1, 1, 1, 1)
        t_next = timesteps[idx + 1].view(-1, 1, 1, 1)
        h = timesteps[idx + 1] - timesteps[idx]

        if labels is None:
            # euler step
            v_curr = model(x, t_curr)
            x_next = x + v_curr * h
            # heun revise step
            x = x + 0.5 * (v_curr + model(x_next, t_next)) * h
        else:
            # use cfg
            if guidance is None:
                guidance = 1.0

            # euler step
            v_curr_cfg = guidance * model(x, t_curr, labels) + (1-guidance) * model(x, t_curr, labels_uncond)
            x_next = x + v_curr_cfg * h
            # heun revise step
            v_next_cfg = guidance * model(x_next, t_next, labels) + (1-guidance) * model(x_next, t_next, labels_uncond)
            x = x + 0.5 * (v_curr_cfg + v_next_cfg) * h

    return x


class FlowMatching:
    def __init__(
            self,
            num_classes: Optional[int],
            cfg_ratio: Optional[float],
            img_shape: Tuple[int]
    ):
        self.img_shape = img_shape
        self.num_classes = num_classes
        self.cfg_ratio = cfg_ratio

    def get_img_shape(self) -> Tuple[int]:
        return self.img_shape

    def loss(
            self,
            model: nn.Module,
            images: torch.Tensor,
            labels: Optional[torch.Tensor] = None,
            loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = mse_loss
    ) -> torch.Tensor:
        """
        Args:
        - model: v_theta(x_t, t | y) where y is optional
        - images: (bs, c, h, w)
        - labels: (bs,), of type torch.int
        Returns:
        - loss: the train loss
        """
        t = torch.rand(images.shape[0], device=images.device).view(-1, 1, 1, 1)
        x_init = torch.randn_like(images)
        x_t = (1 - t) * images + t * x_init
        v_cond = x_init - images

        if labels is None:
            v_theta = model(x_t, t)
        else:
            mask = torch.rand(images.shape[0], device=images.device)
            labels[mask < self.cfg_ratio] = self.num_classes
            v_theta = model(x_t, t, labels)

        loss = loss_func(v_cond, v_theta)
        return loss

    @torch.no_grad()
    def sample(
            self,
            model: nn.Module,
            num_timesteps: int,
            num_imgs: int,
            labels: Optional[torch.Tensor],
            guidance_scale: Optional[float],
            device: torch.device,
            simulator: Callable = euler_simulator
    ) -> torch.Tensor:
        """
        to generate images
        Args:
        - model: v_theta(x_t, t, | y) where y is optional
        - num_timesteps: number of steps to simulate ODE
        - labels: (bs,), of type torch.int
        - guidance_scale: cfg guidance scale,
        - simulator: function to simulate ODE
        Returns:
        - x: (bs, c, h, w)
        """
        x_init = torch.randn(num_imgs, *self.img_shape, device=device)
        labels = labels.to(device)
        x = simulator(
            model = model,
            num_timesteps = num_timesteps,
            x_init = x_init,
            labels = labels,
            guidance = guidance_scale,
            num_classes = self.num_classes
        )

        return x
