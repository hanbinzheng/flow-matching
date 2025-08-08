import torch
import torch.nn as nn
import math
from typing import List


def time_embedding(t: torch.Tensor, dim: int, max_period: float = 1e4) -> torch.Tensor:
    """
    The sinusoidal positional time embedding
    Args:
    - t: (bs,), (bs, 1) or (bs, 1, 1, 1)
    - dim: embedding dimension
    Returns:
    - t_embed: (bs, dim)
    """
    t = t.reshape(-1)
    half_dim = dim // 2
    # angular_freqs_i = (1 / max_period)^{i / half_dim}
    angular_freqs = torch.exp(
        - math.log(max_period)
        * torch.arange(0, half_dim, dtype=torch.float32)
        / half_dim
    ).to(t.device)  # angular_freqs: (half_dim, )

    args = t[:, None].float() * angular_freqs[None]  # args: (bs, half_dim)
    t_embed = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (bs, half_dim * 2)

    if dim % 2:
        t_embed = torch.cat([t_embed, torch.zeros_like(t_embed[:, :1])], dim=-1)

    return t_embed


class TimeEmbedder(nn.Module):
    def __init__(self, dim: int, num_freq: int = 256, time_scale: float = 1e3):
        super().__init__()
        self.num_freq = num_freq
        self.time_scale = time_scale
        self.mlp = nn.Sequential(
            nn.Linear(num_freq, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - t: (bs,), (bs, 1) or (bs, 1, 1, 1)
        Returns:
        - t_embed: (bs, dim)
        """
        t = t * self.time_scale
        t_embed = time_embedding(t, self.num_freq)
        t_embed = self.mlp(t_embed)
        return t_embed


class ResidualLayer(nn.Module):
    def __init__(self, num_channels: int, t_embed_dim: int, y_embed_dim: int):
        super().__init__()
        self.init_conv = nn.Sequential(
            nn.SiLU(),
            nn.BatchNorm2d(num_channels),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        )
        self.main_conv = nn.Sequential(
            nn.SiLU(),
            nn.BatchNorm2d(num_channels),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        )
        # change t from (bs, t_embed_dim) into (bs, num_channels)
        self.t_adapter = nn.Sequential(
            nn.Linear(t_embed_dim, t_embed_dim),
            nn.SiLU(),
            nn.Linear(t_embed_dim, num_channels)
        )
        # change label from (bs, y_embed_dim) into (bs, num_channels)
        self.y_adapter = nn.Sequential(
            nn.Linear(y_embed_dim, y_embed_dim),
            nn.SiLU(),
            nn.Linear(y_embed_dim, num_channels)
        )

    def forward(self, x: torch.Tensor,
                t_embed: torch.Tensor, y_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, h, w)
        - t_embed: (bx, t_embed_dim)
        - y_embed: (bs, y_embed_dim)
        """
        res = x.clone()
        x = self.init_conv(x)

        t_embed = self.t_adapter(t_embed)
        x = x + t_embed[:, :, None, None]
        y_embed = self.y_adapter(y_embed)
        x = x + y_embed[:, :, None, None]

        x = self.main_conv(x)
        x = x + res

        return x


class Encoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 num_residual_layers: int, t_embed_dim: int, y_embed_dim: int):
        super().__init__()
        self.residual_blocks = nn.ModuleList([
           ResidualLayer(
               num_channels = in_channels,
               t_embed_dim = t_embed_dim,
               y_embed_dim = y_embed_dim
           ) for _ in range(num_residual_layers)
        ])
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )

    def forward(self, x: torch.Tensor,
                t_embed: torch.Tensor, y_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, h, w)
        - t_embed: (bs, t_embed_dim)
        - y_embed: (bs, y_embed_dim)
        """
        for block in self.residual_blocks:
            x = block(x, t_embed, y_embed)

        x = self.downsample(x)
        return x


class Bottleneck(nn.Module):
    def __init__(self, num_channels: int,
                 num_residual_layers: int, t_embed_dim: int, y_embed_dim: int):
        super().__init__()
        self.residual_blocks = nn.ModuleList([
            ResidualLayer(
                num_channels = num_channels,
                t_embed_dim = t_embed_dim,
                y_embed_dim = y_embed_dim
            ) for _ in range(num_residual_layers)
        ])

    def forward(self, x: torch.Tensor,
                t_embed: torch.Tensor, y_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, h, w)
        - t_embed: (bs, t_embed_dim)
        - y_embed: (bs, y_embed_dim)
        """
        for block in self.residual_blocks:
            x = block(x, t_embed, y_embed)

        return x


class Decoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 num_residual_layers: int, t_embed_dim: int, y_embed_dim: int):
        super().__init__()
        fusion_channels = in_channels // 2
        self.fusion_block = nn.Sequential(
            nn.Conv2d(in_channels, fusion_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(fusion_channels),
            nn.SiLU()
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(fusion_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )
        self.residual_blocks = nn.ModuleList([
            ResidualLayer(
                num_channels = out_channels,
                t_embed_dim = t_embed_dim,
                y_embed_dim = y_embed_dim,
            ) for _ in range(num_residual_layers)
        ])

    def forward(self, x: torch.Tensor,
                t_embed: torch.Tensor, y_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, h, w)
        - t_embed: (bs, t_embed_dim)
        - y_embed: (bs, y_embed_dim)
        """
        x = self.fusion_block(x)
        x = self.upsample(x)

        for block in self.residual_blocks:
            x = block(x, t_embed, y_embed)

        return x


class FMUNet(nn.Module):
    def __init__(self, channels: List[int], num_residual_layers: int,
                 t_embed_dim: int, y_embed_dim: int, num_classes: int = 10):
        super().__init__()
        self.init_conv = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[1]),
            nn.SiLU()
        )
        self.t_embedder = TimeEmbedder(t_embed_dim)
        self.y_embedder = nn.Embedding(num_classes + 1, y_embed_dim)

        encoders = []
        decoders = []
        # [3, 7, 15, 20, 30]
        for (curr, next) in zip(channels[1:-1], channels[2:]):
            encoders.append(Encoder(curr, next, num_residual_layers, t_embed_dim, y_embed_dim))
            decoders.append(Decoder(next * 2, curr, num_residual_layers, t_embed_dim, y_embed_dim))

        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(reversed(decoders))
        self.bottleneck = Bottleneck(channels[-1], num_residual_layers, t_embed_dim, y_embed_dim)
        self.final_conv = nn.Conv2d(channels[1], channels[0], kernel_size=3, padding=1)


    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, h, w)
        - t: (bs,), (bs, 1) or (bs, 1, 1, 1)
        - y: (bs,), of type torch.int
        Returns:
        - u^theta(x, t | y): (bs, c, h, w)
        """
        res = []
        t_embed = self.t_embedder(t)  # (bs, t_embed_dim)
        y_embed = self.y_embedder(y)  # (bs, y_embed_dim)

        x = self.init_conv(x)  # (bs, channel[1], h, w)

        for encoder in self.encoders:
            x = encoder(x, t_embed, y_embed)  # (bs, channel[i], h, w)
            res.append(x.clone())

        x = self.bottleneck(x, t_embed, y_embed)

        for decoder in self.decoders:
            x_res = res.pop()  # (bs, channel[i], h, w)
            x = torch.cat([x, x_res], dim=1)  # (bs, channel[i] * 2, h, w)
            x = decoder(x, t_embed, y_embed)  # (bs, channel[i-1], h, w)

        x = self.final_conv(x)
        return x

    def initialize_weights(self):
        def init_model(model):
            if isinstance(model, nn.Linear):
                nn.init.kaiming_normal_(model.weight, nonlinearity='relu')
                if model.bias is not None:
                    nn.init.zeros_(model.bias)
            elif isinstance(model, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(model.weight, mode='fan_out', nonlinearity='relu')
                if model.bias is not None:
                    nn.init.zeros_(model.bias)
            elif isinstance(model, nn.Embedding):
                nn.init.normal_(model.weight, 0.0, 0.02)
        self.apply(init_model)
