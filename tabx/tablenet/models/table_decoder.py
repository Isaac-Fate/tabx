import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F

from ..spec import TableNetSpec


class TableDecoder(nn.Module):

    def __init__(self) -> None:

        super().__init__()

        # Conv layer
        self.conv7 = nn.Conv2d(512, 512, kernel_size=1)

        # Up sampling layers
        self.upsample_pool4 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsample_pool3 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.out = nn.Sequential(
            nn.ConvTranspose2d(
                1280,
                TableNetSpec.n_output_channels,
                kernel_size=2,
                stride=2,
                dilation=1,
            ),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: Tensor,
        *,
        pool3: Tensor,
        pool4: Tensor,
    ) -> Tensor:

        # Input x has shape (N, 512, 32, 32)

        # (N, 512, 32, 32)
        x = self.conv7(x)

        # Upsampling and concatenate with pool4
        # pool4 has shape (N, 512, 64, 64)

        # (N, 512, 64, 64)
        x = self.upsample_pool4(x)

        # (N, 1024, 64, 64)
        x = torch.cat((x, pool4), dim=1)

        # Upsampling and concatenate with pool3
        # pool3 has shape (N, 256, 128, 128)

        # (N, 1024, 128, 128)
        x = self.upsample_pool3(x)

        # (N, 1280, 128, 128)
        x = torch.cat((x, pool3), dim=1)

        # Further upsampling

        # (N, 1280, 256, 256)
        x = F.interpolate(x, scale_factor=2)

        # (N, 1280, 512, 512)
        x = F.interpolate(x, scale_factor=2)

        # Ouput layer
        # (N, 1, 1024, 1024)
        x = self.out(x)

        return x
