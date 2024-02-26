from torch import nn
from torch import Tensor
from torchvision.models import vgg19, VGG19_Weights

from ..spec import TableNetSpec
from .mask import Mask
from .table_decoder import TableDecoder
from .column_decoder import ColumnDecoder


class TableNet(nn.Module):

    def __init__(
        self,
        vgg19_weights: VGG19_Weights = VGG19_Weights.IMAGENET1K_V1,
    ) -> None:

        super().__init__()

        # Weights for the VGG19 network
        self._vgg19_weights = vgg19_weights

        # VGG19 base network
        self.encoder = vgg19(weights=self._vgg19_weights).features

        # VGG19 Blocks
        self.vgg19_block1 = self.encoder[0:5]
        self.vgg19_block2 = self.encoder[5:10]
        self.vgg19_block3 = self.encoder[10:19]
        self.vgg19_block4 = self.encoder[19:28]
        self.vgg19_block5 = self.encoder[28:37]

        # Two more conv layers after the VGG19 features

        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.8),
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.8),
        )

        # Table decoder
        self.table_decoder = TableDecoder()

        # Column decoder
        self.column_decoder = ColumnDecoder()

    def forward(self, image: Tensor) -> Mask:

        # Input image has shape (N, 3, 1024, 1024)

        # (N, 64, 512, 512)
        x = self.vgg19_block1(image)

        # (N, 128, 256, 256)
        x = self.vgg19_block2(x)

        # (N, 256, 128, 128)
        pool3 = self.vgg19_block3(x)

        # (N, 512, 64, 64)
        pool4 = self.vgg19_block4(pool3)

        # (N, 512, 32, 32)
        x = self.vgg19_block5(pool4)

        # Table mask
        table_mask = self.table_decoder(x, pool3=pool3, pool4=pool4)

        # Column mask
        column_mask = self.column_decoder(x, pool3=pool3, pool4=pool4)

        return Mask(
            table=table_mask,
            column=column_mask,
        )
