from typing import ClassVar
from abc import ABC
from pydantic import BaseModel


class TableNetSpec(BaseModel, ABC):

    input_image_width: ClassVar[int] = 1024
    input_image_height: ClassVar[int] = 1024
