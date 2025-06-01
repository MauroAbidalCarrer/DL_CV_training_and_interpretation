from collections.abc import Sequence

from torch import Tensor
from torch import device as dev
from torch.utils.data import Dataset, DataLoader


class DeviceDataLoader(DataLoader):

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        device: dev,
        # store_on_dev: bool,
        **kwargs,
        ):
        super().__init__(dataset, batch_size, **kwargs)
        self.device = device

    def __iter__(self):
        for batch in super().__iter__():
            if isinstance(batch, Sequence):
                yield type(batch)(map(self._to_device, batch))
            else:
                yield self._to_device(batch)

    def _to_device(self, tensor: Tensor) -> Tensor:
        if not isinstance(tensor, Tensor):
            raise ValueError(f"Batch of the wrapped dataset does not yield tensor, got {type(tensor)}")
        return tensor.to(self.device)