import torchvision
from typing import Dict, Any, List
from cvtrainer.data.base import BaseDataAdapter


class CIFAR10DataAdapter(BaseDataAdapter):
    """CIFAR-10 DataAdapter"""

    def __init__(
        self,
        data_path: str,
        transforms=None,
        tensorizer=None,
        train: bool = True,
    ):
        super().__init__(data_path, transforms, tensorizer)
        self.train = train
        self.dataset = torchvision.datasets.CIFAR10(
            root=data_path, train=train, download=True
        )
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx

    def __len__(self) -> int:
        return len(self.dataset)

    def _load_sample(self, idx: int) -> Dict[str, Any]:
        image, label = self.dataset[idx]
        return {
            "x": image,
            "target": {"class_id": label},
        }
