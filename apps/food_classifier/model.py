from cvtrainer.models.base import BaseModel
from cvtrainer.models.backbones import ResNet18Backbone, ResNet50Backbone
from cvtrainer.models.adapters import IdentityAdapter, ConvAdapter, LinearAdapter
from cvtrainer.models.heads import MulticlassClassifier, MultiLabelClassifier


class MyModel(BaseModel):
    """示例模型：ResNet18 + Conv + Classifier"""

    def __init__(self, num_classes: int = 101, dropout_rate: float = 0.5):
        backbone = ResNet18Backbone(pretrained=True)
        adapter = ConvAdapter(backbone.backbone_channel, head_channel=256)
        head = MulticlassClassifier(adapter.head_channel, num_classes, dropout_rate)
        super().__init__(backbone, adapter, head)
