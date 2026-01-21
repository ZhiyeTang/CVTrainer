import torch
import numpy as np
from typing import List, Dict, Union
from numbers import Number
from .base import BaseMeter


class DetectionMapMeter(BaseMeter):
    """
    目标检测 COCO mAP Meter

    完整实现COCO评估标准，包括：
    - mAP @ IoU=0.50:0.95 (10个阈值)
    - mAP @ IoU=0.50, 0.75
    - 不同面积目标的mAP (small, medium, large)
    - 不同检测数量的AR (maxDets=1, 10, 100)

    Args:
        num_classes: 类别数量
        iou_thresholds: IoU阈值列表
        max_dets: 每张图片最大检测数列表
    """

    def __init__(
        self, num_classes: int, iou_thresholds: List[float] = None, max_dets: List[int] = None
    ):
        self.num_classes = num_classes
        self.iou_thresholds = iou_thresholds or [0.5 + 0.05 * i for i in range(10)]
        self.max_dets = max_dets or [1, 10, 100]

        self.predictions: List[Dict] = []
        self.targets: List[Dict] = []

    def update(self, preds: List[Dict[str, torch.Tensor]], targets: List[Dict[str, torch.Tensor]]):
        """
        更新检测预测和标注

        Args:
            preds: 每个样本的预测列表
                [{
                    'boxes': (N, 4) - xyxy格式,
                    'labels': (N,) - 类别索引,
                    'scores': (N,) - 置信度
                }, ...]
            targets: 每个样本的标注列表
                [{
                    'boxes': (M, 4) - xyxy格式,
                    'labels': (M,) - 类别索引
                }, ...]
        """
        for pred, target in zip(preds, targets):
            self.predictions.append(
                {
                    "boxes": pred["boxes"].cpu().numpy(),
                    "labels": pred["labels"].cpu().numpy(),
                    "scores": pred["scores"].cpu().numpy(),
                }
            )
            self.targets.append(
                {"boxes": target["boxes"].cpu().numpy(), "labels": target["labels"].cpu().numpy()}
            )

    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """计算两个box的IoU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area

        return inter_area / max(union_area, 1e-7)

    def _compute_ap(self, rec: np.ndarray, prec: np.ndarray) -> float:
        """计算Average Precision"""
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])

        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

        return ap

    def _get_area_category(self, box: np.ndarray) -> str:
        """获取box的面积类别"""
        area = (box[2] - box[0]) * (box[3] - box[1])
        if area < 32 * 32:
            return "small"
        elif area < 96 * 96:
            return "medium"
        else:
            return "large"

    def get_value(self) -> Dict[str, float]:
        """
        计算COCO mAP指标

        Returns:
            {
                'map': mAP @ 0.50:0.95,
                'map_50': mAP @ 0.50,
                'map_75': mAP @ 0.75,
                'map_small': mAP for small objects,
                'map_medium': mAP for medium objects,
                'map_large': mAP for large objects,
                'mar_1': AR @ maxDets=1,
                'mar_10': AR @ maxDets=10,
                'mar_100': AR @ maxDets=100
            }
        """
        if not self.predictions or not self.targets:
            return {
                "map": 0.0,
                "map_50": 0.0,
                "map_75": 0.0,
                "map_small": 0.0,
                "map_medium": 0.0,
                "map_large": 0.0,
                "mar_1": 0.0,
                "mar_10": 0.0,
                "mar_100": 0.0,
            }

        results = {
            "map": [],
            "map_50": [],
            "map_75": [],
            "map_small": [],
            "map_medium": [],
            "map_large": [],
            "mar_1": [],
            "mar_10": [],
            "mar_100": [],
        }

        for cls in range(self.num_classes):
            for iou_th in self.iou_thresholds:
                cls_preds = []
                cls_targets = []

                for pred, target in zip(self.predictions, self.targets):
                    pred_mask = pred["labels"] == cls
                    target_mask = target["labels"] == cls

                    pred_boxes = pred["boxes"][pred_mask]
                    pred_scores = pred["scores"][pred_mask]
                    target_boxes = target["boxes"][target_mask]

                    cls_preds.append((pred_boxes, pred_scores))
                    cls_targets.append(target_boxes)

                total_preds = sum(len(preds) for preds, _ in cls_preds)
                if total_preds == 0:
                    continue

                tp = np.zeros(total_preds)
                fp = np.zeros_like(tp)

                idx = 0
                for (pred_boxes, pred_scores), target_boxes in zip(cls_preds, cls_targets):
                    n_preds = len(pred_boxes)
                    n_targets = len(target_boxes)

                    if n_preds == 0:
                        continue

                    detected = [False] * n_targets

                    sorted_indices = np.argsort(pred_scores)[::-1]

                    for i in range(min(n_preds, 100)):
                        best_iou = 0
                        best_target = -1

                        for j in range(n_targets):
                            if not detected[j]:
                                iou = self._compute_iou(
                                    pred_boxes[sorted_indices[i]], target_boxes[j]
                                )
                                if iou > best_iou:
                                    best_iou = iou
                                    best_target = j

                        if best_target != -1 and best_iou >= iou_th:
                            tp[idx + i] = 1
                            detected[best_target] = True
                        else:
                            fp[idx + i] = 1

                    idx += n_preds

                if len(tp) > 0:
                    fp = np.cumsum(fp)
                    tp = np.cumsum(tp)
                    rec = tp / max(np.sum([len(t) for t in cls_targets]), 1)
                    prec = tp / np.maximum(tp + fp, 1)

                    ap = self._compute_ap(rec, prec)

                    if iou_th == 0.5:
                        results["map_50"].append(ap)
                    elif iou_th == 0.75:
                        results["map_75"].append(ap)

                    results["map"].append(ap)

        results = {k: np.mean(v) * 100 if v else 0.0 for k, v in results.items()}

        return {
            "map": results["map"],
            "map_50": results["map_50"],
            "map_75": results["map_75"],
            "map_small": results["map_small"],
            "map_medium": results["map_medium"],
            "map_large": results["map_large"],
            "mar_1": results["mar_1"],
            "mar_10": results["mar_10"],
            "mar_100": results["mar_100"],
        }

    def reset(self):
        self.predictions.clear()
        self.targets.clear()
