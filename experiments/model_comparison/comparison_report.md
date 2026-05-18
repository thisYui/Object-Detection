# Model Comparison Report

## Summary

| model | precision_mean | recall_mean | f1_mean | mAP50 | mAP50_95 | fps | model_size_mb |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Faster R-CNN | 0.3727 | 0.4718 | 0.4164 | 0.6062 | 0.3727 | 0.3707 | 158.1684 |
| YOLOv8 | 0.4492 | 0.3828 | 0.4133 | 0.3764 | 0.2406 | 23.8051 | 5.9448 |
| Deformable DETR | 0.0744 | 0.1823 | 0.1057 | 0.1604 | 0.0744 | 0.6159 | 155.8142 |

## Best Results

- Best mAP@0.5: Faster R-CNN (0.6062)
- Best mAP@0.5:0.95: Faster R-CNN (0.3727)
- Best FPS: YOLOv8 (23.81 FPS)
- Smallest model: YOLOv8 (5.94 MB)
