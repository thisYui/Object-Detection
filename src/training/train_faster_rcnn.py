import os
import argparse
import yaml
import torch
import torchvision
from torchvision.datasets import CocoDetection
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
from torchvision import transforms

# ── Try to import torchmetrics; fall back gracefully ──────────────────────────
try:
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    HAS_TORCHMETRICS = True
except ImportError:
    HAS_TORCHMETRICS = False
    print("[WARNING] torchmetrics not found. Validation mAP will be skipped.")
    print("          Install with: pip install torchmetrics")

torch.backends.cudnn.benchmark = True

# ──────────────────────────────────────────────────────────────────────────────
# 1. Dataset
# ──────────────────────────────────────────────────────────────────────────────
class COCOTrafficDataset(CocoDetection):
    """
    Custom Dataset wrapping torchvision's CocoDetection.
    - Remaps COCO category_ids to a contiguous label space starting at 1
      (0 is reserved for background by Faster R-CNN).
    - Returns (image_tensor, target_dict) ready for torchvision detection models.
    """

    def __init__(self, img_dir: str, ann_file: str):
        super().__init__(img_dir, ann_file)
        self.to_tensor = transforms.ToTensor()

        # Build a stable category_id → contiguous label mapping.
        # Sorted so the mapping is deterministic regardless of JSON order.
        cat_ids = sorted(self.coco.getCatIds())
        self.cat_id_to_label = {cat_id: idx + 1 for idx, cat_id in enumerate(cat_ids)}
        print(f"[Dataset] {len(cat_ids)} categories → label mapping: {self.cat_id_to_label}")

    def __getitem__(self, idx: int):
        img, annotations = super().__getitem__(idx)
        img = img.convert("RGB")
        image = self.to_tensor(img)

        boxes, labels = [], []
        for obj in annotations:
            x, y, w, h = obj["bbox"]
            # Skip degenerate boxes that would crash the loss computation
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_id_to_label[obj["category_id"]])

        if boxes:
            boxes_t  = torch.as_tensor(boxes,  dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes_t  = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,),   dtype=torch.int64)

        target = {
            "boxes":    boxes_t,
            "labels":   labels_t,
            "image_id": torch.tensor([idx]),
        }
        return image, target


def collate_fn(batch):
    """Variable-length bounding boxes require a custom collate."""
    return tuple(zip(*batch))


# ──────────────────────────────────────────────────────────────────────────────
# 2. Model
# ──────────────────────────────────────────────────────────────────────────────
def get_model(num_classes: int) -> torch.nn.Module:
    """
    Load ImageNet-pretrained Faster R-CNN ResNet-50 FPN and replace the
    classification head for `num_classes` (including background).
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# ──────────────────────────────────────────────────────────────────────────────
# 3. Validation
# ──────────────────────────────────────────────────────────────────────────────
def evaluate(model: torch.nn.Module, data_loader: DataLoader, device: torch.device) -> dict:
    """
    Run inference on the validation set and compute mAP using torchmetrics.
    Returns an empty dict if torchmetrics is unavailable.
    """
    if not HAS_TORCHMETRICS:
        return {}

    model.eval()
    metric = MeanAveragePrecision()

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            preds = [
                {
                    "boxes":  o["boxes"].cpu(),
                    "scores": o["scores"].cpu(),
                    "labels": o["labels"].cpu(),
                }
                for o in outputs
            ]
            tgts = [
                {
                    "boxes":  t["boxes"].cpu(),
                    "labels": t["labels"].cpu(),
                }
                for t in targets
            ]
            metric.update(preds, tgts)

    return metric.compute()


# ──────────────────────────────────────────────────────────────────────────────
# 4. Training
# ──────────────────────────────────────────────────────────────────────────────
def train_faster_rcnn(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Training Faster R-CNN on device: {device}")

    save_dir = config["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_dataset = COCOTrafficDataset(
        config["dataset"]["train_images"],
        config["dataset"]["train_annotations"],
    )
    val_dataset = COCOTrafficDataset(
        config["dataset"]["val_images"],
        config["dataset"]["val_annotations"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["model"]["batch_size"],
        shuffle=True,
        num_workers=config["model"].get("num_workers", 2),
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,           # Faster R-CNN inference is memory-heavy; keep at 1
        shuffle=False,
        num_workers=config["model"].get("num_workers", 2),
        collate_fn=collate_fn,
        pin_memory=True
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    num_classes = config["model"]["num_classes"]  # includes background (class 0)
    model = get_model(num_classes)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)

    model.to(device)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=config["model"]["learning_rate"],
        momentum=0.9,
        weight_decay=0.0005,
    )

    # MultiStepLR: decay x0.1 at epoch 12 and 17 (sensible for 20-epoch runs)
    num_epochs = config["model"]["epochs"]
    milestone_1 = int(num_epochs * 0.60)   # 60 % of total
    milestone_2 = int(num_epochs * 0.85)   # 85 % of total
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[milestone_1, milestone_2], gamma=0.1
    )
    print(f"[Info] LR milestones at epochs {milestone_1} and {milestone_2}")

    # ── Training loop ─────────────────────────────────────────────────────────
    best_loss = float("inf")
    val_every  = config["model"].get("val_every", 5)   # evaluate every N epochs

    print(f"[Info] Starting training for {num_epochs} epochs …\n")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (images, targets) in enumerate(train_loader):
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss.mean() for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

            if (batch_idx + 1) % 50 == 0:
                print(
                    f"  Epoch [{epoch+1}/{num_epochs}]  "
                    f"Step [{batch_idx+1}/{len(train_loader)}]  "
                    f"Loss: {losses.item():.4f}"
                )

        lr_scheduler.step()

        avg_loss = epoch_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\n── Epoch {epoch+1:>3}/{num_epochs}  avg_loss: {avg_loss:.4f}  lr: {current_lr:.6f}")

        # Save last checkpoint every epoch
        state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        torch.save(state_dict, os.path.join(save_dir, "last.pth"))

        # Save best checkpoint only when loss improves
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(state_dict, os.path.join(save_dir, "best.pth"))
            print(f"   ✓ Best model updated (loss: {best_loss:.4f})")

        # Periodic validation
        if (epoch + 1) % val_every == 0:
            print(f"   Running validation …")
            val_metrics = evaluate(model, val_loader, device)
            if val_metrics:
                map50    = val_metrics.get("map_50",   torch.tensor(0.0)).item()
                map5095  = val_metrics.get("map",      torch.tensor(0.0)).item()
                print(f"   mAP@50: {map50:.3f}  |  mAP@50-95: {map5095:.3f}")
            print()

    print(f"\n[Done] Training complete. Weights saved to: {save_dir}")
    print(f"       Best train loss: {best_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Faster R-CNN")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/faster_rcnn_config.yaml",
        help="Path to the YAML config file",
    )
    args = parser.parse_args()
    train_faster_rcnn(args.config)