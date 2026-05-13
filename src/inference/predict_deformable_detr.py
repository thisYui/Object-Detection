import os
import argparse
import yaml
import json
import torch

from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection

try:
    from transformers import (
        DeformableDetrForObjectDetection,
        DeformableDetrImageProcessor
    )
except ImportError:
    raise ImportError("[ERROR] Missing required libraries. Run: pip install transformers timm")


# ──────────────────────────────────────────────────────────────────────────────
# 1. Dataset
# ──────────────────────────────────────────────────────────────────────────────
class COCOTrafficDatasetDETR(CocoDetection):
    """
    COCO dataset wrapper for Hugging Face Deformable DETR.

    The processor expects each target in this format:
    {
        "image_id": image_id,
        "annotations": [...]
    }
    """

    def __init__(self, img_dir: str, ann_file: str):
        super().__init__(img_dir, ann_file)

    def __getitem__(self, idx: int):
        img, target = super().__getitem__(idx)

        # Force RGB to avoid channel errors
        img = img.convert("RGB")

        image_id = self.ids[idx]

        formatted_target = {
            "image_id": image_id,
            "annotations": target
        }

        return img, formatted_target


# ──────────────────────────────────────────────────────────────────────────────
# 2. Collate Function
# ──────────────────────────────────────────────────────────────────────────────
def get_collate_fn(processor):
    def collate_fn(batch):
        images = [item[0] for item in batch]
        targets = [item[1] for item in batch]

        encoding = processor(
            images=images,
            annotations=targets,
            return_tensors="pt"
        )

        return {
            "pixel_values": encoding["pixel_values"],
            "pixel_mask": encoding["pixel_mask"],
            "labels": encoding["labels"]
        }

    return collate_fn


# ──────────────────────────────────────────────────────────────────────────────
# 3. Utility Functions
# ──────────────────────────────────────────────────────────────────────────────
def load_yaml(config_path: str):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"[ERROR] Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_paths(config):
    required_paths = [
        config["dataset"]["train_images"],
        config["dataset"]["train_annotations"],
        config["dataset"]["val_images"],
        config["dataset"]["val_annotations"],
    ]

    for path in required_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"[ERROR] Path not found: {path}")


def check_coco_category_ids(annotation_file: str, expected_ids: set):
    """
    Check whether COCO category_id values match the id2label mapping.

    This project assumes category_id is 0-based:
    0: person
    1: bicycle
    ...
    7: stop sign
    """

    with open(annotation_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    category_ids_in_categories = {
        cat["id"] for cat in data.get("categories", [])
    }

    category_ids_in_annotations = {
        ann["category_id"] for ann in data.get("annotations", [])
    }

    all_ids = category_ids_in_categories | category_ids_in_annotations

    if not all_ids.issubset(expected_ids):
        print("[WARNING] COCO category_id values do not match expected 0-based IDs.")
        print(f"[WARNING] Expected IDs: {sorted(expected_ids)}")
        print(f"[WARNING] Found IDs:    {sorted(all_ids)}")
        print("[WARNING] If you are using original COCO IDs, you must remap them to 0-based IDs before training.")
    else:
        print(f"[Info] COCO category_id check passed for: {annotation_file}")


def move_labels_to_device(labels, device):
    """
    Move all tensors inside labels to device.
    Hugging Face processor returns labels as a list of dictionaries.
    """

    moved_labels = []

    for target in labels:
        moved_target = {}
        for key, value in target.items():
            if torch.is_tensor(value):
                moved_target[key] = value.to(device)
            else:
                moved_target[key] = value
        moved_labels.append(moved_target)

    return moved_labels


# ──────────────────────────────────────────────────────────────────────────────
# 4. Main Training Function
# ──────────────────────────────────────────────────────────────────────────────
def train_deformable_detr(config_path: str):
    config = load_yaml(config_path)
    check_paths(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    print(f"[Info] Training Deformable DETR on device: {device}")
    print(f"[Info] AMP enabled: {use_amp}")

    save_dir = config["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # IMPORTANT:
    # These IDs assume your COCO annotations were converted to 0-based category_id.
    id2label = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        4: "bus",
        5: "truck",
        6: "traffic light",
        7: "stop sign"
    }

    label2id = {v: k for k, v in id2label.items()}
    expected_ids = set(id2label.keys())

    check_coco_category_ids(config["dataset"]["train_annotations"], expected_ids)
    check_coco_category_ids(config["dataset"]["val_annotations"], expected_ids)

    print("[Info] Loading image processor...")
    processor = DeformableDetrImageProcessor.from_pretrained(
        "SenseTime/deformable-detr",
        size={
            "shortest_edge": 480,
            "longest_edge": 800
        }
    )

    print("[Info] Loading datasets...")
    train_dataset = COCOTrafficDatasetDETR(
        config["dataset"]["train_images"],
        config["dataset"]["train_annotations"]
    )

    val_dataset = COCOTrafficDatasetDETR(
        config["dataset"]["val_images"],
        config["dataset"]["val_annotations"]
    )

    batch_size = config["model"]["batch_size"]
    num_workers = config["model"].get("num_workers", 2)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=get_collate_fn(processor),
        pin_memory=use_amp,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=get_collate_fn(processor),
        pin_memory=use_amp,
        drop_last=False
    )

    if len(train_loader) == 0:
        raise ValueError("[ERROR] train_loader is empty. Check dataset size and batch_size.")

    if len(val_loader) == 0:
        raise ValueError("[ERROR] val_loader is empty. Check validation dataset size and batch_size.")

    print(f"[Info] Train images: {len(train_dataset)}")
    print(f"[Info] Val images:   {len(val_dataset)}")
    print(f"[Info] Train steps per epoch: {len(train_loader)}")
    print(f"[Info] Val steps per epoch:   {len(val_loader)}")

    print("[Info] Loading pre-trained Deformable DETR weights...")
    model = DeformableDetrForObjectDetection.from_pretrained(
        "SenseTime/deformable-detr",
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

    model.to(device)

    learning_rate = config["model"]["learning_rate"]
    lr_backbone = config["model"]["lr_backbone"]
    weight_decay = config["model"]["weight_decay"]
    num_epochs = config["model"]["epochs"]

    param_dicts = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if "backbone" not in n and p.requires_grad
            ],
            "lr": learning_rate
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if "backbone" in n and p.requires_grad
            ],
            "lr": lr_backbone
        }
    ]

    optimizer = torch.optim.AdamW(
        param_dicts,
        weight_decay=weight_decay
    )

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs
    )

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_loss = float("inf")

    print(f"\n[Info] Starting training for {num_epochs} epochs...\n")

    for epoch in range(num_epochs):
        # ──────────────────────────────────────────────────────────────────────
        # Training Phase
        # ──────────────────────────────────────────────────────────────────────
        model.train()
        epoch_train_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            labels = move_labels_to_device(batch["labels"], device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(
                    pixel_values=pixel_values,
                    pixel_mask=pixel_mask,
                    labels=labels
                )
                loss = outputs.loss

            scaler.scale(loss).backward()

            # Important when using AMP:
            # unscale before gradient clipping
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=0.1
            )

            scaler.step(optimizer)
            scaler.update()

            epoch_train_loss += loss.item()

            if (batch_idx + 1) % 50 == 0:
                print(
                    f"  [Train] Epoch [{epoch + 1}/{num_epochs}] "
                    f"Step [{batch_idx + 1}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        avg_train_loss = epoch_train_loss / len(train_loader)

        # ──────────────────────────────────────────────────────────────────────
        # Validation Phase
        # ──────────────────────────────────────────────────────────────────────
        model.eval()
        epoch_val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(device)
                pixel_mask = batch["pixel_mask"].to(device)
                labels = move_labels_to_device(batch["labels"], device)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs = model(
                        pixel_values=pixel_values,
                        pixel_mask=pixel_mask,
                        labels=labels
                    )
                    val_loss = outputs.loss

                epoch_val_loss += val_loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader)

        lr_scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"── Epoch {epoch + 1:>3}/{num_epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"LR: {current_lr:.8f} ──"
        )

        # Save last checkpoint every epoch
        last_dir = os.path.join(save_dir, "last")
        model.save_pretrained(last_dir)
        processor.save_pretrained(last_dir)

        # Save best checkpoint based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

            best_dir = os.path.join(save_dir, "best")
            model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)

            print(f"   ✓ Best model updated. Val Loss: {best_val_loss:.4f}")

    print("\n[Done] Training complete.")
    print(f"[Done] Best model saved to: {os.path.join(save_dir, 'best')}")
    print(f"[Done] Last model saved to: {os.path.join(save_dir, 'last')}")


# ──────────────────────────────────────────────────────────────────────────────
# 5. Entry Point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Deformable DETR on COCO-format traffic dataset"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/deformable_detr_config.yaml",
        help="Path to Deformable DETR config YAML file"
    )

    args = parser.parse_args()
    train_deformable_detr(args.config)