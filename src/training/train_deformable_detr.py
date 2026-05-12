import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection

try:
    from transformers import DeformableDetrForObjectDetection, DeformableDetrImageProcessor
except ImportError:
    raise ImportError("[ERROR] Missing required libraries. Run: pip install transformers timm")

# ──────────────────────────────────────────────────────────────────────────────
# 1. Dataset & Collate Function
# ──────────────────────────────────────────────────────────────────────────────
class COCOTrafficDatasetDETR(CocoDetection):
    def __init__(self, img_dir: str, ann_file: str):
        super().__init__(img_dir, ann_file)

    def __getitem__(self, idx: int):
        img, target = super().__getitem__(idx)
        img = img.convert("RGB") # Ép kiểu RGB để tránh lỗi tensor
        image_id = self.ids[idx]
        formatted_target = {'image_id': image_id, 'annotations': target}
        return img, formatted_target

def get_collate_fn(processor):
    def collate_fn(batch):
        pixel_values = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        encoding = processor(images=pixel_values, annotations=targets, return_tensors="pt")
        return {
            'pixel_values': encoding['pixel_values'],
            'pixel_mask': encoding['pixel_mask'],
            'labels': encoding['labels']
        }
    return collate_fn

# ──────────────────────────────────────────────────────────────────────────────
# 2. Main Training Loop (Single GPU with Validation)
# ──────────────────────────────────────────────────────────────────────────────
def train_deformable_detr(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Training Deformable DETR on Single Device: {device}")

    save_dir = config["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # NHẮC LẠI: ID giữ nguyên từ 0 vì ta đã chạy convert_to_coco.py ép về 0-based
    id2label = {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
        4: 'bus', 5: 'truck', 6: 'traffic light', 7: 'stop sign'
    }
    label2id = {v: k for k, v in id2label.items()}

    processor = DeformableDetrImageProcessor.from_pretrained("SenseTime/deformable-detr")

    # --- SETUP TRAIN & VAL DATASETS ---
    train_dataset = COCOTrafficDatasetDETR(config["dataset"]["train_images"], config["dataset"]["train_annotations"])
    val_dataset = COCOTrafficDatasetDETR(config["dataset"]["val_images"], config["dataset"]["val_annotations"])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["model"]["batch_size"],
        shuffle=True,
        num_workers=config["model"].get("num_workers", 2),
        collate_fn=get_collate_fn(processor),
        pin_memory=True,
        drop_last=True 
    )

    # Validation DataLoader (Không cần shuffle, drop_last=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["model"]["batch_size"],
        shuffle=False,
        num_workers=config["model"].get("num_workers", 2),
        collate_fn=get_collate_fn(processor),
        pin_memory=True
    )

    print("[Info] Downloading pre-trained Deformable DETR weights...")
    model = DeformableDetrForObjectDetection.from_pretrained(
        "SenseTime/deformable-detr",
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

    model.to(device)

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad], "lr": config["model"]["learning_rate"]},
        {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad], "lr": config["model"]["lr_backbone"]},
    ]
    
    optimizer = torch.optim.AdamW(param_dicts, weight_decay=config["model"]["weight_decay"])
    num_epochs = config["model"]["epochs"]
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_loss = float("inf")
    print(f"\n[Info] Starting training loop for {num_epochs} epochs on 1 GPU...\n")

    for epoch in range(num_epochs):
        # ─── TRAINING PHASE ───
        model.train()
        epoch_train_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()

            epoch_train_loss += loss.item()

            if (batch_idx + 1) % 50 == 0:
                print(f"  [Train] Epoch [{epoch+1}/{num_epochs}] Step [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

        lr_scheduler.step()
        avg_train_loss = epoch_train_loss / len(train_loader)

        # ─── VALIDATION PHASE 
        model.eval()
        epoch_val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(device)
                pixel_mask = batch["pixel_mask"].to(device)
                labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]
                
                outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
                epoch_val_loss += outputs.loss.item()
                
        avg_val_loss = epoch_val_loss / len(val_loader)

        print(f"── Epoch {epoch+1:>3}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} ──")

        # Lưu checkpoint Last cho mục đích fault tolerance
        model.save_pretrained(os.path.join(save_dir, "last"))
        
        # So sánh bằng Val Loss thay vì Train Loss để lưu Best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(os.path.join(save_dir, "best"))
            processor.save_pretrained(os.path.join(save_dir, "best")) 
            print(f"   ✓ Best model updated (Val Loss: {best_val_loss:.4f})")

    print(f"\n[Done] Training complete. Weights saved to: {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Deformable DETR (Single GPU)")
    parser.add_argument("--config", type=str, default="configs/deformable_detr_config.yaml")
    args = parser.parse_args()
    train_deformable_detr(args.config)