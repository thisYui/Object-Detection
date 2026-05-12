# Requirement: pip install pycocotools
from pycocotools.coco import COCO
import json
import random
import os
import urllib.request


def extract_traffic_dataset(coco_annotation_path, output_path, target_classes, max_images=3000):
    # Initialize COCO API for instance annotations
    coco = COCO(coco_annotation_path)

    # Get category IDs for the target classes
    cat_ids = coco.getCatIds(catNms=target_classes)

    # Gather all image IDs that contain at least one of the target classes
    img_ids_with_targets = set()
    for cat_id in cat_ids:
        ids = coco.getImgIds(catIds=[cat_id])
        img_ids_with_targets.update(ids)

    img_ids = list(img_ids_with_targets)

    # Randomly sample images to strictly limit training dataset size
    if len(img_ids) > max_images:
        img_ids = random.sample(img_ids, max_images)

    # Load annotations specifically for the selected images and target categories
    ann_ids = coco.getAnnIds(imgIds=img_ids, catIds=cat_ids, iscrowd=None)
    annotations = coco.loadAnns(ann_ids)

    # Load metadata for the sampled images
    images = coco.loadImgs(img_ids)

    # Construct a new COCO-compliant dictionary
    filtered_coco = {
        "info": coco.dataset.get('info', {}),
        "licenses": coco.dataset.get('licenses', []),
        "categories": coco.loadCats(cat_ids),
        "images": images,
        "annotations": annotations
    }

    # Dump the filtered dataset into a new JSON file
    with open(output_path, 'w') as f:
        json.dump(filtered_coco, f)

    print(f"Successfully extracted {len(img_ids)} images and {len(annotations)} annotations.")
    print(f"Filtered dataset saved to: {output_path}")

def download_subset_images(json_file, output_dir):
    # Load the filtered JSON dataset
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    images = data['images']
    total_images = len(images)
    print(f"Starting download of {total_images} images...")

    # Iterate through the image list and download each one
    for i, img in enumerate(images):
        img_url = img['coco_url']
        file_name = img['file_name']
        file_path = os.path.join(output_dir, file_name)

        # Skip download if the file already exists
        if not os.path.exists(file_path):
            try:
                urllib.request.urlretrieve(img_url, file_path)
            except Exception as e:
                print(f"Error downloading {file_name}: {e}")

        # Print progress every 100 images
        if (i + 1) % 100 == 0:
            print(f"Downloaded {i + 1}/{total_images} images.")

    print(f"Download complete! Images saved to: {output_dir}")

# Execution example
if __name__ == "__main__":
    traffic_classes = ['person', 'car', 'motorcycle', 'bus', 'truck', 'bicycle', 'traffic light', 'stop sign']
    coco_annotation_path = 'data/raw/annotations/instances_train2017.json'
    input_json = 'data/raw/traffic_subset_train.json'
    output_images_dir = 'data/processed/train/images'
    
    extract_traffic_dataset(coco_annotation_path, input_json, traffic_classes, max_images=5000)

    download_subset_images(input_json, output_images_dir)