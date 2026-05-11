import json
import os

def clean_and_format_coco(input_json, output_json):
    """
    Standardize COCO JSON file: Remap class IDs to the 0-N range and ensure
    a valid structure for Faster R-CNN and Deformable DETR.
    """
    if not os.path.exists(input_json):
        print(f"Error: Input file not found at {input_json}")
        return

    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 1. Remap Category IDs to ensure continuity (0, 1, 2... 7)
    # Sorting by old ID ensures the class name order remains unchanged
    old_categories = sorted(data['categories'], key=lambda x: x['id'])
    
    id_map = {}
    new_categories = []
    
    print("Starting class ID remapping for consistency:")
    for i, cat in enumerate(old_categories):
        new_id = i
        id_map[cat['id']] = new_id
        
        new_cat = cat.copy()
        new_cat['id'] = new_id
        new_categories.append(new_cat)
        print(f"  - {cat['name']}: {cat['id']} -> {new_id}")

    # 2. Update annotations with the new IDs
    new_annotations = []
    for ann in data['annotations']:
        new_ann = ann.copy()
        new_ann['category_id'] = id_map[ann['category_id']]
        new_annotations.append(new_ann)

    # 3. Create complete COCO structure
    cleaned_coco = {
        "info": data.get('info', {"description": "Traffic Subset for Object Detection Project"}),
        "licenses": data.get('licenses', []),
        "images": data['images'],
        "annotations": new_annotations,
        "categories": new_categories
    }

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(cleaned_coco, f, ensure_ascii=False, indent=4)

    print(f"\nSuccess: Standardized COCO file saved at {output_json}")
    print(f"Total: {len(data['images'])} images, {len(new_annotations)} annotations.")

if __name__ == "__main__":
    # Path to the filtered file from the previous step
    raw_json = 'data/raw/traffic_subset_train.json'
    
    # Path to save the standardized file for training Faster R-CNN / DETR
    processed_json = 'data/processed/annotations/train.json'
    
    clean_and_format_coco(raw_json, processed_json)
