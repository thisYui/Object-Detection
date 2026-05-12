import json
import os

def clean_format_and_split_coco(input_json, processed_base_dir):
    if not os.path.exists(input_json):
        print(f"Error: Original file not found at {input_json}")
        return

    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 1. Remap Category IDs
    old_categories = sorted(data['categories'], key=lambda x: x['id'])
    id_map = {}
    new_categories = []
    
    for i, cat in enumerate(old_categories):
        new_id = i
        id_map[cat['id']] = new_id
        new_cat = cat.copy()
        new_cat['id'] = new_id
        new_categories.append(new_cat)

    # 2. Update annotations with the new IDs
    new_annotations = []
    for ann in data['annotations']:
        new_ann = ann.copy()
        new_ann['category_id'] = id_map[ann['category_id']]
        new_annotations.append(new_ann)

    # 3. Split COCO file based on README directory structure
    splits = ['train', 'val', 'test']
    
    # Create directory for COCO JSON files required by Faster R-CNN/DETR
    annotations_out_dir = os.path.join(processed_base_dir, 'annotations')
    os.makedirs(annotations_out_dir, exist_ok=True)

    print("Starting generation of split JSON files...")
    for split in splits:
        # Image path complies strictly with: data/processed/train/images
        img_dir = os.path.join(processed_base_dir, split, 'images')
        
        if not os.path.exists(img_dir):
            print(f"Skipping {split} split: Directory {img_dir} does not exist.")
            continue

        valid_filenames = set([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
        
        filtered_images = [img for img in data['images'] if img['file_name'] in valid_filenames]
        valid_image_ids = set([img['id'] for img in filtered_images])
        filtered_annotations = [ann for ann in new_annotations if ann['image_id'] in valid_image_ids]

        split_coco_data = {
            "info": data.get('info', {"description": f"Traffic Subset - {split.capitalize()}"}),
            "licenses": data.get('licenses', []),
            "images": filtered_images,
            "annotations": filtered_annotations,
            "categories": new_categories
        }

        out_json_path = os.path.join(annotations_out_dir, f"{split}.json")
        with open(out_json_path, 'w', encoding='utf-8') as f:
            json.dump(split_coco_data, f, ensure_ascii=False, indent=4)

        print(f"Created {split}.json: {len(filtered_images)} images, {len(filtered_annotations)} annotations.")

if __name__ == "__main__":
    raw_json = 'data/raw/traffic_subset_train.json'
    
    # Points to the root 'processed' directory containing train/, val/, test/
    processed_base_dir = 'data/processed' 
    
    clean_format_and_split_coco(raw_json, processed_base_dir)