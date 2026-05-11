import json
import os

def convert_coco_to_yolo(json_file, output_dir):
    # Load JSON dataset
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Create output directory for YOLO labels
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Map COCO category IDs to continuous YOLO class IDs (0 to N-1)
    # Sorting ensures the ID assignment is consistent with the YAML file
    categories = sorted(data['categories'], key=lambda x: x['id'])
    coco_id_to_yolo_id = {cat['id']: i for i, cat in enumerate(categories)}

    # Print the mapping logic for verification
    print("Class ID Mapping:")
    for cat in categories:
        print(f"  {cat['name']}: {coco_id_to_yolo_id[cat['id']]}")

    # Create a lookup dictionary for images
    images_info = {img['id']: img for img in data['images']}

    # Process each annotation
    for ann in data['annotations']:
        image_id = ann['image_id']
        category_id = ann['category_id']
        bbox = ann['bbox']

        # Get image dimensions for normalization
        img = images_info[image_id]
        img_w = img['width']
        img_h = img['height']

        # Get the new mapped YOLO class ID
        yolo_class_id = coco_id_to_yolo_id[category_id]

        # Convert COCO bbox [x_min, y_min, width, height] 
        # to YOLO bbox [x_center, y_center, width, height] normalized
        x_min, y_min, w, h = bbox
        
        x_center = (x_min + w / 2.0) / img_w
        y_center = (y_min + h / 2.0) / img_h
        norm_w = w / img_w
        norm_h = h / img_h

        # Generate YOLO label file path
        base_name = os.path.splitext(img['file_name'])[0]
        output_path = os.path.join(output_dir, f"{base_name}.txt")

        # Write annotation to file
        with open(output_path, 'a') as out_f:
            out_f.write(f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

    print(f"\nSuccessfully converted annotations. Labels saved to: {output_dir}")

# Execution example
if __name__ == "__main__":
    # Point this to the JSON file you generated in the previous step
    input_json = 'data/raw/traffic_subset_train.json'
    
    # Directory where the .txt files will be saved
    output_labels_dir = 'data/processed/labels/train'
    
    convert_coco_to_yolo(input_json, output_labels_dir)