import json
import os

def convert_coco_to_yolo(json_file, output_dir):
    with open(json_file, 'r') as f:
        data = json.load(f)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    categories = sorted(data['categories'], key=lambda x: x['id'])
    coco_id_to_yolo_id = {cat['id']: i for i, cat in enumerate(categories)}

    print("Class ID Mapping:")
    for cat in categories:
        print(f"  {cat['name']}: {coco_id_to_yolo_id[cat['id']]}")

    images_info = {img['id']: img for img in data['images']}

    for ann in data['annotations']:
        image_id = ann['image_id']
        category_id = ann['category_id']
        bbox = ann['bbox']

        img = images_info[image_id]
        img_w = img['width']
        img_h = img['height']

        yolo_class_id = coco_id_to_yolo_id[category_id]

        x_min, y_min, w, h = bbox
        
        x_center = (x_min + w / 2.0) / img_w
        y_center = (y_min + h / 2.0) / img_h
        norm_w = w / img_w
        norm_h = h / img_h

        base_name = os.path.splitext(img['file_name'])[0]
        output_path = os.path.join(output_dir, f"{base_name}.txt")

        with open(output_path, 'a') as out_f:
            out_f.write(f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

    print(f"\nSuccessfully converted annotations. Labels saved to: {output_dir}")

if __name__ == "__main__":
    input_json = 'data/raw/traffic_subset_train.json'
    
    output_labels_dir = 'data/processed/train/labels'
    
    convert_coco_to_yolo(input_json, output_labels_dir)