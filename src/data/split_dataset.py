import os
import random
import shutil

def split_dataset(base_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    # Corrected source directories based on README structure
    img_train_dir = os.path.join(base_dir, 'train', 'images')
    lbl_train_dir = os.path.join(base_dir, 'train', 'labels')

    # Define target directories for val and test
    dirs_to_make = [
        os.path.join(base_dir, 'val', 'images'),
        os.path.join(base_dir, 'test', 'images'),
        os.path.join(base_dir, 'val', 'labels'),
        os.path.join(base_dir, 'test', 'labels')
    ]
    
    for d in dirs_to_make:
        os.makedirs(d, exist_ok=True)

    files = [f[:-4] for f in os.listdir(lbl_train_dir) if f.endswith('.txt')]
    
    random.seed(42)
    random.shuffle(files)

    total_files = len(files)
    train_end = int(total_files * train_ratio)
    val_end = train_end + int(total_files * val_ratio)

    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]

    def move_files(file_list, split_name):
        for f in file_list:
            src_img = os.path.join(img_train_dir, f + '.jpg')
            dst_img = os.path.join(base_dir, split_name, 'images', f + '.jpg')
            if os.path.exists(src_img):
                shutil.move(src_img, dst_img)

            src_lbl = os.path.join(lbl_train_dir, f + '.txt')
            dst_lbl = os.path.join(base_dir, split_name, 'labels', f + '.txt')
            if os.path.exists(src_lbl):
                shutil.move(src_lbl, dst_lbl)

    move_files(val_files, 'val')
    move_files(test_files, 'test')

    print("Dataset splitting completed successfully.")
    print(f"Total files : {total_files}")
    print(f"Train set   : {len(train_files)} files")
    print(f"Val set     : {len(val_files)} files")
    print(f"Test set    : {len(test_files)} files")

if __name__ == "__main__":
    dataset_root = r'data/processed'
    split_dataset(dataset_root)