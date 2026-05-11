import os
import random
import shutil

def split_dataset(base_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    # Define current train directories
    img_train_dir = os.path.join(base_dir, 'images/train')
    lbl_train_dir = os.path.join(base_dir, 'labels/train')

    # Define target directories for val and test
    dirs_to_make = [
        os.path.join(base_dir, 'images/val'),
        os.path.join(base_dir, 'images/test'),
        os.path.join(base_dir, 'labels/val'),
        os.path.join(base_dir, 'labels/test')
    ]
    
    # Create directories if they do not exist
    for d in dirs_to_make:
        os.makedirs(d, exist_ok=True)

    # Get all label files (excluding extensions)
    files = [f[:-4] for f in os.listdir(lbl_train_dir) if f.endswith('.txt')]
    
    # Randomly shuffle the data to ensure even distribution
    random.seed(42)
    random.shuffle(files)

    total_files = len(files)
    train_end = int(total_files * train_ratio)
    val_end = train_end + int(total_files * val_ratio)

    # Split the list of files
    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]

    def move_files(file_list, split_name):
        for f in file_list:
            # Move images
            src_img = os.path.join(img_train_dir, f + '.jpg')
            dst_img = os.path.join(base_dir, f'images/{split_name}', f + '.jpg')
            if os.path.exists(src_img):
                shutil.move(src_img, dst_img)

            # Move labels
            src_lbl = os.path.join(lbl_train_dir, f + '.txt')
            dst_lbl = os.path.join(base_dir, f'labels/{split_name}', f + '.txt')
            if os.path.exists(src_lbl):
                shutil.move(src_lbl, dst_lbl)

    # Move val and test files to their respective folders
    # Train files remain in the original 'train' folder
    move_files(val_files, 'val')
    move_files(test_files, 'test')

    print("Dataset splitting completed successfully.")
    print(f"Total files : {total_files}")
    print(f"Train set   : {len(train_files)} files")
    print(f"Val set     : {len(val_files)} files")
    print(f"Test set    : {len(test_files)} files")

# Execution example
if __name__ == "__main__":
    # Point this to the root directory containing 'images' and 'labels'
    dataset_root = 'data/processed'
    split_dataset(dataset_root)