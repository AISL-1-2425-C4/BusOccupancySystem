import os
import shutil
import random

# Config
DATASET_PATH = "path/dataset"

FROM_SPLIT = "train" 
TO_SPLIT = "valid"

IMAGE_SUBFOLDER = "images"
LABEL_SUBFOLDER = "labels"

NUM_TO_MOVE = 57

def move_files():
    from_images_dir = os.path.join(DATASET_PATH, FROM_SPLIT, IMAGE_SUBFOLDER)
    from_labels_dir = os.path.join(DATASET_PATH, FROM_SPLIT, LABEL_SUBFOLDER)

    to_images_dir = os.path.join(DATASET_PATH, TO_SPLIT, IMAGE_SUBFOLDER)
    to_labels_dir = os.path.join(DATASET_PATH, TO_SPLIT, LABEL_SUBFOLDER)

    os.makedirs(to_images_dir, exist_ok=True)
    os.makedirs(to_labels_dir, exist_ok=True)

    images = [f for f in os.listdir(from_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    selected_images = random.sample(images, NUM_TO_MOVE)

    moved_count = 0
    for img_file in selected_images:
        # Move image
        src_img_path = os.path.join(from_images_dir, img_file)
        dst_img_path = os.path.join(to_images_dir, img_file)
        shutil.move(src_img_path, dst_img_path)

        # Corresponding label file has same base filename but with .txt extension
        base_name, _ = os.path.splitext(img_file)
        label_file = base_name + ".txt"
        src_label_path = os.path.join(from_labels_dir, label_file)
        dst_label_path = os.path.join(to_labels_dir, label_file)

        if os.path.exists(src_label_path):
            shutil.move(src_label_path, dst_label_path)
        else:
            print(f"Label file not found for image: {img_file}")

        moved_count += 1

    print(f"\nMoved {moved_count} images and their labels from '{FROM_SPLIT}' to '{TO_SPLIT}'.")

if __name__ == "__main__":
    move_files()
