import os
import cv2
import shutil
import random
from tqdm import tqdm

# ====== CONFIG ======
ORIGINAL_DIR = "Mango Dataset/process data"
AUGMENTED_DIR = "Mango Dataset/Augmented Image"
OUTPUT_DIR = "final_dataset"

IMG_SIZE = 224   # Resize size (224x224 recommended for CNN)
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
# ====================

# Create output folders
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

# Get class names
classes = os.listdir(ORIGINAL_DIR)

for class_name in classes:
    print(f"\nProcessing {class_name}...")

    original_path = os.path.join(ORIGINAL_DIR, class_name)
    augmented_path = os.path.join(AUGMENTED_DIR, class_name)

    # Merge original + augmented images
    all_images = []

    for folder in [original_path, augmented_path]:
        if os.path.exists(folder):
            for img in os.listdir(folder):
                all_images.append(os.path.join(folder, img))

    random.shuffle(all_images)

    total_images = len(all_images)

    train_end = int(TRAIN_SPLIT * total_images)
    val_end = train_end + int(VAL_SPLIT * total_images)

    splits = {
        "train": all_images[:train_end],
        "val": all_images[train_end:val_end],
        "test": all_images[val_end:]
    }

    for split_name, images in splits.items():

        split_class_path = os.path.join(OUTPUT_DIR, split_name, class_name)
        os.makedirs(split_class_path, exist_ok=True)

        for idx, img_path in enumerate(tqdm(images)):

            img = cv2.imread(img_path)

            if img is None:
                continue

            # Resize image
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            # Rename image
            new_name = f"{class_name}_{idx}.jpg"

            save_path = os.path.join(split_class_path, new_name)

            cv2.imwrite(save_path, img)

print("\n Dataset processing completed successfully!")
