import os
import cv2

# ===== CONFIG =====
BASE_DIR = r"C:\Users\Vansh\Downloads\Mango Dataset\final_dataset"
IMG_SIZE = 224
# ==================

splits = ["train", "val", "test"]

for split in splits:

    split_path = os.path.join(BASE_DIR, split)

    print(f"\nProcessing {split} folder...")

    classes = os.listdir(split_path)

    for class_name in classes:

        class_path = os.path.join(split_path, class_name)

        if not os.path.isdir(class_path):
            continue

        print(f"  Resizing & renaming: {class_name}")

        images = os.listdir(class_path)

        count = 1

        for img_name in images:

            img_path = os.path.join(class_path, img_name)

            img = cv2.imread(img_path)

            if img is None:
                continue

            # Resize
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            # Rename format
            new_name = f"{class_name.replace(' ', '_')}_{str(count).zfill(4)}.jpg"

            save_path = os.path.join(class_path, new_name)

            cv2.imwrite(save_path, img)

            # Remove old file if name changed
            if img_name != new_name:
                os.remove(img_path)

            count += 1

        print(f"    Done. Total images: {count-1}")

print("\n All images resized and renamed successfully!")
