import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, hinge_loss
from sklearn.metrics import classification_report, confusion_matrix

# ================= CONFIG =================
BASE_PATH = r"C:\Users\Vansh\Downloads\Mango Dataset\final_dataset"
IMG_SIZE = 224
MAX_EPOCHS = 50
PATIENCE = 3
# ==========================================

X_train = []
y_train = []
X_val = []
y_val = []

print("Loading training and validation images...")

# -------- LOAD TRAIN DATA --------
train_path = os.path.join(BASE_PATH, "train")

for class_name in os.listdir(train_path):
    class_path = os.path.join(train_path, class_name)

    if not os.path.isdir(class_path):
        continue

    for img_name in os.listdir(class_path):

        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(class_path, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img / 255.0
        img = img.flatten()

        X_train.append(img)
        y_train.append(class_name)

# -------- LOAD VALIDATION DATA --------
val_path = os.path.join(BASE_PATH, "val")

for class_name in os.listdir(val_path):
    class_path = os.path.join(val_path, class_name)

    if not os.path.isdir(class_path):
        continue

    for img_name in os.listdir(class_path):

        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(class_path, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img / 255.0
        img = img.flatten()

        X_val.append(img)
        y_val.append(class_name)

# Convert to numpy
X_train = np.array(X_train)
X_val = np.array(X_val)

print("Train shape:", X_train.shape)
print("Validation shape:", X_val.shape)

# Encode labels
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_val = le.transform(y_val)

# ================= SVM MODEL =================
svm = SGDClassifier(loss="hinge", warm_start=True, random_state=42)

train_loss_list = []
val_loss_list = []
train_acc_list = []
val_acc_list = []

best_val_loss = float("inf")
early_stop_counter = 0

print("\nTraining SVM...")

for epoch in range(MAX_EPOCHS):

    svm.max_iter = 1
    svm.fit(X_train, y_train)

    train_pred = svm.predict(X_train)
    val_pred = svm.predict(X_val)

    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)

    train_loss = hinge_loss(y_train, svm.decision_function(X_train))
    val_loss = hinge_loss(y_val, svm.decision_function(X_val))

    train_loss_list.append(train_loss)
    val_loss_list.append(val_loss)
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)

    print(f"Epoch {epoch+1}: "
          f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
          f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
    else:
        early_stop_counter += 1

    if early_stop_counter >= PATIENCE:
        print("Early stopping triggered!")
        break

print("\nTraining Complete.")

# ================= CLASSIFICATION REPORT =================
final_val_pred = svm.predict(X_val)

print("\nClassification Report:\n")
print(classification_report(y_val, final_val_pred,
                            target_names=le.classes_))

# ================= CONFUSION MATRIX =================
cm = confusion_matrix(y_val, final_val_pred)

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.show()

# ================= LOSS GRAPH =================
plt.figure()
plt.plot(train_loss_list)
plt.plot(val_loss_list)
plt.title("Train vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Train Loss", "Validation Loss"])
plt.show()

# ================= ACCURACY GRAPH =================
plt.figure()
plt.plot(train_acc_list)
plt.plot(val_acc_list)
plt.title("Train vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train Accuracy", "Validation Accuracy"])
plt.show()
