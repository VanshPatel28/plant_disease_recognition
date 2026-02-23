import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TRAIN_DIR = r"C:\Users\Vansh\Downloads\Mango Dataset\final_dataset\train"
VAL_DIR   = r"C:\Users\Vansh\Downloads\Mango Dataset\final_dataset\val"
TEST_DIR  = r"C:\Users\Vansh\Downloads\Mango Dataset\final_dataset\test"

IMG_SIZE    = (224, 224)   # VGG16 expects 224x224
BATCH_SIZE  = 32
EPOCHS      = 50
LR          = 1e-4

CLASS_NAMES = [
    "Anthracnose",
    "Bacterial Canker",
    "Cutting Weevil",
    "Die Back",
    "Gall Midge",
    "Healthy",
    "Powdery Mildew",
    "Sooty Mould"
]
NUM_CLASSES = len(CLASS_NAMES)

# ─────────────────────────────────────────────
# DATA GENERATORS
# (Data is pre-augmented, so only rescale all splits)
# ─────────────────────────────────────────────
train_datagen = ImageDataGenerator(rescale=1./255)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_gen = val_test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_gen = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# ─────────────────────────────────────────────
# BUILD VGG16 MODEL
# ─────────────────────────────────────────────
base_model = VGG16(
    weights='imagenet',
    include_top=False,          # remove original FC layers
    input_shape=(224, 224, 3)
)

# Freeze all base layers initially
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification head
x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=LR),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ─────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
    ModelCheckpoint('vgg16_mango_best.keras', monitor='val_accuracy', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)
]

# ─────────────────────────────────────────────
# PHASE 1: Train only the custom head
# ─────────────────────────────────────────────
print("\n=== PHASE 1: Training custom head (base frozen) ===")
history1 = model.fit(
    train_gen,
    epochs=15,
    validation_data=val_gen,
    callbacks=callbacks
)

# ─────────────────────────────────────────────
# PHASE 2: Fine-tune last 4 conv layers of VGG16
# ─────────────────────────────────────────────
print("\n=== PHASE 2: Fine-tuning last conv block ===")

# Unfreeze the last conv block (block5: layers -4 onwards)
for layer in base_model.layers[-4:]:
    layer.trainable = True

# Use a lower learning rate for fine-tuning
model.compile(
    optimizer=Adam(learning_rate=LR / 10),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history2 = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks
)

# ================= EARLY STOPPING =================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# ─────────────────────────────────────────────
# EVALUATION ON TEST SET
# ─────────────────────────────────────────────
print("\n=== TEST SET EVALUATION ===")
test_loss, test_acc = model.evaluate(test_gen)
print(f"Test Accuracy : {test_acc * 100:.2f}%")
print(f"Test Loss     : {test_loss:.4f}")

# Classification Report
test_gen.reset()
y_pred = model.predict(test_gen, verbose=1)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_gen.classes

print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=CLASS_NAMES))

# ─────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────
def merge_histories(h1, h2):
    merged = {}
    for key in h1.history:
        merged[key] = h1.history[key] + h2.history[key]
    return merged

history = merge_histories(history1, history2)

# Accuracy plot
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Train Accuracy')
plt.plot(history['val_accuracy'], label='Val Accuracy')
plt.axvline(x=len(history1.history['accuracy'])-1, color='gray', linestyle='--', label='Fine-tune start')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.axvline(x=len(history1.history['loss'])-1, color='gray', linestyle='--', label='Fine-tune start')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('vgg16_training_curves.png', dpi=150)
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Confusion Matrix - VGG16')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('vgg16_confusion_matrix.png', dpi=150)
plt.show()

print("\nModel saved as: vgg16_mango_best.keras")
print("Plots saved as: vgg16_training_curves.png, vgg16_confusion_matrix.png")
