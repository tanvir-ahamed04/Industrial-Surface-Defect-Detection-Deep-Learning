import os

# Paths
DATA_ROOT = "data/NEU-DET"
IMAGE_DIR = os.path.join(DATA_ROOT, "IMAGES")
ANNOTATION_DIR = os.path.join(DATA_ROOT, "ANNOTATIONS")

# Classes (NEU-DET)
CLASS_NAMES = [
    "background",
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]

CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

# Training
BATCH_SIZE = 4
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
IMAGE_SIZE = 512
DEVICE = "cuda"
