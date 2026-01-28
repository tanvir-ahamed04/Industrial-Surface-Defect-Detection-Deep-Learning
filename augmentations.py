# CutMix / Copy-Paste Augmentation (Detection)

import random
import numpy as np
def copy_paste_augmentation(image1, target1, image2, target2):
    """
    Paste objects from image2 into image1
    """
    img_h, img_w, _ = image1.shape

    new_boxes = []
    new_labels = []

    for box, label in zip(target2["boxes"], target2["labels"]):
        x1, y1, x2, y2 = map(int, box)
        patch = image2[y1:y2, x1:x2]

        if patch.size == 0:
            continue

        px = random.randint(0, img_w - patch.shape[1])
        py = random.randint(0, img_h - patch.shape[0])

        image1[py:py+patch.shape[0],
               px:px+patch.shape[1]] = patch

        new_boxes.append([
            px, py,
            px + patch.shape[1],
            py + patch.shape[0]
        ])
        new_labels.append(label)

    if len(new_boxes) > 0:
        target1["boxes"] = np.vstack(
            (target1["boxes"], np.array(new_boxes))
        )
        target1["labels"] = np.hstack(
            (target1["labels"], np.array(new_labels))
        )

    return image1, target1
