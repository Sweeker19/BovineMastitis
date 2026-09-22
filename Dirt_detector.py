"""
Shed Dirtiness Scorer - Absolute Color/Texture Based (v3)
--------------------------------------------------------------
Ground truth (Clean_shed.png) is used ONCE to learn what "clean" floor color
looks like in this shed. Every image (including ground truth) then gets its
own absolute dirtiness score. "Dirtiness change" = test_score - baseline_score.
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO

ANIMAL_CLASS_IDS = {14, 15, 16, 17, 18, 19, 20, 21, 22, 23}
CONF_THRESHOLD = 0.10
DILATE_PX = 15

COLOR_DIST_THRESHOLD = 18
TEXTURE_THRESHOLD = 12

IMG_DIR = r"D:\College Work\Sem 5\SIH\Images"
GROUND_TRUTH = "Clean_shed.png"
TEST_IMAGES = ["Clean_cow.png", "Dirt_cow.png", "Dirt_shed.png"]

model = YOLO("yolov8n-seg.pt")


def get_animal_mask(image_path, target_shape):
    results = model.predict(image_path, conf=CONF_THRESHOLD, verbose=False)
    h, w = target_shape
    mask_total = np.zeros((h, w), dtype=np.uint8)
    r = results[0]
    if r.masks is None:
        return mask_total
    for cls_id, seg_mask in zip(r.boxes.cls.cpu().numpy(), r.masks.data.cpu().numpy()):
        if int(cls_id) in ANIMAL_CLASS_IDS:
            m = cv2.resize(seg_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_total = np.maximum(mask_total, (m > 0.5).astype(np.uint8) * 255)
    kernel = np.ones((DILATE_PX, DILATE_PX), np.uint8)
    return cv2.dilate(mask_total, kernel, iterations=1) > 0


def learn_clean_reference(gt_img, ignore_mask):
    lab = cv2.cvtColor(gt_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    valid = ~ignore_mask
    a_mean = lab[:, :, 1][valid].mean()
    b_mean = lab[:, :, 2][valid].mean()
    return a_mean, b_mean


def dirtiness_score(img, ignore_mask, ref_a, ref_b):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    a, b = lab[:, :, 1], lab[:, :, 2]
    color_dist = np.sqrt((a - ref_a) ** 2 + (b - ref_b) ** 2)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    local_mean = cv2.blur(gray, (9, 9))
    local_sqmean = cv2.blur(gray ** 2, (9, 9))
    local_std = np.sqrt(np.clip(local_sqmean - local_mean ** 2, 0, None))

    dirty_pixels = (color_dist > COLOR_DIST_THRESHOLD) | (local_std > TEXTURE_THRESHOLD)

    valid = ~ignore_mask
    total_valid = np.count_nonzero(valid)
    if total_valid == 0:
        return 0.0, dirty_pixels

    dirty_pct = 100.0 * np.count_nonzero(dirty_pixels & valid) / total_valid
    return dirty_pct, dirty_pixels & valid


def save_debug_visual(dirty_map, out_path):
    vis = (dirty_map.astype(np.uint8)) * 255
    cv2.imwrite(out_path, vis)


if __name__ == "__main__":
    gt_path = os.path.join(IMG_DIR, GROUND_TRUTH)
    gt_img = cv2.imread(gt_path)
    if gt_img is None:
        raise FileNotFoundError(f"Could not read ground truth: {gt_path}")

    gt_ignore_mask = get_animal_mask(gt_path, gt_img.shape[:2])
    ref_a, ref_b = learn_clean_reference(gt_img, gt_ignore_mask)
    baseline_score, _ = dirtiness_score(gt_img, gt_ignore_mask, ref_a, ref_b)

    for test_img_name in TEST_IMAGES:
        test_path = os.path.join(IMG_DIR, test_img_name)
        img = cv2.imread(test_path)
        if img is None:
            print(f"Could not read {test_path}, skipping.")
            continue

        img = cv2.resize(img, (gt_img.shape[1], gt_img.shape[0]))
        ignore_mask = get_animal_mask(test_path, img.shape[:2])

        score, dirty_map = dirtiness_score(img, ignore_mask, ref_a, ref_b)
        change = round(score - baseline_score, 2)

        prefix = os.path.splitext(test_img_name)[0] + "_debug"
        save_debug_visual(dirty_map, f"{prefix}_dirtmap.png")

        animal_pct = round(100.0 * np.count_nonzero(ignore_mask) / (img.shape[0] * img.shape[1]), 2)

        print(f"{test_img_name}: Dirtiness change = {change}% | Animal coverage = {animal_pct}%")
