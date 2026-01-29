import cv2
import numpy as np

def extract_leaf_mask_grabcut(rgb_img):
    img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    mask = np.zeros(img.shape[:2], np.uint8)

    h, w = img.shape[:2]
    rect = (10, 10, w - 20, h - 20)

    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)

    cv2.grabCut(img, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)

    return (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)

def compute_severity_topk(cam, rgb_img, k_percent):
    cam = cv2.resize(cam, (rgb_img.shape[1], rgb_img.shape[0]))
    leaf_mask = extract_leaf_mask_grabcut(rgb_img)

    cam_leaf = cam[leaf_mask]
    if cam_leaf.size == 0:
        return 0.0

    thresh = np.percentile(cam_leaf, 100 - k_percent)
    infected = (cam >= thresh) & leaf_mask

    return infected.sum() / leaf_mask.sum() * 100
