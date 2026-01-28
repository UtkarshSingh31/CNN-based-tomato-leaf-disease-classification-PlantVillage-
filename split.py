from pathlib import Path
import random
import shutil

random.seed(42)

RAW_DIR=Path("data/raw/plantvillage/tomato")
OUT_DIR = Path("data/processed")


SPLITS = {
    "train": 0.7,
    "val": 0.2,
    "test": 0.1
}

for split in SPLITS:
    (OUT_DIR/split).mkdir(parents=True,exist_ok=True)


for class_dir in RAW_DIR.iterdir():
    if not class_dir.is_dir():
        continue

    images=[
        img for img in class_dir.iterdir()
        if img.suffix.lower() in [".jpg", ".png", ".jpeg"]
    ]


    n = len(images)
    n_train = int(SPLITS["train"] * n)
    n_val = int(SPLITS["val"] * n)

    split_map = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:]
    }

    for split, imgs in split_map.items():
        target_dir = OUT_DIR / split / class_dir.name
        target_dir.mkdir(parents=True, exist_ok=True)

        for img in imgs:
            shutil.copy(img, target_dir / img.name)

print(" Dataset split completed.")