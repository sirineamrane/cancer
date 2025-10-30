import os
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from torchvision import transforms

# 1️⃣ Charger le dataset
def load_dataset(data_path):
    data = []
    for split in ["train", "test"]:
        split_path = os.path.join(data_path, split)
        if not os.path.exists(split_path):
            continue
        classes = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
        for c in classes:
            files = os.listdir(os.path.join(split_path, c))
            for f in files:
                data.append((os.path.join(split_path, c, f), c, split))
    df = pd.DataFrame(data, columns=["filepath", "label", "split"])
    return df

# 2️⃣ Vérifier le déséquilibre
def check_imbalance(df):
    counts = Counter(df["label"])
    print("Distribution des classes :", counts)

# 3️⃣ Découpage train/val/test stratifié
def split_dataset(df, val_size=0.1, random_state=42):
    train_df = df[df["split"]=="train"]
    train_set, val_set = train_test_split(
        train_df,
        test_size=val_size,
        stratify=train_df["label"],
        random_state=random_state
    )
    test_df = df[df["split"]=="test"]
    return train_set, val_set, test_df

# 4️⃣ Transformations (pré-traitement + augmentations + resizing)
def get_transforms(resize=(128,128)):
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
