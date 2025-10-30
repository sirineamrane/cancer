import hydra
from omegaconf import DictConfig
from data import load_dataset, split_dataset, get_transforms
from collections import Counter
import pandas as pd

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):

    print("modele :", cfg.model.name)
    print("epoch :", cfg.training.epochs)

    # 1️⃣ Chargement du dataset
    df = load_dataset(cfg.dataset.path)

    # 2️⃣ Analyse initiale du dataset
    print("\n📊 === Analyse du dataset ===")
    print(f"Nombre total d’images : {len(df)}")

    # Distribution des classes
    label_counts = Counter(df["label"])
    print("\n🧬 Distribution des classes :")
    for label, count in label_counts.items():
        print(f"  - {label} : {count}")

    # Distribution par split
    split_counts = Counter(df["split"])
    print("\n📁 Répartition par split :")
    for split, count in split_counts.items():
        print(f"  - {split} : {count}")

    # Tableau split x label
    print("\n🔍 Détail par split et par classe :")
    table = pd.crosstab(df["split"], df["label"])
    print(table)

    # 3️⃣ Découpage train/val/test
    train_df, val_df, test_df = split_dataset(df, val_size=cfg.dataset.val_size)

    # Concaténation pour inclure val dans l'analyse
    df_split = pd.concat([
        train_df.assign(split="train"),
        val_df.assign(split="val"),
        test_df.assign(split="test")
    ])

    print("\n📂 Nombre d'exemples par split après découpage :")
    split_counts_post = Counter(df_split["split"])
    for split, count in split_counts_post.items():
        print(f"  - {split} : {count}")

    print("\n🔍 Détail par split et par classe après découpage :")
    table_post = pd.crosstab(df_split["split"], df_split["label"])
    print(table_post)

    # 4️⃣ Préparation des transformations
    transforms = get_transforms(resize=(cfg.training.resize, cfg.training.resize))
    print("\n🧩 Transformations appliquées :", transforms)


if __name__ == "__main__":
    main()
