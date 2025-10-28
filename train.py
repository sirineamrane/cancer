import os
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print("=== 🚀 TEST HYDRA - CANCER CLASSIFICATION ===\n")

    # 1️⃣ Afficher toute la configuration actuelle
    print("📘 Configuration complète :")
    print(OmegaConf.to_yaml(cfg))

    # 2️⃣ Accéder aux paramètres principaux
    print("\n--- Paramètres principaux ---")
    print(f"🧠 Modèle        : {cfg.model.name}")
    print(f"⚙️  Optimiseur    : {cfg.optimizer.name}")
    print(f"📊 Dataset       : {cfg.dataset.name}")
    print(f"📁 Chemin dataset: {cfg.dataset.path}")
    print(f"🔁 Époques       : {cfg.training.epochs}")
    print(f"📦 Batch size    : {cfg.training.batch_size}")

    # 3️⃣ Vérifier que le dataset est accessible
    data_path = cfg.dataset.path
    if os.path.exists(data_path):
        files = os.listdir(data_path)
        print(f"\n✅ Dataset accessible ({len(files)} fichiers trouvés)")
        print("Exemple de contenu :", files[:5])
    else:
        print("\n❌ ERREUR : le chemin du dataset n'existe pas !")

    # 4️⃣ Exemple d’utilisation des hyperparamètres
    print(f"\n🔧 Entraînement simulé avec lr={cfg.optimizer.lr} pendant {cfg.training.epochs} époques...")
    print("➡️  (ici tu pourrais appeler ta vraie fonction d'entraînement PyTorch)")

    print("\n✅ Test Hydra terminé avec succès !")


if __name__ == "__main__":
    main()

