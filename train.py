import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print("=== Configuration complète ===")
    print(OmegaConf.to_yaml(cfg))

    # Exemple : affichage des paramètres
    print(f"\nModèle : {cfg.model.name}")
    print(f"Optimiseur : {cfg.optimizer.name}")
    print(f"Dataset : {cfg.dataset.name}")
    print(f"Époques : {cfg.training.epochs}")

if __name__ == "__main__":
    main()
