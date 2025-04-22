import hydra
from omegaconf import OmegaConf
from postgrad_class.conf.config import Config, LinearModelConfig, NeuralNetConfig
from sklearn.datasets import make_regression
from logging import getLogger
from hydra.core.config_store import ConfigStore

logger = getLogger(__name__)

X, y = make_regression()


# Register the configuration classes with Hydra
cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)
cs.store(name="base_neural", node=NeuralNetConfig, group="model")
cs.store(name="base_linear", node=LinearModelConfig, group="model")

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: Config):
    logger.info(OmegaConf.to_yaml(cfg))

    # Dynamically instantiate the model
    model = hydra.utils.instantiate(cfg.model)

    logger.info(f"Fitting model")
    model.fit(X, y)
    preds = model.predict(X)
    logger.info(f"Predictions: {preds}")
    return preds

if __name__ == "__main__":
    main()