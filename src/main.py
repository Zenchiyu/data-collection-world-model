import hydra

from omegaconf import DictConfig, OmegaConf
from collector import Collector
from utils import skip_if_run_is_over


OmegaConf.register_new_resolver("eval", eval)  # to evaluate expressions


@hydra.main(version_base="1.3.2", config_path="../config", config_name="collector")
def main(cfg: DictConfig):
    run(cfg)

@skip_if_run_is_over
def run(cfg):
    Collector(cfg).run()


if __name__ == "__main__":
    main()