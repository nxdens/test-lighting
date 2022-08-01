import hydra
from omegaconf import DictConfig, OmegaConf
from project.misc.utils import ROOT_DIR
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config_folder", help="name of the folder contained in /config")
args = parser.parse_args()


@hydra.main(
    version_base=None,
    config_path=os.path.join(ROOT_DIR, "config", args.config_folder),
    config_name="config",
)
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    my_app()
