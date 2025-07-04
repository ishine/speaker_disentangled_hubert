import fire
from omegaconf import OmegaConf

from src.bigvgan.train import train_bigvgan
from src.flow_matching.data import extract_features, resample, tokenize_dataset
from src.flow_matching.train import train_flow_matching


class TaskRunner:
    def resample(self, config: str = "configs/unit2speech/default.yaml"):
        config = OmegaConf.load(config)
        resample(config)

    def extract_features(self, config: str = "configs/unit2speech/default.yaml"):
        config = OmegaConf.load(config)
        extract_features(config)

    def tokenize_dataset(self, config: str = "configs/unit2speech/default.yaml"):
        config = OmegaConf.load(config)
        tokenize_dataset(config)

    def train_bigvgan(self, config: str = "configs/unit2speech/default.yaml"):
        config = OmegaConf.load(config)
        train_bigvgan(config)

    def train_flow_matching(self, config: str = "configs/unit2speech/default.yaml"):
        config = OmegaConf.load(config)
        train_flow_matching(config)


if __name__ == "__main__":
    fire.Fire(TaskRunner)
