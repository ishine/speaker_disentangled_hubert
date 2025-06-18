import fire
from omegaconf import OmegaConf

from src.speechlm.data import tokenize_slm21, tokenize_trainset
from src.speechlm.eval import evaluate
from src.speechlm.train import train


class TaskRunner:
    def tokenize_trainset(self, config: str = "configs/speechlm/default.yaml", spkids: str = "123456789"):
        config = OmegaConf.load(config)
        tokenize_trainset(config, spkids)

    def tokenize_slm21(self, config: str = "configs/speechlm/default.yaml"):
        config = OmegaConf.load(config)
        tokenize_slm21(config)

    def train(self, config: str = "configs/speechlm/default.yaml"):
        config = OmegaConf.load(config)
        train(config)

    def eval(self, config: str = "configs/speechlm/default.yaml"):
        config = OmegaConf.load(config)
        evaluate(config)

    def __call__(self, config: str = "configs/speechlm/default.yaml", spkids: str = "123456789"):
        config = OmegaConf.load(config)
        tokenize_trainset(config, spkids)
        tokenize_slm21(config)
        train(config)


if __name__ == "__main__":
    fire.Fire(TaskRunner)
