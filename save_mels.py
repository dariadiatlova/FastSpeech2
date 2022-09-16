import hydra
import glob
import os

from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from model.lightning_model import FastSpeechLightning
from utils.model import get_dataloader


@hydra.main(version_base=None, config_path="config/SingleESD", config_name="train")
def generate_mels(cfg) -> None:
    seed_everything(cfg.seed)

    loader = get_dataloader(config=cfg.preprocess, train=False)
    model = FastSpeechLightning(cfg)
    assert cfg.checkpoint_path is not None, "Model supposed to be trained!"

    post_processed_path = cfg.postprocessing.output_mel_dir
    vocoder_json_path = cfg.postprocessing.vocoder_config_path
    os.makedirs(post_processed_path, exist_ok=True)

    device = cfg.postprocessing.device
    model = model.load_from_checkpoint(checkpoint_path=cfg.checkpoint_path, config=cfg, strict=True).to(device)
    model.generate_mels(loader, post_processed_path, vocoder_json_path)
    all_mel_paths = glob.glob(f"{post_processed_path}/*.pt")
    print(f"{len(all_mel_paths)} mels were written to {post_processed_path} directory!")


if __name__ == "__main__":
    generate_mels()
