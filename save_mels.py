import hydra
import glob
import os

from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from model.lightning_model import FastSpeechLightning
from utils.model import get_dataloader


@hydra.main(version_base=None, config_path="config/EmoEnglish", config_name="train")
def generate_mels(cfg) -> None:
    seed_everything(cfg.seed)

    train_loader = get_dataloader(config=cfg.preprocess, train=True)
    val_loader = get_dataloader(config=cfg.preprocess, train=False)
    model = FastSpeechLightning(cfg)
    assert cfg.checkpoint_path is not None, "Model supposed to be trained!"

    post_processed_path = cfg.postprocessing.output_mel_dir
    vocoder_json_dirpath = cfg.postprocessing.vocoder_json_dirpath
    os.makedirs(post_processed_path, exist_ok=True)
    os.makedirs(vocoder_json_dirpath, exist_ok=True)

    device = cfg.postprocessing.device
    model = model.load_from_checkpoint(checkpoint_path=cfg.checkpoint_path, config=cfg, strict=True).to(device)
    model.generate_mels(train_loader, post_processed_path, f"{vocoder_json_dirpath}/train_dataset.json")
    model.generate_mels(val_loader, post_processed_path, f"{vocoder_json_dirpath}/val_dataset.json")

    all_mel_paths = glob.glob(f"{post_processed_path}/*.pt")
    print(f"{len(all_mel_paths)} mels were written to {post_processed_path} directory!")


if __name__ == "__main__":
    generate_mels()
