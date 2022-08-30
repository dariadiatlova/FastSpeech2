import os
import json
import hydra
import numpy as np
import torch.nn as nn

from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from model.lightning_model import FastSpeechLightning
from utils.model import get_dataloader


@hydra.main(version_base=None, config_path="config/EmoEnglish", config_name="train")
def train(cfg) -> None:
    seed_everything(cfg.seed)
    wandb_logger = WandbLogger(save_dir=cfg.wandb.save_dir, project=cfg.wandb.project, log_model=False,
                               offline=cfg.wandb.offline, config=cfg)

    callbacks = ModelCheckpoint(dirpath=str(wandb_logger.experiment.dir),
                                monitor="train_loss/total_loss",
                                save_top_k=-1,
                                every_n_train_steps=cfg.wandb.save_step,
                                save_last=cfg.wandb.save_last)

    progress_bar = TQDMProgressBar(refresh_rate=cfg.wandb.progress_bar_refresh_rate)

    trainer = Trainer(max_steps=cfg.wandb.total_step,
                      gradient_clip_algorithm="norm",
                      gradient_clip_val=cfg.model.optimizer.grad_clip_thresh,
                      check_val_every_n_epoch=cfg.wandb.val_step,
                      log_every_n_steps=cfg.wandb.log_every_n_steps,
                      logger=wandb_logger,
                      gpus=cfg.n_gpus,
                      callbacks=[callbacks, progress_bar],
                      strategy="ddp")

    train_loader = get_dataloader(config=cfg.preprocess, train=True)
    synthesis_loader = get_dataloader(config=cfg.preprocess, train=False)
    model = FastSpeechLightning(cfg)
    if cfg.checkpoint_path is not None:
        model = model.load_from_checkpoint(checkpoint_path=cfg.checkpoint_path, config=cfg, strict=False)
        if cfg.model.retrain_speaker_embed:
            with open(os.path.join(cfg.preprocess.path.preprocessed_path, "speakers.json"), "r") as f:
                speaker_dict = json.load(f)
                n_speaker = max(np.unique([*speaker_dict.values()]).astype(int)) + 1
                model.speaker_emb = nn.Embedding(n_speaker, cfg.model.transformer.encoder_hidden)
    trainer.fit(model, train_loader, synthesis_loader)


if __name__ == "__main__":
    train()
