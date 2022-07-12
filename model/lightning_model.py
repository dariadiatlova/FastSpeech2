import os
from typing import Dict

import torch
import torchaudio
import wandb
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import ExponentialLR

from model import FastSpeech2, FastSpeech2Loss
from utils.tools import synthesize_predicted_wav, torch_from_numpy


class FastSpeechLightning(LightningModule):
    def __init__(self, config: Dict):
        super().__init__()
        self.save_hyperparameters()
        os.makedirs(config.preprocess.path.raw_path, exist_ok=True)
        os.makedirs(config.preprocess.path.preprocessed_path, exist_ok=True)
        train_log_path = os.path.join(config.preprocess.path.log_path, "train")
        val_log_path = os.path.join(config.preprocess.path.log_path, "val")
        os.makedirs(train_log_path, exist_ok=True)
        os.makedirs(val_log_path, exist_ok=True)

        self.ground_truth_audio_path = config.preprocess.path.raw_path
        self.sampling_rate = config.preprocess.audio.sampling_rate
        self.preprocess_config = config.preprocess
        self.model_config = config.model
        self.model = FastSpeech2(preprocess_config=config.preprocess, model_config=config.model)
        self.loss = FastSpeech2Loss(self.preprocess_config)
        self.vocoder = torch.jit.load(config.vocoder_path, map_location=config.preprocess.device)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     betas=self.model_config["optimizer"]["betas"],
                                     eps=self.model_config["optimizer"]["eps"],
                                     weight_decay=self.model_config["optimizer"]["weight_decay"])
        scheduler = ExponentialLR(optimizer, gamma=self.model_config["optimizer"]["weight_decay"])
        return [optimizer], [scheduler]

    def _shared_step(self, input, output):
        total_loss, mel_loss, postnet_mel_loss, pitch_loss, energy_loss, duration_loss = self.loss(self.device, input, output)
        gen_log_dict = {f"train_loss/total_loss": total_loss,
                        f"train_loss/mel_loss": mel_loss,
                        f"train_loss/postnet_mel_loss": postnet_mel_loss,
                        f"train_loss/pitch_loss": pitch_loss,
                        f"train_loss/energy_loss": energy_loss,
                        f"train_loss/duration_loss": duration_loss}

        self.log_dict(gen_log_dict, on_step=True, on_epoch=False)
        return total_loss

    def forward(self, basenames, speakers, texts, text_lens, max_text_lens):
        predictions = self.model(self.device,
                                 speakers=speakers, texts=texts, src_lens=text_lens, max_src_len=max_text_lens)
        synthesized_wav = synthesize_predicted_wav(0, predictions, self.vocoder)
        return synthesized_wav

    def training_step(self, batch, batch_idx):
        batch = torch_from_numpy(batch[0])
        speakers, texts, text_lens, max_src_len, mels, mel_lens, max_mel_len, p_targets, e_targets, d_targets = batch[2:]
        batch_output = self.model(self.device, speakers, texts, text_lens, max_src_len, mels,
                                  mel_lens, max_mel_len, p_targets, e_targets, d_targets)
        return self._shared_step(batch, batch_output)

    def validation_step(self, batch, batch_idx):
        batch = torch_from_numpy(batch[0])
        speakers, texts, text_lens, max_src_len = batch[2:6]
        basenames = batch[0]
        predictions = self.model(device=self.device,
                                 speakers=speakers.to(self.device),
                                 texts=texts.to(self.device),
                                 src_lens=text_lens.to(self.device),
                                 max_src_len=max_src_len)

        for i, tag in zip(range(len(basenames)), basenames):
            synthesized_wav = synthesize_predicted_wav(i, predictions, self.vocoder)
            ground_truth_audio_path = f"{self.ground_truth_audio_path}/{tag}.wav"
            ground_truth_wav, sr = torchaudio.load(ground_truth_audio_path)
            ground_truth_wav = ground_truth_wav.squeeze(0)

            self.logger.experiment.log(
                {"Validation_audio/predicted": wandb.Audio(
                    synthesized_wav, caption=f"Generated_{tag}", sample_rate=self.sampling_rate)}
            )

            # log original audios only ones
            if self.global_step == 0:

                self.logger.experiment.log(
                    {"Validation_audio/original": wandb.Audio(
                        ground_truth_wav, caption=f"Original_{tag}", sample_rate=self.sampling_rate)}
                )
