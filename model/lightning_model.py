from typing import Dict

import numpy as np
import os
import torch
import torchaudio
import wandb
from pytorch_lightning import LightningModule

from model import FastSpeech2, FastSpeech2Loss
from utils.tools import synthesize_predicted_wav, torch_from_numpy, synthesize_from_gt_mel, pad_or_trim_mel, write_json


class FastSpeechLightning(LightningModule):
    def __init__(self, config: Dict):
        super().__init__()
        self.save_hyperparameters()
        self.ground_truth_audio_path = config.preprocess.path.raw_path
        self.sampling_rate = config.preprocess.audio.sampling_rate
        self.preprocess_config = config.preprocess
        self.model_config = config.model
        self.model = FastSpeech2(preprocess_config=config.preprocess, model_config=config.model)
        self.loss = FastSpeech2Loss(self.preprocess_config)
        self.vocoder = torch.jit.load(config.vocoder_path, map_location="cpu")
        self.init_lr = np.power(config.model.transformer["encoder_hidden"], -0.5)
        self.n_warmup_steps = config.model.optimizer["warm_up_step"]
        self.anneal_steps = config.model.optimizer["anneal_steps"]
        self.anneal_rate = config.model.optimizer["anneal_rate"]

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=1,
                                          betas=self.model_config["optimizer"]["betas"],
                                          eps=self.model_config["optimizer"]["eps"],
                                          weight_decay=self.model_config["optimizer"]["weight_decay"])
        scheduler = {"scheduler": self._scheduler(self.optimizer),
                     "interval": 'step',
                     "frequency": 1}
        return [self.optimizer], [scheduler]

    def _scheduler(self, optimizer):
        def lr_lambda(current_step: int):
            current_step += 1
            lr = np.min([np.power(current_step, -0.5), np.power(self.n_warmup_steps, -1.5) * current_step])
            for s in self.anneal_steps:
                if current_step > s:
                    lr = lr * self.anneal_rate
            return self.init_lr * lr
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def _shared_step(self, input, output):
        total_loss, mel_loss, postnet_mel_loss, pitch_loss, energy_loss, duration_loss, lpips_loss = self.loss(
            self.device, input, output)
        gen_log_dict = {f"train_loss/total_loss": total_loss,
                        f"train_loss/mel_loss": mel_loss,
                        f"train_loss/postnet_mel_loss": postnet_mel_loss,
                        f"train_loss/pitch_loss": pitch_loss,
                        f"train_loss/energy_loss": energy_loss,
                        f"train_loss/duration_loss": duration_loss,
                        f"train_loss/lpips_loss": lpips_loss,
                        f"optimizer_rate/optimizer": self.optimizer.param_groups[0]['lr']}
        self.log_dict(gen_log_dict, on_step=True, on_epoch=False)
        return total_loss

    def generate_mels(self, loader, post_processed_path, vocoder_json_path):
        for batch in loader:
            batch = torch_from_numpy(batch[0])
            ids, _, speakers, emotions, texts, text_lens, max_src_lens, _, mel_lens = batch[:9]
            batch_output = self.model(self.device, speakers=speakers, emotions=emotions, texts=texts,
                                      src_lens=text_lens, max_src_len=max_src_lens)
            postnet_mel_predictions = batch_output[1]
            mel_lens = batch[8]
            for i, mel, in enumerate(postnet_mel_predictions):
                predicted_mel = pad_or_trim_mel(mel, mel_lens[i])
                wav_path = f"{self.ground_truth_audio_path}/{ids[i]}.wav"
                mel_path = f"{post_processed_path}/{ids[i]}.pt"
                if os.path.exists(wav_path):
                    dict_to_write = {"audio_filepath": wav_path, "mel_filepath": mel_path, "duration": None}
                    write_json(dict_to_write, vocoder_json_path)
                    torch.save(predicted_mel, mel_path)
                else:
                    print(f"Ooops {wav_path} not found :(")

    def forward(self, speakers: torch.Tensor, emotions: torch.Tensor, texts: torch.Tensor, src_lens: torch.Tensor):
        """
        return: output, postnet_output, p_predictions, e_predictions, log_d_predictions, d_rounded, \
        src_masks, mel_masks, src_lens, mel_lens
        """
        return self.model(device=self.device,
                          emotions=emotions.to(self.device),
                          speakers=speakers.to(self.device),
                          texts=texts.to(self.device),
                          src_lens=src_lens.to(self.device),
                          max_src_len=max(src_lens))

    def training_step(self, batch, batch_idx):
        batch = torch_from_numpy(batch[0])
        speakers, emotions, texts, text_lens, max_src_len, mels, mel_lens, max_mel_len, \
        p_targets, e_targets, d_targets = batch[2:]
        batch_output = self.model(self.device, speakers, emotions, texts, text_lens, max_src_len, mels,
                                  mel_lens, max_mel_len, p_targets, e_targets, d_targets)
        return self._shared_step(batch, batch_output)

    def validation_step(self, batch, batch_idx):
        batch = torch_from_numpy(batch[0])
        basenames = batch[0]
        speakers, emotions, texts, text_lens, max_src_len = batch[2:7]
        gt_mel, gt_mel_lens, gt_max_mel_len = batch[7:10]
        gt_durations = batch[11]
        batch_size = len(speakers)
        with torch.no_grad():
            predictions = self.forward(speakers, emotions, texts, text_lens)
            pitch_loss, energy_loss, duration_loss = self.loss(self.device, batch, predictions, compute_mel_loss=False)

            validation_predictions = self.model(device=self.device,
                                                speakers=speakers.to(self.device),
                                                emotions=emotions.to(self.device),
                                                texts=texts.to(self.device),
                                                src_lens=text_lens.to(self.device),
                                                max_src_len=max_src_len,
                                                mel_lens=gt_mel_lens.to(self.device),
                                                max_mel_len=gt_max_mel_len,
                                                d_targets=gt_durations.to(self.device))

            loss = self.loss(self.device, batch, validation_predictions)
            total_loss, mel_loss, postnet_mel_loss = loss[:3]
            lpips_loss = loss[-1]

            gen_log_dict = {f"val_loss/total_loss": total_loss,
                            f"val_loss/mel_loss": mel_loss,
                            f"val_loss/postnet_mel_loss": postnet_mel_loss,
                            f"val_loss/pitch_loss": pitch_loss,
                            f"val_loss/energy_loss": energy_loss,
                            f"val_loss/duration_loss": duration_loss,
                            f"val_loss/lpips_loss": lpips_loss}

            self.log_dict(gen_log_dict, on_step=True, on_epoch=False, batch_size=batch_size)

            for i, tag in enumerate(basenames):
                speaker = speakers[i]
                synthesized_wav = synthesize_predicted_wav(i, predictions, self.vocoder)

                self.logger.experiment.log(
                    {f"{speaker}/generated_{tag}": wandb.Audio(
                        synthesized_wav, caption=f"generated_{tag}",
                        sample_rate=self.sampling_rate)}
                )

                if self.global_step == 0:
                    # log original audios only ones
                    vocoder_synthesized_from_gt = synthesize_from_gt_mel(gt_mel[i, :gt_mel_lens[i]], self.vocoder)

                    ground_truth_audio_path = f"{self.ground_truth_audio_path}/{tag}.wav"
                    ground_truth_wav, sr = torchaudio.load(ground_truth_audio_path)
                    ground_truth_wav = ground_truth_wav.squeeze(0)

                    self.logger.experiment.log(
                        {f"{speaker}/original_{tag}": wandb.Audio(
                            ground_truth_wav, caption=f"original_{tag}",
                            sample_rate=self.sampling_rate)}
                    )

                    self.logger.experiment.log(
                        {f"{speaker}/reconstructed_{tag}": wandb.Audio(
                            vocoder_synthesized_from_gt, caption=f"reconstructed_{tag}",
                            sample_rate=self.sampling_rate)}
                    )

            return total_loss
