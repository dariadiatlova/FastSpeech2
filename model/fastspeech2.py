import json
import os

import torch.nn as nn
import numpy as np
import torch
from transformer import Encoder, Decoder, PostNet
from utils.tools import get_mask_from_lengths
from .modules import VarianceAdaptor


class FastSpeech2(nn.Module):
    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config
        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(model_config["transformer"]["decoder_hidden"],
                                    preprocess_config["mel"]["n_mel_channels"])
        self.postnet = PostNet()
        self.speaker_emb = None

        if model_config["multi_speaker"]:
            self.speaker_emb = nn.Embedding(model_config["old_speaker_count"],
                                            model_config["transformer"]["encoder_hidden"])

        with open(os.path.join(preprocess_config["path"]["esd_vctk_speaker_mapping_path"])) as f:
            self.speakers_dict = json.load(f)

        self.emotion_emb = None
        if model_config["emotion"]:
            with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "emotions.json"), "r") as f:
                emotion_dict = json.load(f)
                n_emotion = np.unique([*emotion_dict.values()]).shape[0]
            self.emotion_emb = nn.Embedding(n_emotion, model_config["transformer"]["encoder_hidden"])

    def forward(self, device, speakers, emotions, texts, src_lens, max_src_len, mels=None, mel_lens=None,
                max_mel_len=None, p_targets=None, e_targets=None, d_targets=None, p_control=1.0, e_control=1.0,
                d_control=1.0):
        src_masks = get_mask_from_lengths(src_lens, device, max_src_len)
        mel_masks = get_mask_from_lengths(mel_lens, device, max_mel_len) if mel_lens is not None else None
        output = self.encoder(texts.to(device), src_masks.to(device))

        if self.speaker_emb is not None:
            speakers = torch.Tensor([self.speakers_dict[str(i.item())] for i in speakers]).long().to(device)
            self.speaker_emb = self.speaker_emb.to(device)
            output = output + self.speaker_emb(speakers.to(device)).unsqueeze(1).expand(-1, max_src_len, -1)
        if self.emotion_emb is not None:
            self.emotion_emb = self.emotion_emb.to(device)
            output = output + self.emotion_emb(emotions.to(device)).unsqueeze(1).expand(-1, max_src_len, -1)

        var_adaptor_output = self.variance_adaptor(device, output, src_masks, mel_masks, max_mel_len, p_targets,
                                                   e_targets, d_targets, p_control, d_control)
        output, p_predictions, e_predictions, log_d_predictions, d_rounded, mel_lens, mel_masks = var_adaptor_output
        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)
        if mels is not None:
            assert output.shape == mels.shape, f"Expected Variational Adapter Output to be equal to the target mel, " \
                                               f"found target: {mels.shape}, output: {output.shape}."
        postnet_output = self.postnet(output) + output
        return output, postnet_output, p_predictions, e_predictions, log_d_predictions, d_rounded,\
               src_masks, mel_masks, src_lens, mel_lens
