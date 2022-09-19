import json
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from ssqueezepy import icwt
from utils.tools import get_mask_from_lengths, pad


class VarianceAdaptor(nn.Module):
    """Variance Adaptor"""

    def __init__(self, preprocess_config, model_config):
        super(VarianceAdaptor, self).__init__()
        with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")) as f:
            stats = json.load(f)
            energy_min, energy_max = stats["energy"][:2]
            pitch_min, pitch_max = stats["pitch"][:2]

        self.duration_predictor = VariancePredictor(model_config)
        self.length_regulator = LengthRegulator(preprocess_config["device"])
        self.pitch_predictor = PitchPredictor(model_config)
        self.energy_predictor = VariancePredictor(model_config)
        self.device = preprocess_config["device"]
        self.pitch_feature_level = preprocess_config["pitch"]["feature"]
        self.energy_feature_level = preprocess_config["energy"]["feature"]

        assert self.pitch_feature_level in ["phoneme_level", "frame_level"]
        assert self.energy_feature_level in ["phoneme_level", "frame_level"]

        pitch_quantization = model_config["variance_embedding"]["pitch_quantization"]
        energy_quantization = model_config["variance_embedding"]["energy_quantization"]
        n_bins = model_config["variance_embedding"]["n_bins"]

        assert pitch_quantization in ["linear", "log"]
        assert energy_quantization in ["linear", "log"]

        if pitch_quantization == "log":
            self.pitch_bins = nn.Parameter(
                torch.exp(torch.linspace(np.log(pitch_min), np.log(pitch_max), n_bins - 1)), requires_grad=False,
            )
        else:
            self.pitch_bins = nn.Parameter(torch.linspace(pitch_min, pitch_max, n_bins - 1), requires_grad=False)
        if energy_quantization == "log":
            self.energy_bins = nn.Parameter(
                torch.exp(torch.linspace(np.log(energy_min), np.log(energy_max), n_bins - 1)), requires_grad=False)
        else:
            self.energy_bins = nn.Parameter(torch.linspace(energy_min, energy_max, n_bins - 1), requires_grad=False)

        self.pitch_embedding = nn.Embedding(n_bins, model_config["transformer"]["encoder_hidden"])
        self.energy_embedding = nn.Embedding(n_bins, model_config["transformer"]["encoder_hidden"])

    def get_pitch_embedding(self, device, x, target, mask, control):
        pitch_prediction, cwt = self.pitch_predictor(x, mask)
        self.pitch_embedding.to(device)
        if target is not None:
            embedding = self.pitch_embedding(torch.bucketize(target.to(device), self.pitch_bins.to(device)))
        else:
            pitch_prediction = pitch_prediction * control
            embedding = self.pitch_embedding(torch.bucketize(pitch_prediction.to(device), self.pitch_bins.to(device)))
        return pitch_prediction, cwt, embedding

    def get_energy_embedding(self, device, x, target, mask, control):
        prediction = self.energy_predictor(x, mask)
        if target is not None:
            embedding = self.energy_embedding(torch.bucketize(target.to(device), self.energy_bins.to(device)))
        else:
            prediction = prediction * control
            # try:
            embedding = self.energy_embedding(torch.bucketize(prediction.to(device), self.energy_bins.to(device)))
            # except Exception:
            #     print(f"Couldn't put on device:")
            #     print(f"prediction: {prediction}")
            #     print(f"energy bins: {self.energy_bins}")
        return prediction, embedding

    def forward(self, device, x, src_mask, mel_mask=None, max_len=None, pitch_target=None, p_mean=None, p_std=None,
                energy_target=None, duration_target=None, p_control=1.0, d_control=1.0):

        log_duration_prediction = self.duration_predictor(x, src_mask)
        if self.pitch_feature_level == "phoneme_level":
            pitch_prediction, cwt, pitch_embedding = self.get_pitch_embedding(
                device, x, pitch_target, src_mask, p_control)
            x = x + pitch_embedding
        if self.energy_feature_level == "phoneme_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(device, x, energy_target, src_mask, p_control)
            x = x + energy_embedding

        if duration_target is not None:
            x, mel_len = self.length_regulator(x, duration_target, max_len)
            duration_rounded = duration_target
        else:
            duration_rounded = torch.clamp((torch.round(torch.exp(log_duration_prediction) - 1) * d_control), min=0)
            x, mel_len = self.length_regulator(x, duration_rounded, max_len)
            mel_mask = get_mask_from_lengths(mel_len, device=self.device)

        if self.pitch_feature_level == "frame_level":
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(device, x, pitch_target, mel_mask, p_control)
            x = x + pitch_embedding
        if self.energy_feature_level == "frame_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(device, x, energy_target, mel_mask, p_control)
            x = x + energy_embedding

        return x, cwt, pitch_prediction, energy_prediction, log_duration_prediction, duration_rounded, mel_len, mel_mask


class LengthRegulator(nn.Module):
    """Length Regulator"""

    def __init__(self, device):
        super(LengthRegulator, self).__init__()
        self.device = device

    def LR(self, x, duration, max_len):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration): # b, t, c
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, torch.LongTensor(mel_len).to(self.device)

    def expand(self, batch, predicted):
        out = list()
        for i, vec in enumerate(batch): # t_phoneme, c -> {t_i, c} /forall i t_mel, c -> t_mel, c; t_mel = /sum t_i
            expand_size = predicted[i].item()
            out.append(vec.expand(max(int(expand_size), 0), -1))
        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len


class VariancePredictor(nn.Module):
    """Duration, Pitch and Energy Predictor"""

    def __init__(self, model_config):
        super(VariancePredictor, self).__init__()

        self.input_size = model_config["transformer"]["encoder_hidden"]
        self.filter_size = model_config["variance_predictor"]["filter_size"]
        self.kernel = model_config["variance_predictor"]["kernel_size"]
        self.conv_output_size = model_config["variance_predictor"]["filter_size"]
        self.dropout = model_config["variance_predictor"]["dropout"]

        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    ("conv1d_1",
                     Conv(self.input_size, self.filter_size, kernel_size=self.kernel, padding=(self.kernel - 1) // 2)),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    ("conv1d_2", Conv(self.filter_size, self.filter_size, kernel_size=self.kernel, padding=1)),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output, mask):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)

        if mask is not None:
            out = out.masked_fill(mask, 0.0)

        return out


class PitchPredictor(nn.Module):
    """Pitch Predictor using CWT"""

    def __init__(self, model_config):
        super(PitchPredictor, self).__init__()

        self.input_size = model_config["transformer"]["encoder_hidden"]
        self.filter_size = model_config["variance_predictor"]["filter_size"]
        self.kernel = model_config["variance_predictor"]["kernel_size"]
        self.conv_output_size = model_config["variance_predictor"]["filter_size"]
        self.dropout = model_config["variance_predictor"]["dropout"]
        self.scale = model_config["variance_predictor"]["scale"]
        self.scales = np.array(list(range(1, self.scale + 1))).astype(np.float32)

        self.stats_conv_layer = Conv(self.input_size,
                                     self.filter_size,
                                     kernel_size=self.kernel,
                                     padding=(self.kernel - 1) // 2)
        self.stats_linear_layer = nn.Linear(self.conv_output_size, 2)

        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    ("conv1d_1",
                     Conv(self.input_size, self.filter_size, kernel_size=self.kernel, padding=(self.kernel - 1) // 2)),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    ("conv1d_2", Conv(self.filter_size, self.filter_size, kernel_size=self.kernel, padding=1)),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.linear_layer = nn.Linear(self.conv_output_size, self.scale * 2)

    def forward(self, encoder_output, mask):
        out = self.linear_layer(self.conv_layer(encoder_output)).permute(0, 2, 1)
        stats = self.stats_linear_layer(torch.mean(self.stats_conv_layer(encoder_output), dim=1))
        bs = out.shape[0]

        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.scale * 2, -1)
            out = out.masked_fill(mask, 0.0)
            cwt_batch = torch.complex(out[:, :self.scale, :], out[:, self.scale:, :])
            means = stats[:, 0]
            stds = stats[:, 1]
            batch = []
            for i in range(bs):
                batch.append(torch.tensor(
                    icwt(cwt_batch.detach().cpu().numpy()[i, :, :], wavelet="cmhat", scales=self.scales),
                    requires_grad=True).to(means.device.type) + means[i] * stds[i])
            pitch = torch.stack(batch)

        return pitch, out


class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, bias=bias)

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)
        return x
