import torch
import torch.nn as nn
import torchaudio

MIN_MEL_VALUE = 1e-05
PAD_MEL_VALUE = -11.52


class ComputeMelEnergy(nn.Module):
    def __init__(self, sample_rate, n_fft, hop_length, win_length, n_mels, device, power=1, center=False,
                 f_min=0, f_max=None, pad_value=PAD_MEL_VALUE):
        super().__init__()
        self.compute_mel = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft,
                                                                hop_length=hop_length, n_mels=n_mels,
                                                                power=power, center=center,
                                                                f_min=f_min, f_max=f_max,
                                                                norm='slaney', mel_scale='slaney')
        self.hop_length = hop_length
        self.padding = n_fft - hop_length
        self.pad_value = pad_value

    def forward(self, wav, wav_lens=None):
        padded_wav = nn.functional.pad(wav, (self.padding // 2, self.padding // 2), mode='reflect')
        spectrogram = self.compute_mel.spectrogram(padded_wav)
        mel = log_mel(self.compute_mel.mel_scale(spectrogram))
        energy = torch.norm(spectrogram, dim=1)

        if wav_lens is None:
            return mel, energy

        mel_lens = wav_lens // self.hop_length
        max_len = mel.size(-1)
        mask = torch.arange(max_len).to(mel.device)
        mask = mask.expand(mel.size(0), max_len) >= mel_lens.unsqueeze(1)
        mel = mel.masked_fill(mask.unsqueeze(1).type(torch.bool).to(device=mel.device), self.pad_value)

        return mel, energy


def log_mel(mel):
    return torch.log(torch.clamp(mel, min=MIN_MEL_VALUE))
