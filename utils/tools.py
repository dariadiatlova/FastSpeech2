import os
import json

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib

from matplotlib import pyplot as plt


matplotlib.use("Agg")


def denormalize(data: np.ndarray, mean: float, std: float) -> np.ndarray:
    return data * std + mean


def get_pitch_from_pitch_spec(pitch_spec):
    result_pitch = np.zeros(pitch_spec.shape[0]).astype(np.float32)
    for t in range(pitch_spec.shape[0]):
        for i in range(pitch_spec.shape[1]):
            result_pitch[t] += pitch_spec[t, i] * (i + 2.5 + 1) ** (-5 / 2)
    return result_pitch


def torch_from_numpy(data):
    if len(data) == 13:
        ids, raw_texts, speakers, texts, src_lens, max_src_len = data[:6]
        mels, mel_lens, max_mel_len, cwt, pitches, energies, durations = data[6:]
        speakers = torch.from_numpy(np.zeros(len(speakers))).long()
        texts = torch.from_numpy(texts).long()
        src_lens = torch.from_numpy(src_lens)
        mels = torch.from_numpy(mels).float()
        mel_lens = torch.from_numpy(mel_lens)
        pitches = torch.from_numpy(pitches).float()
        cwt = torch.from_numpy(cwt).float()
        energies = torch.from_numpy(energies)
        durations = torch.from_numpy(durations).long()
        return ids, raw_texts, speakers, texts, src_lens, max_src_len, mels, \
               mel_lens, max_mel_len, cwt, pitches, energies, durations

    if len(data) == 6:
        ids, raw_texts, speakers, texts, src_lens, max_src_len = data
        speakers = torch.from_numpy(np.zeros(len(speakers))).long()
        texts = torch.from_numpy(texts).long()
        src_lens = torch.from_numpy(src_lens)
        return ids, raw_texts, speakers, texts, src_lens, max_src_len


def to_device(data, device):
    if len(data) == 12:
        (
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            pitches,
            energies,
            durations,
        ) = data

        speakers = torch.from_numpy(np.zeros(len(speakers))).long().to(device)
        # speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)
        mels = torch.from_numpy(mels).float().to(device)
        mel_lens = torch.from_numpy(mel_lens).to(device)
        pitches = torch.from_numpy(pitches).float().to(device)
        energies = torch.from_numpy(energies).to(device)
        durations = torch.from_numpy(durations).long().to(device)

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            pitches,
            energies,
            durations,
        )

    if len(data) == 6:
        ids, raw_texts, speakers, texts, src_lens, max_src_len = data
        # speakers = torch.from_numpy(speakers).long().to(device)
        speakers = torch.from_numpy(np.zeros(len(speakers))).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)

        return ids, raw_texts, speakers, texts, src_lens, max_src_len


def log(
    logger, step=None, losses=None, fig=None, audio=None, sampling_rate=22050, tag=""
):
    if losses is not None:
        logger.add_scalar("Loss/total_loss", losses[0], step)
        logger.add_scalar("Loss/mel_loss", losses[1], step)
        logger.add_scalar("Loss/mel_postnet_loss", losses[2], step)
        logger.add_scalar("Loss/pitch_loss", losses[3], step)
        logger.add_scalar("Loss/energy_loss", losses[4], step)
        logger.add_scalar("Loss/duration_loss", losses[5], step)

    if fig is not None:
        logger.add_figure(tag, fig)

    if audio is not None:
        logger.add_audio(
            tag,
            audio / max(abs(audio)),
            sample_rate=sampling_rate,
        )


def get_mask_from_lengths(lengths, device, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
    lengths = lengths.to(device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)
    # B, T ; B, T
    # 1, 2, 3, .... 7 >= 5, 5, 5, 5, .... 5
    # 1, 2, 3, .... 7 >= 7, 7, 7, 7
    # 0, 0, 0, 0, 0, 1, 1
    # 0, 0, 0, 0, 0, 0, 0
    # 0, 0, 0, 1, 1, 1, 1
    # 5, 7, 3 => 5, 5, 5, ..., 5;
    return mask


def expand(values, durations):
    out = list()
    for value, d in zip(values, durations):
        out += [value] * max(0, int(d))
    return np.array(out)


def synth_one_sample(targets, predictions, vocoder, preprocess_config, i):
    basename = targets[0][0]
    src_len = predictions[8][0].item()
    mel_len = predictions[9][0].item()
    mel_target = targets[6][0, :mel_len].detach().transpose(0, 1)
    mel_prediction = predictions[1][0, :mel_len].detach().transpose(0, 1)
    duration = targets[11][0, :src_len].detach().cpu().numpy()

    if preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
        pitch = targets[9][0, :src_len].detach().cpu().numpy()
        pitch = expand(pitch, duration)
    else:
        pitch = targets[9][0, :mel_len].detach().cpu().numpy()
    if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
        energy = targets[10][0, :src_len].detach().cpu().numpy()
        energy = expand(energy, duration)
    else:
        energy = targets[10][0, :mel_len].detach().cpu().numpy()

    with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")) as f:
        stats = json.load(f)
        stats = stats["pitch"] + stats["energy"][:2]

    fig = plot_mel(
        [
            (mel_prediction.cpu().numpy(), pitch, energy),
            (mel_target.cpu().numpy(), pitch, energy),
        ],
        stats,
        ["Synthetized Spectrogram", "Ground-Truth Spectrogram"],
    )

    if vocoder is not None:
        wav_reconstruction = vocoder(mel_target.unsqueeze(0))[0].squeeze(0).detach().cpu().numpy()
        wav_prediction = vocoder(mel_prediction.unsqueeze(0))[0].squeeze(0).detach().cpu().numpy()
    else:
        wav_reconstruction = wav_prediction = None

    return fig, wav_reconstruction, wav_prediction, basename


def synthesize_predicted_wav(i, predictions, vocoder):
    mel_len = predictions[9][i]
    mel_prediction = predictions[1][i, :mel_len].detach().transpose(0, 1)
    wav_prediction = vocoder(mel_prediction.unsqueeze(0).detach().cpu())[0].squeeze(0).detach().cpu().numpy()
    return wav_prediction


def synthesize_from_gt_mel(mel, vocoder):
    mel = mel.detach().transpose(0, 1)
    wav_reconstructed = vocoder(mel.unsqueeze(0).detach().cpu())[0].squeeze(0).detach().cpu().numpy()
    return wav_reconstructed


def reconstruct_wav(target, mel_len, vocoder):
    mel_target = target[6][:mel_len].detach().transpose(0, 1)
    wav_reconstruction = vocoder(mel_target.unsqueeze(0))[0].squeeze(0).detach().cpu().numpy()
    wav_prediction = vocoder(wav_reconstruction.unsqueeze(0))[0].squeeze(0).detach().cpu().numpy()
    return wav_prediction


def plot_mel(data, stats, titles):
    fig, axes = plt.subplots(len(data), 1, squeeze=False)
    if titles is None:
        titles = [None for i in range(len(data))]
    pitch_min, pitch_max, pitch_mean, pitch_std, energy_min, energy_max = stats
    pitch_min = pitch_min * pitch_std + pitch_mean
    pitch_max = pitch_max * pitch_std + pitch_mean

    def add_axis(fig, old_ax):
        ax = fig.add_axes(old_ax.get_position(), anchor="W")
        ax.set_facecolor("None")
        return ax

    for i in range(len(data)):
        mel, pitch, energy = data[i]
        pitch = pitch * pitch_std + pitch_mean
        axes[i][0].imshow(mel, origin="lower")
        axes[i][0].set_aspect(2.5, adjustable="box")
        axes[i][0].set_ylim(0, mel.shape[0])
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[i][0].set_anchor("W")

        ax1 = add_axis(fig, axes[i][0])
        ax1.plot(pitch, color="tomato")
        ax1.set_xlim(0, mel.shape[1])
        ax1.set_ylim(0, pitch_max)
        ax1.set_ylabel("F0", color="tomato")
        ax1.tick_params(
            labelsize="x-small", colors="tomato", bottom=False, labelbottom=False
        )

        ax2 = add_axis(fig, axes[i][0])
        ax2.plot(energy, color="darkviolet")
        ax2.set_xlim(0, mel.shape[1])
        ax2.set_ylim(energy_min, energy_max)
        ax2.set_ylabel("Energy", color="darkviolet")
        ax2.yaxis.set_label_position("right")
        ax2.tick_params(
            labelsize="x-small",
            colors="darkviolet",
            bottom=False,
            labelbottom=False,
            left=False,
            labelleft=False,
            right=True,
            labelright=True,
        )

    return fig


def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None, pitch_spec=False):
    def pad(x, max_len):
        PAD = 0
        if pitch_spec:
            if np.shape(x)[1] > max_len:
                raise ValueError("not max_len")
            s = np.shape(x)[0]
            x_padded = np.pad(x, (max_len - np.shape(x)[1], 0), mode="constant", constant_values=PAD)
            return x_padded[:s, :]
        else:
            if np.shape(x)[0] > max_len:
                raise ValueError("not max_len")

            s = np.shape(x)[1]
            x_padded = np.pad(x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD)
            return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        if pitch_spec:
            max_len = max(np.shape(x)[1] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded
