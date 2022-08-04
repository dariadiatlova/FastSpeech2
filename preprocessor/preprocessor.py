import os
import random
import json

import tgt
import librosa
import numpy as np
import pyworld as pw
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import audio as Audio
from audio.compute_mel import PAD_MEL_VALUE


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.include_empty_intervals = config["preprocessing"]["include_empty_intervals"]
        self.in_dir = config["path"]["raw_path"]
        self.text_grids_dir = config["path"]["text_grids_path"]
        self.out_dir = config["path"]["preprocessed_path"]
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = config["preprocessing"]["stft"]["hop_length"]
        self.n_mels = config["preprocessing"]["mel"]["n_mel_channels"]

        assert config["preprocessing"]["pitch"]["feature"] in ["phoneme_level", "frame_level"]
        assert config["preprocessing"]["energy"]["feature"] in ["phoneme_level", "frame_level"]
        self.pitch_phoneme_averaging = (config["preprocessing"]["pitch"]["feature"] == "phoneme_level")
        self.energy_phoneme_averaging = (config["preprocessing"]["energy"]["feature"] == "phoneme_level")

        self.pitch_normalization = config["preprocessing"]["pitch"]["normalization"]
        self.energy_normalization = config["preprocessing"]["energy"]["normalization"]

        self.val_ids = open(config["path"]["val_ids_path"]).readlines()[0].split("|")[:-1]
        assert len(self.val_ids) == config["preprocessing"]["val_size"], \
            f"Expected to find {config['preprocessing']['val_size']} ids, found {len(self.val_ids)}."

        self.compute_mel_energy = Audio.compute_mel.ComputeMelEnergy(
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["device"],
            f_max=config["preprocessing"]["mel"]["mel_fmax"]
        )

    def build_from_path(self):
        os.makedirs((os.path.join(self.out_dir, "mel")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "pitch")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "energy")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "duration")), exist_ok=True)

        print("Processing Data ...")
        out = list()
        n_frames = 0
        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()
        self.emotions = {}

        # Compute pitch, energy, duration, and mel-spectrogram
        for filename in tqdm(os.listdir(self.text_grids_dir)):
            if ".TextGrid" not in filename:
                continue
            short_filename = filename[:-9]
            speaker_idx, filename_idx, emotion_id = short_filename.split("_")
            self.emotions[short_filename] = emotion_id
            tg_path = os.path.join(self.text_grids_dir, "{}.TextGrid".format(short_filename))
            wav_path = os.path.join(self.in_dir, "{}.wav".format(short_filename))
            txt_path = os.path.join(self.in_dir, "{}.txt".format(short_filename))
            if os.path.exists(tg_path) and os.path.exists(wav_path) and os.path.exists(txt_path):
                ret = self.process_utterance(short_filename, self.include_empty_intervals)
                if ret is None:
                    continue
                else:
                    info, pitch, energy, n = ret
                    out.append(info)

                    if len(pitch) > 0:
                        pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
                    if len(energy) > 0:
                        energy_scaler.partial_fit(energy.reshape((-1, 1)))

                    n_frames += n

        print("Computing statistic quantities ...")
        # Perform normalization if necessary
        if self.pitch_normalization:
            pitch_mean = pitch_scaler.mean_[0]
            pitch_std = pitch_scaler.scale_[0]
        else:
            # A numerical trick to avoid normalization...
            pitch_mean = 0
            pitch_std = 1
        if self.energy_normalization:
            energy_mean = energy_scaler.mean_[0]
            energy_std = energy_scaler.scale_[0]
        else:
            energy_mean = 0
            energy_std = 1

        pitch_min, pitch_max = self.normalize(os.path.join(self.out_dir, "pitch"), pitch_mean, pitch_std)
        energy_min, energy_max = self.normalize(os.path.join(self.out_dir, "energy"), energy_mean, energy_std)

        # Save files
        with open(os.path.join(self.out_dir, "emotions.json"), "w") as f:
            f.write(json.dumps(self.emotions))

        with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
            stats = {
                "pitch": [float(pitch_min), float(pitch_max), float(pitch_mean), float(pitch_std)],
                "energy": [float(energy_min), float(energy_max), float(energy_mean), float(energy_std)],
            }
            f.write(json.dumps(stats))

        print("Total time: {} hours".format(n_frames * self.hop_length / self.sampling_rate / 3600))
        random.shuffle(out)
        out = [r for r in out if r is not None]

        # Write metadata
        train_data, val_data = self.train_val_split(out)
        with open(os.path.join(self.out_dir, "train.txt"), "w", encoding="utf-8") as f:
            for m in train_data:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "val.txt"), "w", encoding="utf-8") as f:
            for m in val_data:
                f.write(m + "\n")
        return out

    def train_val_split(self, metadata):
        train_set = []
        val_set = []
        for sample in metadata:
            speaker_idx, filename_idx, emotion_id, text, raw_text = sample.split("|")
            if f"{speaker_idx}_{filename_idx}_{emotion_id}" in self.val_ids:
                val_set.append(sample)
            else:
                train_set.append(sample)
        assert len(train_set) + len(val_set) == len(metadata)
        return train_set, val_set

    def process_utterance(self, short_filename, include_empty_intervals):
        speaker_id, file_id, emotion_id = short_filename.split("_")
        wav_path = os.path.join(self.in_dir, f"{short_filename}.wav")
        text_path = os.path.join(self.in_dir, f"{short_filename}.txt")
        tg_path = os.path.join(self.text_grids_dir, f"{short_filename}.TextGrid")

        # Get alignments
        textgrid = tgt.io.read_textgrid(tg_path, include_empty_intervals=include_empty_intervals)
        phone, duration, start, end = self.get_alignment(textgrid.get_tier_by_name("phones"))

        text = "{" + " ".join(phone) + "}"
        if start >= end:
            return None

        # Read and trim wav files
        wav, _ = librosa.load(wav_path, sr=self.sampling_rate)
        wav = wav[int(self.sampling_rate * start):int(self.sampling_rate * end)].astype(np.float32)

        # Read raw text
        with open(text_path, "r") as f:
            raw_text = f.readline().strip("\n")

        # Compute fundamental frequency
        pitch, t = pw.dio(
            wav.astype(np.float64),
            self.sampling_rate,
            frame_period=self.hop_length / self.sampling_rate * 1000,
        )
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)

        # Compute mel-scale spectrogram and energy
        mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.compute_mel_energy)
        mel_count = mel_spectrogram.shape[1]

        if pitch.shape[0] - mel_count == 1:
            pitch = pitch[:-1]

        assert pitch.shape[0] == mel_count, f"Pitch isn't count for each mel. Mel count: {mel_count}, pitch " \
                                            f"count {pitch.shape[0]}"

        assert energy.shape[0] == mel_count, f"Energy isn't count for each mel. Mel count: {mel_count}, energy " \
                                             f"count {energy.shape[0]}"

        pitch = pitch[:sum(duration)]
        if np.sum(pitch != 0) <= 1:
            return None

        # Duration check
        mel_count = mel_spectrogram.shape[1]
        duration_sum = sum(duration)
        if duration_sum - mel_count == 1:
            mel_spectrogram = np.pad(mel_spectrogram,
                                     ((0, 0), (0, duration_sum - mel_count)),
                                     mode="constant", constant_values=PAD_MEL_VALUE)
        mel_count = mel_spectrogram.shape[1]

        assert mel_count == duration_sum, f"Mels and durations mismatch, mel count: {mel_count}, " \
                                          f"duration count: {duration_sum}."

        assert mel_spectrogram.shape[0] == self.n_mels, f"Incorrect padding, supposed to have: {self.n_mels}, got " \
                                                        f"{mel_spectrogram.shape[0]}."
        energy = energy[:sum(duration)]

        if self.pitch_phoneme_averaging:
            # perform linear interpolation
            nonzero_ids = np.where(pitch != 0)[0]
            interp_fn = interp1d(
                nonzero_ids,
                pitch[nonzero_ids],
                fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
                bounds_error=False,
            )
            pitch = interp_fn(np.arange(0, len(pitch)))

            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    pitch[i] = np.mean(pitch[pos: pos + d])
                else:
                    pitch[i] = 0
                pos += d
            pitch = pitch[:len(duration)]

        if self.energy_phoneme_averaging:
            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    energy[i] = np.mean(energy[pos: pos + d])
                else:
                    energy[i] = 0
                pos += d
            energy = energy[:len(duration)]

        # Save files

        assert not np.isnan(duration).any(), f"{short_filename} sample duration contains nan"
        dur_filename = f"{speaker_id}-duration-{file_id}-{emotion_id}.npy"
        np.save(os.path.join(self.out_dir, "duration", dur_filename), duration)

        assert not np.isnan(pitch).any(), f"{short_filename} sample pitch contains nan"
        pitch_filename = f"{speaker_id}-pitch-{file_id}-{emotion_id}.npy"
        np.save(os.path.join(self.out_dir, "pitch", pitch_filename), pitch)

        assert not np.isnan(energy).any(), f"{short_filename} sample energy contains nan"
        energy_filename = f"{speaker_id}-energy-{file_id}-{emotion_id}.npy"
        np.save(os.path.join(self.out_dir, "energy", energy_filename), energy)

        assert not np.isnan(mel_spectrogram).any(), f"{short_filename} sample mel_spectrogram contains nan"
        mel_filename = f"{speaker_id}-mel-{file_id}-{emotion_id}.npy"
        np.save(os.path.join(self.out_dir, "mel", mel_filename), mel_spectrogram.T)

        return (
            "|".join([str(speaker_id), str(file_id), str(emotion_id), text, raw_text]),
            self.remove_outlier(pitch),
            self.remove_outlier(energy),
            mel_spectrogram.shape[1]
        )

    def get_alignment(self, tier):
        sil_phones = ["sil", "sp", "spn"]

        phones = []
        durations = []
        start_time = 0
        end_time = 0
        end_idx = 0
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text

            # Trim leading silences
            if phones == []:
                if p in sil_phones:
                    continue
                else:
                    start_time = s

            if p not in sil_phones:
                # For ordinary phones
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                # For silent phones
                phones.append(p)

            durations.append(
                int(np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length))
            )
        assert len(phones) == len(durations), f"Phones and durations mismatch phones count {phones.shape[0]}," \
                                              f"durations count {durations.shape[0]}"
        # Trim tailing silences
        phones = phones[:end_idx]
        durations = durations[:end_idx]

        return phones, durations, start_time, end_time

    def remove_outlier(self, values):
        values = np.array(values)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)

        return values[normal_indices]

    def normalize(self, in_dir, mean, std):
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        for filename in os.listdir(in_dir):
            filename = os.path.join(in_dir, filename)
            values = (np.load(filename) - mean) / std
            np.save(filename, values)

            max_value = max(max_value, max(values))
            min_value = min(min_value, min(values))

        return min_value, max_value
