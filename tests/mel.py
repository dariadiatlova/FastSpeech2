import argparse

import yaml
import numpy as np
import os
import librosa
import tgt

from audio.compute_mel import ComputeMelEnergy
from audio.tools import get_mel_from_wav
from preprocessor.preprocessor import Preprocessor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to mel.yaml")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    compute_mel_energy = ComputeMelEnergy(config["sample_rate"], config["filter_length"], config["hop_length"],
                                          config["win_length"], config["n_mels"], config["device"])

    for filename in config["test_filenames"]:
        # get mel online
        basename = filename.split(".")[0]
        tg_path = os.path.join(config["source_wav_directory"], "{}.TextGrid".format(filename))
        wav_path = os.path.join(config["source_wav_directory"], "{}.wav".format(filename))
        textgrid = tgt.io.read_textgrid(tg_path)

        preprocessor_config = yaml.load(open(config["preprocess_config_path"], "r"), Loader=yaml.FullLoader)
        preprocessor = Preprocessor(preprocessor_config)

        _, duration, start, end = preprocessor.get_alignment(textgrid.get_tier_by_name("phones"))
        wav = librosa.load(wav_path, sr=config["sample_rate"])[0]
        wav = wav[int(config["sample_rate"] * start):int(config["sample_rate"] * end)].astype(np.float32)
        online_mel, energy = get_mel_from_wav(wav, compute_mel_energy)
        online_mel = online_mel[:, :sum(duration)]

        # load mel from saved
        saved_mel = np.load(os.path.join(config["source_mel_directory"], "0-mel-{}.npy".format(filename)))
        assert np.allclose(online_mel, saved_mel), "Mel Spectrograms aren't closed due to the set tolerance"
