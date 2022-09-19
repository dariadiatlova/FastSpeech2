import os
import shutil
import torch
import torchaudio
from tqdm import tqdm
import torchaudio.functional as F


def main(resampled_wav_path, root_path, txt_path, target_sr=22050):
    speakers = os.listdir(root_path)
    os.makedirs(resampled_wav_path, exist_ok=True)
    for speaker in tqdm(speakers):
        prev_dir_path = f"{root_path}/{speaker}"
        if os.path.isdir(prev_dir_path):
            filenames = os.listdir(prev_dir_path)
            for filename in filenames:
                audio_filepath = f"{prev_dir_path}/{filename}"
                if "mic2" in audio_filepath:
                    continue
                wav, sr = torchaudio.load(audio_filepath)
                resampled = F.resample(wav, sr, target_sr)
                short_filename = filename[:8]
                txt_filepath = f"{txt_path}/{speaker}/{short_filename}.txt"
                if os.path.exists(txt_filepath):
                    shutil.copy(txt_filepath, f"{resampled_wav_path}/{short_filename}.txt")
                    torchaudio.save(f"{resampled_wav_path}/{short_filename}.wav", resampled, sample_rate=target_sr)


if __name__ == "__main__":
    resampled_wav_path = "/root/storage/dasha/data/vctk/wavs22050"
    root_path = "/root/storage/dasha/data/vctk/wav48_silence_trimmed"
    txt_path = "/root/storage/dasha/data/vctk/txt"
    main(resampled_wav_path, root_path, txt_path)
