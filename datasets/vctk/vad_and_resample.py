import os
import shutil
import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
from tqdm import tqdm


def main(dir_path: str, sample_rate: int = 48000, resample_rate: int = 16000):
    speakers = os.listdir(f"{dir_path}/wav48_silence_trimmed")
    vad = T.Vad(sample_rate)
    new_dir_path = f"{dir_path}/wavs16vad2"
    os.makedirs(new_dir_path, exist_ok=True)
    counter = 0
    for speaker in tqdm(speakers):
        prev_dir_path = f"{dir_path}/wav48_silence_trimmed/{speaker}"
        if os.path.isdir(prev_dir_path):
            filenames = os.listdir(prev_dir_path)
            for filename in tqdm(filenames):
                audio_filepath = f"{prev_dir_path}/{filename}"
                if "mic2" in audio_filepath:
                    continue
                wav, sr = torchaudio.load(audio_filepath)
                vad_wav = torch.flip(vad(torch.flip(vad(wav), dims=[1])), dims=[1])
                resampled = F.resample(vad_wav, sample_rate, resample_rate)
                short_filename = filename[:8]
                txt_filepath = f"{dir_path}/txt/{speaker}/{short_filename}.txt"
                if os.path.exists(txt_filepath):
                    shutil.copy(txt_filepath, f"{new_dir_path}/{short_filename}.txt")
                    torchaudio.save(f"{new_dir_path}/{short_filename}.wav", resampled, sample_rate=resample_rate)
                else:
                    counter += 1
    print(f"Resampled and added wavs are saved to the {new_dir_path}!\n Couldn't find txt for {counter} samples :c")


if __name__ == "__main__":
    dir_path = "/root/storage/dasha/data/vctk"
    main(dir_path)
