import os
import shutil
import torch
import torchaudio
import librosa
from tqdm import tqdm


def main(vad_wav_path, root_path, txt_path, target_sr=16000):
    speakers = os.listdir(root_path)
    os.makedirs(vad_wav_path, exist_ok=True)
    for speaker in tqdm(speakers):
        prev_dir_path = f"{root_path}/{speaker}"
        if os.path.isdir(prev_dir_path):
            filenames = os.listdir(prev_dir_path)
            for filename in filenames:
                audio_filepath = f"{prev_dir_path}/{filename}"
                if "mic2" in audio_filepath:
                    continue
                wav, sr = torchaudio.load(audio_filepath)
                vad_wav = librosa.effects.trim(wav[0, :].numpy(), top_db=30)[0]
                resampled_wav = librosa.resample(vad_wav, orig_sr=sr, target_sr=target_sr)
                short_filename = filename[:8]
                txt_filepath = f"{txt_path}/{speaker}/{short_filename}.txt"
                if os.path.exists(txt_filepath):
                    shutil.copy(txt_filepath, f"{vad_wav_path}/{short_filename}.txt")
                    torchaudio.save(f"{vad_wav_path}/{short_filename}.wav",
                                    torch.tensor(resampled_wav).unsqueeze(0),
                                    sample_rate=target_sr)


if __name__ == "__main__":
    vad_wav_path = "/root/storage/dasha/data/vctk/librosa_vad30_16"
    root_path = "/root/storage/dasha/data/vctk/wav48_silence_trimmed"
    txt_path = "/root/storage/dasha/data/vctk/txt"
    main(vad_wav_path, root_path, txt_path)
