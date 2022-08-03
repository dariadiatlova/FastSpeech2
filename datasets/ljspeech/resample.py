import shutil
import os
import torchaudio
import torchaudio.functional as F
from tqdm import tqdm


def main(source_path, target_path, resample_rate: int = 16000):
    os.makedirs(target_path, exist_ok=True)
    filenames = os.listdir(source_path)
    for f in tqdm(filenames):
        if ".wav" not in f:
            continue
        wav, sample_rate = torchaudio.load(f"{source_path}/{f}")
        if sample_rate != resample_rate:
            wav = F.resample(wav, sample_rate, resample_rate)
        txt_path = f"{source_path}/{f[:-4]}.txt"
        textgrid_path = f"{source_path}/{f[:-4]}.TextGrid"
        torchaudio.save(f"{target_path}/{f}", wav, sample_rate=resample_rate)
        shutil.copy(txt_path, target_path)
        shutil.copy(textgrid_path, target_path)


if __name__ == "__main__":
    source_path = "/root/storage/dasha/data/lj_mfa_data/aligned_corpus/"
    target_path = "/root/storage/dasha/data/lj_mfa_data/wav16"
    main(source_path, target_path)
