import os
import shutil


def main(source_path, target_path, target_speaker: str = "15"):
    text_grids_path = f"{source_path}/TextGrid"
    wavs_path = f"{source_path}/wavs"
    filenames = os.listdir(text_grids_path)
    counter = 0
    for filename in filenames:
        if filename:
            short_name = filename[:-9]
            speaker, _, _ = short_name.split("_")
            if speaker == target_speaker:
                txt_path = f"{wavs_path}/{short_name}.txt"
                wav_path = f"{wavs_path}/{short_name}.wav"
                if os.path.exists(txt_path) and os.path.exists(wav_path):
                    counter += 1
                    shutil.copy(f"{text_grids_path}/{filename}", f"{target_path}/TextGrid/{filename}")
                    shutil.copy(txt_path, f"{target_path}/wavs/{short_name}.txt")
                    shutil.copy(wav_path, f"{target_path}/wavs/{short_name}.wav")
    print(f"Moved {counter} files to {target_path}/TextGrid and {target_path}/wavs directories!")


if __name__ == "__main__":
    target_path = "/root/storage/dasha/data/emo-single-speaker"
    source_path = "/root/storage/dasha/data/emo-data/english_esd"
    main(source_path, target_path)
