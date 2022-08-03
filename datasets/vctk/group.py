import os
import shutil


def main(main_dir_path: str):
    speakers = os.listdir(f"{main_dir_path}/wav16vad")
    target_dir_path = f"{main_dir_path}/wavs"
    os.makedirs(target_dir_path)
    speaker_mapping = dict(zip(speakers, list(range(1, len(speakers) + 1))))
    for speaker in speakers:
        speaker_id = speaker_mapping[speaker]
        speaker_dir_path = f"{main_dir_path}/wav16vad/{speaker}"
        files = os.listdir(speaker_dir_path)
        for file in files:
            shutil.copy(file, f"{target_dir_path}/{speaker_id}_{file}")


if __name__ == "__main__":
    main_dir_path = "/root/storage/dasha/data/vctk"
    main(main_dir_path)
