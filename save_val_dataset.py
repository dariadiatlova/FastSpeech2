import os
import numpy as np


def set_check(dir_path, speaker_id, file_id):
    emotions = [0, 1, 2, 3, 4]
    paths_bool = [os.path.exists(f"{dir_path}/TextGrid/{speaker_id}_{file_id}_{i}.TextGrid") for i in emotions]
    paths = tuple([f"{speaker_id}_{file_id}_{i}" for i in emotions])
    return np.all(paths_bool), paths


def main(source_dir: str, target_dir: str, target_speaker: str = "15"):
    test_ids = list(range(100, 133))
    target_val_filenames = []
    for test_id in test_ids:
        res, paths = set_check(source_dir, target_speaker, test_id)
        if res:
            target_val_filenames.extend(paths)

    counter = 0
    txt_path = f"{target_dir}/val_names.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for sample in target_val_filenames:
            counter += 1
            f.write(sample + "|")
    print(f"Wrote {counter} ids to {txt_path}")


if __name__ == "__main__":
    source_dir = "/root/storage/dasha/data/emo-data/english_esd"
    target_dir = "/root/storage/dasha/data/emo-single-speaker"
    main(source_dir, target_dir)
