import os
import numpy as np


def set_check(dir_path, speaker_id, file_id, emotions):
    paths_bool = [os.path.exists(f"{dir_path}/{speaker_id}_{file_id}_{i}.TextGrid") for i in emotions]
    paths = tuple([f"{speaker_id}_{file_id}_{i}" for i in emotions])
    return np.all(paths_bool), paths


def main(dir_path, txt_path):
    """
    Function iterates through all .TextGrid files in a directory and writes to the .txt file ids of each
    speaker present in a dataset, pronouncing one sentence with all existed emotions (5 emotions for speaker 1-10).
    In total txt file should consist of 50 sentences.
    """
    filenames = os.listdir(dir_path)
    emotions = ["0", "1", "2", "3", "4"]
    validation_set = set()
    speaker_ids = ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20"]
    speaker_dictionary = dict(zip(speaker_ids, np.zeros(10)))
    for file in filenames:
        if np.all(list(speaker_dictionary.values())):
            break
        short_name = file[:-9]
        speaker_id, file_id, emotion_id = short_name.split("_")
        if speaker_dictionary[speaker_id] == 0:
            res, paths = set_check(dir_path, speaker_id, file_id, emotions)
            if res:
                speaker_dictionary[speaker_id] = 1
                validation_set.add(paths)
    counter = 0
    with open(txt_path, "w", encoding="utf-8") as f:
        for sample in validation_set:
            for s in sample:
                counter += 1
                f.write(s + "|")
    print(f"Wrote {counter} ids to {txt_path}")


if __name__ == "__main__":
    dir_path = "/root/storage/dasha/data/emo-data/clean/TextGrids"
    txt_path = "/root/storage/dasha/data/emo-data/clean/copied_wavs_val_paths.txt"
    main(dir_path, txt_path)
