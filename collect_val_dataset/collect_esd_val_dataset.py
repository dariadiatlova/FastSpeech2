import os
import numpy as np


def set_check(dir_path, speaker_id, file_id, emotions):
    paths_bool = [os.path.exists(f"{dir_path}/{speaker_id}_{file_id}_{i}.TextGrid") for i in emotions]
    # print(paths_bool)
    paths = tuple([f"{speaker_id}_{file_id}_{i}" for i in emotions])
    return np.all(paths_bool), paths


def main(dir_path, original_val_ids_path, shorten_val_ids_path):
    """
    Function iterates through all .TextGrid files in a directory and writes to the .txt file ids of each
    speaker present in a dataset, pronouncing one sentence with all existed emotions (5 emotions for speaker 1-10).
    In total txt file should consist of 50 sentences.
    """
    emotions = ["0", "1", "2", "3", "4"]
    validation_set = set()
    speaker_ids = ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20"]
    speaker_dictionary = dict(zip(speaker_ids, np.zeros(10)))
    ids = open(original_val_ids_path).readlines()[0].split("|")
    for _id in ids:
        speaker_id, file_id, emotion_id = _id.split("_")
        if np.all(list(speaker_dictionary.values())):
            break
        if speaker_dictionary[speaker_id] == 0:
            res, paths = set_check(dir_path, speaker_id, file_id, emotions)
            if res:
                speaker_dictionary[speaker_id] = 1
                validation_set.add(paths)
    counter = 0
    with open(shorten_val_ids_path, "w", encoding="utf-8") as f:
        for sample in validation_set:
            for s in sample:
                counter += 1
                f.write(s + "|")
    print(f"Wrote {counter} ids to {shorten_val_ids_path}")


if __name__ == "__main__":
    dir_path = "/root/storage/dasha/data/emo-data/english_esd/TextGrid"
    original_val_ids_path = "/root/storage/dasha/data/emo-data/english_esd/val_ids.txt"
    shorten_val_ids_path = "/root/storage/dasha/data/emo-data/english_esd/short_val_ids.txt"
    main(dir_path, original_val_ids_path, shorten_val_ids_path)
