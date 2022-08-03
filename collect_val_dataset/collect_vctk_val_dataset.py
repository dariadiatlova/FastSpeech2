import os
import numpy as np
import json


def main(dir_path, txt_path, speaker_ids_mapping_path):
    """
    Function iterates through all .TextGrid files in a directory and writes to the .txt file ids of each
    speaker present in a dataset and writes to .json file mapping of <p***> formatted speaker id with corresponded
    integer ids.
    """
    filenames = os.listdir(dir_path)
    speakers = np.unique([s[:4] for s in filenames])

    validation_set = set()
    speaker_ids_dict = dict(zip(speakers, np.arange(speakers.shape[0]).astype(str)))
    speaker_dictionary = dict(zip(np.arange(speakers.shape[0]).astype(str), np.zeros(speakers.shape[0])))

    for file in filenames:
        if np.all(list(speaker_dictionary.values())):
            break
        short_name = file[:8]
        speaker, file_id = short_name.split("_")
        speaker_id = speaker_ids_dict[speaker]
        if speaker_dictionary[speaker_id] == 0:
            speaker_dictionary[speaker_id] = 1
            validation_set.add(short_name)

    counter = 0
    with open(txt_path, "w", encoding="utf-8") as f:
        for sample in validation_set:
            counter += 1
            f.write(sample + "|")
    print(f"Wrote {counter} ids to {txt_path}")

    with open(speaker_ids_mapping_path, "w") as f:
        f.write(json.dumps(speaker_ids_dict))
    print(f"Wrote speaker id mapping to {speaker_ids_mapping_path}")


if __name__ == "__main__":
    speaker_ids_mapping_path = "/root/storage/dasha/data/vctk/speaker_ids_mapping.json"
    dir_path = "/root/storage/dasha/data/vctk/TextGrids"
    txt_path = "/root/storage/dasha/data/vctk/val_ids.txt"

    main(dir_path, txt_path, speaker_ids_mapping_path)
