import os
import numpy as np


def main(dir_path, txt_path):
    """
    Function iterates through all .TextGrid files in a directory and writes to the .txt file ids of 64 files for val.
    """
    filenames = os.listdir(dir_path)
    np.random.shuffle(filenames)
    validation_set = [file[:-9] for file in filenames[:64]]
    with open(txt_path, "w", encoding="utf-8") as f:
        for sample in validation_set:
            f.write(sample + "|")
    print(f"Wrote {len(validation_set)} ids to {txt_path}")


if __name__ == "__main__":
    dir_path = "/root/storage/dasha/data/lj_mfa_data/TextGrids"
    txt_path = "/root/storage/dasha/data/lj_mfa_data/lj16_val_paths.txt"
    main(dir_path, txt_path)
