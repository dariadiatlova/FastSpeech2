import pandas as pd
import os


def main(file_path, dir_path):
    df = pd.read_csv(file_path)
    filenames = df.file.unique()
    for filename in filenames:
        os.remove(os.path.join(dir_path, f"{filename}.wav"))
        os.remove(os.path.join(dir_path, f"{filename}.txt"))
    print(f"Removed {filenames.shape[0]} from {dir_path} directory")


if __name__ == "__main__":
    unaligned_path = "/root/Documents/MFA/wavs16_validate_pretrained/unalignable_files.csv"
    source_directory = "/root/storage/dasha/data/vctk/wavs16"
    main(unaligned_path, source_directory)
