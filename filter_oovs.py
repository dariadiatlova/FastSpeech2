import os


def main(filepath: str, dir_path: str):
    file = open(filepath)
    lines = file.readlines()
    for line in lines:
        filename = line.split(" ")[0][1:-1]
        os.remove(os.path.join(dir_path, f"{filename}.wav"))
        os.remove(os.path.join(dir_path, f"{filename}.txt"))
    print(f"Removed {len(lines)} from {dir_path} directory")


if __name__ == "__main__":
    oovs_file_path = "/root/Documents/MFA/wavs_validate_pretrained/utterance_oovs.txt"
    data_dir_path = "/root/storage/dasha/emo-data/etts/vk_etts_data/wavs"
    main(filepath=oovs_file_path, dir_path=data_dir_path)
