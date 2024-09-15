import tgt
import argparse
import glob
import json
from tqdm import tqdm

from typing import List


def main(textgrids: List[str], target_path: str):
    mapping = dict()
    mapping[""] = 0
    counter = 1
    for textgrid_path in tqdm(textgrids):
        textgrid = tgt.io.read_textgrid(textgrid_path, include_empty_intervals=True)
        txt_phones = textgrid.get_tier_by_name("phones")
        for t in txt_phones._objects:
            phone = t.text
            if phone not in mapping.keys():
                counter += 1
                mapping[phone] = counter
    json.dump(mapping, open(target_path, 'w'), ensure_ascii=False)
    print(mapping.keys())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_grids_directory", type=str, default="/root/dasha/mfa/aligned_corpus")
    parser.add_argument("--target_path", type=str, default="/root/dasha/mfa/phones_mapping.json")
    args = parser.parse_args()
    all_grids = glob.glob(f"{args.text_grids_directory}/*.TextGrid")
    main(all_grids, args.target_path)
