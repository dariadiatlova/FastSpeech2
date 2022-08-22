from tqdm import tqdm
import glob
import tgt


def main(text_grids_path, logs_path):
    spn_files = []
    textgrids = glob.glob(f"{text_grids_path}/*.TextGrid")
    for textgrid_path in tqdm(textgrids):
        textgrid = tgt.io.read_textgrid(textgrid_path, include_empty_intervals=True)
        txt_phones = textgrid.get_tier_by_name("phones")
        for t in txt_phones._objects:
            phone = t.text
            if phone == "spn":
                spn_files.append(textgrid_path)
    with open(logs_path, 'w') as f:
        for line in spn_files:
            f.write(f"{line}\n")

    print(f"Found {len(spn_files)} files containing spn in their transcriptions.")


if __name__ == "__main__":
    text_grids_path = "/root/storage/dasha/data/vctk/TextGridsVad2"
    logs_path = "/root/storage/dasha/data/vctk/filenames_with_spt.txt"
    main(text_grids_path, logs_path)
