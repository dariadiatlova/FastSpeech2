import json
import os
from typing import Optional

import numpy as np
from torch.utils.data import Dataset

from utils.tools import pad_1D, pad_2D


class Dataset(Dataset):
    def __init__(self, filename, preprocess_config, synthesis_size: Optional[int] = None, sort=False, drop_last=False,
                 limit: int = None):
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.batch_size = preprocess_config["batch_size"]
        self.synthesis_size = synthesis_size
        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(filename)
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        self.speaker_map = dict(zip(list(self.speaker_map.values()), list(self.speaker_map.keys())))
        with open(preprocess_config["path"]["phones_mapping_path"], "r") as f:
            self.phones_mapping = json.load(f)
        self.sort = sort
        self.drop_last = drop_last
        self.limit = limit

    def __len__(self):
        if self.limit is not None:
            return self.limit
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker_id = self.speaker_map[int(self.speaker[idx])]
        raw_text = self.raw_text[idx]
        phone = np.array([self.phones_mapping[i] for i in self.text[idx][1:-1].split(" ")])
        mel = np.load(os.path.join(self.preprocessed_path, "mel", "{}-mel-{}.npy".format("0", basename)))
        pitch = np.load(os.path.join(self.preprocessed_path, "pitch", "{}-pitch-{}.npy".format("0", basename)))
        energy = np.load(os.path.join(self.preprocessed_path, "energy", "{}-energy-{}.npy".format("0", basename)))
        duration = np.load(os.path.join(self.preprocessed_path, "duration", "{}-duration-{}.npy".format("0", basename)))

        assert duration.shape == phone.shape, f"Duration and phone shapes do not match. Phone shape {phone.shape}, " \
                                              f"duration: {duration.shape} for sample: {self.basename[idx]}."

        sample = {"id": basename, "speaker": speaker_id, "text": phone, "raw_text": raw_text, "mel": mel,
                  "pitch": pitch, "energy": energy, "duration": duration}

        return sample

    def process_meta(self, filename):
        with open(os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8") as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
            return name, speaker, text, raw_text

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]

        text_lens = np.array([text.shape[0] for text in texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])

        speakers = np.array(speakers)
        texts = pad_1D(texts)
        mels = pad_2D(mels)
        pitches = pad_1D(pitches)
        energies = pad_1D(energies)
        durations = pad_1D(durations)

        return ids, raw_texts, speakers, texts, text_lens, max(text_lens), mels, mel_lens, max(mel_lens), pitches,\
               energies, durations

    def collate_fn(self, data):
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size):]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = [self.reprocess(data, idx) for idx in idx_arr]

        return output


class TextDataset(Dataset):
    def __init__(self, filepath, preprocess_config, train_config):
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        with open(train_config["phones_mapping_path"], "r") as f:
            self.phones_mapping = json.load(f)

        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(filepath)
        with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "speakers.json")) as f:
            self.speaker_map = json.load(f)
        # change to the view: {"0": "LJ039-0161"}
        self.speaker_map = dict(zip(list(self.speaker_map.values()), list(self.speaker_map.keys())))

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker = int(speaker)
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        # phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
        phone = np.array([self.phones_mapping[i] for i in self.text[idx][1:-1].split(" ")])
        return basename, speaker_id, phone, raw_text

    def process_meta(self, filename):
        with open(os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8") as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
            return name, speaker, text, raw_text

    def collate_fn(self, data):
        ids = [d[0] for d in data]
        speakers = np.array([d[1] for d in data])
        texts = [d[2] for d in data]
        raw_texts = [d[3] for d in data]
        text_lens = np.array([text.shape[0] for text in texts])

        texts = pad_1D(texts)

        return ids, raw_texts, speakers, texts, text_lens, max(text_lens)


if __name__ == "__main__":
    # Test
    import yaml
    from torch.utils.data import DataLoader
    from utils.utils import to_device

    preprocess_config = yaml.load(open("./config/LJSpeech/preprocess.yaml", "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open("./config/LJSpeech/train.yaml", "r"), Loader=yaml.FullLoader)
    device = train_config["device"]

    train_dataset = Dataset("train.txt", preprocess_config, train_config, sort=True, drop_last=True)
    val_dataset = Dataset("val.txt", preprocess_config, train_config, sort=False, drop_last=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["optimizer"]["batch_size"] * 4,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config["optimizer"]["batch_size"],
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
    )

    n_batch = 0
    for batchs in train_loader:
        for batch in batchs:
            to_device(batch, device)
            n_batch += 1
    print("Training set  with size {} is composed of {} batches.".format(len(train_dataset), n_batch))

    n_batch = 0
    for batchs in val_loader:
        for batch in batchs:
            to_device(batch, device)
            n_batch += 1
    print("Validation set  with size {} is composed of {} batches.".format(len(val_dataset), n_batch))
