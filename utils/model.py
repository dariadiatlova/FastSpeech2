from typing import Optional, Dict

from torch.utils.data import DataLoader

from dataset import Dataset


def get_dataloader(config: Dict, train: bool = True):
    txt_name = "train.txt" if train else "val.txt"
    shuffle = True if train else False
    sort = True if train else False
    batch_size = config["batch_size"]
    dataset = Dataset(filename=txt_name, preprocess_config=config, sort=sort, batch_size=batch_size)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn,
                        shuffle=shuffle, num_workers=16, pin_memory=True, drop_last=False)
    return loader
