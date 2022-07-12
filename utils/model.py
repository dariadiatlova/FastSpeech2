from typing import Optional, Dict

from torch.utils.data import DataLoader

from dataset import Dataset


def get_dataloader(config: Dict, train: bool = True, to_synthesis: Optional[int] = None):
    txt_name = "train.txt" if train else "val.txt"
    shuffle = True if train else False
    dataset = Dataset(filename=txt_name, preprocess_config=config, synthesis_size=to_synthesis)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=shuffle, num_workers=16,
                        collate_fn=dataset.collate_fn, pin_memory=True)
    return loader
