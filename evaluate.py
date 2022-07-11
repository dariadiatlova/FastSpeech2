import argparse

import torch
import yaml
from torch.utils.data import DataLoader

from dataset import Dataset, TextDataset
from model import FastSpeech2Loss
from utils.model import get_model
from utils.tools import to_device, synth_one_sample


def evaluate(model, step, total_step, configs, device, vocoder=None):
    preprocess_config, model_config, train_config, wandb_config = configs
    # Get dataset
    data_to_return = []
    dataset = Dataset("val.txt", preprocess_config, train_config, sort=False, drop_last=False)
    text_dataset = TextDataset("val.txt", preprocess_config, train_config)
    batch_size = train_config["optimizer"]["batch_size"]
    val_text_loader = DataLoader(text_dataset, batch_size=batch_size, collate_fn=text_dataset.collate_fn)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn)

    # Get loss function
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)

    # Evaluation
    loss_sums = [0 for _ in range(6)]
    for batchs, text_batch in zip(loader, val_text_loader):
        for batch in batchs:
            batch = to_device(batch, device)
            text_batch = to_device(text_batch, device)
            with torch.no_grad():
                # Forward
                output_target_durations = model(*(batch[2:]))
                output_predicted_durations = model(*(batch[2:6]))

                # Cal Loss
                losses = Loss(batch, output_target_durations)

                for i in range(len(losses)):
                    loss_sums[i] += losses[i].item() * len(batch[0])

    loss_means = [loss_sum / len(dataset) for loss_sum in loss_sums]

    message1 = f"Validation Step {step}/{total_step}"
    message2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, " \
               "Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(*[l for l in loss_means])
    message = message1 + message2 + "\n"

    fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(batch, output_predicted_durations,
                                                                    vocoder, preprocess_config)

    if loss_means is not None:
        data_to_return.append(loss_means)

    if fig is not None:
        data_to_return.append(fig)

    if wav_reconstruction is not None:
        data_to_return.append(wav_reconstruction)

    if wav_prediction is not None:
        data_to_return.append(wav_prediction)

    if tag is not None:
        data_to_return.append(tag)

    return message, data_to_return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=30000)
    parser.add_argument("-p", "--preprocess_config", type=str, required=True, help="path to preprocess.yaml")
    parser.add_argument("-m", "--model_config", type=str, required=True, help="path to model.yaml")
    parser.add_argument( "-t", "--train_config", type=str, required=True, help="path to train.yaml")

    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(open(args.preprocess_config, "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    device = train_config["device"]
    model = get_model(args, configs, device, train=False).to(device)

    message = evaluate(model, args.restore_step, configs, device)
    print(message)
