import argparse
import os

import torch
import torch.nn as nn
import wandb
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Dataset
from evaluate import evaluate
from model import FastSpeech2Loss
from utils.model import get_model, get_vocoder, get_param_num
from utils.tools import to_device, synth_one_sample


def main(args, configs):
    print("Prepare training ...")
    preprocess_config, model_config, train_config, wandb_config = configs
    device = train_config["device"]
    run = wandb.init(project=wandb_config["project"])

    # Get dataset
    dataset = Dataset("train.txt", preprocess_config, train_config, sort=True, drop_last=True)
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 4  # Set this larger than 1 to enable sorting in Dataset
    assert batch_size * group_size < len(dataset)
    loader = DataLoader(dataset, batch_size=batch_size * group_size, shuffle=True, collate_fn=dataset.collate_fn)

    # Prepare model
    model, optimizer = get_model(args, configs[:-1], device, train=True)
    model = nn.DataParallel(model)

    wandb.watch(model, log_freq=wandb_config["log_every_n_steps"])
    num_param = get_param_num(model)
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)
    print("Number of FastSpeech2 Parameters:", num_param)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Init logger
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)

    # Training
    step = args.restore_step + 1
    epoch = 1
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]

    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = args.restore_step
    outer_bar.update()

    while True:
        inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
        for batchs in loader:
            for batch in batchs:
                batch = to_device(batch, device)

                # Forward
                output = model(*(batch[2:]))

                # Cal Loss
                losses = Loss(batch, output)
                total_loss = losses[0]

                # Backward
                total_loss = total_loss / grad_acc_step
                total_loss.backward()
                if step % grad_acc_step == 0:
                    # Clipping gradients to avoid gradient explosion
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)

                    # Update weights
                    optimizer.step_and_update_lr()
                    optimizer.zero_grad()

                if step % log_step == 0:
                    losses = [l.item() for l in losses]
                    message1 = "Step {}/{}, ".format(step, total_step)
                    message2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, " \
                               "Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(*losses)

                    with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                        f.write(message1 + message2 + "\n")

                    outer_bar.write(message1 + message2)
                    wandb.log({"Train Total_loss": losses[0]})
                    wandb.log({"Train Mel_loss": losses[1]})
                    wandb.log({"Train Mel_postnet_loss": losses[2]})
                    wandb.log({"Train Pitch_loss": losses[3]})
                    wandb.log({"Train Energy_loss": losses[4]})
                    wandb.log({"Train Duration_loss": losses[5]})

                if step % synth_step == 0:
                    fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(batch, output, vocoder,
                                                                                    model_config, preprocess_config)
                    if fig is not None:
                        images = wandb.Image(fig, caption="Training/step_{}_{}".format(step, tag))
                        wandb.log({"train_spectrograms": images})

                    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
                    if wav_reconstruction is not None:

                        reconstructed_audio = wandb.Audio(wav_reconstruction,
                                                          caption="Training/step_{}_{}".format(step, tag),
                                                          sample_rate=sampling_rate)
                        wandb.log({"train_wav_reconstruction": reconstructed_audio})

                    if wav_prediction is not None:
                        predicted_audio = wandb.Audio(wav_prediction,
                                                      caption="Training/step_{}_{}".format(step, tag),
                                                      sample_rate=sampling_rate)
                        wandb.log({"train_wav_predicted": predicted_audio})

                if step % val_step == 0:
                    model.eval()
                    message, logged_data = evaluate(model, step, total_step, configs, device, None, vocoder)
                    val_losses = logged_data[0]
                    val_fig, val_wav_reconstruction, val_wav_prediction = logged_data[1], logged_data[2], logged_data[3]
                    tag = logged_data[4]
                    with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                        f.write(message + "\n")
                    outer_bar.write(message)
                    wandb.log({"Val_total_loss": val_losses[0]})
                    wandb.log({"Train_mel_loss": val_losses[1]})
                    wandb.log({"Train_mel_postnet_loss": val_losses[2]})
                    wandb.log({"Train_pitch_loss": val_losses[3]})
                    wandb.log({"Train_energy_loss": val_losses[4]})
                    wandb.log({"Train_duration_loss": val_losses[5]})
                    # "gt/..."
                    # "rec/..."
                    # "gen/..."

                    if val_fig is not None:
                        images = wandb.Image(val_fig, caption="Val/step_{}_{}".format(step, tag))
                        wandb.log({"val_spectrograms": images})

                    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]

                    if val_wav_reconstruction is not None:
                        reconstructed_audio = wandb.Audio(val_wav_reconstruction,
                                                          caption="Val/step_{}_{}".format(step, tag),
                                                          sample_rate=sampling_rate)
                        wandb.log({"val_wav_reconstruction": reconstructed_audio})

                    if val_wav_prediction is not None:
                        predicted_audio = wandb.Audio(val_wav_prediction,
                                                      caption="Val/step_{}_{}".format(step, tag),
                                                      sample_rate=sampling_rate)
                        wandb.log({"val_wav_predicted": predicted_audio})

                    model.train()

                if step % save_step == 0:
                    model_save_path = os.path.join(train_config["path"]["ckpt_path"], "{}.pth.tar".format(step))
                    torch.save({"model": model.state_dict(),
                                "optimizer": optimizer._optimizer.state_dict()},
                               model_save_path)
                    artifact = wandb.Artifact('model', type='model')
                    artifact.add_file(model_save_path)
                    run.log_artifact(artifact)

            if step == total_step:
                quit()
            step += 1
            outer_bar.update(1)

        inner_bar.update(1)
        epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument("-p", "--preprocess_config", type=str, required=True, help="path to preprocess.yaml")
    parser.add_argument("-m", "--model_config", type=str, required=True, help="path to model.yaml")
    parser.add_argument("-t", "--train_config", type=str, required=True, help="path to train.yaml")
    parser.add_argument("-w", "--wandb_config", type=str, required=True, help="path to wandb.yaml")
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(open(args.preprocess_config, "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    wandb_config = yaml.load(open(args.wandb_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config, wandb_config)

    main(args, configs)
