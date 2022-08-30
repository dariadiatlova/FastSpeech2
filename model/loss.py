import torch
import torch.nn as nn
import lpips


class FastSpeech2Loss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self, preprocess_config):
        super(FastSpeech2Loss, self).__init__()
        self.pitch_feature_level = preprocess_config["pitch"]["feature"]
        self.energy_feature_level = preprocess_config["energy"]["feature"]
        self.batch_size = preprocess_config["batch_size"]
        self.n_mels = preprocess_config["mel"]["n_mel_channels"]
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.lpips_loss = lpips.LPIPS(net='alex')
        self.scale = preprocess_config["scale"]

    def forward(self, device, inputs, predictions, compute_mel_loss: bool = True):
        mel_targets = inputs[7]
        pitch_targets, energy_targets, duration_targets = inputs[10:]
        mel_predictions = predictions[0]
        postnet_mel_predictions, pitch_predictions, energy_predictions, log_duration_predictions = predictions[1:5]
        src_masks, mel_masks = predictions[6:8]
        src_masks = ~src_masks
        mel_masks = ~mel_masks
        log_duration_targets = torch.log(duration_targets.float() + 1)
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]

        log_duration_targets.requires_grad = False
        pitch_targets.requires_grad = False
        energy_targets.requires_grad = False
        mel_targets.requires_grad = False

        pitch_targets = pitch_targets.to(device)
        energy_targets = energy_targets.to(device)
        log_duration_targets = log_duration_targets.to(device)
        mel_targets = mel_targets.to(device)

        if self.pitch_feature_level == "phoneme_level":
            pitch_predictions = pitch_predictions.masked_select(src_masks).to(device)
            pitch_targets = pitch_targets.masked_select(src_masks.to(device))
        elif self.pitch_feature_level == "frame_level":
            pitch_predictions = pitch_predictions.masked_select(mel_masks).to(device)
            pitch_targets = pitch_targets.masked_select(mel_masks.to(device))

        if self.energy_feature_level == "phoneme_level":
            energy_predictions = energy_predictions.masked_select(src_masks).to(device)
            energy_targets = energy_targets.masked_select(src_masks.to(device))
        if self.energy_feature_level == "frame_level":
            energy_predictions = energy_predictions.masked_select(mel_masks)
            energy_targets = energy_targets.masked_select(mel_masks.to(device))

        log_duration_predictions = log_duration_predictions.masked_select(src_masks)
        log_duration_targets = log_duration_targets.masked_select(src_masks)

        pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)
        energy_loss = self.mse_loss(energy_predictions, energy_targets)
        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)

        if not compute_mel_loss:
            return pitch_loss, energy_loss, duration_loss

        # reshape mask to 3d size -> normalize mels to [-1, 1] -> change pad values to 0
        mask3d = mel_masks.unsqueeze(2).expand(-1, -1, self.n_mels)

        mel_predicted_lpips = mel_predictions
        mel_predicted_lpips = 2 * (mel_predicted_lpips - torch.min(mel_predicted_lpips)) / \
                              (torch.max(mel_predicted_lpips) - torch.min(mel_predicted_lpips)) - 1
        mel_predicted_lpips[~mask3d] = 0

        mel_target_lpips = mel_targets
        mel_target_lpips = 2 * (mel_target_lpips - torch.min(mel_target_lpips)) / \
                           (torch.max(mel_target_lpips) - torch.min(mel_target_lpips)) - 1
        mel_target_lpips[~mask3d] = 0

        mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1)) # b, t, 1 -> b, t, c
        postnet_mel_predictions = postnet_mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))
        mel_loss = self.mae_loss(mel_predictions, mel_targets)
        lpips_loss = torch.mean(self.lpips_loss(mel_predicted_lpips.unsqueeze(1), mel_target_lpips.unsqueeze(1))) * self.scale
        postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)

        total_loss = mel_loss + postnet_mel_loss + duration_loss + pitch_loss + energy_loss + lpips_loss

        return total_loss, mel_loss, postnet_mel_loss, pitch_loss, energy_loss, duration_loss, lpips_loss
