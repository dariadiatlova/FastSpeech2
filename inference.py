import soundfile as sf
import hydra

from model.lightning_model import FastSpeechLightning
from utils.model import get_dataloader
from utils.tools import torch_from_numpy


@hydra.main(version_base=None, config_path="config/VCTK", config_name="train")
def run(cfg):
    model_weights = cfg.checkpoint_path
    model = FastSpeechLightning.load_from_checkpoint(checkpoint_path=model_weights, config=cfg)
    model.to(cfg.preprocess.device).eval()
    loader = get_dataloader(cfg.preprocess, train=False)
    dataset = [i for i in loader][0][0]
    dataset = torch_from_numpy(dataset)
    speakers, texts, text_lens, max_text_lens = dataset[2:6]
    output = model(speakers.cuda(), texts.cuda(), text_lens.cuda(), max_text_lens)
    sf.write(cfg.filepath, output, cfg.preprocess.audio.sampling_rate)


if __name__ == "__main__":
    run()
