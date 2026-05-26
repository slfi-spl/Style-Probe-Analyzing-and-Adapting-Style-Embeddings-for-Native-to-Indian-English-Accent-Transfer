import os
import time
import yaml
import torch
import random
import numpy as np
import librosa
import torchaudio
from munch import Munch
from nltk.tokenize import word_tokenize
import soundfile as sf

from models import *
from utils import *
from text_utils import TextCleaner
from Utils.PLBERT.util import load_plbert
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
import phonemizer


# -------------------------------------------------
# Reproducibility
# -------------------------------------------------
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
random.seed(0)
np.random.seed(0)


# -------------------------------------------------
# Device
# -------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


# -------------------------------------------------
# Utility Functions
# -------------------------------------------------

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(
        lengths.shape[0], -1
    ).type_as(lengths)
    mask = torch.gt(mask + 1, lengths.unsqueeze(1))
    return mask


# Mel
to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300
)
mean, std = -4, 4


def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor


# -------------------------------------------------
# Load Model
# -------------------------------------------------

def load_styletts2(config_path, checkpoint_path):
    print("Loading config...")
    config = yaml.safe_load(open(config_path))

    # Load ASR
    ASR_config = config.get("ASR_config", False)
    ASR_path = config.get("ASR_path", False)
    text_aligner = load_ASR_models(ASR_path, ASR_config)

    # Load F0
    F0_path = config.get("F0_path", False)
    pitch_extractor = load_F0_models(F0_path)

    # Load BERT
    BERT_path = config.get("PLBERT_dir", False)
    plbert = load_plbert(BERT_path)

    # Build model
    model = build_model(
        recursive_munch(config["model_params"]),
        text_aligner,
        pitch_extractor,
        plbert,
    )

    model = {k: v.to(device).eval() for k, v in model.items()}

    print("Loading checkpoint...")
    params_whole = torch.load(checkpoint_path, map_location="cpu")
    params = params_whole["net"]

    for key in model:
        if key in params:
            print(f"{key} loaded")
            try:
                model[key].load_state_dict(params[key])
            except:
                from collections import OrderedDict
                state_dict = params[key]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]
                    new_state_dict[name] = v
                model[key].load_state_dict(new_state_dict, strict=False)

    return model


# -------------------------------------------------
# Sampler
# -------------------------------------------------

def build_sampler(model):
    sampler = DiffusionSampler(
        model["diffusion"].diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(
            sigma_min=0.0001,
            sigma_max=3.0,
            rho=9.0,
        ),
        clamp=False,
    )
    return sampler


# -------------------------------------------------
# Inference
# -------------------------------------------------

textcleaner = TextCleaner()
global_phonemizer = phonemizer.backend.EspeakBackend(
    language="en-us",
    preserve_punctuation=True,
    with_stress=True,
)


def inference(model, sampler, text,
              diffusion_steps=5,
              embedding_scale=1):

    text = text.strip().replace('"', '')
    ps = global_phonemizer.phonemize([text])
    ps = word_tokenize(ps[0])
    ps = " ".join(ps)

    tokens = textcleaner(ps)

    if isinstance(tokens, tuple):
        tokens = tokens[0]

    tokens = list(tokens)
    tokens = [0] + tokens
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)
    print(type(tokens))
    print(tokens)


    with torch.no_grad():

        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = length_to_mask(input_lengths).to(device)

        t_en = model["text_encoder"](tokens, input_lengths, text_mask)

        bert_dur = model["bert"](tokens, attention_mask=(~text_mask).int())
        d_en = model["bert_encoder"](bert_dur).transpose(-1, -2)

        noise = torch.randn(1, 1, 256).to(device)

        s_pred = sampler(
            noise,
            embedding=bert_dur[0].unsqueeze(0),
            num_steps=diffusion_steps,
            embedding_scale=embedding_scale,
        ).squeeze(0)

        s = s_pred[:, 128:]
        ref = s_pred[:, :128]

        d = model["predictor"].text_encoder(
            d_en, s, input_lengths, text_mask
        )

        x, _ = model["predictor"].lstm(d)
        duration = model["predictor"].duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)

        pred_dur[-1] += 5

        pred_aln_trg = torch.zeros(
            input_lengths, int(pred_dur.sum().item())
        ).to(device)

        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i])] = 1
            c_frame += int(pred_dur[i])

        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0))

        F0_pred, N_pred = model["predictor"].F0Ntrain(en, s)

        out = model["decoder"](
            (t_en @ pred_aln_trg.unsqueeze(0)),
            F0_pred,
            N_pred,
            ref.squeeze().unsqueeze(0),
        )

    return out.squeeze().cpu().numpy()


# -------------------------------------------------
# Main
# -------------------------------------------------

if __name__ == "__main__":

    CONFIG_PATH = "Models/SPICOR_ft_full/config_full.yml"
    CHECKPOINT_PATH = "Models/SPICOR_ft_full/epoch_2nd_00009.pth"
    OUTPUT_DIR = "Results_ft/misc"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = load_styletts2(CONFIG_PATH, CHECKPOINT_PATH)
    sampler = build_sampler(model)

    texts = {
        "exp_01": "The quick brown fox jumps over the lazy dog.",
        "exp_02": "She sells seashells by the seashore.",
        "exp_03": "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
        "exp_04": "Peter Piper picked a peck of pickled peppers.",
        "exp_05": "I scream, you scream, we all scream for ice cream.",
        "exp_06": "A journey of a thousand miles begins with a single step.",
        "exp_07": "To be or not to be, that is the question.",
        "exp_08": "All that glitters is not gold.",
        "exp_09": "The pen is mightier than the sword.",
        "exp_10": "When in Rome, do as the Romans do."
    }


    for name, text in texts.items():

        print(f"Synthesizing: {name}")

        start = time.time()
        wav = inference(
            model,
            sampler,
            text,
            diffusion_steps=10,
            embedding_scale=1.5,
        )
        rtf = (time.time() - start) / (len(wav) / 24000)
        print("RTF:", rtf)

        save_path = os.path.join(OUTPUT_DIR, f"{name}.wav")
        sf.write(save_path, wav, 24000)

    print("Done. Files saved to:", OUTPUT_DIR)
