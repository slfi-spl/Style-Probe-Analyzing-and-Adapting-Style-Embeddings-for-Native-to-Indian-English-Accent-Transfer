# =========================
# Standard imports
# =========================
import os
# os.environ["NUMBA_DISABLE_JIT"] = "1"

import yaml
import numpy as np
import torch
import torchaudio
# import librosa
import scipy.io as sio
from tqdm import tqdm
from collections import Counter

# =========================
# StyleTTS2 imports
# =========================
from models import *
from utils import *
from text_utils import TextCleaner
from Utils.PLBERT.util import load_plbert

import phonemizer

# =========================
# PATHS
# =========================
INDIC_ROOT = "IndicTIMIT"
TRANSCRIPT_FILE = os.path.join(INDIC_ROOT, "transcripts_L2.txt")

OUTPUT_MAT = "indic_style_full_ft.mat"

CONFIG_PATH = "Models/SPICOR_ft_full/config_full.yml"
CHECKPOINT_PATH = "Models/SPICOR_ft_full/epoch_2nd_00009.pth"
PLBERT_DIR = "Utils/PLBERT"

# =========================
# AUDIO PARAMS (from config)
# =========================
SAMPLE_RATE = 24000
N_MELS = 80

to_mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_mels=N_MELS,
    n_fft=2048,
    win_length=1200,
    hop_length=300
)

MEAN, STD = -4.0, 4.0

def preprocess_audio(wav):
    wav = torch.from_numpy(wav).float()
    mel = to_mel(wav)
    mel = (torch.log(1e-5 + mel.unsqueeze(0)) - MEAN) / STD
    return mel

# =========================
# DEVICE
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# TEXT PROCESSING
# =========================
textcleaner = TextCleaner()

global_phonemizer = phonemizer.backend.EspeakBackend(
    language="en-us",
    preserve_punctuation=True,
    with_stress=True
)

# =========================
# LOAD MODEL
# =========================
print("Loading model...")

config = yaml.safe_load(open(CONFIG_PATH))

text_aligner = load_ASR_models(
    config["ASR_path"], config["ASR_config"]
)
pitch_extractor = load_F0_models(config["F0_path"])
plbert = load_plbert(PLBERT_DIR)

model = build_model(
    recursive_munch(config["model_params"]),
    text_aligner,
    pitch_extractor,
    plbert
)

ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")["net"]

for k in model:
    if k in ckpt:
        model[k].load_state_dict(ckpt[k], strict=False)

for k in model:
    model[k].eval()
    model[k].to(device)

print("Model ready.")

# =========================
# LOAD TRANSCRIPTS
# =========================
print("Loading transcripts...")

transcript_dict = {}

with open(TRANSCRIPT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        key, value = line.split("|", 1)
        transcript_dict[key.strip()] = value.strip()

print("Total transcripts:", len(transcript_dict))

# =========================
# BUILD METADATA
# =========================
print("Building metadata...")

languages = sorted([
    d for d in os.listdir(INDIC_ROOT)
    if os.path.isdir(os.path.join(INDIC_ROOT, d))
])

filename_list = []
text_list = []
language_list = []

for lang in languages:
    lang_path = os.path.join(INDIC_ROOT, lang)
    wav_files = sorted([
        f for f in os.listdir(lang_path)
        if f.endswith(".wav")
    ])

    if len(wav_files) != 195:
        raise RuntimeError(f"{lang} does not contain 195 files.")

    for wav_file in wav_files:
        fname = wav_file.replace(".wav", "")
        if fname not in transcript_dict:
            raise RuntimeError(f"Missing transcript for {fname}")

        filename_list.append(fname)
        text_list.append(transcript_dict[fname])
        language_list.append(lang)

N = len(filename_list)
print("Total samples:", N)

if N != 195 * 10:
    raise RuntimeError("Total sample count is incorrect.")

# =========================
# ALLOCATE STORAGE
# =========================
sa_all = np.zeros((N, 128), dtype=np.float32)
sp_all = np.zeros((N, 128), dtype=np.float32)
s_all  = np.zeros((N, 256), dtype=np.float32)

htext_all = [None] * N
hbert_all = [None] * N

# =========================
# SAFE OBJECT ARRAY
# =========================
def safe_object_array(lst, feat_dim):
    out = []
    for x in lst:
        if x is None:
            out.append(np.zeros((0, feat_dim), dtype=np.float32))
        else:
            out.append(x)
    return np.array(out, dtype=object)

# =========================
# EXTRACTION
# =========================
@torch.no_grad()
def extract_one(idx):
    wav_path = os.path.join(
        INDIC_ROOT,
        language_list[idx],
        filename_list[idx] + ".wav"
    )

    wav, sr = torchaudio.load(wav_path)

    # Convert to mono if needed
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)

    # Resample safely
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        wav = resampler(wav)

    wav = wav.squeeze(0).numpy()


    mel = preprocess_audio(wav).to(device)

    sa = model.style_encoder(mel.unsqueeze(1)).squeeze(0)
    sp = model.predictor_encoder(mel.unsqueeze(1)).squeeze(0)
    s = torch.cat([sa, sp], dim=0)

    phonemes = global_phonemizer.phonemize([text_list[idx]])[0]
    phoneme_ids, _ = textcleaner(phonemes)
    
    print(type(phoneme_ids))
    print(phoneme_ids[:10])
    
    tokens = torch.LongTensor(phoneme_ids).unsqueeze(0).to(device)

    lengths = torch.tensor([tokens.shape[1]]).to(device)
    mask = model.text_encoder.length_to_mask(lengths)

    htext = model.text_encoder(tokens, lengths, mask)
    htext = htext.squeeze(0).transpose(0, 1).cpu().numpy()

    bert = model.bert(tokens, attention_mask=~mask)
    hbert = model.bert_encoder(bert)
    hbert = hbert.squeeze(0).cpu().numpy()

    return sa.cpu().numpy(), sp.cpu().numpy(), s.cpu().numpy(), htext, hbert

# =========================
# MAIN LOOP
# =========================
print("Starting extraction...")

for i in tqdm(range(N)):
    sa, sp, s, htext, hbert = extract_one(i)

    sa_all[i] = sa
    sp_all[i] = sp
    s_all[i]  = s
    htext_all[i] = htext
    hbert_all[i] = hbert

# =========================
# SAVE MAT FILE
# =========================
print("Saving MAT file...")

out = {}
out["filename"] = np.array(filename_list, dtype=object)
out["text"] = np.array(text_list, dtype=object)
out["language"] = np.array(language_list, dtype=object)
out["sa"] = sa_all
out["sp"] = sp_all
out["s"] = s_all
out["htext"] = safe_object_array(htext_all, 512)
out["hbert"] = safe_object_array(hbert_all, 512)

sio.savemat(OUTPUT_MAT, out)

print("Saved:", OUTPUT_MAT)

# =========================
# VERIFICATION
# =========================
print("Verifying...")

mat = sio.loadmat(OUTPUT_MAT)

print("Total entries:", len(mat["filename"]))

langs = [l[0] for l in mat["language"].squeeze()]
print("Language counts:", Counter(langs))

print("Done.")
