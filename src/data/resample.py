import os
import torchaudio
from pathlib import Path
from tqdm import tqdm

# ============================
# CONFIG
# ============================

INPUT_ROOT = "/media/nayan/g/harshitha/interns/namrata/StyleTTS2/SPICOR_LJ_22k"
OUTPUT_ROOT = "/media/nayan/g/harshitha/interns/namrata/StyleTTS2/SPICOR_LJ_24k"
TARGET_SR = 24000

# ============================

def resample_file(input_path, output_path):
    waveform, sr = torchaudio.load(input_path)

    if sr != TARGET_SR:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
        waveform = resampler(waveform)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(output_path, waveform, TARGET_SR)


def main():
    print("===== RESAMPLING TO 24kHz =====")

    input_root = Path(INPUT_ROOT)
    output_root = Path(OUTPUT_ROOT)

    wav_files = list(input_root.rglob("*.wav"))

    print(f"Found {len(wav_files)} wav files.")

    for wav in tqdm(wav_files):
        relative_path = wav.relative_to(input_root)
        output_path = output_root / relative_path
        resample_file(wav, output_path)

    print("===== DONE =====")


if __name__ == "__main__":
    main()
