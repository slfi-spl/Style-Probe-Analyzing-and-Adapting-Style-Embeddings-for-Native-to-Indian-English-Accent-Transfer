import random
from phonemizer import phonemize

# ---------------- CONFIG ---------------- #

RAW_TRANSCRIPT_FILE = "SPICOR_Female_LJSpeech/ljspeech_text_with_speaker.txt"
OUTPUT_TRAIN_FILE = "train.txt"
OUTPUT_VAL_FILE = "val.txt"

WAV_PREFIX = "IISc_SPICORProject_EN_F_LJSP_"
WAV_SUFFIX = ".wav"

SPEAKER_ID = "0"
VAL_RATIO = 0.05
RANDOM_SEED = 42

# phonemizer settings (match LJSpeech-style setup)
PHONEMIZER_KWARGS = dict(
    language="en-us",
    backend="espeak",
    strip=True,
    preserve_punctuation=True,
    with_stress=True,
)

# ---------------------------------------- #

def read_transcripts(path):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            utt_id, text = line.split("|", 1)
            pairs.append((utt_id.strip(), text.strip()))
    return pairs


def phonemize_texts(texts):
    phonemes = phonemize(
        texts,
        **PHONEMIZER_KWARGS
    )
    return phonemes


def main():
    random.seed(RANDOM_SEED)

    pairs = read_transcripts(RAW_TRANSCRIPT_FILE)
    random.shuffle(pairs)

    val_size = int(len(pairs) * VAL_RATIO)
    val_pairs = pairs[:val_size]
    train_pairs = pairs[val_size:]

    def write_file(pairs, out_path):
        texts = [text for _, text in pairs]
        phonemes = phonemize_texts(texts)

        with open(out_path, "w", encoding="utf-8") as f:
            for (utt_id, _), ph in zip(pairs, phonemes):
                wav_name = f"{WAV_PREFIX}{utt_id}{WAV_SUFFIX}"
                ph = ph.strip()
                if ph == "":
                    continue  # safety
                f.write(f"{wav_name}|{ph}|{SPEAKER_ID}\n")

    write_file(train_pairs, OUTPUT_TRAIN_FILE)
    write_file(val_pairs, OUTPUT_VAL_FILE)

    print(f"Done.")
    print(f"Train samples: {len(train_pairs)}")
    print(f"Val samples: {len(val_pairs)}")


if __name__ == "__main__":
    main()
