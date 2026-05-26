# Datasets

This repository uses three publicly available English speech datasets for accent analysis and accent adaptation experiments.

---

# 1. LJSpeech

**Source:** https://keithito.com/LJ-Speech-Dataset/

LJSpeech is a single-speaker American English corpus containing approximately 13,000 high-quality recordings spoken by a female speaker.

### Usage in this work

- Source accent (Native American English)
- Base StyleTTS2 pretrained model
- Accent comparison with Indian English speech
- Native reference in accent adaptation experiments

### Citation

```bibtex
@misc{ljspeech17,
  author = {Keith Ito and Linda Johnson},
  title = {The LJ Speech Dataset},
  year = {2017},
  url = {https://keithito.com/LJ-Speech-Dataset/}
}
```

---

# 2. SPICOR TTS 1.0

**Source:** https://spiredatasets.iisc.ac.in/spicortts10

SPICOR TTS 1.0 is a large-scale Indian English speech corpus containing over 97 hours of speech.

For this work, we use the **SPICOR LJSP subset**, which contains recordings aligned with LJSpeech prompts, enabling parallel Native-to-Indian English accent analysis.

### Usage in this work

- Target accent (Indian English)
- Fine-tuning StyleTTS2
- Accent adaptation experiments
- Objective evaluation
- Embedding-space analysis

### Dataset Split

| Split | Utterances |
|---------|---------|
| Train | 4260 |
| Validation | 236 |
| Test | 238 |

### Citation

```bibtex
@misc{spicor2025,
  author = {Anoop Kunchukuttan et al.},
  title = {SPICOR TTS 1.0 Corpus: A 97+ Hour Domain-Rich Indian English TTS Corpus},
  year = {2025},
  note = {Released under CC-BY-4.0}
}
```

---

# 3. IndicTIMIT

IndicTIMIT is a multi-speaker Indian English speech corpus containing speakers influenced by diverse Indian languages.

### Usage in this work

IndicTIMIT is used exclusively for accent separability analysis of StyleTTS2 embeddings.

### Accents Analyzed

- Bengali
- Gujarati
- Hindi
- Kannada
- Malayalam
- Marathi
- Odia
- Punjabi
- Tamil
- Telugu

### Citation

```bibtex
@inproceedings{indicTIMIT2019,
  author = {Chandra Yarra and Rohan Aggarwal and Ayush Rajpal and Preeti Rao Ghosh},
  title = {Indic TIMIT and Indic English Lexicon: A Speech Database of Indian Speakers Using TIMIT Stimuli and a Lexicon from Their Mispronunciations},
  booktitle = {Oriental COCOSDA},
  year = {2019}
}
```

---

# Data Preparation

The repository does not distribute any dataset files.

Users must independently obtain the datasets and organize them according to the paths specified in the configuration files.

The following preprocessing scripts are provided:

```bash
python src/data/resample.py

python src/data/create_metadata.py

python src/data/make_phonemized_metadata.py
```

Audio is resampled to 24 kHz before training and evaluation.

---

# Dataset Availability

| Dataset | Distributed in Repository |
|----------|----------|
| LJSpeech | No |
| SPICOR TTS 1.0 | No |
| IndicTIMIT | No |

Users are responsible for complying with the respective dataset licenses and usage restrictions.