# Methodology

## Overview

This repository investigates how accent information is represented and modified within the style embeddings of StyleTTS2.

The work consists of three stages:

1. Accent separability analysis in pretrained StyleTTS2 embeddings.
2. Native-to-Indian English accent adaptation.
3. Embedding-space analysis before and after adaptation.

The overall goal is to determine whether accent-related information is encoded primarily in the acoustic embedding, the prosodic embedding, or both.

---

# Style Representation in StyleTTS2

StyleTTS2 represents speaking style using two independent embeddings:

- Prosodic Embedding (128 dimensions)
- Acoustic Embedding (128 dimensions)

These embeddings are concatenated to form the final style vector:

```text
Style Vector = [Prosodic Embedding, Acoustic Embedding]
```

The prosodic embedding conditions:

- Duration
- Pitch
- Energy

The acoustic embedding conditions:

- Spectral characteristics
- Timbre
- Vowel formants
- Acoustic realization of speech

---

# Stage 1: Accent Separability Analysis

We first examine whether accent information is encoded in the pretrained StyleTTS2 style embeddings.

## Datasets

- LJSpeech
- SPICOR TTS 1.0
- IndicTIMIT

## Pipeline

### Step 1: Embedding Extraction

Style embeddings are extracted using:

```bash
python src/embeddings/extracting_embeddings_interspeech.py
```

### Step 2: Preprocessing

The extracted embeddings are:

1. Checked for invalid values
2. Clipped when necessary
3. L2 normalized
4. Reduced using PCA

### Step 3: Clustering Analysis

The processed embeddings are analyzed using:

- Principal Component Analysis (PCA)
- t-SNE visualization
- Silhouette scores

The corresponding notebooks are:

```text
notebooks/Clustering_SPICOR_accent.ipynb

notebooks/IndicTIMIT_Analysis.ipynb
```

---

# Stage 2: Accent Adaptation

We study Native-to-Indian English accent adaptation using StyleTTS2.

## Fine-Tuning (FT)

The pretrained StyleTTS2 model is fine-tuned on the SPICOR LJSP subset.

Configuration:

```text
configs/config_ft_full_slm.yml
```

Training script:

```bash
python src/training/train_ft_full_slm.py
```

All model parameters are updated during fine-tuning.

---

## Style Vector Substitution (SVS)

The pretrained model remains frozen.

During inference:

1. A style vector is generated using the baseline model.
2. A style vector is generated using the fine-tuned model.
3. The fine-tuned style vector is substituted into the synthesis pipeline.

This allows accent transfer through style manipulation without modifying the acoustic model parameters.

---

# Stage 3: Evaluation

## Subjective Evaluation

Accent similarity is evaluated using a listening test.

Listeners compare synthesized speech against:

- Native English references
- Indian English references

and rate accent similarity on a five-point scale.

---

## Objective Evaluation

The following objective metrics are computed:

### Acoustic Metrics

- Mel-Cepstral Distortion (MCD)
- Log-F0 RMSE
- Energy RMSE
- Duration Mean Absolute Deviation (Duration MAD)

### Pronunciation Metrics

- Vowel Formant Distance
- PPG-DTW

Evaluation code is provided in:

```text
notebooks/Objective_and_embedding_analysis.ipynb
```

---

# Embedding Space Analysis

To understand how adaptation affects the internal representation, we compare embeddings extracted from:

- Baseline StyleTTS2
- Fine-Tuned StyleTTS2

The following embeddings are analyzed independently:

- Full Style Embedding
- Acoustic Embedding
- Prosodic Embedding

The comparison includes:

- PCA visualization
- Centroid displacement
- Cluster separation analysis

The notebook used for this analysis is:

```text
notebooks/four_models_comparison.ipynb
```

---

# Expected Outputs

The repository can reproduce:

- Accent clustering visualizations
- PCA projections
- t-SNE projections
- Silhouette scores
- Accent adaptation metrics
- Embedding-space comparisons

These analyses support the findings reported in the accompanying paper.