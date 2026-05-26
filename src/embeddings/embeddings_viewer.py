import scipy.io as sio
import numpy as np

MAT_PATH = "/media/nayan/g/harshitha/interns/namrata/StyleTTS2/SPICOR/libri_accent_spicor.mat"

data = sio.loadmat(MAT_PATH)

idx = 0  # debug only first utterance

# ---------- Correct extraction ----------
filename = data["filename"][0, idx][0]
text     = data["text"][0, idx][0]
speaker  = data["speaker"][0, idx][0]

sa = data["sa"][idx]
sp = data["sp"][idx]
s  = data["s"][idx]

htext = data["htext"][0, idx]
hbert = data["hbert"][0, idx]

print("\n=========== UTTERANCE DEBUG (MAT FILE) ===========")
print(f"Index        : {idx}")
print(f"Filename     : {filename}")
print(f"Speaker      : {speaker}")
print(f"Text         : {text}")
print(f"No of Words       : {len(text.split())}")
print(f"No of Characters       : {len(text)}")

print("\n----------- EMBEDDING SHAPES -----------")
print(f"sa shape     : {sa.shape} ")
print(f"sp shape     : {sp.shape} ")
print(f"s shape      : {s.shape}  ")

print(f"htext shape  : {htext.shape} ")
print(f"hbert shape  : {hbert.shape} ")

print("\n----------- PHONEME CONFIRMATION -----------")
print(f"No of Phonemes (from htext): {htext.shape[0]}")
print(f"No of Phonemes (from hbert): {hbert.shape[0]}")

print("\n----------- SAMPLE VECTORS -----------")
print("First phoneme htext vector (first 5 dims):")
print(htext[0][:5])

print("\nFirst phoneme hbert vector (first 5 dims):")
print(hbert[0][:5])

print("\nvector sa (first 5 dims):")
print(sa[:5])

print("\nvector sp (first 5 dims):")
print(sp[:5])

print("==============================================\n")
