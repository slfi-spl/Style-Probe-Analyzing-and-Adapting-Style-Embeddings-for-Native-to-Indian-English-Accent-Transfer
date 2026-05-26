import os
import shutil

# ==============================
# PATHS (edit if needed)
# ==============================

BASE_DIR = os.getcwd()

NATIVE_SRC = os.path.join(BASE_DIR, "SPICOR", "LJ_Native")
INDIAN_SRC = os.path.join(BASE_DIR, "SPICOR_LJ_24k")

DEST_BASE = os.path.join(BASE_DIR, "Results_ft")
DEST_NATIVE = os.path.join(DEST_BASE, "gt_native_lj")
DEST_INDIAN = os.path.join(DEST_BASE, "gt_indian_spicor")

NUM_FILES = 40

# ==============================
# CREATE DESTINATION FOLDERS
# ==============================

os.makedirs(DEST_NATIVE, exist_ok=True)
os.makedirs(DEST_INDIAN, exist_ok=True)

print("Destination folders ready.")

# ==============================
# HELPER: extract LJ ID
# ==============================

def extract_lj_id(filename):
    """
    Extracts LJ ID like LJ001-0003 from:
    IISC_SPICORProject_EN_F_LJSP_LJ001-0003.wav
    """
    base = os.path.basename(filename)
    parts = base.split("_")
    
    for p in parts:
        if p.startswith("LJ") and "-" in p:
            return p.replace(".wav", "")
    
    return None


# ==============================
# PROCESS NATIVE
# ==============================

native_files = sorted([f for f in os.listdir(NATIVE_SRC) if f.endswith(".wav")])[:NUM_FILES]

print(f"Copying {len(native_files)} native LJ files...")

for file in native_files:
    src_path = os.path.join(NATIVE_SRC, file)
    lj_id = extract_lj_id(file)
    
    if lj_id is None:
        print(f"Skipping {file}, no LJ ID found.")
        continue
    
    dest_filename = f"{lj_id}.wav"
    dest_path = os.path.join(DEST_NATIVE, dest_filename)
    
    shutil.copy(src_path, dest_path)

print("Native files copied.")


# ==============================
# PROCESS INDIAN SPICOR
# ==============================

indian_files = sorted([f for f in os.listdir(INDIAN_SRC) if f.endswith(".wav")])[:NUM_FILES]

print(f"Copying {len(indian_files)} Indian SPICOR files...")

for file in indian_files:
    src_path = os.path.join(INDIAN_SRC, file)
    lj_id = extract_lj_id(file)
    
    if lj_id is None:
        print(f"Skipping {file}, no LJ ID found.")
        continue
    
    dest_filename = f"{lj_id}.wav"
    dest_path = os.path.join(DEST_INDIAN, dest_filename)
    
    shutil.copy(src_path, dest_path)

print("Indian SPICOR files copied.")

print("Done.")
