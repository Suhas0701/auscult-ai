import os
import glob
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

CACHE_FILE = "dsp-ml/data/features_cache.npz"
TEST_VEC_DIR = "dsp-ml/test_vectors"
FIG_DIR = "docs/figures"

os.makedirs(FIG_DIR, exist_ok=True)

# ==============================================================================
# Plot 1: Multi-Class Confusion Matrix (Normalized & Counts)
# ==============================================================================
print("Generating Confusion Matrix Plot...")

if not os.path.exists(CACHE_FILE):
    raise FileNotFoundError(f"Feature cache {CACHE_FILE} not found. Run 04_train_baseline_ml.py first.")

cached = np.load(CACHE_FILE)
X = cached['X']
y = cached['y']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay.from_estimator(
    rf, 
    X_test, 
    y_test, 
    display_labels=rf.classes_, 
    cmap="Blues", 
    normalize="true", 
    values_format=".2f",
    ax=ax
)
plt.title("ICBHI Baseline Random Forest: Normalized Confusion Matrix", fontsize=12, pad=12)
plt.xlabel("Predicted Clinical Label", fontsize=11)
plt.ylabel("True Clinical Label", fontsize=11)
plt.tight_layout()

cm_path = os.path.join(FIG_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=300)
plt.close()
print(f"Saved: {cm_path}")

# ==============================================================================
# Plot 2: 5-Panel Spectrogram Comparison of Golden Test Vectors
# ==============================================================================
print("Generating 5-Panel Test Vector Spectrogram Plot...")

vector_files = sorted(glob.glob(os.path.join(TEST_VEC_DIR, "vector_*.wav")))
if len(vector_files) != 5:
    raise FileNotFoundError(f"Expected 5 test vectors in {TEST_VEC_DIR}, found {len(vector_files)}.")

fig, axes = plt.subplots(5, 1, figsize=(10, 14), sharex=True)

titles = [
    "Vector 0: Clean Normal Vesicular Breath (Passband: 100–1200 Hz)",
    "Vector 1: Pathological Wheeze (High-Pitched Musical Resonances)",
    "Vector 2: Fabric Friction Artifact (Broadband Uniform Flat Noise)",
    "Vector 3: 50 Hz Mains Hum & Chest Air-Gap (Sub-100 Hz Out-of-Band)",
    "Vector 4: Ambient Speech Bleed (1500 Hz & 1850 Hz Out-of-Band)"
]

target_sr = 4000

for idx, (vpath, ax, title) in enumerate(zip(vector_files, axes, titles)):
    data, sr = sf.read(vpath)
    
    # Compute spectrogram using 128-point FFT slices with 75% overlap
    Pxx, freqs, bins, im = ax.specgram(
        data, 
        NFFT=128, 
        Fs=sr, 
        noverlap=96, 
        cmap="magma"
    )
    
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.set_ylabel("Frequency (Hz)", fontsize=9)
    ax.set_ylim(0, 2000)
    
    # Highlight the 100 Hz to 1200 Hz diagnostic passband
    ax.axhline(100, color="cyan", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axhline(1200, color="cyan", linestyle="--", linewidth=0.8, alpha=0.7)

axes[-1].set_xlabel("Time (Seconds)", fontsize=10)
fig.suptitle("AUSCULT-AI Test Vectors: Time-Frequency Spectral Signatures", fontsize=13, y=0.99)
plt.tight_layout()

spec_path = os.path.join(FIG_DIR, "test_vectors_spectrogram.png")
plt.savefig(spec_path, dpi=300)
plt.close()
print(f"Saved: {spec_path}")
