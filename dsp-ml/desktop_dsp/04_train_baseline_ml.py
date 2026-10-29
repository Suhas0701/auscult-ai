import os
import glob
import numpy as np
import librosa
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

PROCESSED_DIR = "dsp-ml/data/processed_cycles"
TEST_VEC_DIR = "dsp-ml/test_vectors"
CACHE_FILE = "dsp-ml/data/features_cache.npz"
TARGET_SR = 4000
N_MFCC = 13

# Quality Gate Thresholds (from Step 5)
TH_BAND_MIN = 0.65
TH_FLAT_MAX = 0.35

def extract_cycle_features(file_path):
    """
    Extracts 26 features (13 MFCC means + 13 MFCC stds) from a 4 kHz cycle.
    """
    audio, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)
    if len(audio) < 256:
        return None
    
    # Compute 13 MFCCs across time
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC, n_fft=512, hop_length=128)
    
    # Summarize temporal distribution with mean and standard deviation
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    
    return np.hstack([mfcc_mean, mfcc_std])

def compute_sqi_check(audio_frame):
    """
    Computes SQI_band and SQI_flat for gating verification.
    """
    if len(audio_frame) != 1024:
        return False, "Invalid Length"
    
    window = np.hanning(1024)
    fft_vals = np.fft.rfft(audio_frame * window, n=1024)
    power = np.abs(fft_vals) ** 2
    
    # Bins: 26 to 307 (~100 Hz to ~1200 Hz)
    total_power = np.sum(power[1:513])
    in_band_power = np.sum(power[26:308])
    sqi_band = (in_band_power / total_power) if total_power > 0 else 0.0
    
    band_power = power[26:308]
    eps = 1e-12
    sqi_flat = np.exp(np.mean(np.log(band_power + eps))) / (np.mean(band_power) + eps)
    
    passed = (sqi_band >= TH_BAND_MIN) and (sqi_flat <= TH_FLAT_MAX)
    return passed, f"band={sqi_band:.2f}, flat={sqi_flat:.2f}"

# ==============================================================================
# 1. Feature Extraction & Dataset Caching
# ==============================================================================
if os.path.exists(CACHE_FILE):
    print(f"Loading cached feature matrix from: {CACHE_FILE}")
    cached = np.load(CACHE_FILE)
    X = cached['X']
    y = cached['y']
else:
    print("Extracting 26-dim MFCC features across all extracted breath cycles...")
    wav_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "*.wav")))
    if not wav_files:
        raise FileNotFoundError(f"No WAV files found in {PROCESSED_DIR}. Run 01_slice_icbhi.py first.")
    
    features_list = []
    labels_list = []
    
    total = len(wav_files)
    for i, fpath in enumerate(wav_files):
        # Extract class label from filename (e.g., 'normal', 'crackle', 'wheeze', 'both')
        label = os.path.basename(fpath).split('_')[-1][:-4]
        feats = extract_cycle_features(fpath)
        
        if feats is not None:
            features_list.append(feats)
            labels_list.append(label)
            
        if (i + 1) % 1000 == 0 or (i + 1) == total:
            print(f"Extracted features [{i + 1}/{total}]...")

    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list)
    
    # Cache to disk for instant subsequent executions
    np.savez_compressed(CACHE_FILE, X=X, y=y)
    print(f"Features cached to: {CACHE_FILE}")

print(f"\nFeature Matrix Shape: {X.shape} (Cycles: {X.shape[0]}, Features: {X.shape[1]})")

# ==============================================================================
# 2. Train / Test Split and Classifier Training
# ==============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Training Random Forest (100 estimators) on {len(X_train)} training cycles...")
rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n" + "=" * 65)
print(f"ICBHI BASELINE MODEL EVALUATION (Test Accuracy: {acc*100:.2f}%)")
print("=" * 65)
print(classification_report(y_test, y_pred))

# ==============================================================================
# 3. Deliverable Proof: Gated vs. Ungated Inference Demonstration
# ==============================================================================
print("=" * 80)
print("PHASE II DELIVERABLE PROOF: GATED VS. UNGATED INFERENCE ON TEST VECTORS")
print("=" * 80)
print(f"{'Vector Name':<20} | {'Ungated Prediction':<20} | {'Gated Output':<25}")
print("-" * 80)

vector_files = sorted(glob.glob(os.path.join(TEST_VEC_DIR, "vector_*.wav")))
for vpath in vector_files:
    vname = os.path.basename(vpath)[:-4]
    data, _ = sf.read(vpath)
    
    # 1. Naive Ungated Inference
    mfcc_feats = extract_cycle_features(vpath)
    ungated_pred = rf.predict([mfcc_feats])[0]
    
    # 2. Confidence-Gated Inference
    passed_gate, gate_info = compute_sqi_check(data)
    if passed_gate:
        gated_output = f"PASS -> {ungated_pred.upper()}"
    else:
        gated_output = "BLOCKED (Artifact Rejected)"
        
    print(f"{vname:<20} | {ungated_pred:<20} | {gated_output:<25}")

print("=" * 80 + "\n")
