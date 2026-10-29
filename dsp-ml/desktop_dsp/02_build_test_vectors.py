import os
import glob
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

TARGET_SR = 4000
N_SAMPLES = 1024
PROCESSED_DIR = "dsp-ml/data/processed_cycles"
TEST_VEC_DIR = "dsp-ml/test_vectors"
C_HEADER_PATH = "firmware/include/test_vectors.h"

os.makedirs(TEST_VEC_DIR, exist_ok=True)
os.makedirs("firmware/include", exist_ok=True)

# Emulate Track A: 4th-order Butterworth Bandpass Filter (100 Hz to 1200 Hz)
sos = butter(4, [100.0, 1200.0], btype='bandpass', fs=TARGET_SR, output='sos')

def extract_best_active_frame(files_list):
    """
    Filters raw audio using the 100-1200 Hz bandpass filter (emulating Track A),
    locates the 1024-sample window with the highest acoustic energy (active breath phase),
    and peak-normalizes to [-1.0, 1.0].
    """
    for file_path in files_list:
        audio, _ = librosa.load(file_path, sr=TARGET_SR, mono=True)
        if len(audio) < N_SAMPLES:
            continue
        
        # Apply the hardware-equivalent bandpass filter
        filtered_audio = sosfiltfilt(sos, audio)

        # Sliding window to find peak RMS energy (active breath sound, not silence)
        hop = 128
        max_energy = -1.0
        best_frame = None

        for start in range(0, len(filtered_audio) - N_SAMPLES, hop):
            frame = filtered_audio[start : start + N_SAMPLES]
            energy = np.sum(frame ** 2)
            if energy > max_energy:
                max_energy = energy
                best_frame = frame

        if best_frame is not None:
            peak = np.max(np.abs(best_frame))
            if peak > 0.01:
                return (best_frame / peak).astype(np.float32)

    raise RuntimeError("Could not locate a valid active frame in dataset.")

print("Extracting filtered physiological frames (simulating 100-1200 Hz Sallen-Key)...")

# 1. Vector 0: Clean Normal Vesicular Breath
normal_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "*_normal.wav")))
v0_clean = extract_best_active_frame(normal_files)

# 2. Vector 1: Pathological Wheeze
wheeze_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "*_wheeze.wav")))
v1_wheeze = extract_best_active_frame(wheeze_files)

print("Synthesizing controlled failure mode vectors...")

# 3. Vector 2: Fabric Friction (Uniform broadband noise)
np.random.seed(42)
v2_friction = np.random.uniform(-0.85, 0.85, N_SAMPLES).astype(np.float32)

# 4. Vector 3: 50 Hz Mains Hum + 25 Hz Baseline Wobble (Air leak / missing ground)
t = np.linspace(0, N_SAMPLES / TARGET_SR, N_SAMPLES, endpoint=False)
v3_hum = (0.75 * np.sin(2 * np.pi * 50 * t) + 0.25 * np.sin(2 * np.pi * 25 * t)).astype(np.float32)

# 5. Vector 4: High-Frequency Ambient Bleed (Speech / room chatter > 1200 Hz)
v4_speech = (0.60 * np.sin(2 * np.pi * 1500 * t) + 0.40 * np.sin(2 * np.pi * 1850 * t)).astype(np.float32)

vectors = {
    "vector_0_clean": v0_clean,
    "vector_1_wheeze": v1_wheeze,
    "vector_2_friction": v2_friction,
    "vector_3_hum": v3_hum,
    "vector_4_speech": v4_speech
}

# Save WAV files
for name, samples in vectors.items():
    wav_path = os.path.join(TEST_VEC_DIR, f"{name}.wav")
    sf.write(wav_path, samples, TARGET_SR)

# Export C Header
with open(C_HEADER_PATH, "w") as f:
    f.write("// =============================================================================\n")
    f.write("// AUSCULT-AI: Standardized Gating Test Vectors (Track C Golden Reference)\n")
    f.write("// Emulates 100-1200 Hz Analog Bandpass Filtered Signals\n")
    f.write(f"// Sampling Rate: {TARGET_SR} Hz | Frame Length: {N_SAMPLES} samples\n")
    f.write("// =============================================================================\n\n")
    f.write("#ifndef TEST_VECTORS_H\n")
    f.write("#define TEST_VECTORS_H\n\n")
    f.write(f"#define FRAME_LEN {N_SAMPLES}\n")
    f.write(f"#define SAMPLE_RATE {TARGET_SR}\n\n")
    
    for name, samples in vectors.items():
        arr_str = ", ".join([f"{x:.6f}f" for x in samples])
        f.write(f"const float {name}[{N_SAMPLES}] = {{\n  {arr_str}\n}};\n\n")
        
    f.write("#endif // TEST_VECTORS_H\n")

print(f"Updated 5 WAV test vectors in: {TEST_VEC_DIR}/")
print(f"Updated firmware C header in:   {C_HEADER_PATH}")
