import os
import glob
import numpy as np
import soundfile as sf

TARGET_SR = 4000
N_SAMPLES = 1024
TEST_VEC_DIR = "dsp-ml/test_vectors"
RESULTS_FILE = "dsp-ml/desktop_dsp/reference_results.txt"

# Frequency Bin Mapping
# Delta f = 4000 / 1024 = 3.90625 Hz per bin
BIN_START = int(np.ceil(100.0 / (TARGET_SR / N_SAMPLES)))   # Bin 26 (~101.56 Hz)
BIN_END   = int(np.floor(1200.0 / (TARGET_SR / N_SAMPLES))) # Bin 307 (~1199.22 Hz)
NYQUIST_BIN = N_SAMPLES // 2                                # Bin 512 (2000 Hz)

# Quality Thresholds
TH_BAND_MIN = 0.65
TH_FLAT_MAX = 0.35

def compute_sqi(frame):
    """
    Computes SQI_band and SQI_flat for a normalized 1024-sample frame.
    """
    # 1. Apply Hann window
    window = np.hanning(N_SAMPLES)
    windowed = frame * window

    # 2. Compute 1024-point Real FFT and Power Spectrum
    fft_vals = np.fft.rfft(windowed, n=N_SAMPLES)
    power_spectrum = np.abs(fft_vals) ** 2

    # 3. Compute In-Band Energy Ratio (SQI_band)
    # Ignore Bin 0 (DC component)
    total_power = np.sum(power_spectrum[1 : NYQUIST_BIN + 1])
    in_band_power = np.sum(power_spectrum[BIN_START : BIN_END + 1])
    
    sqi_band = (in_band_power / total_power) if total_power > 0 else 0.0

    # 4. Compute Spectral Flatness (SQI_flat) over the diagnostic band
    band_power = power_spectrum[BIN_START : BIN_END + 1]
    eps = 1e-12  # Guard against log(0)
    
    geometric_mean = np.exp(np.mean(np.log(band_power + eps)))
    arithmetic_mean = np.mean(band_power) + eps
    sqi_flat = geometric_mean / arithmetic_mean

    # 5. Gating Decision Logic
    if sqi_band < TH_BAND_MIN:
        status = "REJECT"
        reason = "Out-of-band energy (Hum/Speech/Leak)"
    elif sqi_flat > TH_FLAT_MAX:
        status = "REJECT"
        reason = "Broadband friction artifact"
    else:
        status = "PASS"
        reason = "Valid physiological signal"

    return sqi_band, sqi_flat, status, reason

# Load and process each test vector
wav_files = sorted(glob.glob(os.path.join(TEST_VEC_DIR, "vector_*.wav")))
if not wav_files:
    raise FileNotFoundError("Test vectors not found. Run 02_build_test_vectors.py first.")

header_str = f"{'Vector Name':<22} | {'SQI_band':<10} | {'SQI_flat':<10} | {'Status':<8} | {'Reason'}"
divider = "-" * 85

print("\n" + divider)
print("AUSCULT-AI TRACK C: PYTHON REFERENCE DSP BENCHMARK (GROUND TRUTH)")
print(divider)
print(header_str)
print(divider)

log_lines = [divider, "AUSCULT-AI TRACK C: PYTHON REFERENCE DSP BENCHMARK", divider, header_str, divider]

for wav_path in wav_files:
    name = os.path.basename(wav_path)[:-4]
    data, sr = sf.read(wav_path)
    if len(data) != N_SAMPLES:
        raise ValueError(f"Vector {name} has length {len(data)}, expected {N_SAMPLES}")
    
    band, flat, status, reason = compute_sqi(data)
    row_str = f"{name:<22} | {band:<10.4f} | {flat:<10.4f} | {status:<8} | {reason}"
    print(row_str)
    log_lines.append(row_str)

print(divider + "\n")
log_lines.append(divider)

with open(RESULTS_FILE, "w") as f:
    f.write("\n".join(log_lines) + "\n")

print(f"Benchmark results successfully exported to: {RESULTS_FILE}")
