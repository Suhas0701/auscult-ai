import os
import glob
import librosa
import soundfile as sf

RAW_DIR = "dsp-ml/data/icbhi_raw"
OUT_DIR = "dsp-ml/data/processed_cycles"
TARGET_SR = 4000  # 4 kHz sampling frequency for ESP32 ADC

os.makedirs(OUT_DIR, exist_ok=True)
txt_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.txt")))
print(f"Found {len(txt_files)} annotation files. Slicing respiratory cycles...")

cycle_count = 0
total_files = len(txt_files)

for i, txt_path in enumerate(txt_files):
    wav_path = txt_path[:-4] + ".wav"
    if not os.path.exists(wav_path):
        continue

    # Load and resample audio to 4 kHz mono
    audio, _ = librosa.load(wav_path, sr=TARGET_SR, mono=True)

    with open(txt_path, "r") as f:
        for idx, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            
            start_sec = float(parts[0])
            end_sec = float(parts[1])
            crackle = int(parts[2])
            wheeze = int(parts[3])

            start_idx = int(start_sec * TARGET_SR)
            end_idx = int(end_sec * TARGET_SR)
            cycle_audio = audio[start_idx:end_idx]

            # Minimum length check (ignore empty or corrupt sub-second glitches)
            if len(cycle_audio) < int(0.2 * TARGET_SR):
                continue

            # Map labels
            if crackle == 0 and wheeze == 0:
                label = "normal"
            elif crackle == 1 and wheeze == 0:
                label = "crackle"
            elif crackle == 0 and wheeze == 1:
                label = "wheeze"
            else:
                label = "both"

            base_name = os.path.basename(txt_path)[:-4]
            out_filename = f"{base_name}_c{idx}_{label}.wav"
            sf.write(os.path.join(OUT_DIR, out_filename), cycle_audio, TARGET_SR)
            cycle_count += 1

    # Print progress every 100 files
    if (i + 1) % 100 == 0 or (i + 1) == total_files:
        print(f"Processed [{i + 1}/{total_files}] files... (Total cycles extracted: {cycle_count})")

print(f"\nCompleted! Successfully extracted {cycle_count} cycles into {OUT_DIR}")
