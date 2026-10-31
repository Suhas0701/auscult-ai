# AUSCULT-AI Phase II: Track D (Member 5) Final Documentation & Test Report

## 1. Executive Summary & Objectives
This document presents the complete Phase II engineering validation, test protocols, and experimental data logs for Track D of the AUSCULT-AI project. As part of our multi-modal digital stethoscope architecture, Track D focuses on physical bench validation, tissue-mimicking phantom integration, and artifact disturbance logging to ensure reliable Edge-AI inference.

---

## 2. Methodology & Test Bench Setup

### 2.1 Silicone Chest Phantom Construction
*   **Material:** Smooth-On Ecoflex 00-30 platinum-cure two-part silicone.
*   **Dimensions:** $120 \times 120 \times 15\text{ mm}$.
*   **Acoustic Rationale:** Cast to mimic the acoustic impedance and density of human soft tissue, backed by soft-tissue characterization studies (Mencarelli et al., 2026; Zell et al., 2007).
*   **Procedure:** Parts A and B were mixed in a 1:1 ratio, poured slowly into an acrylic mold to avoid air bubbles, cured for 4 hours at room temperature, and acoustically coupled via ultrasound gel to a $3\text{W}$ flat exciter speaker.

### 2.2 Kinematic & Spectral Disturbance Rigs
To rigorously test the edge-AI gating mechanism under noisy conditions (Design and Comparative Performance of a Robust Lung Auscultation System for Noisy Clinical Settings, 2021), three mechanical rigs were designed:
1.  **Static Force Calibration Rig:** A vertical sliding PVC collar with calibrated lab weights ($50\text{ g}$ to $500\text{ g}$) to establish the FSR operational window.
2.  **Tremor Simulator:** A $3\text{V}$ Eccentric Rotating Mass (ERM) disc motor clamped to the bell housing to induce $5\text{--}12\text{ Hz}$ hand tremors.
3.  **Friction Rig:** A linear sliding rail mechanism using 100% Cotton and Polyester swatches stroked across the bell rim at $\approx 5\text{ cm/s}$ to simulate clothing friction.

---

## 3. Thresholds & Experimental Results Summary

### 3.1 Force Sensitive Resistor (FSR) Calibration
*   **Operating Window:** $0.8\text{ N}$ to $4.0\text{ N}$ contact force.
*   **Findings:** Forces below $0.8\text{ N}$ trigger a `"Press Firmer"` gate rejection, while forces above $4.0\text{ N}$ trigger saturation warnings, ensuring consistent acoustic coupling.

### 3.2 Kinematic Gating (IMU Jerk Threshold)
*   **Threshold:** $|\mathbf{J}| > 1.2\text{ m/s}^3$.
*   **Findings:** During resting state, jerk magnitude remained $\approx 0.20\text{ m/s}^3$ (PASS). When the $8\text{ Hz}$ tremor was injected, jerk spiked to $2.45\text{ m/s}^3$, successfully triggering a `"Hold Steady"` rejection.

### 3.3 Spectral Gating (Audio Wiener Entropy / SQI_flat)
*   **Threshold:** $\text{SQI}_{\text{flat}} > 0.65$.
*   **Findings:** Clean breath frames maintained a low spectral flatness ($\approx 0.15$, PASS). When cotton or polyester friction occurred, flatness rose to $0.76\text{--}0.82$, triggering a `"Friction Detected"` rejection.

---

## 4. Verification Data Logs
All raw and processed verification datasets are stored under `bench/data_logs/`:
*   `fsr_calibration_real.csv`: Applied mass, force, and ADC voltages.
*   `imu_tremor_test_real.csv`: Acceleration vectors, computed jerk magnitude, and gate status.
*   `audio_friction_test_real.csv`: In-band energy, Wiener entropy, and gate status.
