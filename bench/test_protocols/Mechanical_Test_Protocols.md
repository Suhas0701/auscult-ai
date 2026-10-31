# AUSCULT-AI Test Protocols (Phase II) - Track D

## 1. Silicone Chest Phantom Fabrication Protocol
*   **Objective:** Fabricate a reusable soft-tissue acoustic phantom for bench testing.
*   **Materials:** Ecoflex 00-30 platinum silicone, mixing cups, vacuum chamber (optional), acrylic mold, 3W exciter speaker.
*   **Step-by-Step Instructions:**
    1. Measure equal parts A and B by weight (1:1 ratio).
    2. Mix thoroughly for 3 minutes, scraping sides to ensure uniform curing.
    3. Pour into a $120 \times 120 \times 15\text{ mm}$ acrylic mold.
    4. Let cure for 4 hours at room temperature.
    5. Mount on top of the exciter speaker using acoustic coupling gel.

## 2. Kinematic Tremor Rig Protocol
*   **Objective:** Inject repeatable $5\text{--}12\text{ Hz}$ hand tremors into the stethoscope probe.
*   **Materials:** 3V ERM disc motor, variable DC power supply, mounting clamp.
*   **Step-by-Step Instructions:**
    1. Secure the ERM motor to the top IMU boss of the stethoscope housing.
    2. Connect power leads to the DC supply and set voltage to $2.2\text{V}$.
    3. Monitor the ESP32 serial output to verify that jerk magnitude exceeds $1.2\text{ m/s}^3$ and triggers the motion gate.

## 3. Friction Rig Protocol
*   **Objective:** Simulate clothing friction against the stethoscope rim during auscultation.
*   **Materials:** Linear rail slider, 100% Cotton swatch, 100% Polyester swatch.
*   **Step-by-Step Instructions:**
    1. Attach fabric swatch to the sliding carriage.
    2. Stroke fabric across the FSR contact rim at $\approx 5\text{ cm/s}$.
    3. Verify that `SQI_flat` exceeds $0.65$ and triggers the friction rejection flag.
