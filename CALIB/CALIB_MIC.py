import sounddevice as sd, numpy as np, json
from typing import Any, cast

CALIB_SPL = 114
CALIB_FREQ = 1000
CALIB_PA = 20e-6 * 10**(CALIB_SPL / 20)

def measure_channel(ch):
    """Interactively measure a single channel sensitivity.

    Returns a native Python float so it can be JSON-serialized directly.
    """
    # Query default input device info (type: dict at runtime); isolate samplerate for clarity
    devinfo = cast(dict[str, Any], sd.query_devices(kind='input'))  # runtime is a dict
    samplerate = int(devinfo.get('default_samplerate') or 48000)
    input(f"Place calibrator on mic {ch} (1 kHz {CALIB_SPL} dB) and press Enter...")
    duration_sec = 2
    frames = int(duration_sec * samplerate)
    rec = sd.rec(frames, samplerate=samplerate, channels=2, dtype='float32')
    sd.wait()
    # Compute RMS while guarding against NaNs / zeros
    chan = rec[:, ch-1]
    rms = float(np.sqrt(np.mean(np.square(chan, dtype=np.float64))))
    if rms == 0.0:
        raise RuntimeError(f"Recorded silence (rms=0) on channel {ch}; check wiring/calibrator.")
    sens_val = CALIB_PA / rms
    return float(sens_val)

sens = {}
for ch in [1, 2]:
    sens[f"mic{ch}_sens"] = measure_channel(ch)  # ensure native float

with open("mic_sensitivity.json", "w") as f:
    json.dump(sens, f, indent=2)

print("Saved sensitivities:", sens)
print("Calibration complete. Place the calibrator back in its holder - reminder ^^ ;))")
