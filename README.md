# FxLMS Active Noise Control (ANC)

Real-time Active Noise Control using the Filtered-x LMS algorithm in Python. Designed for ultra-low latency audio with `sounddevice` and `numba`.

## What you need

- Hardware
  - 2-channel audio input (stereo):
    - CH1 = reference mic (near noise source or a reference pickup)
    - CH2 = error mic (near the ear/quiet zone)
  - 2-channel audio output (stereo) to the loudspeaker/actuator - only 1 channel used.
  - macOS tip: create an Aggregate Device (Audio MIDI Setup) to get a single 2-in/2-out device
  - 1 kHz acoustic calibrator (94 or 114 dB SPL) for mic sensitivity calibration

- Software
  - Python 3.12 (used for the development)
  - macOS: CoreAudio works with `sounddevice`
  - Dependencies listed in `requirements.txt`

## Repository layout

- `anc_realtime.py` – CLI entry point; opens the audio stream and runs ANC - run this code to do ANC
- `src/anc_processor.py` – FxLMS core (Numba-accelerated) and config loader
- `config_FxLMS.json` – runtime/configuration (rate, block size, mu, filters, delays, mic sensitivities, etc.)
- `filter_coeffs.json` – calibrated secondary-path IIR (A, B, tau_ms) used at runtime
- `CALIB/CALIB_MIC.py` – measures mic sensitivities with a 1 kHz calibrator; writes `mic_sensitivity.json`
- `CALIB/NEW_COEFFS.py` – scales RAW secondary-path B by mic2 sensitivity; writes `filter_coeffs.json`
- `CALIB/filter_coeffs_RAW.json` – RAW secondary-path IIR to start from (you provide A, B, tau_ms)
- `MATLAB/IRMeasurer_CODE.m` – records the secondary-path impulse response via swept-sine (Audio Toolbox)
- `MATLAB/PRONY.m` – fits an IIR model (A,B) + pure delay tau_ms from the measured IR and saves JSON
- `requirements.txt` – Python dependencies

## Step 1 — Measure the secondary path (MATLAB)

Use this when you need to generate or refresh `CALIB/filter_coeffs_RAW.json` from a real acoustic path.

Prereqs (MATLAB): Audio Toolbox (sweeptone, audioPlayerRecorder), DSP System Toolbox (dsp.AsyncBuffer), Signal Processing Toolbox.

1) Record IR
  - Launch ImpulseRespMeasurer via the commande window OR 
  - Open `MATLAB/IRMeasurer_CODE.m` in MATLAB
  - Set `device` to your audio interface name, verify `fs=8000`, `L`, and channel maps
  - Run the script; it plays a 30–300 Hz sweep and records the response (watch for underrun/overrun)
  - Save the produced measurement `.mat` (contains `measurementData`)

2) Fit IIR (Prony) and export RAW coefficients
  - Run `MATLAB/PRONY.m`, select the `.mat` file when prompted
  - The script grid-searches stable IIR orders, reports best model, and asks to save JSON
  - Save as `filter_coeffs_RAW.json` (contains `{ "A": [...], "B": [...], "tau_ms": <float> }`)
  - Move/copy it to `CALIB/filter_coeffs_RAW.json`

Notes
- Keep the same sample rate (8 kHz) as your Python runtime
- On macOS, using an Aggregate Device helps to align 2-in/2-out channels
- Target the 50–250 Hz band as in the provided scripts

## Configuration essentials

- Inputs/outputs
  - Input expects stereo: CH1=reference, CH2=error
  - Output writes the anti-noise to all output channels
- `config_FxLMS.json` keys (minimal):
  - `sample_rate`, `block_size`, `L`, `mu`
  - `secondary_path`: path to `filter_coeffs.json`
  - `sens_ref`, `sens_err`: microphone sensitivities (Pa per input unit) – set these manually after calibration
  - Optional: `bandpass`, `dc_blocker`, `lms.leakage`, `adaptation`, `phase.extra_output_delay_*`

## Correct startup workflow (IR → calibration → coefficients → manual config)

0) If you don’t have `CALIB/filter_coeffs_RAW.json`: run the MATLAB step above (IR measurement + PRONY)

1) Microphone calibration
   - Connect both mics; place the calibrator on each mic (one after another)
   - Run `CALIB/CALIB_MIC.py` and follow prompts (1 kHz, 114 dB SPL by default)
   - This writes `mic_sensitivity.json` at repo root with `mic1_sens` and `mic2_sens` (Pa/unit)

2) Prepare calibrated secondary-path IIR
   - Ensure `CALIB/filter_coeffs_RAW.json` exists with:
     - `{ "A": [...], "B": [...], "tau_ms": <float_ms> }`
   - Run `CALIB/NEW_COEFFS.py` to scale B by `mic2_sens` and generate `filter_coeffs.json` in the repo root

3) Manually set mic sensitivities in config
   - Open `config_FxLMS.json`
   - Copy the values from `mic_sensitivity.json` to:
     - `sens_ref`  ← sensitivity of the reference mic (usually mic1)
     - `sens_err`  ← sensitivity of the error mic (usually mic2)

4) Run ANC
   - Activate your virtual environment and install dependencies
   - Start: `python anc_realtime.py`
   - Pick input/output device IDs when prompted (use your Aggregate Device if available)

## Quick start (macOS, zsh)

Optional commands, assuming you are inside the project folder:

```bash
# Create venv (Python 3.12) and install deps
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 1) Calibrate mics
python CALIB/CALIB_MIC.py

# 2) Build calibrated secondary-path (A,B,tau_ms must be in CALIB/filter_coeffs_RAW.json)
python CALIB/NEW_COEFFS.py

# 3) Edit config_FxLMS.json and copy mic sensitivities to sens_ref / sens_err

# 4) Run
python anc_realtime.py
```

## Tips

- Start with `sample_rate=8000` and small `block_size` (e.g., 32–128) to minimize latency
- If xruns occur, increase `block_size` or reduce CPU usage
- Ensure `filter_coeffs.json` exists and is stable (A/B must be valid IIR)
- On macOS, allow microphone access for Python in System Settings

## Safety

- Keep output levels modest for initial tests; watch for oscillation/feedback
- Protect your speakers (hardware limiter is nice)

## Real-time Python audio

Pre-flight (system)
- [ ] Use a wired 2-in/2-out interface; fix sample rate to your config (e.g., 8 kHz).
- [ ] Close heavy apps; plug into power; disable OS sound effects/alerts.
- [ ] macOS: Use an Aggregate Device if needed; match sample rates; disable App Nap for Terminal.
- [ ] Linux: Use PipeWire/JACK low-latency profile; enable rtkit; governor=performance; raise memlock and rtprio (ulimits).
- [ ] Windows: WASAPI Exclusive; High performance power plan; disable audio “enhancements”.

Python/env
- [ ] Use Python 3.12 in a venv; pin deps from requirements.txt.
- [ ] Set Numba threads to 1: export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
- [ ] Warm up JIT before opening the stream (call numba kernels once).
- [ ] Preallocate arrays/buffers; no allocations in the callback; use float32 and contiguous arrays.
- [ ] Disable logging/printing in the callback; consider gc.disable() during streaming (re-enable after).

sounddevice/PortAudio
- [ ] Choose the lowest-latency stable device/host API (CoreAudio/ASIO/WASAPI Exclusive/JACK).
- [ ] Set block_size small but safe (e.g., 32–128); dtype=float32; channels as needed.
- [ ] Keep callback code minimal; push heavy math into pre-compiled numba functions.
- [ ] Avoid Python objects, locks, and exceptions in the callback; use ring buffers for state.

Priority and stability
- [ ] macOS: optionally renice: sudo renice -n -10 -p $PID
- [ ] Linux: if permitted, chrt -f -p 80 $PID; ensure user has rtprio and memlock limits.
- [ ] Windows: start Python High priority (Task Manager or Start-Process -Priority High).

Monitoring (during run)
- [ ] Track callback time vs frame time (block_size / sample_rate). Keep <50% headroom.
- [ ] Watch for xruns in console; if seen, act immediately (see below).

If you get xruns/underruns
- [ ] Increase block_size one step; or increase requested latency.
- [ ] Reduce CPU: lower filter length L; simplify processing; disable debug; keep arrays preallocated.
- [ ] Verify sample rate/device matches config; prefer Exclusive/ASIO/JACK modes.

