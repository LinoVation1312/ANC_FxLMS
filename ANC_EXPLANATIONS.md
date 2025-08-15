# REAL-TIME ANC SYSTEM – FxLMS Implementation

## System Overview

This real-time ANC (Active Noise Control) system implements a residual-error **FxLMS (Filtered-x Least Mean Squares)** algorithm using the Python `sounddevice` backend for low-latency audio I/O. Configuration is fully driven by `config_FxLMS.json`. The current tuned operating band (band-pass filter) is **40–250 Hz**; the secondary path IIR was measured primarily inside the broader low‑frequency range (original notes 35–400 Hz). All values below reflect the harmonized configuration.

## Real-Time System Architecture

```
                    LIVE FEEDFORWARD ANC SYSTEM WITH FxLMS ADAPTATION
                    
Reference Microphone                             Error Microphone
        │                                               │
        │ ref_input                                     │
        │                                               │
        ▼                                               ▼
┌─────────────────┐                              ┌─────────────┐
│ Audio Device    │        Primary path       +  │ Audio Device│───> error_signal
│ Input Channel 1 │ -------------------------->  │ Input Ch 2  │
│                 │                              │             │
└─────────────────┘                              └─────────────┘
        │                                         +    ▲
        │                                              │
        ▼                                              │
┌─────────────────┐      ┌─────────────────┐           │
│ Adaptive Filter │ y(n) │ Audio Output    │ output    │
│    W(z)         │────► │ Speaker         │───────────┘
│   256 taps      │      │ + Environment   │
└─────────────────┘      └─────────────────┘
        ▲                                              
        │                                               
        │ Real-time Coefficient Update                  
        │ W(k+1) = W(k) - μ_norm * error * x_filt(k)    
        │                                               
┌─────────────────┐      ┌─────────────────┐           
│   FxLMS         │      │ Secondary Path  │            
│  Adaptation     │◄──── │ Model H(z)      │◄────────
│   Engine        │      │ (IIR Filter)    │ ref_input
└─────────────────┘      └─────────────────┘
        ▲                                              
        │                                               
        │   error


Real-Time Components:
- ref_input: Live reference microphone signal
- error_signal: Residual error microphone signal (already contains primary + anti-noise result)
- W(z): Adaptive FIR filter (256 coefficients @8kHz in current config --> 3.2 ms length)
- S(z): Secondary path acoustic model (IIR + delay) loaded from `filter_coeffs.json` (fields: B, A, tau_ms)
- Audio I/O: `sounddevice` stream (full‑duplex, float32, block = 32 samples)
```

## Core Components

### 1. Audio Input System
- **Reference Input**: Channel 0
- **Error Input**: Channel 1
- **Sample Rate**: 8000 Hz (config: `sample_rate`)
- **Block Size**: 32 samples (config: `block_size`) → 4.0 ms frame @ 8 kHz
        (Algorithm still updates weights sample-by-sample inside each block.)

### 2. ANCProcessor
- **Adaptive Filter Length (L)**: 256 taps (config: `L`)
- **Step Size (μ)**: 0.00015 (config: `mu`)
- **Leakage**: 0.0008 (config: `lms.leakage`) applied as (1 - leakage) * W per sample
- **Initialization**: Small random weights (Normal σ=1e-3);
- **Band-Pass**: IIR 2nd order (LP+HP 1st order) 40–250 Hz applied to reference and error if enabled
- **DC Blocker** (optionnal): 1st-order high-pass (config: `dc_blocker.r` = 0.995) on reference and adaptive output before band-pass
- **Coefficient Clipping**: Each W[i] constrained to [-10, +10] for numerical safety

### 3. Secondary Path Model (S(z))
- **File**: `filter_coeffs.json` (referenced by `secondary_path` in main config)
- **Fields**: `B`, `A`, `tau_ms` (physical delay excluding block buffering)
- **Effective Delay τ**: τ = round(tau_ms * Fs / 1000) + block_size (block I/O latency) + optional `phase.extra_output_delay_{samples|blocks}`
- **Usage**: Reference is filtered through S(z) and delayed (pure delay line) to produce filtered-x for adaptation.

## Real-Time FxLMS Algorithm

### Overview
Residual-error FxLMS operates per incoming audio sample inside each block:
1. Reference preprocessing: DC blocker → band-pass.
2. Anti-noise synthesis: y(n) = Σ W[i] * x_ref(n - i).
3. Output preprocessing: DC blocker (optional) → band-pass (mirrors reference shaping).
4. Error signal e(n) is already the residual (primary + anti-noise) from the error mic.
5. Filtered-x path: Reference (after preproc) → S(z)=B/A IIR → pure delay τ → circular buffer.
6. Weight update (if adaptation enabled):
        W_i ← (1 - leakage) * W_i - μ * e(n) * x_filt_i(n)
        then clip W_i to [-10, 10].

### Configuration Mapping (config_FxLMS.json → Runtime)
- `sample_rate` → audio sample rate
- `block_size` → processing frame size & implicit component of τ
- `L` → adaptive FIR length
- `mu` → LMS step size
- `secondary_path` → path to S(z) JSON (B, A, tau_ms)
- `bandpass.enabled` / `low_hz` / `high_hz` / `order` / `window` → FIR taps auto-designed (scipy.firwin); disabled → passthrough
- `bandpass.unit_gain_hz` (informational) → design target center (not explicitly used in code)
- `dc_blocker.enabled` + `r` → enable 1st-order DC blocker (y[n]=x[n]-x[n-1]+r*y[n-1])
- `lms.leakage` → leakage factor per update
- `phase.extra_output_delay_samples` or `phase.extra_output_delay_blocks` → additional delay (blocks overrides samples)

### Notation Consistency
- Adaptive filter: W(z), length L.
- Secondary path: S(z) = B(z)/A(z) with added pure delay τ.
- Filtered reference used for adaptation: x_filt(n).
- Residual/error: e(n).

### Notes
- All adaptation occurs sample-by-sample inside each block for minimal bias w.r.t. block edges.
- Clipping guards against divergence if large transients occur.
- Changing `mu` or enabling/disabling band-pass/DC only requires editing `config_FxLMS.json` and restarting.
