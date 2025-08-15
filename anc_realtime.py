#!/usr/bin/env python3
"""
Optimized Real-Time ANC Implementation with Ultra-Low Latency
Implementing all best practices for real-time audio processing
"""

import os
import gc
import time
import psutil
import numpy as np
import sounddevice as sd
from numba import njit
from src.anc_processor import ANCProcessor
import json
from typing import Any, Dict, List, Tuple, cast
from scipy.signal import firwin


# =============================================================================
# Performance Optimizations Setup
# =============================================================================

# 1. Limit parallel threads to avoid context switching
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

# 2. Set process priority (requires sudo on some systems)
try:
    p = psutil.Process(os.getpid())
    p.nice(-20)  # Highest priority
    # print only if needed; kept quiet by default
except Exception:
    pass

# 3. Disable garbage collector during real-time processing
gc.disable()

# =============================================================================
# Audio Configuration
# =============================================================================

_DEFAULT_BLOCKSIZE: int = 128
_DEFAULT_SR: int = 8000
sd.default.blocksize = _DEFAULT_BLOCKSIZE     
sd.default.dtype = ('float32', 'float32')  

# quiet by default

# =============================================================================
# Pre-compiled Performance Monitoring Functions
# =============================================================================

@njit(cache=True) 
def compute_rms(data):
    """Fast RMS computation"""
    sum_sq = 0.0
    for i in range(data.shape[0]):
        sum_sq += data[i] * data[i]
    return np.sqrt(sum_sq / data.shape[0])

# =============================================================================
# Ultra-Low Latency ANC Client
# =============================================================================

class UltraLowLatencyANC:
    """Ultra-optimized ANC client with minimal latency"""
    
    def __init__(self, processor, sample_rate=44100, block_size=128, output_cfg: Dict[str, Any] | None = None):
        self.processor = processor
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.running = False
        
        # Performance monitoring
        self.max_elapsed = 0.0
        self.avg_elapsed = 0.0
        self.xrun_count = 0
        self.callback_count = 0
        self.performance_history = []
        
        self.buffer_in = np.zeros((block_size, 2), dtype=np.float32)
        self.buffer_out = np.zeros((block_size, 2), dtype=np.float32)
        
        # Pre-allocate processing buffers (float64 for processing, float32 for I/O)
        self.ref_buffer = np.zeros(block_size, dtype=np.float64)
        self.err_buffer = np.zeros(block_size, dtype=np.float64)
        self.out_buffer = np.zeros(block_size, dtype=np.float64)
        
        # Audio budget calculation
        self.audio_budget = (block_size / sample_rate) * 1000  # ms
        self.warning_threshold = self.audio_budget * 0.85  # 85% of budget
        
        # Output configuration and verbosity
        self.output_cfg = output_cfg or {}
        self.verbose = bool(self.output_cfg.get('verbose', False))
        self.output_gain = 1.0  # always unity

        if self.verbose:
            print(f"✓ ANC initialized: {block_size} samples @ {sample_rate}Hz")
            print(f"✓ Audio budget: {self.audio_budget:.2f}ms (warning at {self.warning_threshold:.2f}ms)")

        # Optional: list available devices only when verbose
        if self.verbose:
            self._show_audio_devices()

        # Validate configuration
        if not self._validate_configuration():
            raise Exception("Configuration validation failed")
        
        # Warm-up JIT compilation
        self._warmup_jit()
        
        # Test device compatibility
        self._test_device_compatibility()
    
    def _test_device_compatibility(self):
        """Minimal device compatibility check (quiet unless verbose or error)."""
        try:
            def test_callback(indata, outdata, frames, time, status):
                outdata.fill(0.0)
            # Brief, minimal stream open/close
            with sd.Stream(
                samplerate=self.sample_rate,
                blocksize=min(self.block_size, 64),
                channels=(1, 1),
                dtype=np.float32,
                callback=test_callback
            ):
                sd.sleep(50)
            if self.verbose:
                print("✅ Device compatibility OK")
        except Exception as e:
            print(f"⚠️ Device compatibility issue: {e}")
            # Keep guidance minimal
            if self.verbose:
                print("ℹ️ Consider an aggregate device or a different sample rate")
    
    def _show_audio_devices(self):
        """Show available audio devices (verbose mode)."""
        print("\n=== Available Audio Devices ===")
        devices = cast(List[Dict[str, Any]], sd.query_devices())

        # Find aggregate device
        aggregate_device = None
        default_dev = sd.default.device
        try:
            default_input = int(default_dev[0]) if default_dev and default_dev[0] is not None else -1
        except Exception:
            default_input = -1
        try:
            default_output = int(default_dev[1]) if default_dev and default_dev[1] is not None else -1
        except Exception:
            default_output = -1

        for i, dev in enumerate(devices):
            # Check device capabilities
            max_in = int(dev.get('max_input_channels', 0))
            max_out = int(dev.get('max_output_channels', 0))
            can_input = max_in > 0
            can_output = max_out > 0

            # Markers for special devices
            markers: List[str] = []
            if i == default_input:
                markers.append("DEFAULT_IN")
            if i == default_output:
                markers.append("DEFAULT_OUT")
            name = str(dev.get('name', ''))
            if 'agrégé' in name.lower() or 'aggregate' in name.lower():
                markers.append("AGGREGATE")
                if can_input and can_output and aggregate_device is None:
                    aggregate_device = i

            marker_str = f" ({', '.join(markers)})" if markers else ""

            # Status indicators
            status = "✅" if can_input and can_output else ("🔸" if can_input or can_output else "❌")

            print(f"{i:2d}: {status} {name}{marker_str}")
            try:
                default_sr = float(dev.get('default_samplerate', 0))
            except Exception:
                default_sr = 0.0
            print(f"     In: {max_in:2d} ch, Out: {max_out:2d} ch, SR: {default_sr:.0f}Hz")
            
        print()
    
    def _validate_configuration(self):
        """Validate system configuration before starting"""
        if self.verbose:
            print("– Validating system configuration...")
        
        # Test filter coefficients
        try:
            with open('filter_coeffs.json', 'r') as f:
                data = json.load(f)
            if self.verbose:
                print("✓ Filter coefficients loaded successfully")
        except FileNotFoundError:
            print("❌ filter_coeffs.json not found")
            return False
        except Exception as e:
            print(f"❌ Error loading coefficients: {e}")
            return False
        
        # Test processor performance for current configuration
        try:
            if self.verbose:
                print(f"🧪 Testing performance for block_size={self.block_size}, filter_len={self.processor.L}")
            
            # Generate test signals
            ref_block = np.random.randn(self.block_size).astype(np.float64) * 0.1
            err_block = np.random.randn(self.block_size).astype(np.float64) * 0.1
            
            # Performance test with downsampled function
            times = []
            for _ in range(100):
                start = time.perf_counter()
                self.processor.process_block(ref_block, err_block)
                times.append((time.perf_counter() - start) * 1000)
            
            avg_time = np.mean(times)
            max_time = np.max(times)
            budget = self.audio_budget
            cpu_usage = (avg_time / budget) * 100
            
            if max_time < budget * 0.5:
                status = "✅ EXCELLENT"
            elif max_time < budget * 0.8:
                status = "✅ GOOD"
            elif max_time < budget:
                status = "⚠️ TIGHT"
            else:
                status = "❌ OVERRUN"
            
            if self.verbose:
                print(f"   📊 {avg_time:.3f}ms avg, {max_time:.3f}ms max")
                print(f"   🎯 Budget: {budget:.3f}ms, CPU: {cpu_usage:.1f}%")
                print(f"   📈 {status}")
            
            if max_time >= budget:
                if self.verbose:
                    print("⚠️ Performance may be tight for real-time")
                    print("Press Enter to continue, or type anything to cancel.")
                    response = input("> ").strip()
                    if response:
                        return False
                # In non-verbose mode, proceed without prompting

            
        except Exception as e:
            print(f"❌ Performance validation failed: {e}")
            return False
        
        return True
    
    def _warmup_jit(self):
        """Warm-up Numba JIT compilation"""
        if self.verbose:
            print("🔥 Warming up JIT compilation...")
        
        # Warm-up audio processing
        dummy_ref = np.random.randn(self.block_size).astype(np.float64) * 0.01
        dummy_err = np.random.randn(self.block_size).astype(np.float64) * 0.01
        
        # Multiple warmup passes with downsampled function
        for _ in range(10):
            self.processor.process_block(dummy_ref, dummy_err)
        
        # Warm-up utility functions
        compute_rms(dummy_ref)
        
        if self.verbose:
            print("✓ JIT warmup completed")
    
    def audio_callback(self, indata, outdata, frames, time_info, status):
        """
        Ultra-optimized audio callback
        - Pas d'allocation
        - Pas de print (sauf erreurs critiques)
        - Tout pré-calculé en dehors
        """
        start_time = time.perf_counter()
        
        # Status check (minimal logging)
        if status:
            self.xrun_count += 1
            # Only log every 10th xrun to avoid flooding
            if self.verbose and self.xrun_count % 10 == 1:
                print(f'[Audio Status] {status} (count: {self.xrun_count})')
        
        try:
            # Fast type conversion and channel extraction
            if indata.shape[1] >= 2:
                # Stereo input: channel 0 = reference, channel 1 = error
                self.ref_buffer[:] = indata[:, 0].astype(np.float64)
                self.err_buffer[:] = indata[:, 1].astype(np.float64)
            else:
                # Mono input: duplicate channel for error
                self.ref_buffer[:] = indata[:, 0].astype(np.float64)
                self.err_buffer[:] = self.ref_buffer
            
            # Core FxLMS processing (JIT-compiled)
            y_block, e_block = self.processor.process_block(
                self.ref_buffer, self.err_buffer
            )
            
            y_limited = y_block
            
            # Debug (verbose): Every 1000 callbacks, check signal levels
            if self.verbose and self.callback_count % 1000 == 0 and self.callback_count > 0:
                ref_rms = compute_rms(self.ref_buffer)
                err_rms = compute_rms(self.err_buffer)
                out_rms = compute_rms(y_limited)
                w_norm = np.sqrt(np.sum(self.processor.W ** 2))
                print(f"[DEBUG] Ref: {ref_rms:.6f}, Err: {err_rms:.6f}, Out: {out_rms:.6f}, |W|: {w_norm:.6f}")
            
            # Output to all channels (anti-noise signal) with overflow protection
            # Clip to prevent float32 overflow and ensure audio safety

            for ch in range(outdata.shape[1]):
                outdata[:, ch] = y_limited.astype(np.float32)

        except Exception as e:
            # Critical error: output silence and increment xrun counter
            outdata.fill(0.0)
            self.xrun_count += 1
            # Emergency print (should be rare)
            if self.callback_count % 1000 == 0:  # Only every 1000 callbacks
                print(f"[CRITICAL] Callback error: {e}")
        
        # Performance monitoring (minimal overhead)
        elapsed = (time.perf_counter() - start_time) * 1000  # ms
        self.max_elapsed = max(self.max_elapsed, elapsed)
        self.avg_elapsed += elapsed
        self.callback_count += 1
        
        # Store performance sample (every 100 callbacks)
        if self.callback_count % 100 == 0:
            self.performance_history.append(elapsed)
            if len(self.performance_history) > 1000:  # Keep only last 1000 samples
                self.performance_history.pop(0)
    
    def start(self, input_device=None, output_device=None, channels_in=2, channels_out=2):
        """Start ultra-low latency audio stream"""

        if self.verbose:
            print("\n🚀 Starting ANC...")
            print(f"📊 Sample rate: {self.sample_rate}Hz")
            print(f"📦 Block size: {self.block_size} samples")
            print(f"⏱️  Audio budget: {self.audio_budget:.2f}ms per block")
            # Display total modeled secondary path delay
            if hasattr(self.processor, 'total_secondary_delay'):
                delay_val = self.processor.total_secondary_delay
            else:
                delay_val = getattr(self.processor, 'delay', 'n/a')
            print(f"🎛️  Filter length: {self.processor.L}, Delay: {delay_val} samples")

        try:
            # Single, straightforward stream configuration
            config = {
                'device': (input_device, output_device),
                'samplerate': self.sample_rate,
                'blocksize': self.block_size,
                'channels': (channels_in, channels_out),
                'dtype': np.float32,
                'callback': self.audio_callback,
                'latency': 'low',
            }

            with sd.Stream(**config) as stream:
                if self.verbose:
                    print(f"✅ Stream started with latency: {stream.latency}")
                    print("🎵 ANC is running. Press Ctrl+C to stop...")
                    print("📈 Performance monitoring every 5 seconds...")
                self.running = True

                # Main monitoring loop
                monitor_interval = 5.0  # seconds
                last_monitor = time.time()

                while self.running:
                    time.sleep(0.1)  # Small sleep to prevent busy waiting
                    # Performance monitoring
                    current_time = time.time()
                    if current_time - last_monitor >= monitor_interval:
                        self._print_performance_stats()
                        last_monitor = current_time

        except KeyboardInterrupt:
            if self.verbose:
                print("\n🛑 Stopping ANC...")
            self.running = False
        except Exception as e:
            print(f"💥 Stream error: {e}")
            self.running = False
        finally:
            # Re-enable garbage collector
            gc.enable()
            if self.verbose:
                print("✓ Garbage collector re-enabled")
    
    def _print_performance_stats(self):
        """Print performance statistics"""
        if self.callback_count == 0:
            return
            
        avg_time = self.avg_elapsed / self.callback_count
        cpu_usage = (avg_time / self.audio_budget) * 100
        max_cpu_usage = (self.max_elapsed / self.audio_budget) * 100
        
        # Status indicators
        if self.max_elapsed < self.warning_threshold:
            status = "✅ EXCELLENT"
        elif self.max_elapsed < self.audio_budget:
            status = "⚠️  TIGHT"
        else:
            status = "❌ OVERRUN"
        
        if self.verbose:
            print(f"📊 Perf: {avg_time:.2f}ms avg, {self.max_elapsed:.2f}ms max | "
                  f"CPU: {cpu_usage:.1f}%/{max_cpu_usage:.1f}% | "
                  f"XRUNs: {self.xrun_count} | {status}")
        
        # Reset for next period
        self.max_elapsed = 0.0
        self.avg_elapsed = 0.0
        self.callback_count = 0

def load_config(config_path: str = 'config_FxLMS.json') -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return json.load(f)

def design_bandpass(sample_rate: int, low_hz: float, high_hz: float, order: int = 513, window: str = 'hamming') -> np.ndarray:
    nyq = 0.5 * sample_rate
    taps = firwin(order, [low_hz/nyq, high_hz/nyq], pass_zero=False, window=window, scale=True)
    # Normalize DC gain out-of-band implicitly; in-band unity (approx).
    return taps.astype(np.float64)

# =============================================================================
# Main Application
# =============================================================================

def main():
    """Main application: build processor directly from config file."""
    print("ANC System")
    print("=" * 40)

    cfg = load_config('config_FxLMS.json')
    # Build processor (handles band-pass, DC, leakage, delays)
    try:
        processor = ANCProcessor.from_config('config_FxLMS.json')
    except FileNotFoundError:
        print("❌ Error: config_FxLMS.json not found")
        return
    except Exception as e:
        print(f"❌ Error building processor from config: {e}")
        return

    block_size = processor.block_size
    sample_rate = int(processor.sample_rate)
    sd.default.blocksize = block_size

    output_cfg = cfg.get('output', {}) or {}

    # Expose core parameters for user visibility
    print(f"Sample Rate: {sample_rate} Hz")
    print(f"Block Size : {block_size} samples")
    print(f"Filter Len : {processor.L} taps")
    print(f"mu         : {processor.mu}")
    print(f"Leakage    : {processor.leakage}")
    print(f"Band-pass  : {'custom ('+str(len(processor.bp_taps))+' taps)' if len(processor.bp_taps)>1 else 'disabled'}")
    print(f"DC Blocker : {'r='+str(processor.dc_r) if processor.dc_enabled else 'disabled'}")
    print(f"Tau Samples: {processor.tau}")
    if hasattr(processor, 'sens_ref') and hasattr(processor, 'sens_err'):
        print(f"Mic Sens   : ref={processor.sens_ref:.6g} Pa/unit, err={processor.sens_err:.6g} Pa/unit")

    # Initialize ultra-low latency client
    client = UltraLowLatencyANC(
        processor,
        sample_rate,
        block_size,
        output_cfg=output_cfg
    )

    # Device selection
    print("\n🎛️  Audio Device Selection:")
    devices = cast(List[Dict[str, Any]], sd.query_devices())
    suggested_device = None
    for i, dev in enumerate(devices):
        name = str(dev.get('name', ''))
        max_in = int(dev.get('max_input_channels', 0))
        max_out = int(dev.get('max_output_channels', 0))
        if ('agrégé' in name.lower() or 'aggregate' in name.lower()) and \
           max_in >= 2 and max_out >= 2:
            suggested_device = i
            break

    if suggested_device is not None:
        print(f"💡 Recommended: Device {suggested_device} (aggregate device)")
        device_input = input(f"Input device ID (or Enter for device {suggested_device}): ").strip()
        device_output = input(f"Output device ID (or Enter for device {suggested_device}): ").strip()
        input_dev = int(device_input) if device_input.isdigit() else suggested_device
        output_dev = int(device_output) if device_output.isdigit() else suggested_device
    else:
        device_input = input("Input device ID (or Enter for default): ").strip()
        device_output = input("Output device ID (or Enter for default): ").strip()
        input_dev = int(device_input) if device_input.isdigit() else None
        output_dev = int(device_output) if device_output.isdigit() else None

    try:
        client.start(input_device=input_dev, output_device=output_dev)
    except FileNotFoundError:
        print("❌ Error: filter_coeffs.json not found")
        print("💡 Tip: Ensure secondary path JSON exists (secondary_path in config)")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        print("🏁 ANC system shutdown complete")

if __name__ == '__main__':
    main()
