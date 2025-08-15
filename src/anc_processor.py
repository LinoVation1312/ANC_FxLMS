import json
from typing import Optional, Dict, Any
import numpy as np
from numba import njit

@njit(cache=True)
def fx_lms_block(ref_block, err_block,
                 buffer_x, buffer_x_filt, W,
                 B, A,
                 iir_state_x,
                 ref_delay_buf, ref_delay_len, ref_delay_idx,
                 bp_taps, buffer_x_bp, buffer_y_bp, buffer_e_bp,
                 do_dc, dc_r, dc_apply_err,
                 dc_x_prev_ref, dc_y_prev_ref,
                 dc_x_prev_out, dc_y_prev_out,
                 dc_x_prev_err, dc_y_prev_err,
                 bp_apply_err,
                 sens_ref, sens_err, noise_floor_pa,
                 mu, leakage, do_adaptation,
                 x_idx, x_filt_idx,
                 x_bp_idx, y_bp_idx, e_bp_idx):
    """FxLMS residual-error kernel (block processing).

    Processing sequence per sample (keywords added for traceability):
      [1] REF_IN       : Acquire reference sample
      [2] DC_REF       : (Optional) DC blocking on reference
      [3] BP_REF       : Band-pass filter on reference
      [4] RING_REF     : Update reference ring buffer (time-domain vector x)
      [5] ADAPT_OUT    : Compute adaptive FIR output y = W^T x
      [6] DC_OUT       : (Optional) DC blocking on output
      [7] BP_OUT       : Band-pass filter on anti-noise output
      [8] ERR_SAMPLE   : Read residual error e(n)
      [9] SEC_PATH_IIR : Filter reference through estimated secondary path S0(z)=B/A
      [10] PURE_DELAY  : Apply pure delay z^{-tau}
      [11] RING_FX     : Store filtered & delayed reference (Fx(n)) for adaptation
      [12] LMS_UPDATE  : Leakage + FxLMS weight update + clipping safeguard
    """
    N = ref_block.shape[0]
    L = W.shape[0]
    y_out_block = np.empty(N, dtype=np.float64)
    e_block = np.empty(N, dtype=np.float64)
    a0_inv = 1.0 / A[0] if A[0] != 0 else 1.0
    state_len_x = iir_state_x.shape[0]
    b_len = B.shape[0]
    bp_len = bp_taps.shape[0]
    for n in range(N):
        # [1] REF_IN: acquire raw reference sample (with noise floor subtraction)
        x_in = ref_block[n] * sens_ref - noise_floor_pa
        # Clamp negative values after noise-floor subtraction
        if x_in < 0.0:
            x_in = 0.0
        # [2] DC_REF: optional DC blocking on reference
        if do_dc:
            x_dc = x_in - dc_x_prev_ref[0] + dc_r * dc_y_prev_ref[0]
            dc_x_prev_ref[0] = x_in
            dc_y_prev_ref[0] = x_dc
        else:
            x_dc = x_in
        # [3] BP_REF: band-pass on reference
        x_bp_idx[0] = (x_bp_idx[0] + 1) % bp_len
        buffer_x_bp[x_bp_idx[0]] = x_dc
        x_bp = 0.0
        for k in range(bp_len):
            x_bp += bp_taps[k] * buffer_x_bp[(x_bp_idx[0] - k) % bp_len]
        # [4] RING_REF: update reference buffer
        x_idx[0] = (x_idx[0] + 1) % L
        buffer_x[x_idx[0]] = x_bp
        # [5] ADAPT_OUT: adaptive FIR output y = W^T x
        y = 0.0
        for i in range(L):
            y += W[i] * buffer_x[(x_idx[0] - i) % L]
        # [6] DC_OUT: optional DC blocking on output
        if do_dc:
            y_dc = y - dc_x_prev_out[0] + dc_r * dc_y_prev_out[0]
            dc_x_prev_out[0] = y
            dc_y_prev_out[0] = y_dc
        else:
            y_dc = y
        # [7] BP_OUT: band-pass on anti-noise output
        y_bp_idx[0] = (y_bp_idx[0] + 1) % bp_len
        buffer_y_bp[y_bp_idx[0]] = y_dc
        y_bp = 0.0
        for k in range(bp_len):
            y_bp += bp_taps[k] * buffer_y_bp[(y_bp_idx[0] - k) % bp_len]
        # [8] ERR_SAMPLE: residual error sample (apply DC + BP like ref/out if enabled)
        e_raw = err_block[n] * sens_err - noise_floor_pa
        if e_raw < 0.0:
            e_raw = 0.0
        # Optional DC on error
        if do_dc and dc_apply_err:
            e_dc = e_raw - dc_x_prev_err[0] + dc_r * dc_y_prev_err[0]
            dc_x_prev_err[0] = e_raw
            dc_y_prev_err[0] = e_dc
        else:
            e_dc = e_raw
        # Optional band-pass on error
        if bp_apply_err:
            e_bp_idx[0] = (e_bp_idx[0] + 1) % bp_len
            buffer_e_bp[e_bp_idx[0]] = e_dc
            e = 0.0
            for k in range(bp_len):
                e += bp_taps[k] * buffer_e_bp[(e_bp_idx[0] - k) % bp_len]
        else:
            e = e_dc
        e_block[n] = e
        # [9] SEC_PATH_IIR: filter reference through S0(z)=B/A
        v_ref = x_bp
        for i in range(A.shape[0] - 1):
            v_ref -= A[i + 1] * iir_state_x[i]
        v_ref *= a0_inv
        ref_filt = B[0] * v_ref
        for i in range(1, min(b_len, state_len_x + 1)):
            ref_filt += B[i] * iir_state_x[i - 1]
        for i in range(state_len_x - 1, 0, -1):
            iir_state_x[i] = iir_state_x[i - 1]
        if state_len_x > 0:
            iir_state_x[0] = v_ref
        # [10] PURE_DELAY: apply modelled pure delay
        if ref_delay_len > 0:
            ref_delay_idx[0] = (ref_delay_idx[0] + 1) % ref_delay_len
            delayed_fx = ref_delay_buf[ref_delay_idx[0]]
            ref_delay_buf[ref_delay_idx[0]] = ref_filt
        else:
            delayed_fx = ref_filt
        # [11] RING_FX: store filtered+delayed reference
        x_filt_idx[0] = (x_filt_idx[0] + 1) % L
        buffer_x_filt[x_filt_idx[0]] = delayed_fx
        # [12] LMS_UPDATE: FxLMS weight update (with leakage + clipping safeguard)
        if do_adaptation:
            for i in range(L):
                pos = (x_filt_idx[0] - i) % L
                if leakage > 0.0:
                    W[i] *= (1.0 - leakage)
                W[i] -= mu * e * buffer_x_filt[pos]
                if W[i] > 10.0:
                    W[i] = 10.0
                elif W[i] < -10.0:
                    W[i] = -10.0
        y_out_block[n] = y_bp
    return y_out_block, e_block


class ANCProcessor:
    def __init__(self,
                 secondary_path_json: str,
                 block_size: int,
                 L: int,
                 mu: float,
                 sample_rate: float,
                 bandpass_taps: Optional[np.ndarray] = None,
                 dc_blocker_r: Optional[float] = None,
                 leakage: float = 0.0,
                 extra_output_delay_samples: int = 0,
                 do_adaptation: bool = True,
                 clip_limit: float = 10.0,
                 weight_init_std: float = 1e-3,
                 random_seed: Optional[int] = None,
                 bp_apply_err: bool = False,
                 dc_apply_err: bool = False,
                 sens_ref: float = 1.0,
                 sens_err: float = 1.0,
                 noise_floor_pa: float = 0.0):
        # Secondary path
        with open(secondary_path_json, 'r') as f:
            sp = json.load(f)
        self.B = np.array(sp['B'], dtype=np.float64)
        self.A = np.array(sp['A'], dtype=np.float64)
        tau_ms = float(sp.get('tau_ms', 0.0))
        self.sample_rate = float(sample_rate)
        self.block_size = int(block_size)
        self.physical_delay_samples = int(round(tau_ms * self.sample_rate / 1000.0))
        self.buffer_delay_samples = self.block_size
        self.tau_samples = self.physical_delay_samples + self.buffer_delay_samples + int(max(0, extra_output_delay_samples))
        # Adaptation params
        self.L = int(L)
        self.mu = float(mu)
        self.leakage = float(leakage) if leakage and leakage > 0 else 0.0
        self.do_adaptation = bool(do_adaptation)
        self.clip_limit = float(clip_limit) if clip_limit and clip_limit > 0 else 0.0
        if random_seed is not None:
            np.random.seed(int(random_seed))
        # Buffers / weights
        self.buffer_x = np.zeros(self.L, dtype=np.float64)
        self.buffer_x_filt = np.zeros(self.L, dtype=np.float64)
        self.W = np.random.randn(self.L).astype(np.float64) * float(weight_init_std)
        self.x_idx = np.array([0], dtype=np.int32)
        self.x_filt_idx = np.array([0], dtype=np.int32)
        # Secondary path state
        iir_len = max(len(self.A), len(self.B)) - 1
        self.iir_state_x = np.zeros(max(0, iir_len), dtype=np.float64)
        # Delay buffer
        self.ref_delay_len = max(0, self.tau_samples)
        self.ref_delay_buf = np.zeros(max(1, self.ref_delay_len), dtype=np.float64)
        self.ref_delay_idx = np.array([0], dtype=np.int32)
        # Band-pass taps
        if bandpass_taps is None or getattr(bandpass_taps, 'size', 0) == 0:
            self.bp_taps = np.array([1.0], dtype=np.float64)
        else:
            self.bp_taps = bandpass_taps.astype(np.float64)
        bp_len = self.bp_taps.shape[0]
        self.buffer_x_bp = np.zeros(bp_len, dtype=np.float64)
        self.buffer_y_bp = np.zeros(bp_len, dtype=np.float64)
        self.buffer_e_bp = np.zeros(bp_len, dtype=np.float64)
        self.x_bp_idx = np.array([0], dtype=np.int32)
        self.y_bp_idx = np.array([0], dtype=np.int32)
        self.e_bp_idx = np.array([0], dtype=np.int32)
        # DC blocker
        if dc_blocker_r is None or dc_blocker_r <= 0.0:
            self.dc_enabled = False
            self.dc_r = 0.0
        else:
            self.dc_enabled = True
            self.dc_r = float(dc_blocker_r)
        self.dc_apply_err = bool(dc_apply_err)
        self.bp_apply_err = bool(bp_apply_err)
        # DC states
        self.dc_x_prev_ref = np.array([0.0], dtype=np.float64)
        self.dc_y_prev_ref = np.array([0.0], dtype=np.float64)
        self.dc_x_prev_out = np.array([0.0], dtype=np.float64)
        self.dc_y_prev_out = np.array([0.0], dtype=np.float64)
        self.dc_x_prev_err = np.array([0.0], dtype=np.float64)
        self.dc_y_prev_err = np.array([0.0], dtype=np.float64)
        # Sensitivities & noise floor
        self.sens_ref = float(sens_ref)
        self.sens_err = float(sens_err)
        self.noise_floor_pa = float(noise_floor_pa)
        # Kernel
        self.fx_lms_step = fx_lms_block

    @classmethod
    def from_config(cls, cfg_path: str) -> "ANCProcessor":
        with open(cfg_path, 'r') as f:
            cfg: Dict[str, Any] = json.load(f)
        sample_rate = int(cfg.get('sample_rate', 8000))
        block_size = int(cfg.get('block_size', 128))
        L = int(cfg.get('L', 256))
        mu = float(cfg.get('mu', 0.01))
        secondary_path = str(cfg.get('secondary_path', 'filter_coeffs.json'))
        sens_ref = float(cfg.get('sens_ref', 1.0))
        sens_err = float(cfg.get('sens_err', 1.0))
        noise_floor_spl = float(cfg.get('noise_floor_SPL', 0.0))
        noise_floor_pa = 20e-6 * 10.0**(noise_floor_spl / 20.0) if noise_floor_spl > 0 else 0.0
        # Phase
        phase_cfg = cfg.get('phase', {}) or {}
        extra_samples = int(phase_cfg.get('extra_output_delay_samples', 0))
        if 'extra_output_delay_blocks' in phase_cfg:
            try:
                extra_samples = int(round(float(phase_cfg.get('extra_output_delay_blocks', 0.0)) * block_size))
            except Exception:
                pass
        # Band-pass
        bp_cfg = cfg.get('bandpass', {}) or {}
        if bp_cfg.get('enabled', False):
            try:
                from scipy.signal import firwin  # type: ignore
                order = int(bp_cfg.get('order', 513))
                low_hz = float(bp_cfg.get('low_hz', 40.0))
                high_hz = float(bp_cfg.get('high_hz', 250.0))
                window = str(bp_cfg.get('window', 'hamming'))
                nyq = 0.5 * sample_rate
                taps = firwin(order, [low_hz/nyq, high_hz/nyq], pass_zero=False, window=window, scale=True).astype(np.float64)
            except Exception:
                taps = np.array([1.0], dtype=np.float64)
        else:
            taps = np.array([1.0], dtype=np.float64)
        apply_list = bp_cfg.get('apply_to', []) if isinstance(bp_cfg, dict) else []
        bp_apply_err = ('error' in apply_list) or bool(bp_cfg.get('apply_error', False))
        # DC
        dc_cfg = cfg.get('dc_blocker', {}) or {}
        dc_r = float(dc_cfg.get('r', 0.995)) if dc_cfg.get('enabled', False) else None
        dc_apply_err = bool(dc_cfg.get('apply_error', False))
        # Leakage
        lms_cfg = cfg.get('lms', {}) or {}
        leakage = float(lms_cfg.get('leakage', 0.0))
        # Adaptation
        adapt_cfg = cfg.get('adaptation', {}) or {}
        do_adapt = bool(adapt_cfg.get('enabled', True))
        clip_limit = float(adapt_cfg.get('clip_limit', 10.0))
        # Init
        init_cfg = cfg.get('init', {}) or {}
        weight_init_std = float(init_cfg.get('weight_init_std', 1e-3))
        random_seed = init_cfg.get('random_seed', None)
        inst = cls(
            secondary_path_json=secondary_path,
            block_size=block_size,
            L=L,
            mu=mu,
            sample_rate=sample_rate,
            bandpass_taps=taps,
            dc_blocker_r=dc_r,
            leakage=leakage,
            extra_output_delay_samples=extra_samples,
            do_adaptation=do_adapt,
            clip_limit=clip_limit,
            weight_init_std=weight_init_std,
            random_seed=random_seed,
            bp_apply_err=bp_apply_err,
            dc_apply_err=dc_apply_err,
            sens_ref=sens_ref,
            sens_err=sens_err,
            noise_floor_pa=noise_floor_pa,
        )
        inst._loaded_config = cfg  # type: ignore[attr-defined]
        return inst

    @property
    def tau(self) -> int:
        return self.ref_delay_len

    def reset_state(self):
        self.buffer_x.fill(0.0)
        self.buffer_x_filt.fill(0.0)
        self.W.fill(0.0)
        self.x_idx[0] = 0
        self.x_filt_idx[0] = 0
        self.iir_state_x.fill(0.0)
        self.ref_delay_buf.fill(0.0)
        self.ref_delay_idx[0] = 0
        self.buffer_x_bp.fill(0.0)
        self.buffer_y_bp.fill(0.0)
        self.buffer_e_bp.fill(0.0)
        self.x_bp_idx[0] = 0
        self.y_bp_idx[0] = 0
        self.e_bp_idx[0] = 0
        self.dc_x_prev_ref[0] = 0.0
        self.dc_y_prev_ref[0] = 0.0
        self.dc_x_prev_out[0] = 0.0
        self.dc_y_prev_out[0] = 0.0
        self.dc_x_prev_err[0] = 0.0
        self.dc_y_prev_err[0] = 0.0

    def process_block(self, ref_block, err_block, do_adaptation: bool = True):
        y_block, e_block = self.fx_lms_step(
            ref_block, err_block,
            self.buffer_x, self.buffer_x_filt, self.W,
            self.B, self.A,
            self.iir_state_x,
            self.ref_delay_buf, self.ref_delay_len, self.ref_delay_idx,
            self.bp_taps, self.buffer_x_bp, self.buffer_y_bp, self.buffer_e_bp,
            self.dc_enabled, self.dc_r, self.dc_apply_err,
            self.dc_x_prev_ref, self.dc_y_prev_ref,
            self.dc_x_prev_out, self.dc_y_prev_out,
            self.dc_x_prev_err, self.dc_y_prev_err,
            self.bp_apply_err,
            self.sens_ref, self.sens_err, self.noise_floor_pa,
            self.mu, self.leakage, (self.do_adaptation and do_adaptation),
            self.x_idx, self.x_filt_idx,
            self.x_bp_idx, self.y_bp_idx, self.e_bp_idx
        )
        if self.clip_limit > 0.0:
            np.clip(self.W, -self.clip_limit, self.clip_limit, out=self.W)
        return y_block, e_block
