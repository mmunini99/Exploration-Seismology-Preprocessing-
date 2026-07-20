import numpy as np
from scipy.signal import detrend, hilbert
from scipy.linalg import solve_toeplitz


def estimate_design_slice(trace, dt, smooth_win, k, noise_frac):
 
    trace = np.asarray(trace, dtype=np.float64)
    N = len(trace)

    envelope = np.abs(hilbert(trace))

    # simple moving average smoothing
    kernel = np.ones(smooth_win) / smooth_win
    smoothed = np.convolve(envelope, kernel, mode="same")

    tail_start = int(N * (1 - noise_frac))
    noise_floor = np.median(smoothed[tail_start:])
    if noise_floor <= 0 or not np.isfinite(noise_floor):
        noise_floor = np.median(smoothed) * 0.1  # fallback

    threshold = k * noise_floor
    active_idx = np.where(smoothed > threshold)[0]

    if len(active_idx) == 0:
        # fallback: use full trace
        P0, P1 = 0, N - 1
    else:
        P0, P1 = active_idx[0], active_idx[-1]

    t0_window = P0 * dt
    tT_window = P1 * dt

    return t0_window, tT_window

def correlation_time_1e(signal, dt, max_lag=None):
    fs = 1.0 / dt
    x = np.asarray(signal, dtype=float)
    x = detrend(x)
    N = len(x)

    if max_lag is None:
        max_lag = N // 2

    ac = np.correlate(x, x, mode='full')
    ac = ac[N - 1:N + max_lag]
    ac /= ac[0]

    threshold = np.exp(-1)
    idx = np.where(ac < threshold)[0]

    if len(idx) == 0:
        return None, None, ac

    lag = idx[0]
    Tc = lag / fs

    return Tc, lag, ac

def estimate_operator_length(trace, dt, t0_window, tT_window, safety_factor, fallback_op_dec_len):

    P0 = int(t0_window / dt)
    P1 = int(tT_window / dt)
    seg = trace[P0:P1]

    if len(seg) < 4:
        return fallback_op_dec_len

    Tc, lag, ac = correlation_time_1e(seg, dt)

    if Tc is None or Tc <= 0:
        return fallback_op_dec_len

    op_dec_len = safety_factor * Tc
    return op_dec_len

def compute_params(op_dec_len, t0_window, tT_window, dt, len_signal):
    L = int(op_dec_len / dt)
    P0 = int(t0_window / dt)
    P1 = min(int(tT_window / dt), len_signal)
    design_slice = slice(P0, P1)
    return L, design_slice


def spiking_decon(trace, op_len, pre_whiten, design_slice):
    trace = np.asarray(trace, dtype=np.float64)

    if (not np.all(np.isfinite(trace))) or np.all(trace == 0):
        return trace.astype(np.float32)

    seg = trace[design_slice].copy()

    if len(seg) < op_len:
        return trace.astype(np.float32)

    if np.all(seg == 0):
        return trace.astype(np.float32)

    seg = detrend(seg) 

    r = np.correlate(seg, seg, mode="full")
    r = r[len(seg) - 1:len(seg) - 1 + op_len]

    if (r[0] <= 0) or (not np.isfinite(r[0])):
        return trace.astype(np.float32)

    r[0] *= (1.0 + pre_whiten)

    y = np.zeros(op_len)
    y[0] = 1.0

    try:
        op = solve_toeplitz((r, r), y)
    except Exception:
        return trace.astype(np.float32)

    out = np.convolve(trace, op, mode="same")

    if not np.all(np.isfinite(out)):
        return trace.astype(np.float32)

    return out.astype(np.float32)


def auto_estimate_params(ref_trace, dt, len_signal, safety_factor, smooth_win, k, noise_frac, fallback_op_dec_len):

    t0_window, tT_window = estimate_design_slice(ref_trace, dt, smooth_win, k, noise_frac)

    op_dec_len = estimate_operator_length(ref_trace, dt, t0_window, tT_window, safety_factor, fallback_op_dec_len)

    L, design_slice = compute_params(op_dec_len, t0_window, tT_window, dt, len_signal)

    print(f"[auto_estimate_params] t0={t0_window:.3f}s  tT={tT_window:.3f}s  "
          f"op_dec_len={op_dec_len:.4f}s  L={L} samples")

    return L, design_slice

def deconvolve_all_traces_auto(data, dt, pre_white, ref_trace_idx, safety_factor, smooth_win, k, noise_frac, fallback_op_dec_len):

    n_traces, len_signal = data.shape

    if ref_trace_idx is None:
        ref_trace = data.mean(axis=0)
    else:
        ref_trace = data[ref_trace_idx]

    L, design_slice = auto_estimate_params(ref_trace, dt, len_signal, safety_factor, smooth_win, k, noise_frac, fallback_op_dec_len)

    traces_dc = np.empty_like(data)

    for i in range(n_traces):
        traces_dc[i] = spiking_decon(data[i], L, pre_white, design_slice)

        if (i + 1) % max(1, n_traces // 10) == 0:
            print(f"{i+1}/{n_traces} traces processed")

    return traces_dc