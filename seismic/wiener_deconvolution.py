import numpy as np
from scipy.linalg import solve_toeplitz

def compute_params(op_dec_len, t0_window, tT_window, dt, len_signal):
    """
    op_dec_len in [s]
    t0_window, tT_window, dt in [s]
    """

    L = int(op_dec_len / dt)
    P0 = int(t0_window / dt)
    P1 = min(int(tT_window / dt), len_signal)

    design_slice = slice(P0, P1)


    return L, design_slice


def spiking_decon(trace, op_len, pre_whiten, design_slice):
    """
    Wiener spiking deconvolution of a single trace.
    """

    trace = np.asarray(trace, dtype=np.float64)

    if (not np.all(np.isfinite(trace))) or np.all(trace == 0):
        return trace.astype(np.float32)

    seg = trace[design_slice].copy()

    if len(seg) < op_len:
        return trace.astype(np.float32)

    if np.all(seg == 0):
        return trace.astype(np.float32)

    # Removing the DC component prevents the autocorrelation from being dominated by a constant offset.
    seg -= seg.mean() # 

    # autocorrelation
    r = np.correlate(seg, seg, mode="full")
    r = r[len(seg)-1:len(seg)-1+op_len]

    if (r[0] <= 0) or (not np.isfinite(r[0])):
        return trace.astype(np.float32)

    # pre-whitening
    r[0] *= (1.0 + pre_whiten)

    # define the desired output
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

def deconvolve_all_traces(data, op_dec_len, t0_window, tT_window, dt, len_signal, pre_white):

    traces_dc = np.empty_like(data) # store new deconvolved traces

    L, design_slice = compute_params(op_dec_len, t0_window, tT_window, dt, len_signal)

    for i in range(data.shape[0]):
        traces_dc[i] = spiking_decon(
                                     data[i],
                                     L,
                                     pre_white,
                                     design_slice,
                                    )

        if (i + 1) % max(1, data.shape[0] // 10) == 0:
            print(f"{i+1}/{data.shape[0]} traces processed")


    return traces_dc
