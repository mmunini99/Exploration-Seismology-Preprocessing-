import numpy as np
from scipy.signal import butter, filtfilt
from scipy.linalg import solve_toeplitz
from scipy.ndimage import uniform_filter1d



def spiking_decon(trace, dt, params):

    op_len = params['deconvolution_operator_len']
    op_len = int(op_len / 1000 / dt)
    start = int(params['deconvolution_start_n'] / dt)
    end = min(int(params['deconvolution_end_n'] / dt), len(trace))
    pre_whiten = params['deconvolution_pre_whitening']
    design_slice = slice(start, end)

    _design_len = end - start
    if _design_len <= op_len:
        raise ValueError('Window N too short') 


    if not np.all(np.isfinite(trace)) or np.all(trace == 0):
        return trace.copy()
    seg = trace[design_slice]
    if not np.all(np.isfinite(seg)) or np.all(seg == 0):
        return trace.copy()

    r = np.array([np.dot(seg[: len(seg) - k], seg[k:]) for k in range(op_len)])
    if r[0] <= 0 or not np.isfinite(r[0]):
        return trace.copy()
    r[0] *= (1.0 + pre_whiten)

    rhs = np.zeros(op_len); rhs[0] = 1.0
    try:
        op = solve_toeplitz(r, rhs)
    except Exception:
        return trace.copy()
    if not np.all(np.isfinite(op)):
        return trace.copy()


    out = np.convolve(trace, op, mode='full')[:len(trace)]
    if not np.all(np.isfinite(out)):
        return trace.copy()
    return out.astype(np.float32)


def get_pulse_wavelet(data_stored, dt, params):

    traces = data_stored['traces']

    traces_dc = np.empty_like(traces)
    for i in range(traces.shape[1]):
        traces_dc[:, i] = spiking_decon(traces[:, i], dt, params)

    return traces_dc

