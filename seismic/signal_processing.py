from scipy.signal import butter, filtfilt, medfilt
from scipy.ndimage import uniform_filter1d
import numpy as np

def butterworth_filter(data, dt, low_hz, high_hz, order):
    
    fs = 1.0 / dt
    nyq = 0.5 * fs # Nyquist frequency
    norm_low_f = low_hz / nyq
    norm_high_f = high_hz / nyq
    b, a = butter(order, [norm_low_f, norm_high_f], btype="band")
    return filtfilt(b, a, data, axis=1)


def median_filter_despike(data, kernel_size):
    
    return medfilt(data, kernel_size=(1, kernel_size))

def spherical_divergence_correction(data, t_axis, dt, power=2.0, v_rms=None):

    t = t_axis.copy()
    t[0] = dt  # avoid zero-division at t=0
    gain = (v_rms ** 2) * t if v_rms is not None else t ** power
    return data * gain[np.newaxis, :]


def AGC(data, dt, window):
    
    n_traces, _ = data.shape
    win = max(int(round(window / dt)), 1)
    kernel = np.ones(win) / win
    out = np.zeros_like(data, dtype=np.float64)
    for i in range(n_traces):
        trace = data[i].astype(np.float64)
        power = np.convolve(trace ** 2, kernel, mode="same")
        rms = np.sqrt(power + 1e-12)
        out[i] = trace / rms
    return out

def balance_amplitude(data, dt, win_sec):

    w = max(int(win_sec / dt), 5)
    if w % 2 == 0:
        w += 1
    env = np.sqrt(uniform_filter1d(data**2, size=w, axis=1) + 1e-12)

    return data / (env + 1e-6)