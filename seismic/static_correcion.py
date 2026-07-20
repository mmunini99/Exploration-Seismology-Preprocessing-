import numpy as np
from scipy.signal import correlate

def water_column_static(data, dt, src_depth, rec_depth, v_water):

    n_traces, n_samples = data.shape
    shifted = np.zeros_like(data)
    for i in range(n_traces):
        t_shift = (src_depth[i] + rec_depth[i]) / v_water
        n_shift = int(round(t_shift / dt))
        if n_shift > 0:
            shifted[i, n_shift:] = data[i, : n_samples - n_shift]
        elif n_shift < 0:
            shifted[i, :n_shift] = data[i, -n_shift:]
        else:
            shifted[i] = data[i]
    return shifted


def get_stack_static_correction(stacked_section, all_cdps, max_shift, pilot_half, dt):
    """
    max_shift in [s]
    """

    max_shift = int(max_shift / dt)

    stack_static = np.zeros_like(stacked_section)          # (n_cdps, n_samples)
    shifts = np.zeros(len(all_cdps), dtype=np.int32)

    for j in range(len(all_cdps)):
        j0, j1 = max(0, j - pilot_half), min(len(all_cdps), j + pilot_half + 1)
        idx = np.r_[j0:j, j + 1:j1]                        # exclude trace j itself from its own pilot
        pilot = stacked_section[idx, :].mean(axis=0)
        tr    = stacked_section[j, :]

        if np.all(pilot == 0) or np.all(tr == 0):
            stack_static[j, :] = tr
            continue

        cc  = correlate(tr, pilot, mode='full')
        lag = np.arange(-len(tr) + 1, len(tr))
        sel = (np.abs(lag) <= max_shift)
        best = lag[sel][np.argmax(cc[sel])]
        shifts[j] = best

        # best = how much tr is delayed relative to pilot 
        if best > 0:
            stack_static[j, :-best] = tr[best:]
        elif best < 0:
            stack_static[j, -best:] = tr[:best]
        else:
            stack_static[j, :] = tr

    return stack_static
