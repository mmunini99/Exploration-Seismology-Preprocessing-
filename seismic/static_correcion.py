import numpy as np


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