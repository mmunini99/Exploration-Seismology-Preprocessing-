import numpy as np


def apply_coord_scalar(coord, scalar):

    scalar = np.where(scalar == 0, 1, scalar)
    return np.where(scalar > 0, coord * scalar, coord / np.abs(scalar))


def get_dx(headers):

    sx = apply_coord_scalar(headers['src_x'].astype(np.float64), headers['scalar'])
    sy = apply_coord_scalar(headers['src_y'].astype(np.float64), headers['scalar'])
    rx = apply_coord_scalar(headers['rec_x'].astype(np.float64), headers['scalar'])
    ry = apply_coord_scalar(headers['rec_y'].astype(np.float64), headers['scalar'])

    cdp_x = (sx + rx) / 2.0
    cdp_y = (sy + ry) / 2.0

    cdp = headers['cdp']
    unique_cdps = np.unique(cdp)
    coords = np.array([[cdp_x[cdp == c].mean(), cdp_y[cdp == c].mean()] for c in unique_cdps])

    dists = np.sqrt(np.diff(coords[:, 0])**2 + np.diff(coords[:, 1])**2)
    dx = dists.mean()

    return dx

def kirchhoff_migration(stack_in, t_axis_, dx, v_t, max_aperture_m):

    nt, nx = stack_in.shape
    dt_   = t_axis_[1] - t_axis_[0]
    out   = np.zeros_like(stack_in, dtype=np.float32)
    half  = int(max_aperture_m / dx)
    dx_ap = np.arange(-half, half + 1) * dx       # offsets relativ (m)

    for it0 in range(nt):
        t0v = t_axis_[it0]
        v   = v_t[it0]
        if v <= 0 or t0v <= 0:
            continue

        t_smear  = np.sqrt(t0v ** 2 + (2.0 * dx_ap / v) ** 2)
        it_smear = np.round(t_smear / dt_).astype(np.int64)
        obliquity = t0v / np.maximum(t_smear, 1e-9)   # cos(theta), 1.0 at apex, ->0 at aperture edge

        sum_out = np.zeros(nx, dtype=np.float32)
        wsum_out = np.zeros(nx, dtype=np.float32)

        for k_idx in range(2 * half + 1):
            ts = it_smear[k_idx]
            if ts < 0 or ts >= nt:
                continue
            k   = k_idx - half
            w   = obliquity[k_idx]
            row = stack_in[ts, :] * w
            if k >= 0:
                sum_out[: nx - k]  += row[k:]
            else:
                sum_out[-k:]       += row[: nx + k]
 
            wsum_out += w

        out[it0, :] = sum_out / np.maximum(wsum_out, 1e-9)

        if (it0 + 1) % max(1, nt // 10) == 0:
            print(f'  Done t={it0 + 1}/{nt}')
    return out