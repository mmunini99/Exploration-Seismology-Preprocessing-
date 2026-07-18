import numpy as np

def nmo_correction(cmp_gather, offsets, dt, velocity_t0, stretch_mute):

    n_samples = cmp_gather.shape[1]
    t0 = np.arange(n_samples) * dt
    nmo_data = np.zeros_like(cmp_gather, dtype=np.float64)

    for i, x in enumerate(offsets):
        t_x = np.sqrt(t0 ** 2 + (x / velocity_t0) ** 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            stretch = np.where(t0 > 0, (t_x - t0) / np.where(t0 > 0, t0, 1), 0)
        amp = np.interp(t_x, t0, cmp_gather[i], left=0, right=0)
        amp[stretch > stretch_mute] = 0
        nmo_data[i] = amp

    return nmo_data


def stack_cmp(nmo_gather):
    """Fold-normalized stack of an NMO-corrected CMP gather."""
    fold = np.sum(nmo_gather != 0, axis=0)
    fold_safe = np.where(fold == 0, 1, fold)
    return np.sum(nmo_gather, axis=0) / fold_safe
 
 
def build_velocity_field(control_picks, all_cdps, full_t0):

    control_cdps = np.array(sorted(control_picks.keys()))
    V_control = np.array([control_picks[cdp] for cdp in control_cdps])  # (n_control, n_samples)
 
    all_cdps = np.asarray(all_cdps)
    n_samples = len(full_t0)
    velocity_field = np.zeros((len(all_cdps), n_samples))
 
    for j in range(n_samples):
        velocity_field[:, j] = np.interp(all_cdps, control_cdps, V_control[:, j])
 
    return velocity_field
 
 
def nmo_stack_full_line(data, headers, dt, all_cdps, velocity_field, stretch_mute, verbose=True):

    n_samples = data.shape[1]
    stacked_section = np.zeros((len(all_cdps), n_samples), dtype=np.float32)
    fold_section = np.zeros((len(all_cdps), n_samples), dtype=np.int32)
 
    for i, cdp in enumerate(all_cdps):
        cmp_mask = headers["cdp"] == cdp
        if cmp_mask.sum() == 0:
            continue
 
        cmp_gather = data[cmp_mask]
        cmp_offsets = headers["offset"][cmp_mask]
        velocity_t0 = velocity_field[i]
 
        nmo_gather = nmo_correction(cmp_gather, cmp_offsets, dt, velocity_t0, stretch_mute=stretch_mute)
        stacked_section[i] = stack_cmp(nmo_gather)
        fold_section[i] = np.sum(nmo_gather != 0, axis=0)
 
        if verbose and (i + 1) % max(1, len(all_cdps) // 10) == 0:
            print(f"  stacked {i + 1}/{len(all_cdps)} CMPs")
 
    return stacked_section, fold_section
 

