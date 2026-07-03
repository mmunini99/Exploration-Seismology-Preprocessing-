import segyio
import numpy as np
import matplotlib.pyplot as plt


def get_qc_analysis_and_statistics(data_stored):

    ffid, cdp_num = data_stored['ffid'], data_stored['cdp']

    unique_source_idx = np.unique(ffid)
    unique_cdp_idx  = np.unique(cdp_num)
    fold = np.array([np.sum(cdp_num == id) for id in unique_cdp_idx])

    print("\n=== QC STATISTICS ===")
     print(f'Count different shots: {int(unique_source_idx)}')

    print(f'Average fold: {int(np.mean(fold))}')
    print(f'Median fold: {int(np.median(fold))}')
    print(f'Max fold: {fold.max()}')
    print(f'Min fold: {fold.min()}')



def geometry_qc(data_stored):

    ffid, src_x, src_y, offsets, cdp = data_stored['ffid'], data_stored['src_x'], data_stored['src_y'], data_stored['offsets'], data_stored['cdp']

    print("\n=== GEOMETRY QC ===")


    print("Unique FFIDs:", len(np.unique(ffid)))
    print("SourceX constant:", np.all(src_x == src_x[0]).item())
    print("SourceY constant:", np.all(src_y == src_y[0]))
    print("Offset is valid:",np.all(np.isfinite(offsets)))
    print("CDP count:",  len(np.unique(cdp)))




def get_CMP_fold_plot(data_stored):

    cdp_num = data_stored['cdp']

    unique_cdp_idx  = np.unique(cdp_num)
    fold = np.array([np.sum(cdp_num == id) for id in unique_cdp_idx])

    plt.bar(unique_cdp_idx, fold, lw=1)
    plt.set_xlabel('CDP')
    plt.set_ylabel('Fold')
    plt.set_title('Fold x CDP')



def shot_qc(data_stored, n_shots):

    traces, ffid, offsets, t_axis = data_stored['traces'], data_stored['ffid'], data_stored['offsets'], data_stored['t_axis']

    print("\n=== SHOT QC ===")

    shots = np.unique(ffid)

    random_shots = np.random.randint(0, len(shots), n_shots)

    for i in random_shots:
        sid = shots[i]

        mask = ffid == sid
        gather = traces[:, mask]
        off = offsets[mask]

        # sort by offset
        order = np.argsort(off)
        gather = gather[:, order]
        off = off[order]

        plt.figure(figsize=(8, 4))
        plt.imshow(
            gather,
            aspect="auto",
            cmap="gray",
            extent=[off.min(), off.max(), t_axis[-1], 0],
            vmin=-np.percentile(np.abs(gather), 95),
            vmax=np.percentile(np.abs(gather), 95),
        )
        plt.title(f"Shot gather FFID={sid}")
        plt.xlabel("Offset")
        plt.ylabel("Time (s)")
        plt.show()



def offset_qc(data_stored):

    ffid, offsets =  data_stored['ffid'], data_stored['offsets']

    print("\n=== OFFSET QC ===")

    print("Offset min:", np.min(offsets))
    print("Offset max:", np.max(offsets))
    print("Offset std:", np.std(offsets))

    # check per shot consistency
    shots = np.unique(ffid)
    inconsistent = 0

    for s in shots:  # sample for speed
        off = offsets[ffid == s]
        if len(off) > 1 and np.any(np.diff(np.sort(off)) == 0):
            inconsistent += 1

    print("Shots with duplicate offsets (sampled):", inconsistent)


def trace_qc(data_stored):

    traces = data_stored['traces']

    print("\n=== TRACE QC ===")

    nan_count = np.sum(~np.isfinite(traces))
    flat_traces = np.sum(np.std(traces, axis=0) == 0)

    print("Non-finite samples:", nan_count)
    print("Flat (dead) traces:", flat_traces)




def amplitude_qc(data_stored):

    traces = data_stored['traces']
    
    print("\n=== AMPLITUDE STATISTICS QC ===")

    # Flatten all amplitudes
    flat = traces.ravel()

    # Basic statistics
    mean_amp = np.mean(flat)
    std_amp  = np.std(flat)
    min_amp  = np.min(flat)
    max_amp  = np.max(flat)

    print(f"Mean amplitude      : {mean_amp:.6f}")
    print(f"Std amplitude       : {std_amp:.6f}")
    print(f"Min / Max amplitude : {min_amp:.3f} / {max_amp:.3f}")

    # Energy and distribution shape
    abs_amp = np.abs(flat)
    p95 = np.percentile(abs_amp, 95)
    p99 = np.percentile(abs_amp, 99)

    print(f"95th percentile abs : {p95:.3f}")
    print(f"99th percentile abs : {p99:.3f}")

    # Clipping detection (important QC)
    clip_ratio = np.mean(abs_amp > p99)
    print(f"Extreme amplitude ratio (>p99): {clip_ratio:.4f}")

    # Histogram view (QC visual)
    plt.figure(figsize=(6, 3))
    plt.hist(flat, bins=200, color='black', alpha=0.8)
    plt.title("Amplitude Distribution QC")
    plt.xlabel("Amplitude")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    # Log warning hints
    if clip_ratio > 0.01:
        print("WARNING: possible clipping or spikes detected")

    if std_amp == 0:
        print("CRITICAL: all traces are constant (dead data)")





def trace_integrity_qc(data_stored):
    
    traces = data_stored['traces']

    print("\n=== TRACE INTEGRITY QC ===")

    n_traces = traces.shape[1]

    # 1. Finite values check
    finite_mask = np.isfinite(traces)
    bad_samples = np.size(traces) - np.sum(finite_mask)

    print(f"Non-finite samples: {bad_samples}")

    # 2. Dead trace detection (zero variance)
    trace_std = np.std(traces, axis=0)
    dead_traces = np.sum(trace_std == 0)

    print(f"Dead (flat) traces: {dead_traces} / {n_traces}")

    # 3. Near-dead traces (very low energy)
    energy = np.mean(traces**2, axis=0)
    low_energy_threshold = np.percentile(energy, 1)
    low_energy_traces = np.sum(energy < low_energy_threshold)

    print(f"Low-energy traces: {low_energy_traces}")

    # 4. Duplicate trace detection (coarse check)
    # (hash-based lightweight comparison)
    rounded = np.round(traces, 5)
    unique_traces = len({tuple(col) for col in rounded.T})

    print(f"Unique traces (approx): {unique_traces}")

    # 5. Integrity flags
    if bad_samples > 0:
        print("WARNING: dataset contains NaN or Inf values")

    if dead_traces > 0:
        print("WARNING: dead traces detected")

    if unique_traces < 0.9 * n_traces:
        print("WARNING: possible duplicated traces or acquisition error")