import numpy as np
import segyio



def inspect_first_header(seismic_data_path):
    with segyio.open(seismic_data_path, "r", ignore_geometry=True) as f:
        h = f.header[0]

        print("=== First Trace Header ===")
        print("Field Record:", h[segyio.TraceField.FieldRecord])
        print("Source X (East-West):", h[segyio.TraceField.SourceX])
        print("Source Y (North-South):", h[segyio.TraceField.SourceY])
        print("CDP:", h[segyio.TraceField.CDP])
        print("Trace Sequence Line:", h[segyio.TraceField.TRACE_SEQUENCE_LINE])


def seismic_summary(seismic_data_path):
    with segyio.open(seismic_data_path, "r", ignore_geometry=True) as f:

        n_traces = f.tracecount
        n_samples = len(f.samples)

        delta_t_us = f.bin[segyio.BinField.Interval]
        fmt   = f.bin[segyio.BinField.Format]
        delta_t     = delta_t_us / 1e3

        print("\n=== SEGY DATA SUMMARY ===")
        print(f"Traces           : {n_traces}")
        print(f"Samples in trace : {n_samples}")
        print(f"sampling time    : {delta_t:.3f} ms")
        print(f"Format           : {fmt} (1 = IBM float, 5 = IEEE)")

        return n_traces, n_samples



def load_seismic_volume(seismic_data_path, n_traces, n_samples):
    with segyio.open(seismic_data_path, "r", ignore_geometry=True) as f:

        delta_t = segyio.tools.dt(f) / 1e6

        # Storage creation
        traces  = np.empty((n_samples, n_traces), dtype=np.float32)

        ffid    = np.zeros(n_traces, dtype=np.int32)
        offsets = np.zeros(n_traces, dtype=np.float32)
        cdp_num = np.zeros(n_traces, dtype=np.int32)
        src_x   = np.zeros(n_traces, dtype=np.float32)
        src_y   = np.zeros(n_traces, dtype=np.float32)
        grp_x   = np.zeros(n_traces, dtype=np.float32)
        grp_y   = np.zeros(n_traces, dtype=np.float32)
        cdp_x   = np.zeros(n_traces, dtype=np.float32)
        cdp_y   = np.zeros(n_traces, dtype=np.float32)

        for i in range(n_traces):
            traces[:, i] = f.trace[i]
            h = f.header[i]

            ffid[i]    = h[segyio.TraceField.FieldRecord]
            offsets[i] = h[segyio.TraceField.offset]
            cdp_num[i] = h[segyio.TraceField.CDP]
            src_x[i]   = h[segyio.TraceField.SourceX]
            grp_x[i]   = h[segyio.TraceField.GroupX]
            cdp_x[i]   = h[segyio.TraceField.CDP_X]
            src_y[i]   = h[segyio.TraceField.SourceY]
            grp_y[i]   = h[segyio.TraceField.GroupY]
            cdp_y[i]   = h[segyio.TraceField.CDP_Y]

        # Time axis
        t_axis = np.arange(n_samples) * delta_t

        # Clean data
        bad = ~np.isfinite(traces)
        if bad.any():
            print(f"WARN: {bad.sum()} non-finite samples → set to 0")
            traces[bad] = 0.0

        print("\n=== LOADED VOLUME INFO ===")
        print(f"n_traces  = {n_traces}")
        print(f"n_samples = {n_samples}")
        print(f"delta_t        = {delta_t*1000:.2f} ms")
        print(f"shots     = {len(np.unique(ffid))}")
        print(f"offset    = [{offsets.min():.1f}, {offsets.max():.1f}] m")
        print(f"CDP range = [{cdp_num.min()}, {cdp_num.max()}]")
        print(f"Trace shape = {traces.shape}")

        return {
            "traces": traces,
            "t_axis": t_axis,
            "ffid": ffid,
            "offsets": offsets,
            "cdp": cdp_num,
            "src_x": src_x,
            "grp_x": grp_x,
            "cdp_x": cdp_x,
            "src_y": src_y,
            "grp_y": grp_y,
            "cdp_y": cdp_y,
            "delta_t": delta_t
        }


def initial_seismic_summary(seismic_data_path):
    inspect_first_header(seismic_data_path)
    n_traces, n_samples = seismic_summary(seismic_data_path)
    data = load_seismic_volume(seismic_data_path, n_traces, n_samples)
    return data