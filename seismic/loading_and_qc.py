import segyio
import numpy as np

def load_segy(path, neg_off=True):
    if neg_off:
        c = -1
    else:
        c = 1
    with segyio.open(path, ignore_geometry=True) as f:
        dt = segyio.tools.dt(f) / 1e6  # microseconds -> seconds
        data = segyio.tools.collect(f.trace[:])  # shape (n_traces, n_samples)
        headers = dict(
            offset=f.attributes(segyio.TraceField.offset)[:]*c,
            ffid=f.attributes(segyio.TraceField.FieldRecord)[:],
            cdp=f.attributes(segyio.TraceField.CDP)[:],
            src_x=f.attributes(segyio.TraceField.SourceX)[:],
            src_y=f.attributes(segyio.TraceField.SourceY)[:],
            rec_x=f.attributes(segyio.TraceField.GroupX)[:],
            rec_y=f.attributes(segyio.TraceField.GroupY)[:],
            src_depth=f.attributes(segyio.TraceField.SourceDepth)[:],
            rec_depth=f.attributes(segyio.TraceField.ReceiverGroupElevation)[:],
            scalar=f.attributes(segyio.TraceField.SourceGroupScalar)[:]
        )
    return data, headers, dt


def qc_summary(data, headers, dt):
   
    n_traces, n_samples = data.shape
    print(f"Traces: {n_traces}, Samples/trace: {n_samples}, dt: {dt * 1000:.3f} ms")
    print(f"Offset range: {headers['offset'].min()} to {headers['offset'].max()} m")
    print(f"Unique FFIDs: {len(np.unique(headers['ffid']))}")
    trace_energy = np.sum(data.astype(np.float64) ** 2, axis=1)
    dead = np.where(trace_energy == 0)[0]
    print(f"Dead traces: {len(dead)}")
    return dead


def remove_dead_bad_traces(data, energy_threshold_factor):
    """Remove traces that do not have enough energy --> too weak signal"""
    trace_energy = np.sum(data.astype(np.float64) ** 2, axis=1)
    nonzero = trace_energy[trace_energy > 0]
    median_energy = np.median(nonzero) if len(nonzero) else 0
    bad = np.where((trace_energy == 0) | (trace_energy < energy_threshold_factor * median_energy))[0]
    data_clean = data.copy()
    data_clean[bad] = 0
    return data_clean, bad