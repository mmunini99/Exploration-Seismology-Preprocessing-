from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from statsmodels.stats.diagnostic import acorr_ljungbox

def _robust_center_scale(x):

    b, a = butter(4, 0.1, btype="low") # remove freq under 12Hz
    trend = filtfilt(b, a, x)

    res = x - trend

    med = np.median(res)
    mad = np.median(np.abs(res - med)) 
    return med, mad if mad > 0 else 1e-12

def _max_zero_run(x, thresh=1e-12):

    is_zero = np.abs(x) < thresh
    if not np.any(is_zero):
        return 0
    # trova run consecutive di True
    diffs = np.diff(np.concatenate(([0], is_zero.astype(int), [0])))
    starts = np.where(diffs == 1)[0]
    ends   = np.where(diffs == -1)[0]
    return int(np.max(ends - starts)) if len(starts) else 0


def _spectral_metrics(x, dt, f_lo, f_hi):

    n = len(x)
    X = np.fft.rfft(x * np.hanning(n)) # A Hanning window is applied to reduce spectral leakage.
    Power_Spectrum = np.abs(X) ** 2
    freqs = np.fft.rfftfreq(n, d=dt)

    Power_Spectrum_safe = Power_Spectrum + 1e-20
    geo_mean = np.exp(np.mean(np.log(Power_Spectrum_safe)))
    arith_mean = np.mean(Power_Spectrum_safe)
    flatness = geo_mean / arith_mean  

    if f_hi is None:
        f_hi = freqs[-1]
    in_band = (freqs >= f_lo) & (freqs <= f_hi)
    tot_energy = np.sum(Power_Spectrum)
    out_band_frac = 1.0 - (np.sum(Power_Spectrum[in_band]) / (tot_energy + 1e-20))

    return flatness, out_band_frac

def _snr_windows(x, dt, noise_win_ms, signal_win_ms):

    n0, n1 = [int(t / 1000 / dt) for t in noise_win_ms] # Convert milliseconds to sample indices
    s0, s1 = [int(t / 1000 / dt) for t in signal_win_ms]
    n1 = min(n1, len(x))
    s1 = min(s1, len(x))
    if n1 <= n0 or s1 <= s0:
        return np.nan
    noise_rms_amplitude  = np.sqrt(np.mean(x[n0:n1] ** 2) + 1e-20) 
    signal_rms_amplitude = np.sqrt(np.mean(x[s0:s1] ** 2) + 1e-20)
    if noise_rms_amplitude < 1e-20:
        return np.inf
    return 20 * np.log10(signal_rms_amplitude / noise_rms_amplitude) # decibels, higher is better

def _flag_energy_outlier_gather(data):
    med = np.median(data['energy'])
    mad = np.median(np.abs(data['energy'] - med)) * 1.4826
    mad = mad if mad > 0 else 1e-12
    data['energy_z'] = (data['energy'] - med) / mad
    return data


def _cmp_summary(data):
    return pd.Series({
        'n_traces'   : len(data),
        'n_fail'     : int((data['status'] == 'FAIL').sum()),
        'n_warn'     : int((data['status'] == 'WARN').sum()),
        'n_ok'       : int((data['status'] == 'OK').sum()),
        'frac_bad'   : float(((data['status'] != 'OK').sum()) / len(data)),
        'mean_snr_db': float(np.nanmean(data['snr_db'])),
    })


def qc_single_trace(tr, dt, params):
 
    res = {}
    n = len(tr)
    finite_mask = np.isfinite(tr)

    # check traces are finite --> no missing values
    if not np.all(finite_mask):
        res['status'] = 'FAIL'
        res['reasons'] = ['NaN/Inf presenti']
        res['n_nan'] = int(np.sum(~finite_mask))
        return res

    # check traces are meaningful --> no all 0s
    if np.all(tr == 0):
        res['status'] = 'FAIL'
        res['reasons'] = ['traccia completamente nulla']
        return res

    reasons = []
    status = 'OK'

    # robust statistics
    med, mad = _robust_center_scale(tr)
    std = np.std(tr)
    res['std'] = std
    res['mad'] = mad 

    if std < params['dead_std_min']:
        reasons.append('traccia piatta (std ~ 0)')
        status = 'FAIL'

    clip_thresh = params['clip_thresh_sigma'] * mad * 1.4826
    clip_frac = np.mean(np.abs(tr - med) > clip_thresh)
    res['clip_frac'] = clip_frac
    if clip_frac > params['clip_frac_max']:
        reasons.append(f'Attention ({clip_frac:.4%} samples above {params["clip_thresh_sigma"]}_x_robusSD)')
        status = 'WARN' if status == 'OK' else status

    # checks gaps in traces
    max_zero_run = _max_zero_run(tr)
    max_zero_ms = max_zero_run * dt * 1000
    res['max_zero_run_ms'] = max_zero_ms
    if max_zero_ms > params['zero_run_max_ms']:
        reasons.append(f'Zeros length {max_zero_ms:.0f} ms')
        status = 'WARN' if status == 'OK' else status

    # pdf shapes
    kurt = stats.kurtosis(tr, fisher=False)  # Gaussian (sigma 3)
    res['kurtosis'] = kurt
    if kurt < params['kurtosis_min']:
        reasons.append(f'kurtosis low ({kurt:.1f}) - flat noise signal like')
        status = 'WARN' if status == 'OK' else status
    elif kurt > params['kurtosis_max']:
        reasons.append(f'kurtosis high ({kurt:.1f}) - possible spike anomaly')
        status = 'WARN' if status == 'OK' else status

    # spectrum of frequencies / FT
    flat, out_band = _spectral_metrics(tr, dt, float(params['freq_low_hz']), float(params['freq_high_hz']))
    res['spec_flatness'] = flat
    res['spec_out_band_frac'] = out_band
    if flat < params['spec_flatness_min']:
        reasons.append(f'monocromatic specturm (flatness={flat:.2e})')
        status = 'WARN' if status == 'OK' else status
    if out_band > params['spec_energy_out_band_max']:
        reasons.append(f'{out_band:.0%} outside energy band')
        status = 'WARN' if status == 'OK' else status

    # autocorrelation and time series periodicityparams
    lags = 20
    lb = acorr_ljungbox(tr, lags=[lags], return_df=True)
    p_value = lb["lb_pvalue"].iloc[0]
    if p_value < params['ac_periodicity_thresh']:
        res['corr'] = True
    else:
        res['corr'] = False

    # SNR naive in decibels unit
    snr_db = _snr_windows(tr, dt, params['noise_win_ms'], params['signal_win_ms'])
    res['snr_db'] = snr_db
    if np.isfinite(snr_db) and snr_db < params['snr_min_db']:
        reasons.append(f'SNR low ({snr_db:.1f} dB)')
        status = 'WARN' if status == 'OK' else status

    res['status'] = status
    res['reasons'] = reasons
    return res



def run_autocorr_qc_dataset(data_stored, dt, params):

    traces, ffid, offsets, cmp = data_stored['traces'], data_stored['ffid'], data_stored['offsets'], data_stored['cdp']

    energy_mad_k = params['energy_mad_k']

    n_tr = traces.shape[1]
    rows = []
    for i in range(n_tr):
        tr = traces[:, i]
        r = qc_single_trace(tr, dt, params)
        r['trace_idx'] = i
        r['cmp'] = cmp[i]
        r['offset'] = offsets[i]
        r['ffid'] = ffid[i]
        rows.append(r)

    df = pd.DataFrame(rows)

    # gather cmp level check --> robust z-score on energy level of cmp signals
    df['energy'] = df['std'] ** 2
    df = df.groupby('cmp', group_keys=False).apply(_flag_energy_outlier_gather)

    outlier_mask = df['energy_z'].abs() > energy_mad_k
    df.loc[outlier_mask, 'status'] = df.loc[outlier_mask, 'status'].replace('OK', 'WARN')
    df.loc[outlier_mask, 'reasons'] = df.loc[outlier_mask].apply(
        lambda r: r['reasons'] + [f'Anomaly energy in CMP samples not adj. or agg. (z={r["energy_z"]:.1f} MAD)'], axis=1
    )


    df_cmp = df.groupby('cmp').apply(_cmp_summary).reset_index()

    return df, df_cmp