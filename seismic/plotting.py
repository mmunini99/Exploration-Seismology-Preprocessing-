import numpy as np
import matplotlib.pyplot as plt


def _clip_val(gather, pctl=98):
    """Symmetric clip value for gray-scale display, based on amplitude percentile."""
    return np.percentile(np.abs(gather), pctl) or 1.0


def plot_shot_gather(data, headers, dt, ffid, ax=None, pctl=98, title=None):

    mask = headers["ffid"] == ffid
    if mask.sum() == 0:
        raise ValueError(f"No traces found for FFID={ffid}")

    gather = data[mask]
    offsets = headers["offset"][mask]

    order = np.argsort(offsets)
    gather = gather[order]
    offsets = offsets[order]

    n_samples = gather.shape[1]
    t_max = n_samples * dt

    clip = _clip_val(gather, pctl)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 8))
    ax.imshow(
        gather.T,
        cmap="gray",
        aspect="auto",
        vmin=-clip,
        vmax=clip,
        extent=[offsets.min(), offsets.max(), t_max, 0],
    )
    ax.set_xlabel("Offset [m]")
    ax.set_ylabel("Time [s]")
    ax.set_title(title or f"Shot gather FFID={ffid}")
    if ax is None:
        plt.show()
    else:
        return ax


def plot_common_offset_gather(data, headers, dt, offset_value, tolerance=1.0,
                               ax=None, pctl=98, title=None, sort_by="cdp"):

    mask = np.where(headers["offset"] == offset_value)[0]
    if mask.sum() == 0:
        raise ValueError(f"No traces found near offset={offset_value} (tol={tolerance})")

    gather = data[mask]

    if sort_by is not None:
        order = np.argsort(headers[sort_by][mask])
        gather = gather[order]

    n_samples = gather.shape[1]
    t_max = n_samples * dt

    clip = _clip_val(gather, pctl)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 8))
    ax.imshow(
        gather.T,
        cmap="gray",
        aspect="auto",
        vmin=-clip,
        vmax=clip,
        extent=[0, gather.shape[0], t_max, 0],
    )
    ax.set_xlabel("Trace index")
    ax.set_ylabel("Time [s]")
    ax.set_title(title or f"Common offset gather (offset≈{offset_value} m)")
    if ax is None:
        plt.show()
    else:
        return ax


def plot_first_and_last_shot(data, headers, dt, pctl=98):
    
    ffids = np.unique(headers["ffid"])
    first_ffid, last_ffid = ffids.min(), ffids.max()

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    plot_shot_gather(data, headers, dt, first_ffid, ax=axes[0], pctl=pctl,
                      title=f"First shot gather (FFID={first_ffid})")
    plot_shot_gather(data, headers, dt, last_ffid, ax=axes[1], pctl=pctl,
                      title=f"Last shot gather (FFID={last_ffid})")
    plt.tight_layout()
    plt.show()


def plot_first_and_last_common_offset(data, headers, dt, pctl=98, sort_by="cdp"):
    
    min_offset = headers["offset"].min()
    max_offset = headers["offset"].max()

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    plot_common_offset_gather(data, headers, dt, min_offset, ax=axes[0], pctl=pctl,
                               sort_by=sort_by, title=f"Nearest common offset (~{min_offset} m)")
    plot_common_offset_gather(data, headers, dt, max_offset, ax=axes[1], pctl=pctl,
                               sort_by=sort_by, title=f"Farthest common offset (~{max_offset} m)")
    plt.tight_layout()
    plt.show()


def plot_decon_qc(data_pre, data_post, headers, dt, ffid, pctl=99):

    mask = headers["ffid"] == ffid
    if mask.sum() == 0:
        raise ValueError(f"No traces found for FFID={ffid}")
 
    offsets = headers["offset"][mask]
    order = np.argsort(offsets)
 
    g_b = data_pre[mask][order].T   # (n_samples, n_traces_in_shot)
    g_a = data_post[mask][order].T
    off_sorted = offsets[order]
 
    n_samples = g_b.shape[0]
    t_axis = np.arange(n_samples) * dt
 
    clip_b = np.percentile(np.abs(g_b), pctl) + 1e-12
    clip_a = np.percentile(np.abs(g_a), pctl) + 1e-12
 
    freqs = np.fft.rfftfreq(n_samples, d=dt)
    spec_b = np.mean(np.abs(np.fft.rfft(g_b, axis=0)), axis=1)
    spec_b = spec_b / spec_b.max()
    spec_a = np.mean(np.abs(np.fft.rfft(g_a, axis=0)), axis=1)
    spec_a = spec_a / spec_a.max()
 
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1], sharey=ax0)
    ax2 = fig.add_subplot(gs[1, :])
 
    ext = [off_sorted[0], off_sorted[-1], t_axis[-1], 0]
    ax0.imshow(g_b, aspect="auto", cmap="gray", vmin=-clip_b, vmax=clip_b, extent=ext)
    ax0.set_title(f"Pre-decon  (clip ±{clip_b:.2e})")
    ax0.set_xlabel("Offset [m]")
    ax0.set_ylabel("Time [s]")
 
    ax1.imshow(g_a, aspect="auto", cmap="gray", vmin=-clip_a, vmax=clip_a, extent=ext)
    ax1.set_title(f"Post-decon (clip ±{clip_a:.2e})")
    ax1.set_xlabel("Offset [m]")
 
    ax2.semilogy(freqs, spec_b, label="Pre-decon", lw=1.2)
    ax2.semilogy(freqs, spec_a, label="Post-decon", lw=1.2)
    ax2.set_xlim(0, 0.5 / dt)
    ax2.set_xlabel("Frequency [Hz]")
    ax2.set_ylabel("Normalized spectrum (log)")
    ax2.set_title(f"Mean spectrum, shot FFID={ffid} — spectral whitening check")
    ax2.legend()
    ax2.grid(alpha=0.3)
 
    plt.tight_layout()
    plt.show()

def plot_cmp_gather(cmp_gather, nmo_gather, cmp_offsets, dt, title="CMP Gather vs NMO",
                     clip_pct=99, cmap="gray"):

    n_traces, n_samples = cmp_gather.shape
    t_axis = np.arange(n_samples) * dt

    # sort traces by offset so the plot reads left-to-right correctly
    order = np.argsort(cmp_offsets)
    gather_sorted = cmp_gather[order]
    offsets_sorted = cmp_offsets[order]

    nmo_sorted = nmo_gather[order]
  

    # symmetric clip for balanced color scale
    vmax = np.percentile(np.abs(gather_sorted), clip_pct)
    vmin = -vmax

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # First plot
    im1 = ax[0].imshow(
        gather_sorted.T,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=[
            offsets_sorted.min(),
            offsets_sorted.max(),
            t_axis[-1],
            t_axis[0],
        ],
    )

    ax[0].set_xlabel("Offset (m)")
    ax[0].set_ylabel("Time (s)")
    ax[0].set_title("Gather")

    # Second plot
    im2 = ax[1].imshow(
        nmo_sorted.T,      # <-- replace with your second gather
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=[
            offsets_sorted.min(),
            offsets_sorted.max(),
            t_axis[-1],
            t_axis[0],
        ],
    )

    ax[1].set_xlabel("Offset (m)")
    ax[1].set_ylabel("Time (s)")
    ax[1].set_title("NMO Gather")

    plt.tight_layout()
    plt.show()


def plot_velocity_field(velocity_field, all_cdps, full_t0):

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(
        velocity_field.T,
        aspect="auto",
        cmap="jet",
        extent=[all_cdps[0], all_cdps[-1], full_t0[-1], 0],
    )
    ax.set_xlabel("CDP")
    ax.set_ylabel("Time [s]")
    ax.set_title("Velocity Field")
    plt.colorbar(im, ax=ax, label="Velocity [m/s]")
    plt.show()


def wiggle_plot(data, dt, scale):

    ntr, ns = data.shape
    t = np.arange(ns) * dt

    data = data / np.max(np.abs(data))  # normalize

    plt.figure(figsize=(10, 8))

    for i in range(ntr):
        tr = data[i]

        x = i + tr * scale

        plt.plot(x, t, 'k', linewidth=0.5)
        plt.fill_betweenx(t, i, x, where=(tr > 0), color='k')

    plt.gca().invert_yaxis()
    plt.xlabel("Trace")
    plt.ylabel("Time (s)")
    plt.title("Shot Gather")
    plt.show()

def plot_full_stacking(stacked_data, cdp_list, t_axis):

    vmax = np.percentile(np.abs(stacked_data.T), 98) + 1e-9
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.imshow(stacked_data.T, aspect='auto', cmap='gray', vmin=-vmax, vmax=vmax,
            extent=[cdp_list[0], cdp_list[-1], t_axis[-1], 0])
    ax.set_xlabel('CMP'); ax.set_ylabel('Tempo [s]')
    ax.set_title('Stack post-NMO')
    plt.tight_layout()
    plt.show()