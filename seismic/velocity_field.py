import numpy as np
import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt

def semblance_panel(cmp_gather, offsets, dt, v_range, t_step=4, win_len=11, stretch_limit=0.4):

    n_samples = cmp_gather.shape[1]
    t0_axis = np.arange(0, n_samples, t_step) * dt
    semblance_store = np.zeros((len(t0_axis), len(v_range)))
    half_win = win_len // 2


    for v_idx, v in enumerate(v_range):
        for t_idx, t0 in enumerate(t0_axis):

            t_x = np.sqrt(t0 ** 2 + (offsets / v) ** 2)
            sample  = t_x / dt # sample coord.
            i0 = np.floor(sample).astype(int) # make them indexes
            f = sample  - i0 # distance between actual time and gird time we have
            with np.errstate(divide="ignore", invalid="ignore"):
                stretch = np.where(t0 > 0, (t_x - t0) / t0, 0.0)

            valid_trace = ((i0 >= 0) & (i0 < n_samples-1) & (stretch <= stretch_limit))

            if valid_trace.sum() < 2:
                continue

            num, den = 0.0, 0.0

            for k in range(-half_win, half_win+1):

                samp = i0 + k
                valid = (valid_trace & (samp >= 0) & (samp < n_samples-1))

                if valid.sum() < 2:
                    continue

                amps = ((1-f[valid]) * cmp_gather[valid, samp[valid]] + f[valid] * cmp_gather[valid, samp[valid]+1])

                num += np.sum(amps)**2
                den += (valid.sum() * np.sum(amps**2))


            if den > 0:
                semblance_store[t_idx, v_idx] = (num / (den + 1e-12))

    return semblance_store, t0_axis

def run_next_key(STORE, keys_iter, semblances, full_t0, v_range, on_complete=None):
    try:
        KEY = next(keys_iter)
    except StopIteration:
        print("All keys done.")
        if on_complete is not None:
            on_complete(STORE)
        return

    print(f"Select velocity for CDP {KEY} please")
    STORE[KEY] = dict()

    picker = InteractiveVelocityPicker(semblances[KEY][0], semblances[KEY][1], v_range)
    picker.show()

    save_btn = widgets.Button(description=f"Save picks for {KEY} & continue")
    display(save_btn)

    def on_save(b):
        times_picked, velocities_picked = picker.get_picks()
        try:
            velocity_t0 = picker.get_velocity_function(full_t0)
        except ValueError as e:
            print(f"{KEY}: {e} — Cannot be picked less than 2 points, otherwise issue on velocity filed estimation")
            return
        STORE[KEY]['times'] = times_picked
        STORE[KEY]['picked_v'] = velocities_picked
        STORE[KEY]['vel_int'] = velocity_t0
        save_btn.disabled = True
        save_btn.description = f"Saved {KEY}"
        run_next_key(STORE, keys_iter, semblances, full_t0, v_range, on_complete)  # advance to the next semblance

    save_btn.on_click(on_save)

def finished(STORE):
    print("Velocity picking completed.")
    print(f"Number of picked CMPs: {len(STORE)}")

    return STORE


class InteractiveVelocityPicker:

 
    def __init__(self, semblance, t0_axis, v_range, cmap="viridis", figsize=(7, 8), pctl=99):
        self.t0_axis = t0_axis
        self.v_range = v_range
        self.times = []
        self.velocities = []
 
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.imshow(
            semblance,
            cmap=cmap,
            aspect="auto",
            vmin=0,
            vmax=np.nanpercentile(semblance, pctl),
            extent=[v_range[0], v_range[-1], t0_axis[-1], 0],
        )
        self.ax.set_xlabel("Velocity [m/s]")
        self.ax.set_ylabel("Time [s]")
        self.ax.set_title("Left-click: add pick | Right-click: remove last")
        (self.line,) = self.ax.plot([], [], "r.-", markersize=8, linewidth=1.5)
 
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
 
    def show(self):
     
        return self.fig
 
    def _on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        if event.button == 1:      # left click: add
            self.velocities.append(event.xdata)
            self.times.append(event.ydata)
        elif event.button == 3:    # right click: remove last
            if self.times:
                self.times.pop()
                self.velocities.pop()
        self._redraw()
 
    def _redraw(self):
        if self.times:
            order = np.argsort(self.times)
            t_sorted = np.array(self.times)[order]
            v_sorted = np.array(self.velocities)[order]
            self.line.set_data(v_sorted, t_sorted)
        else:
            self.line.set_data([], [])
        self.fig.canvas.draw_idle()
 
    def get_picks(self):
        
        if not self.times:
            return np.array([]), np.array([])
        order = np.argsort(self.times)
        t_sorted = np.array(self.times)[order]
        v_sorted = np.array(self.velocities)[order]
        return t_sorted, v_sorted
 
    def get_velocity_function(self, full_t0):
        
        t_sorted, v_sorted = self.get_picks()
        if len(t_sorted) < 2:
            raise ValueError("Need at least 2 picks to build a velocity function")
        return np.interp(full_t0, t_sorted, v_sorted)