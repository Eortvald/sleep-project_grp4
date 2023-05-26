from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from librosa.display import specshow
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.patches as patches

def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique), loc='upper right')

def plot_data(
    data: np.ndarray,
    events: np.ndarray,
    fs: int,
    channel_names: Optional[List[str]] = None,
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:

    # Get current axes or create new
    if ax is None:
        fig, ax = plt.subplots(figsize=(25, 4))
        ax.set_xlabel("Time (s)")
    else:
        fig = ax.get_figure()
        ax.tick_params(
            axis="x",  # changes apply to the x-axis
            which="both",  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False,
        )  # labels along the bottom edge are off

    C, T = data.shape
    time_vector = np.arange(T) / fs

    assert (
        len(channel_names) == C
    ), f"Channel names are inconsistent with number of channels in data. Received {channel_names=} and {data.shape=}"

    # Plot events
    for event_label in np.unique(events[:, -1]):
        event_label_dict = {0.0: 'Arousal', 1.0: 'Leg Movement', 2.0: 'Sleep-disordered Breathing'}
        if event_label == 0.0:
            color = "r"
        elif event_label == 1.0:
            color = "yellow"
        elif event_label == 2.0:
            color = "cornflowerblue"
        class_events = events[events[:, -1] == event_label, :-1] * T / fs
        for evt_start, evt_stop in class_events:
            label = np.unique(event_label_dict[event_label])
            ax.axvspan(evt_start, evt_stop, facecolor=color, alpha=0.6 if not color=='yellow' else 0.8, edgecolor=None, label = label)
            legend_without_duplicate_labels(ax)

    # Calculate the offset between signals
    data = (
        2
        * (data - data.min(axis=-1, keepdims=True))
        / (data.max(axis=-1, keepdims=True) - data.min(axis=-1, keepdims=True))
        - 1
    )
    offset = np.zeros((C, T))
    for idx in range(C - 1):
        # offset[idx + 1] = -(np.abs(np.min(data[idx])) + np.abs(np.max(data[idx + 1])))
        offset[idx + 1] = -2 * (idx + 1)

    # Plot signals
    ax.plot(time_vector, data.T + offset.T, color="gray", linewidth=0.1)

    # Adjust plot visuals
    ax.set_xlim(time_vector[0], time_vector[-1])
    ax.set_yticks(ticks=offset[:, 0], labels=channel_names)
    ax.set_title(title)

    font = {'family': 'Times new roman',
            'weight': 'bold',
            'size': 20}

    plt.rc('font', **font)
    return fig, ax


def plot_spectrogram(
    data: np.ndarray,
    fs: int,
    step_size: int,
    window_length: int,
    nfft: int,
    ax: Axes = None,
    display_type: str = "hz",
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
) -> Tuple[Figure, Axes]:

    # Get current axes or create new
    if ax is None:
        fig, ax = plt.subplots(3, 3, figsize=(24,9))
    else:
        fig = ax.get_figure()
    itn = {1: 'c4', 2: 'eog l', 3: 'eog r', 4: 'chin', 5: 'leg l', 6: 'leg r', 7: 'nasal', 8: 'abdo', 9: 'thor'}
    for i in range(3):
        for j in range(3):
            specshow(
                data[3 * i + j + 1],
                sr=fs,
                hop_length=step_size,
                win_length=window_length,
                n_fft=nfft,
                y_axis=display_type if j == 0 else None,
                x_axis="time" if i == 2 else None,
                ax=ax[i,j],
                fmin=fmin,
                fmax=fmax,
                cmap='magma'
            )
            # Build a rectangle in axes coords
            left, width = .25, .5
            bottom, height = .25, .5
            right = left + width
            top = bottom + height
            p = plt.Rectangle((left, bottom), width, height, fill=False)
            p.set_transform(ax[i, j].transAxes)
            p.set_clip_on(False)
            ax[i, j].add_patch(p)
            ax[i, j].text(0.95, top, itn[3 * i + j + 1],
                    horizontalalignment='right',
                    verticalalignment='bottom',
                    transform=ax[i, j].transAxes,
                    color='w',
                    size=24)

    plt.savefig('D:/10channel/3x3_spect.png', dpi=200)

    return fig, ax
