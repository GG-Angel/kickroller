import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.signal import get_window

_CHUNK_FRAMES = 1024


def stft_bins(
    signal: np.ndarray, n_fft: int, hop: int, first_bin: int, last_bin: int
) -> np.ndarray:
    """Magnitude of the STFT bins `first_bin`..`last_bin` (inclusive).

    Hann window; frames are centered, so frame n is centered on sample
    n * hop. Only the requested bins are computed (a windowed DFT basis), in
    chunks, so memory stays small for full tracks.
    """
    bins = np.arange(first_bin, last_bin + 1)
    window = get_window("hann", n_fft)
    phase = 2.0 * np.pi * np.outer(np.arange(n_fft), bins) / n_fft
    basis = np.hstack(
        [window[:, None] * np.cos(phase), window[:, None] * np.sin(phase)]
    )
    basis = basis.astype(np.float32)

    padded = np.pad(signal.astype(np.float32), n_fft // 2)
    n_frames = 1 + len(signal) // hop
    frames = np.lib.stride_tricks.sliding_window_view(padded, n_fft)[::hop][:n_frames]

    n_bins = len(bins)
    magnitude = np.empty((n_frames, n_bins), dtype=np.float32)
    for start in range(0, n_frames, _CHUNK_FRAMES):
        stop = start + _CHUNK_FRAMES
        projection = frames[start:stop] @ basis
        magnitude[start:stop] = np.hypot(projection[:, :n_bins], projection[:, n_bins:])
    return magnitude


def bin_range(
    n_fft: int, sample_rate: int, fmin: float, fmax: float
) -> tuple[int, int]:
    """First and last STFT bin with a frequency in [fmin, fmax]."""
    bin_hz = sample_rate / n_fft
    first = int(np.ceil(fmin / bin_hz))
    last = min(n_fft // 2, int(np.floor(fmax / bin_hz)))
    if last < first:
        raise ValueError(f"no STFT bin between {fmin} and {fmax} Hz at n_fft={n_fft}")
    return first, last


def log_filterbank(
    n_fft: int, sample_rate: int, fmin: float, fmax: float, bands_per_octave: int
) -> tuple[int, np.ndarray]:
    """Unit-sum triangular filters with log-spaced centers from `fmin`.

    Returns (first_bin, filters[bins, bands]); row 0 is STFT bin `first_bin`.
    """
    bin_hz = sample_rate / n_fft
    steps = np.arange(0, np.log2(fmax / fmin) * bands_per_octave + 1)
    centers = fmin * 2.0 ** (steps / bands_per_octave)
    edges = np.unique(np.round(centers / bin_hz).astype(int))
    edges = edges[edges <= n_fft // 2]
    if len(edges) < 3:
        raise ValueError(f"band {fmin}-{fmax} Hz is too narrow at n_fft={n_fft}")

    first_bin = edges[0]
    filters = np.zeros((edges[-1] - first_bin + 1, len(edges) - 2), dtype=np.float32)
    for band, (lo, mid, hi) in enumerate(zip(edges[:-2], edges[1:-1], edges[2:])):
        lo, mid, hi = lo - first_bin, mid - first_bin, hi - first_bin
        filters[lo : mid + 1, band] = np.linspace(0.0, 1.0, mid - lo + 1)
        filters[mid : hi + 1, band] = np.linspace(1.0, 0.0, hi - mid + 1)
        filters[:, band] /= filters[:, band].sum()
    return int(first_bin), filters


def superflux(
    spectrogram: np.ndarray, lag: int = 1, max_size: int = 3, log_scale: float = 1.0
) -> np.ndarray:
    """SuperFlux onset detection function (Böck & Widmer, 2013).

    Log-compressed magnitude, compared with a frequency max-filtered copy of
    the frame `lag` frames earlier; positive differences are summed over all
    bands. The max filter suppresses flux from small pitch movements.
    """
    log_mag = np.log10(1.0 + log_scale * spectrogram)
    reference = maximum_filter1d(log_mag, size=max_size, axis=1)
    diff = np.zeros_like(log_mag)
    diff[lag:] = log_mag[lag:] - reference[:-lag]
    return np.maximum(diff, 0.0).sum(axis=1)
