"""Binary Signal Matrix — seed sketch (mother).
Source: archive 'Important 14' (2026-05-30), section 6.

Mechanism: translate market peaks and troughs into a clean digital circuit
pulse for automated logic gates.
  - BINARY MAPPING: every confirmed macro peak -> 1 (expansion/sell trigger);
    every macro dip -> 0 (compression/buy trigger).
  - ZERO-NOISE EXECUTION: strip narrative; execute on rhythmic circuit waves.

Toy: synthetic series with KNOWN peak/trough locations; local-extremum
detector with confirmation lag; measured precision/recall of the 1/0 mapping.
Real market data: [UNMEASURABLE] here.

NOTE from project #3: fixed-block rhythm detection is confirmation, not
discovery (34% false rhythms on pure noise; period misattribution). The
binary matrix inherits that bound — a '1' is a claim about the detector,
not the market's rhythm, until proven otherwise.
"""
import numpy as np


def synth(n_days=1800, seed=7, trend=0.0, noise=1.0, period=180, amp=8.0):
    rng = np.random.default_rng(seed)
    t = np.arange(n_days, dtype=float)
    y = trend * t + amp * np.sin(2 * np.pi * t / period)
    y += rng.normal(0, noise, n_days)
    # ground truth: peaks at t = period/4 + k*period, troughs at 3*period/4 + k*period
    peaks = (period // 4 + np.arange(0, n_days, period)).astype(int)
    troughs = (3 * period // 4 + np.arange(0, n_days, period)).astype(int)
    peaks = peaks[peaks < n_days]
    troughs = troughs[troughs < n_days]
    return y, peaks, troughs


def detect(y, window=30, confirm=10):
    """Local extrema with confirmation lag. Returns (idx, bit) events."""
    n = len(y)
    events = []
    for i in range(window, n - window - confirm):
        seg = y[i - window:i + window + 1]
        c = window
        if y[i] == seg.max() and (y[i] - seg.min()) > 0:
            # confirm it holds for `confirm` more days
            if y[i] == y[i:i + confirm + 1].max():
                events.append((i, 1))
        elif y[i] == seg.min():
            if y[i] == y[i:i + confirm + 1].min():
                events.append((i, 0))
    # dedupe nearby
    out, last = [], -10**9
    for i, b in events:
        if i - last > window:
            out.append((i, b))
            last = i
    return out


def score(events, truth_idx, tol=15):
    hits = sum(any(abs(i - t) <= tol for t in truth_idx) for i, _ in events)
    return hits, len(events), len(truth_idx)


if __name__ == "__main__":
    for noise in (1.0, 4.0, 8.0):
        y, peaks, troughs = synth(noise=noise)
        ev = detect(y)
        hp, ne1, npk = score([(i, b) for i, b in ev if b == 1], peaks)
        ht, ne0, ntr = score([(i, b) for i, b in ev if b == 0], troughs)
        p1 = hp / max(1, ne1)
        p0 = ht / max(1, ne0)
        print(f"noise={noise}: peaks {hp}/{npk} (prec {p1:.2f}) | "
              f"troughs {ht}/{ntr} (prec {p0:.2f})")
    # [UNMEASURABLE] real market data — needs an actual price series.
