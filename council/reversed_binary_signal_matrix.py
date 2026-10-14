"""Binary Signal Matrix — chain: Alpha -> Shuffled -> ...
Project #4 (archive "Important 14" §6). Peaks -> 1, dips -> 0, confirmation
logic, zero-noise execution.

ALPHA'S TURN (INN-123): smooth-first peak/trough detector with confirmation.
Measured: noise 1.0 -> 1.00/1.00 prec; 4.0 -> 1.00/1.00 (beats mother's
0.73/0.67); 8.0 -> 0.89/1.00. Frontier: delta 1->8 costs lag 11->39 at fixed
1.00 precision. Null: 0.0 false/block (noise 1.0), 1.33 (noise 4.0).
Load-bearing: extrema are smooth-scale features.

SHUFFLED'S TURN (this file): the accounting. Alpha's code kept intact;
[SHUFFLED] marks additions.
  M1: EV ledger — slippage (measured late-pulse cost), false-pulse cost,
      miss cost, all in amplitude units. Optimal delta as function of F/V.
      Finding: precision is free, lag is expensive (~30% of amplitude at
      lag 23 on period-180 sine).
  M2: model-risk haircut — stress suite on non-sinusoidal shapes (spike,
      drift, asymmetric, double-top). EV_stressed/EV_sine = discount factor.
      This is the #3 inheritance priced: a "1" confirms a queried pattern;
      the haircut is what pattern-mismatch costs.
  M3: scale-selection — multi-scale voting vs tuned single-scale, on known
      AND unknown periods. (Alpha's open problem.)
  GUARD: shuffle-test rate check — detection rate on live vs shuffled series.
      If indistinguishable, the "1"s are worthless. No ground truth needed.
EXECUTED on synthetic. Numbers measured.
"""
import numpy as np

# =====================================================================
# ALPHA'S CONSTRUCTION (INN-123) — byte-identical.
# =====================================================================
PERIOD = 180
N = 1800  # 10 periods

def generate(noise_std=1.0, seed=0, pure_noise=False, amp=10.0):
    rng = np.random.default_rng(seed)
    t = np.arange(N, dtype=float)
    if pure_noise:
        return rng.normal(0, noise_std, N), None
    signal = amp * np.sin(2 * np.pi * t / PERIOD)
    return signal + rng.normal(0, noise_std, N), signal

def true_extrema(signal):
    """Ground truth peaks/troughs of the clean sine."""
    peaks, troughs = [], []
    for i in range(1, len(signal) - 1):
        if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
            peaks.append(i)
        elif signal[i] < signal[i-1] and signal[i] < signal[i+1]:
            troughs.append(i)
    fp, ft = [], []
    for p in peaks:
        per = int(p // (PERIOD / 2))
        if not fp or int(fp[-1] // (PERIOD / 2)) != per:
            lo, hi = per * PERIOD // 2, (per + 1) * PERIOD // 2
            fp.append(lo + int(np.argmax(signal[lo:hi])))
    for p in troughs:
        per = int(p // (PERIOD / 2))
        if not ft or int(ft[-1] // (PERIOD / 2)) != per:
            lo, hi = per * PERIOD // 2, (per + 1) * PERIOD // 2
            ft.append(lo + int(np.argmin(signal[lo:hi])))
    return np.array(fp), np.array(ft)

def smooth(series, w=21):
    """Centered moving average. Lesson from #3: squash static before analysis."""
    kernel = np.ones(w) / w
    return np.convolve(series, kernel, mode='same')

def detect(series, w=45, delta=3.0, sustain=5, smooth_w=21):
    """Streaming peak/trough detector with confirmation, on smoothed series."""
    s = smooth(series, smooth_w)
    n = len(s)
    out = []
    i = w
    while i < n - w:
        v = s[i]
        window = s[i - w:i + w + 1]
        is_peak = (v == window.max()) and (v > s[i-1]) and (v > s[i+1])
        is_trough = (v == window.min()) and (v < s[i-1]) and (v < s[i+1])
        if not (is_peak or is_trough):
            i += 1
            continue
        confirmed_at = None
        for t2 in range(i + 1, n - sustain):
            if is_peak:
                drop = v - s[i+1:t2+1].min()
                still_down = np.all(s[t2-sustain+1:t2+1] < v - delta / 2)
                if drop >= delta and still_down:
                    confirmed_at = t2
                    break
            else:
                rise = s[i+1:t2+1].max() - v
                still_up = np.all(s[t2-sustain+1:t2+1] > v + delta / 2)
                if rise >= delta and still_up:
                    confirmed_at = t2
                    break
        if confirmed_at is not None:
            out.append((confirmed_at, i, 1 if is_peak else 0))
            i = confirmed_at + 1
        else:
            i += 1
    return out

def score(detections, true_peaks, true_troughs, tol=20):
    """Precision/recall/mean-lag."""
    tp_p = tp_t = 0
    lags = []
    used_p, used_t = set(), set()
    for ct, tt, kind in detections:
        if kind == 1:
            d = np.abs(true_peaks - tt)
            j = int(np.argmin(d))
            if d[j] <= tol and j not in used_p:
                tp_p += 1; used_p.add(j); lags.append(ct - true_peaks[j])
        else:
            d = np.abs(true_troughs - tt)
            j = int(np.argmin(d))
            if d[j] <= tol and j not in used_t:
                tp_t += 1; used_t.add(j); lags.append(ct - true_troughs[j])
    n_p = sum(1 for _, _, k in detections if k == 1)
    n_t = sum(1 for _, _, k in detections if k == 0)
    prec_p = tp_p / max(n_p, 1); prec_t = tp_t / max(n_t, 1)
    rec_p = tp_p / max(len(true_peaks), 1); rec_t = tp_t / max(len(true_troughs), 1)
    return prec_p, prec_t, rec_p, rec_t, np.mean(lags) if lags else float('nan'), len(detections)


# =====================================================================
# SHUFFLED'S TURN: the accounting.
# =====================================================================

def slippage_ledger(detections, clean, true_peaks, true_troughs, tol=20, amp=10.0):
    """Measured late-pulse cost: for each TRUE-positive detection, the value
    foregone by acting at confirm_time instead of the true extremum, in
    amplitude units. Returns (mean_slip, slips[], n_tp, n_false, n_miss)."""
    slips = []; tp = 0; false = 0
    used_p, used_t = set(), set()
    for ct, tt, kind in detections:
        if kind == 1:
            if len(true_peaks) == 0:
                false += 1; continue
            d = np.abs(true_peaks - tt); j = int(np.argmin(d))
            if d[j] <= tol and j not in used_p:
                used_p.add(j); tp += 1
                slips.append((clean[true_peaks[j]] - clean[ct]) / amp)  # peak: sell high
            else:
                false += 1
        else:
            if len(true_troughs) == 0:
                false += 1; continue
            d = np.abs(true_troughs - tt); j = int(np.argmin(d))
            if d[j] <= tol and j not in used_t:
                used_t.add(j); tp += 1
                slips.append((clean[ct] - clean[true_troughs[j]]) / amp)  # trough: buy low
            else:
                false += 1
    n_true = len(true_peaks) + len(true_troughs)
    return (float(np.mean(slips)) if slips else 0.0, slips, tp, false, n_true - tp)


def ev_per_block(detections, clean, true_peaks, true_troughs, amp=10.0, F_over_V=0.5,
                 tol=20, n_blocks=10):
    """Expected value per 180-sample block, in amplitude units.
    V = amp (full swing). Miss costs V (foregone). False costs F.
    EV = [tp*(V - slip) - miss*V - false*F] / n_blocks, all / amp.
    [SHUFFLED] F_over_V is a DOMAIN parameter, not a detector parameter —
    the ledger prices the tradeoff; the domain sets the price."""
    mean_slip, _, tp, false, miss = slippage_ledger(
        detections, clean, true_peaks, true_troughs, tol, amp)
    V = 1.0  # in amplitude units
    F = F_over_V
    ev = (tp * (V - mean_slip) - miss * V - false * F) / n_blocks
    return ev, dict(slip=mean_slip, tp=tp, false=false, miss=miss)


def stress_shapes(noise_std=4.0, seed=0, amp=10.0):
    """Non-sinusoidal shapes: the model-risk suite. Each returns (series, clean,
    true_peaks, true_troughs, name). Ground truth hand-labeled per shape."""
    rng = np.random.default_rng(seed)
    t = np.arange(N, dtype=float)
    out = {}
    # 1. sharp spike train: single-sample impulses every 180 (smoother should erase)
    s = np.zeros(N); pk = []
    for k in range(10):
        idx = k * 180 + 90; s[idx] = amp; pk.append(idx)
    out["spike"] = (s + rng.normal(0, noise_std, N), s, np.array(pk), np.array([]))
    # 2. slow drift: linear trend, NO extrema (detector should stay silent)
    s = 0.05 * t
    out["drift"] = (s + rng.normal(0, noise_std, N), s, np.array([]), np.array([]))
    # 3. asymmetric: fast rise (45) slow fall (135), period 180
    s = np.zeros(N); pk = []
    for k in range(10):
        base = k * 180; pk.append(base + 45)
        for i in range(180):
            ph = i
            s[base + i] = amp * (ph / 45 if ph < 45 else 1 - (ph - 45) / 135)
    out["asymmetric"] = (s + rng.normal(0, noise_std, N), s, np.array(pk), np.array([]))
    # 4. double-top: two peaks 40 apart per period (does it merge or double-count?)
    s = np.zeros(N); pk = []
    for k in range(10):
        for off in (70, 110):
            idx = k * 180 + off; pk.append(idx)
            s[idx - 8:idx + 9] += amp * np.hanning(17)
    out["double_top"] = (s + rng.normal(0, noise_std, N), s, np.array(pk), np.array([]))
    return out


def detect_multiscale(series, scales=((11, 25), (21, 45), (41, 90)), delta=3.0,
                      sustain=5, votes=2, tol=20):
    """[SHUFFLED] Scale-selection answer: run the detector at several
    (smooth_w, w) scales; cluster detections within tol; keep clusters with
    >= votes scales agreeing. No period knowledge required."""
    all_det = []
    for sw, w in scales:
        all_det.extend(detect(series, w=w, delta=delta, sustain=sustain, smooth_w=sw))
    # cluster by true_time within tol, same kind
    clusters = []
    for ct, tt, kind in sorted(all_det, key=lambda d: d[1]):
        placed = False
        for cl in clusters:
            if cl["kind"] == kind and abs(cl["tt"] - tt) <= tol:
                cl["members"].append((ct, tt)); placed = True; break
        if not placed:
            clusters.append({"kind": kind, "tt": tt, "members": [(ct, tt)]})
    out = []
    for cl in clusters:
        if len(cl["members"]) >= votes:
            ct_med = int(np.median([m[0] for m in cl["members"]]))
            tt_med = int(np.median([m[1] for m in cl["members"]]))
            out.append((ct_med, tt_med, cl["kind"]))
    return sorted(out)


def deploy_null_test(series, smooth_w=21, horizon=90, n_sims=20, seed=0,
                     **det_kw):
    """[SHUFFLED] Operative rate check (final form). History marked in code:
    attempt 1 (global shuffle) inflated the marginal; attempt 2 (per-position
    permutation) had signal-contaminated nulls + selection bias; attempt 3
    (drop-at-confirmation) was threshold-determined (~delta by construction).
    Correct statistic: EVENTUAL drop after the extremum (horizon=90). True
    peaks keep falling (~2*amp); noise blips recover. Null: full detector on
    pure-noise sims at residual-estimated noise; detector-selected eventual
    drops vs detector-selected eventual drops (selection-safe)."""
    rng = np.random.default_rng(seed)
    resid_std = float((series - smooth(series, smooth_w)).std())

    def eventual_drops(s_raw, det):
        sm_ = smooth(s_raw, smooth_w); n = len(sm_); out = []
        for ct, tt, kind in det:
            hi = min(tt + 1 + horizon, n)
            if hi <= tt + 1:
                continue
            out.append(sm_[tt] - sm_[tt + 1:hi].min() if kind == 1
                       else sm_[tt + 1:hi].max() - sm_[tt])
        return out

    live_det = detect(series, **det_kw)
    live_ev = eventual_drops(series, live_det)
    null_med = []
    for _ in range(n_sims):
        nz = rng.normal(0, resid_std, len(series))
        ev = eventual_drops(nz, detect(nz, **det_kw))
        null_med.append(float(np.median(ev)) if ev else 0.0)
    live_med = float(np.median(live_ev)) if live_ev else 0.0
    p = float(np.mean(np.array(null_med) >= live_med))
    return resid_std, len(live_det), live_med, float(np.mean(null_med)), p


if __name__ == "__main__":
    print("### ALPHA'S REPORT (kept) ###")
    for ns in (1.0, 4.0, 8.0):
        series, clean = generate(noise_std=ns, seed=3)
        tp_, tt_ = true_extrema(clean)
        det = detect(series)
        pp, pt, rp, rt, lag, nd = score(det, tp_, tt_)
        print(f"noise={ns}: prec {pp:.2f}/{pt:.2f} rec {(rp+rt)/2:.2f} lag {lag:.0f}")

    print("\n### SHUFFLED M1: EV ledger — pricing lag vs certainty ###")
    print("units: amplitude (V=1.0). slip measured, not modeled.")
    print(f"{'delta':>6} {'lag':>5} {'slip':>6} {'recall':>7} {'EV(F/V=0.1)':>11} {'EV(F/V=1)':>10}")
    series, clean = generate(noise_std=4.0, seed=3)
    tp_, tt_ = true_extrema(clean)
    for delta in (1.0, 2.0, 3.0, 5.0, 8.0):
        det = detect(series, delta=delta)
        pp, pt, rp, rt, lag, _ = score(det, tp_, tt_)
        ev_lo, info = ev_per_block(det, clean, tp_, tt_, F_over_V=0.1)
        ev_hi, _ = ev_per_block(det, clean, tp_, tt_, F_over_V=1.0)
        print(f"{delta:>6.1f} {lag:>5.0f} {info['slip']:>6.2f} {(rp+rt)/2:>7.2f} "
              f"{ev_lo:>11.2f} {ev_hi:>10.2f}")
    print("-> at lag 23, slip ~0.28 amplitude: precision is free, lag costs ~30%.")
    print("   At noise=4.0 F/V does NOT bind (zero false pulses) -> smallest delta")
    print("   wins outright. Even at noise=8.0 small delta wins: lag cost dominates")
    print("   false-pulse cost across the measured range. F/V sensitivity latent.")
    print(f"{'delta':>6} {'lag':>5} {'slip':>6} {'prec':>6} {'EV(F/V=0.1)':>11} {'EV(F/V=2)':>10}")
    s8, c8 = generate(noise_std=8.0, seed=3)
    p8, t8 = true_extrema(c8)
    for delta in (1.0, 3.0, 8.0):
        det = detect(s8, delta=delta)
        pp, pt, rp, rt, lag, _ = score(det, p8, t8)
        ev_lo, info = ev_per_block(det, c8, p8, t8, F_over_V=0.1)
        ev_hi, _ = ev_per_block(det, c8, p8, t8, F_over_V=2.0)
        print(f"{delta:>6.1f} {lag:>5.0f} {info['slip']:>6.2f} {(pp+pt)/2:>6.2f} "
              f"{ev_lo:>11.2f} {ev_hi:>10.2f}   false={info['false']}")

    print("\n### SHUFFLED M2: model-risk haircut (confirmation, not discovery) ###")
    print("stress suite @ noise=4.0, delta=3.0. EV as fraction of sine EV:")
    s0, c0 = generate(noise_std=4.0, seed=3); p0, t0 = true_extrema(c0)
    ev_sine, _ = ev_per_block(detect(s0), c0, p0, t0, F_over_V=0.5)
    print(f"sine baseline EV/block = {ev_sine:.2f}")
    for name, (ss, cc, pk, tr) in stress_shapes().items():
        det = detect(ss)
        if len(pk) + len(tr) == 0:
            n_false = len(det)
            print(f"{name:>10}: detections={len(det)} on NO extrema "
                  f"-> {n_false/10:.2f} false/block (pure model-risk cost)")
            continue
        ev, info = ev_per_block(det, cc, pk, tr, F_over_V=0.5)
        haircut = ev / ev_sine if ev_sine > 0 else 0
        print(f"{name:>10}: EV/block={ev:>6.2f} haircut={haircut:.2f} "
              f"(tp={info['tp']} miss={info['miss']} false={info['false']})")
    print("-> the haircut IS the #3 inheritance priced: a '1' confirms the")
    print("   queried sine-pattern; on other shapes the trigger economics collapse")
    print("   by the haircut factor. Deploy EV = sine EV x haircut x P(shape~sine).")

    print("\n### SHUFFLED M3: scale-selection (multi-scale voting) ###")
    print("tuned single-scale vs voting, on known AND unknown periods:")
    print(f"{'period':>7} {'tuned P/R':>11} {'voted P/R':>11} {'tuned lag':>10} {'voted lag':>10}")
    for per in [30, 90, 180, 360]:
        t_ = np.arange(N, dtype=float)
        sig = 10.0 * np.sin(2 * np.pi * t_ / per)
        rng = np.random.default_rng(3)
        s = sig + rng.normal(0, 4.0, N)
        # tuned single-scale (Alpha's windows: built for 180)
        det_t = detect(s)
        # voted multi-scale (no period knowledge)
        det_v = detect_multiscale(s)
        # ground truth for this period: extrema of clean sine
        pk = np.array([int(k * per + per / 4) for k in range(N // per)])
        tr = np.array([int(k * per + 3 * per / 4) for k in range(N // per)])
        pp, pt, rp, rt, lag_t, _ = score(det_t, pk, tr, tol=20)
        pv, tv, rv, tv2, lag_v, _ = score(det_v, pk, tr, tol=20)
        print(f"{per:>7} {(pp+pt)/2:>5.2f}/{(rp+rt)/2:<5.2f} {(pv+tv)/2:>5.2f}/{(rv+tv2)/2:<5.2f} "
              f"{lag_t:>10.0f} {lag_v:>10.0f}")
    print("-> scale-selection is REAL at fast periods (30: recall 0.32); voting")
    print("   recovers only 0.06. Full answer needs delta scaled to window size")
    print("   (small windows attenuate signal below fixed delta) — marked for")
    print("   synthesis. At 90-360 the tuned windows already generalize.")

    print("\n### SHUFFLED GUARD: deploy null test (eventual-drop) ###")
    print("statistic: median eventual drop (horizon=90) vs noise-sim null.")
    for ns, tag in [(1.0, "clean"), (4.0, "noisy")]:
        s, _ = generate(noise_std=ns, seed=3)
        nest, nlive, lmed, nmed, p = deploy_null_test(s, n_sims=20)
        print(f"noise={ns} ({tag:>5}): nest={nest:.2f} live={nlive} "
              f"live_med_drop={lmed:.1f} null_med_drop={nmed:.1f} p={p:.3f} -> "
              f"{'TRUST' if p < 0.05 else 'STAND DOWN'}")
    nz, _ = generate(noise_std=4.0, seed=3, pure_noise=True)
    nest, nlive, lmed, nmed, p = deploy_null_test(nz, n_sims=20)
    print(f"pure noise     : nest={nest:.2f} live={nlive} "
          f"live_med_drop={lmed:.1f} null_med_drop={nmed:.1f} p={p:.3f} -> "
          f"{'TRUST?' if p < 0.05 else 'STAND DOWN (correct)'}")

    print("\n### UNMEASURED (at the line) ###")
    print("- F/V, V: domain parameters. Ledger prices; the market sets prices.")
    print("- slippage assumes acting at confirm_time on the CLEAN series; real")
    print("  slippage includes spread, fees, and market impact — unmeasured.")
    print("- stress shapes are synthetic; real regime shapes unknown.")
    print("- multi-scale cost: 3x detect calls; wall-clock trivial at N=1800.")
    # [CANNOT RUN — real market data: thresholds tuned on sine do not transfer.]


# =====================================================================
# [REVERSED] TURN: the audit. Every "1" is a claim; the claim carries
# its evidence; the audit tri-states it without ground truth.
# =====================================================================
from typing import NamedTuple


class SignalClaim(NamedTuple):
    """[REVERSED] A fired 1/0 with its evidence attached.
    Kills the silent misfire: Alpha's bare (confirmed_at, true_time, kind)
    tuple is a claim with no support. The claim carries the support, so a
    downstream consumer never acts on a signal it cannot inspect."""
    confirmed_at: int
    true_time: int
    kind: int              # 1 = peak, 0 = trough
    move: float            # measured drop (peak) / rise (trough) at confirmation
    delta: float           # threshold the detector used
    sustain_margin: float  # min margin of sustain samples past the delta/2
                           # line, in delta units (comfort past the bar)
    local_noise: float     # residual std in window around true_time
    smooth_w: int
    w: int


def claims_from_detections(detections, series, w=45, delta=3.0, sustain=5,
                           smooth_w=21):
    """[REVERSED] Wrap each detection in its evidence, recomputed from the
    series. The detector's internals are not trusted — only its outputs.
    Evidence is measured independently so the audit cannot inherit a
    detector bug."""
    s = smooth(series, smooth_w)
    resid = series - s
    # [REVERSED] Scale correction: the move lives on the SMOOTHED series,
    # so the faking-noise is the smoothed noise, ~ resid_std/sqrt(smooth_w)
    # for white residuals. Approximate; marked. Without this the snr
    # compares smoothed moves to raw noise — wrong scale, audit too strict.
    import math as _m
    claims = []
    for ct, tt, kind in detections:
        lo, hi = max(0, tt - w), min(len(s), tt + w + 1)
        raw_n = float(resid[lo:hi].std()) if hi > lo else float("nan")
        local_noise = raw_n / _m.sqrt(smooth_w)
        v = s[tt]
        sus = s[max(0, ct - sustain + 1):ct + 1]
        if kind == 1:
            move = float(v - s[tt + 1:ct + 1].min()) if ct > tt else 0.0
            sm = float(((v - delta / 2) - sus).min() / delta) if len(sus) else 0.0
        else:
            move = float(s[tt + 1:ct + 1].max() - v) if ct > tt else 0.0
            sm = float((sus - (v + delta / 2)).min() / delta) if len(sus) else 0.0
        claims.append(SignalClaim(ct, tt, kind, move, delta, sm,
                                 local_noise, smooth_w, w))
    return claims


def detect_claims(series, w=45, delta=3.0, sustain=5, smooth_w=21):
    """[REVERSED] detect() + evidence wrapper. The kwargs ARE the contract:
    what the detector used is what the claim records. No silent divergence."""
    det = detect(series, w=w, delta=delta, sustain=sustain, smooth_w=smooth_w)
    return claims_from_detections(det, series, w=w, delta=delta,
                                 sustain=sustain, smooth_w=smooth_w)


# [REVERSED] Tri-state thresholds — CHOSEN, not derived. Heuristic.
# Marked loud: these are judgment calls. Validated below against ground
# truth (R1); the agreement rate is the check on the choice.
AUDIT_MARGIN = 2.0   # move/delta for CONFIRMED
AUDIT_SNR = 3.0      # move/local_noise for CONFIRMED
AUDIT_EV = 2.0       # eventual-move/delta for CONFIRMED
AUDIT_STALE = 0.5    # captured fraction above which CONFIRMED is STALE


def audit_claim(claim, series, horizon=90):
    """[REVERSED] Tri-state per detection. NO ground truth used — this is
    the deploy condition. Returns (verdict, detail).
    Verdicts: CONFIRMED | CONFIRMED_STALE | FALSE_PULSE | UNKNOWN.
    Four checks:
      1. snr = move/smoothed-local-noise — distinguishable from noise
         at the detector's own scale? (margin=move/delta is reported but
         NOT gated: the stopping rule pins it ~1.0 by construction.)
      2. eventual follow-through over horizon — did the series keep moving
         in the claimed direction? (Shuffled's statistic, per-claim. Kills
         blips: a confirmed drop that recovers was never a swing.)
      3. captured fraction — how much of the eventual move had ALREADY
         happened by confirmation? (Kills stale signals: on a cliff the
         evidence is fantastic and the signal is worthless.)"""
    s = smooth(series, claim.smooth_w)
    n = len(s)
    tt, ct, kind = claim.true_time, claim.confirmed_at, claim.kind
    margin = claim.move / claim.delta if claim.delta > 0 else 0.0
    snr = claim.move / max(claim.local_noise, 1e-9)
    hi = min(tt + 1 + horizon, n)
    trunc = " [horizon truncated]" if hi < tt + 1 + horizon else ""
    if hi <= tt + 1:
        return ("UNKNOWN", "no horizon past extremum" + trunc)
    if kind == 1:
        eventual = float(s[tt] - s[tt + 1:hi].min())
        captured = float((s[tt] - s[ct]) / eventual) if eventual > 1e-9 else 0.0
    else:
        eventual = float(s[tt + 1:hi].max() - s[tt])
        captured = float((s[ct] - s[tt]) / eventual) if eventual > 1e-9 else 0.0
    ev_ratio = eventual / claim.delta if claim.delta > 0 else 0.0

    # FALSE first: the move didn't follow through — confirmation was a blip.
    if ev_ratio < 1.0:
        return ("FALSE_PULSE",
                f"eventual {eventual:.2f} < delta {claim.delta:.2f} "
                f"(ev {ev_ratio:.2f}): blip, not a swing" + trunc)
    # CONFIRMED: distinguishable from noise AND real follow-through.
    # (margin dropped from the gate: the detector's stopping rule pins
    # move~=delta by construction, so margin is uninformative. Kept below
    # as a diagnostic.)
    if snr >= AUDIT_SNR and ev_ratio >= AUDIT_EV:
        base = (f"margin {margin:.1f} snr {snr:.1f} ev {ev_ratio:.1f} "
                f"captured {captured:.0%}" + trunc)
        if captured > AUDIT_STALE:
            return ("CONFIRMED_STALE", base + ": move already happened")
        return ("CONFIRMED", base)
    # UNKNOWN: thin evidence — name exactly what's thin.
    thin = []
    if margin < 1.0:
        thin.append(f"margin {margin:.1f}<1.0 DETECTOR-BUG?")
    if snr < AUDIT_SNR:
        thin.append(f"snr {snr:.1f}<{AUDIT_SNR}")
    if ev_ratio < AUDIT_EV:
        thin.append(f"ev {ev_ratio:.1f}<{AUDIT_EV}")
    return ("UNKNOWN", "; ".join(thin) + trunc)


def adversarial_cliff(noise_std=1.0, seed=0, amp=10.0):
    """[REVERSED] The #3-inversion analog, constructed. A step drop at t=900.
    The detector finds the local max at the cliff edge and confirms with a
    HUGE drop — the evidence looks fantastic. But the signal is STALE: the
    move completed before confirmation was possible. The '1' says 'act now';
    acting now captures nothing. This is the silent misfire with the
    strongest evidence — the one no threshold can catch, only the audit."""
    rng = np.random.default_rng(seed)
    t = np.arange(N, dtype=float)
    clean = np.where(t < 900, amp, -amp)
    return clean + rng.normal(0, noise_std, N), clean


if __name__ == "__main__":
    print("\n### [REVERSED] R1: audit validation — verdict vs ground truth ###")
    print("sine, noise=4.0, delta=3.0. Audit uses NO ground truth; we check it")
    print("against truth here to validate the thresholds (chosen, not derived).")
    series, clean = generate(noise_std=4.0, seed=3)
    tp_, tt_ = true_extrema(clean)
    claims = detect_claims(series)
    # ground-truth match per claim (same tol logic as score())
    used_p, used_t = set(), set()
    agree_c = agree_f = 0
    n_c = n_f = n_u = n_cs = 0
    for cl in claims:
        v, detail = audit_claim(cl, series)
        # truth: does tt match a real extremum?
        if cl.kind == 1:
            d = np.abs(tp_ - cl.true_time); j = int(np.argmin(d))
            is_true = d[j] <= 20 and j not in used_p
            if is_true: used_p.add(j)
        else:
            d = np.abs(tt_ - cl.true_time); j = int(np.argmin(d))
            is_true = d[j] <= 20 and j not in used_t
            if is_true: used_t.add(j)
        tag = "TRUE " if is_true else "false"
        if v == "CONFIRMED":
            n_c += 1; agree_c += is_true
        elif v == "CONFIRMED_STALE":
            n_cs += 1
        elif v == "FALSE_PULSE":
            n_f += 1; agree_f += (not is_true)
        else:
            n_u += 1
        if len(claims) <= 24:
            print(f"  {'P' if cl.kind==1 else 'T'} tt={cl.true_time:>4} "
                  f"-> {v:>15} [{tag}] {detail}")
    print(f"-> CONFIRMED {n_c} (agree w/ truth {agree_c}/{n_c}), "
          f"FALSE_PULSE {n_f} (agree {agree_f}/{n_f}), "
          f"UNKNOWN {n_u}, STALE {n_cs}")

    print("\n### [REVERSED] R2: misfire battery — pure noise ###")
    print("detections the detector fires with NO signal present, tri-stated.")
    for ns in (1.0, 4.0, 8.0, 16.0):
        nz, _ = generate(noise_std=ns, seed=3, pure_noise=True)
        claims = detect_claims(nz)
        vs = [audit_claim(cl, nz)[0] for cl in claims]
        from collections import Counter
        c = Counter(vs)
        print(f"  noise={ns:>4}: {len(claims):>3} detections -> "
              f"{dict(c) if c else 'silent (correct)'}")

    print("\n### [REVERSED] R3: adversarial cliff — what the detector does ###")
    print("step drop at t=900. Note what the detector ACTUALLY fires on:")
    cs, cc = adversarial_cliff(noise_std=1.0, seed=3)
    claims = detect_claims(cs)
    for cl in claims:
        v, detail = audit_claim(cl, cs)
        print(f"  {'P' if cl.kind==1 else 'T'} tt={cl.true_time:>4} "
              f"ct={cl.confirmed_at:>4} move={cl.move:>6.1f} -> {v}: {detail}")
    print("-> the detector fired at tt=179 (noise wiggle), confirmed at ct=895")
    print("   by BORROWING the cliff's drop. The audit kills it as FALSE_PULSE")
    print("   (the wiggle never followed through). Finding: the detector cannot")
    print("   fire AT a cliff edge — the cliff's drop always confirms whatever")
    print("   came before it. Staleness here is structural (wrong extremum),")
    print("   not late confirmation.")
    print("-> STALE path, constructed (detector never emits it; logic test):")
    cl_s = SignalClaim(910, 899, 1, 12.0, 3.0, 2.0, 0.5/21**0.5, 21, 45)
    v_s, d_s = audit_claim(cl_s, cs)
    print(f"   edge claim tt=899 ct=910 -> {v_s}: {d_s}")
    print("   (if the detector ever fired at the edge, captured-fraction")
    print("   would mark it STALE: strongest evidence, worthless signal.)")

    print("\n### [REVERSED] R4: drift — the audit's boundary, marked ###")
    print("drift has NO extrema by construction. What does the audit say?")
    ss, cc_, pk, tr = stress_shapes(noise_std=4.0, seed=3)["drift"]
    claims = detect_claims(ss)
    vs = [audit_claim(cl, ss) for cl in claims]
    from collections import Counter
    c = Counter(v for v, _ in vs)
    print(f"  drift: {len(claims)} detections -> {dict(c) if c else 'silent'}")
    for cl, (v, detail) in list(zip(claims, vs))[:6]:
        print(f"    {'P' if cl.kind==1 else 'T'} tt={cl.true_time:>4} -> {v}: {detail}")
    print("-> BOUNDARY MARKED: the audit confirms LOCAL SWINGS, not pattern")
    print("   extrema. The drift-troughs ARE local swings (real dips, real")
    print("   follow-through from the drift). The misfire is at the PATTERN")
    print("   level — 'is this the sine-peak I want to trade?' — which belongs")
    print("   to Shuffled's haircut (M2), not to per-detection audit.")
    print("   Composition: audit (per-detection, kills blips) x haircut")
    print("   (per-pattern, prices mismatch). Neither alone is sufficient.")

    print("\n### [REVERSED] UNMEASURED (at the line) ###")
    print("- Tri-state thresholds (snr 3.0 / ev 2.0 / stale 0.5) are chosen")
    print("  heuristics; validated on one sine seed (19/19), not swept.")
    print("- Noise scale correction (resid/sqrt(smooth_w)) is approximate;")
    print("  assumes white residuals. Marked in code.")
    print("- STALE demonstrated on a constructed claim; the detector never")
    print("  emitted a stale case in testing (it fires early instead).")
    print("- The audit covers per-detection blips, NOT pattern mismatch;")
    print("  the drift CONFIRMEDs prove the boundary. Needs the haircut.")
    print("- Real market data: unmeasured. Thresholds tuned on sine.")
    # [CANNOT RUN — live deployment: audit needs a live series, not synthetic.]
