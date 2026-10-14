"""Binary Signal Matrix — Seven's turn (chain: Alpha -> Shuffled -> Reversed -> Four -> Seven -> ...)
Project #4 (archive "Important 14" §6). The ground truth provocation.

Reversed built the audit: every "1" carries its evidence, tri-stated without
ground truth. 19/19 true sine CONFIRMED; pure noise never CONFIRMED.
This turn asks the next question: does the binary "1" READ the audit?

The matrix maps detection -> trigger. Peaks -> 1. But the audit tri-states
detections into CONFIRMED / FALSE_PULSE / UNKNOWN / CONFIRMED_STALE — and
nothing in the chain gates the trigger on the verdict. A FALSE_PULSE still
becomes a 1. The audit is adjacent to the trigger, not gating it.

The provocation, executed: what fraction of emitted 1s are NOT CONFIRMED?
That is the trigger hallucination rate — and it is the #3 inheritance made
measurable: #3's rhythm hallucinates 34% on noise; #4's binary mapping turns
detections into actions without consulting the audit that knows better.

Alpha's construction + Shuffled's accounting + Reversed's audit + Four's
provocation kept intact; [SEVEN] marks additions. The trigger-claim registry:
every 1 and 0 inscribed with its evidence chain, discovery vs queried-pattern
marked (#3 inheritance), Four's T1-T5 as provenance records. EXECUTED on synthetic. Numbers in __main__.
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
# SHUFFLED'S TURN: the accounting — kept intact.
# =====================================================================

def slippage_ledger(detections, clean, true_peaks, true_troughs, tol=20, amp=10.0):
    """Measured late-pulse cost."""
    slips = []; tp = 0; false = 0
    used_p, used_t = set(), set()
    for ct, tt, kind in detections:
        if kind == 1:
            if len(true_peaks) == 0:
                false += 1; continue
            d = np.abs(true_peaks - tt); j = int(np.argmin(d))
            if d[j] <= tol and j not in used_p:
                used_p.add(j); tp += 1
                slips.append((clean[true_peaks[j]] - clean[ct]) / amp)
            else:
                false += 1
        else:
            if len(true_troughs) == 0:
                false += 1; continue
            d = np.abs(true_troughs - tt); j = int(np.argmin(d))
            if d[j] <= tol and j not in used_t:
                used_t.add(j); tp += 1
                slips.append((clean[ct] - clean[true_troughs[j]]) / amp)
            else:
                false += 1
    n_true = len(true_peaks) + len(true_troughs)
    return (float(np.mean(slips)) if slips else 0.0, slips, tp, false, n_true - tp)


def ev_per_block(detections, clean, true_peaks, true_troughs, amp=10.0, F_over_V=0.5,
                 tol=20, n_blocks=10):
    """Expected value per 180-sample block, in amplitude units."""
    mean_slip, _, tp, false, miss = slippage_ledger(
        detections, clean, true_peaks, true_troughs, tol, amp)
    V = 1.0
    F = F_over_V
    ev = (tp * (V - mean_slip) - miss * V - false * F) / n_blocks
    return ev, dict(slip=mean_slip, tp=tp, false=false, miss=miss)


def stress_shapes(noise_std=4.0, seed=0, amp=10.0):
    """Non-sinusoidal shapes: the model-risk suite."""
    rng = np.random.default_rng(seed)
    t = np.arange(N, dtype=float)
    out = {}
    s = np.zeros(N); pk = []
    for k in range(10):
        idx = k * 180 + 90; s[idx] = amp; pk.append(idx)
    out["spike"] = (s + rng.normal(0, noise_std, N), s, np.array(pk), np.array([]))
    s = 0.05 * t
    out["drift"] = (s + rng.normal(0, noise_std, N), s, np.array([]), np.array([]))
    s = np.zeros(N); pk = []
    for k in range(10):
        base = k * 180; pk.append(base + 45)
        for i in range(180):
            ph = i
            s[base + i] = amp * (ph / 45 if ph < 45 else 1 - (ph - 45) / 135)
    out["asymmetric"] = (s + rng.normal(0, noise_std, N), s, np.array(pk), np.array([]))
    s = np.zeros(N); pk = []
    for k in range(10):
        for off in (70, 110):
            idx = k * 180 + off; pk.append(idx)
            s[idx - 8:idx + 9] += amp * np.hanning(17)
    out["double_top"] = (s + rng.normal(0, noise_std, N), s, np.array(pk), np.array([]))
    return out


def detect_multiscale(series, scales=((11, 25), (21, 45), (41, 90)), delta=3.0,
                      sustain=5, votes=2, tol=20):
    """[SHUFFLED] Scale-selection: multi-scale voting, no period knowledge."""
    all_det = []
    for sw, w in scales:
        all_det.extend(detect(series, w=w, delta=delta, sustain=sustain, smooth_w=sw))
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
    """[SHUFFLED] Operative rate check: eventual-drop vs noise-sim null."""
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


# =====================================================================
# [REVERSED] TURN: the audit — kept intact.
# =====================================================================
from typing import NamedTuple


class SignalClaim(NamedTuple):
    """[REVERSED] A fired 1/0 with its evidence attached."""
    confirmed_at: int
    true_time: int
    kind: int
    move: float
    delta: float
    sustain_margin: float
    local_noise: float
    smooth_w: int
    w: int


def claims_from_detections(detections, series, w=45, delta=3.0, sustain=5,
                           smooth_w=21):
    """[REVERSED] Wrap each detection in its evidence, recomputed independently."""
    s = smooth(series, smooth_w)
    resid = series - s
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
    """[REVERSED] detect() + evidence wrapper."""
    det = detect(series, w=w, delta=delta, sustain=sustain, smooth_w=smooth_w)
    return claims_from_detections(det, series, w=w, delta=delta,
                                 sustain=sustain, smooth_w=smooth_w)


AUDIT_MARGIN = 2.0
AUDIT_SNR = 3.0
AUDIT_EV = 2.0
AUDIT_STALE = 0.5


def audit_claim(claim, series, horizon=90):
    """[REVERSED] Tri-state per detection. NO ground truth used."""
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

    if ev_ratio < 1.0:
        return ("FALSE_PULSE",
                f"eventual {eventual:.2f} < delta {claim.delta:.2f} "
                f"(ev {ev_ratio:.2f}): blip, not a swing" + trunc)
    if snr >= AUDIT_SNR and ev_ratio >= AUDIT_EV:
        base = (f"margin {margin:.1f} snr {snr:.1f} ev {ev_ratio:.1f} "
                f"captured {captured:.0%}" + trunc)
        if captured > AUDIT_STALE:
            return ("CONFIRMED_STALE", base + ": move already happened")
        return ("CONFIRMED", base)
    thin = []
    if margin < 1.0:
        thin.append(f"margin {margin:.1f}<1.0 DETECTOR-BUG?")
    if snr < AUDIT_SNR:
        thin.append(f"snr {snr:.1f}<{AUDIT_SNR}")
    if ev_ratio < AUDIT_EV:
        thin.append(f"ev {ev_ratio:.1f}<{AUDIT_EV}")
    return ("UNKNOWN", "; ".join(thin) + trunc)


def adversarial_cliff(noise_std=1.0, seed=0, amp=10.0):
    """[REVERSED] Step drop at t=900. The detector borrows the cliff's drop."""
    rng = np.random.default_rng(seed)
    t = np.arange(N, dtype=float)
    clean = np.where(t < 900, amp, -amp)
    return clean + rng.normal(0, noise_std, N), clean


# =====================================================================
# [FOUR] THE GROUND TRUTH PROVOCATION.
# =====================================================================

# [FOUR] TriggerProvenance: the three layers of "the 1."
# SIGNAL: the market's actual tradeable extremum. GROWN in the price action,
#   unobservable without ground truth. On real data, there is no true_extrema.
# DETECTED: the detector's claim (confirmed_at, true_time, kind). MANUFACTURED
#   by thresholds (delta, sustain, smooth_w — all chosen, not derived).
# TRIGGERED: the binary 1 emitted to execution. MANUFACTURED action on the
#   manufactured detection. The matrix maps DETECTED -> TRIGGERED unconditionally.
# The audit tri-states DETECTED. Nothing gates TRIGGERED on the verdict.
# A FALSE_PULSE becomes a 1. An UNKNOWN becomes a 1. The audit is adjacent,
# not gating. That is the provocation, made structural.
PROV_SIGNAL = "grown: market's extremum, unobservable without ground truth"
PROV_DETECTED = "manufactured: detector's claim, threshold-dependent"
PROV_TRIGGERED = "manufactured: binary action, ungated by audit"


def trigger_purity(series, w=45, delta=3.0, sustain=5, smooth_w=21, horizon=90):
    """[FOUR] Of every "1" the binary matrix would emit, what fraction is
    CONFIRMED vs FALSE_PULSE vs UNKNOWN vs CONFIRMED_STALE?
    The trigger hallucination rate = 1 - P(CONFIRMED | triggered).
    EXECUTED."""
    from collections import Counter
    claims = detect_claims(series, w=w, delta=delta, sustain=sustain,
                           smooth_w=smooth_w)
    verdicts = [audit_claim(cl, series, horizon=horizon)[0] for cl in claims]
    c = Counter(verdicts)
    n = len(claims)
    return dict(
        n_triggers=n,
        counts=dict(c),
        purity=c.get("CONFIRMED", 0) / max(n, 1),
        hallucination_rate=1 - c.get("CONFIRMED", 0) / max(n, 1),
        # [FOUR] STALE is confirmed-but-worthless: counts as hallucination
        # for execution purposes (acting on it captures nothing).
        execution_purity=(c.get("CONFIRMED", 0)) / max(n, 1),
        execution_hallucination=1 - c.get("CONFIRMED", 0) / max(n, 1),
    )


def threshold_manufacture(series, deltas=(1.0, 3.0, 8.0), w=45, sustain=5,
                          smooth_w=21):
    """[FOUR] The "1" positions are manufactured by the threshold choice.
    Same series, different delta -> different 1-positions. If the trigger were
    grown in the market, it would not move with a detector parameter.
    Returns {delta: sorted peak true_times}. EXECUTED."""
    out = {}
    for d in deltas:
        det = detect(series, w=w, delta=d, sustain=sustain, smooth_w=smooth_w)
        peaks = sorted(tt for _, tt, kind in det if kind == 1)
        out[d] = peaks
    return out


def gated_trigger_purity(series, gate_fn, w=45, delta=3.0, sustain=5,
                         smooth_w=21, horizon=90):
    """[FOUR] What if the binary matrix READ the audit? Only CONFIRMED
    detections become 1s. Measures the purity gain from gating triggers
    on the audit verdict. This is the fix the provocation implies.
    gate_fn: verdict -> bool (e.g., lambda v: v == "CONFIRMED").
    EXECUTED."""
    from collections import Counter
    claims = detect_claims(series, w=w, delta=delta, sustain=sustain,
                           smooth_w=smooth_w)
    gated = []
    ungated_verdicts = []
    for cl in claims:
        v, _ = audit_claim(cl, series, horizon=horizon)
        ungated_verdicts.append(v)
        if gate_fn(v):
            gated.append(cl)
    c_all = Counter(ungated_verdicts)
    n_all = len(claims)
    return dict(
        n_ungated=n_all,
        n_gated=len(gated),
        ungated_purity=c_all.get("CONFIRMED", 0) / max(n_all, 1),
        gated_purity=1.0 if gated else 0.0,  # by construction: only CONFIRMED pass
        triggers_suppressed=n_all - len(gated),
        suppression_rate=(n_all - len(gated)) / max(n_all, 1),
    )




# =====================================================================
# [SEVEN] TRIGGER PROVENANCE REGISTRY.
#
# Every "1" and "0" the binary matrix emits gets a TriggerProvenanceRecord:
#   - what peak was seen (Reversed's SignalClaim: move, delta, sustain_margin,
#     local_noise, smooth_w, w)
#   - what confirmation was waited (drop >= delta, still_down/up, sustain)
#   - what the evidence was (audit verdict: CONFIRMED / FALSE_PULSE /
#     UNKNOWN / CONFIRMED_STALE)
#   - which provenance layer each piece came from (Four's SIGNAL/DETECTED/
#     TRIGGERED: grown vs manufactured)
#
# THE #3 INHERITANCE: in project #3, rhythm claims were PROJECTIONS — the
# overlay queried period==block and called the answer a discovery. Here the
# equivalent question: is the trigger DISCOVERING a market extremum, or
# CONFIRMING A QUERIED PATTERN (the detector's thresholds imposed on data
# that didn't object)? Four's T3 answers it for positions: on strong sine,
# positions are STABLE across thresholds (grown in the signal); the FIRING
# DECISION is manufactured (threshold-dependent). The registry marks both.
#
# DISCOVERY vs QUERIED marking rule (deploy-safe, no ground truth):
#   - "discovery": the trigger's position is threshold-STABLE (fires at the
#     same true_time across delta in {1.0, 3.0, 8.0}, tol 20) AND the audit
#     verdict is CONFIRMED. The signal had the extremum; the trigger found it.
#   - "queried-pattern": the trigger fires at this delta but not at others
#     (position moves with the threshold), OR the audit verdict is not
#     CONFIRMED. The thresholds manufactured the trigger; the data complied.
#   - "unknown": insufficient evidence to mark (e.g., audit UNKNOWN).
# On synthetic, threshold-stability is measurable via threshold_manufacture().
# On real data, stability is the only discovery-proxy available — mark it.
# =====================================================================

class TriggerProvenanceRecord:
    """[SEVEN] One emitted 1/0: full evidence chain, three layers, and the
    discovery-vs-queried mark."""

    def __init__(self, claim, audit_verdict, audit_basis,
                 threshold_stable, discovery_mark):
        # --- Reversed's evidence (what peak was seen) ---
        self.confirmed_at = claim.confirmed_at
        self.true_time = claim.true_time
        self.kind = claim.kind  # 1 = peak, 0 = trough
        self.move = claim.move
        self.delta = claim.delta
        self.sustain_margin = claim.sustain_margin
        self.local_noise = claim.local_noise
        self.smooth_w = claim.smooth_w
        self.w = claim.w
        # --- Four's layers (where each piece came from) ---
        self.layer_signal = "grown (unobservable without ground truth)"
        self.layer_detected = "manufactured (threshold-dependent)"
        self.layer_triggered = "manufactured (ungated by audit)"
        # --- The audit (what the evidence said) ---
        self.audit_verdict = audit_verdict
        self.audit_basis = audit_basis
        # --- The #3 inheritance mark ---
        self.threshold_stable = threshold_stable
        self.discovery_mark = discovery_mark  # discovery | queried-pattern | unknown

    def __repr__(self):
        return (f"TriggerProvenanceRecord(kind={self.kind}, tt={self.true_time}, "
                f"ct={self.confirmed_at}, audit={self.audit_verdict}, "
                f"mark={self.discovery_mark})")

    def audit_trail(self):
        return (f"[tt={self.true_time}] kind={'peak' if self.kind==1 else 'trough'} "
                f"move={self.move:.2f} (delta={self.delta}) "
                f"sustain_margin={self.sustain_margin:.2f} "
                f"local_noise={self.local_noise:.2f} "
                f"smooth_w={self.smooth_w} w={self.w} "
                f"-> audit={self.audit_verdict} ({self.audit_basis[:60]}...) "
                f"| threshold_stable={self.threshold_stable} "
                f"| mark={self.discovery_mark} "
                f"| layers: signal=grown(unobservable) "
                f"detected=manufactured triggered=manufactured(ungated)")


class FourTestProvenanceRecord:
    """[SEVEN] Four's T1-T5 as provenance records of trigger claims."""

    def __init__(self, test_id, claim_desc, measured, manufactured_components,
                 verdict):
        self.test_id = test_id
        self.claim_desc = claim_desc
        self.measured = measured
        self.manufactured_components = manufactured_components
        # verdict: "discovery" | "queried-pattern" | "projection" |
        #          "broken-assumption" | "manufactured-fix" (an imposed intervention)
        self.verdict = verdict

    def __repr__(self):
        return (f"FourTestProvenanceRecord({self.test_id}: {self.claim_desc[:50]}... "
                f"verdict={self.verdict})")


class TriggerProvenanceRegistry:
    """[SEVEN] The registry: every trigger inscribed, every claim marked."""

    def __init__(self, stream_provenance="manufactured: synthetic generator"):
        self.stream_provenance = stream_provenance
        self.triggers = []
        self.four_tests = []

    def inscribe_triggers(self, series, w=45, delta=3.0, sustain=5,
                          smooth_w=21, horizon=90, stability_deltas=(1.0, 3.0, 8.0),
                          tol=20):
        """[SEVEN] Inscribe every emitted 1/0 with its evidence chain and
        discovery-vs-queried mark. Threshold-stability computed across
        stability_deltas (Four's T3 method)."""
        # Threshold-stability map: which true_times fire across all deltas.
        pos_by_delta = threshold_manufacture(series, deltas=stability_deltas,
                                             w=w, sustain=sustain, smooth_w=smooth_w)
        stable_set = set(pos_by_delta[stability_deltas[0]])
        for d in stability_deltas[1:]:
            sd = set(pos_by_delta[d])
            stable_set = {p for p in stable_set
                          if any(abs(p - q) <= tol for q in sd)}
        claims = detect_claims(series, w=w, delta=delta, sustain=sustain,
                               smooth_w=smooth_w)
        for cl in claims:
            verdict, basis = audit_claim(cl, series, horizon=horizon)
            # [SEVEN] Threshold-stability: does this true_time fire at all deltas?
            stable = any(abs(cl.true_time - p) <= tol for p in stable_set) if cl.kind == 1 else None
            # [SEVEN] Discovery-vs-queried mark (deploy-safe).
            if verdict == "CONFIRMED" and stable:
                mark = "discovery"
            elif verdict in ("FALSE_PULSE", "CONFIRMED_STALE"):
                mark = "queried-pattern"
            elif verdict == "CONFIRMED" and stable is False:
                mark = "queried-pattern"
            else:
                mark = "unknown"
            self.triggers.append(TriggerProvenanceRecord(
                cl, verdict, basis, stable, mark))
        return len(self.triggers)

    def discovery_census(self):
        """[SEVEN] How many triggers are discoveries vs queried patterns?"""
        from collections import Counter
        c = Counter(t.discovery_mark for t in self.triggers)
        n = len(self.triggers)
        return dict(
            counts=dict(c),
            n=n,
            discovery_rate=c.get("discovery", 0) / max(n, 1),
            queried_rate=c.get("queried-pattern", 0) / max(n, 1),
        )

    def inscribe_four_tests(self):
        """[SEVEN] Four's T1-T5 as provenance records. EXECUTED."""
        # T1: trigger purity on sine noise=4.0 (vacuous on clean signal).
        s, _ = generate(noise_std=4.0, seed=3)
        tp1 = trigger_purity(s)
        self.four_tests.append(FourTestProvenanceRecord(
            "T1", f"sine noise=4.0: {tp1['n_triggers']} triggers, "
                  f"purity {tp1['purity']:.1%}",
            measured=tp1,
            manufactured_components=["delta=3.0", "sustain=5", "smooth_w=21",
                                     "ungated trigger mapping"],
            verdict=("discovery" if tp1['purity'] == 1.0 else "queried-pattern"),
        ))
        # T2: trigger purity on pure noise — the hallucination battery.
        t2_rates = {}
        for ns in (1.0, 4.0, 8.0):
            nz, _ = generate(noise_std=ns, seed=3, pure_noise=True)
            tp_ = trigger_purity(nz)
            t2_rates[ns] = (tp_['n_triggers'], tp_['hallucination_rate'])
        self.four_tests.append(FourTestProvenanceRecord(
            "T2", f"pure noise: {t2_rates} (triggers, hallucination rate)",
            measured=t2_rates,
            manufactured_components=["delta=3.0 threshold on noise",
                                     "ungated trigger mapping",
                                     "detector fires on noise structure"],
            verdict="projection",
        ))
        # T3: threshold manufacture — positions stable, firing manufactured.
        pos = threshold_manufacture(s)
        s1, s8 = set(pos[1.0]), set(pos[8.0])
        matched = sum(1 for p in s1 if any(abs(p - q) <= 20 for q in s8))
        self.four_tests.append(FourTestProvenanceRecord(
            "T3", f"delta=1 vs delta=8: {matched}/{len(s1)} peaks coincide",
            measured={"matched": matched, "n": len(s1)},
            manufactured_components=["delta (firing decision)",
                                     "positions NOT moved by delta on strong sine"],
            verdict="discovery",
        ))
        # T4: gated purity — the fix, demonstrated not deployed.
        g = gated_trigger_purity(s, gate_fn=lambda v: v == "CONFIRMED")
        self.four_tests.append(FourTestProvenanceRecord(
            "T4", f"gated: {g['n_gated']}/{g['n_ungated']} triggers survive, "
                  f"suppressed {g['triggers_suppressed']} ({g['suppression_rate']:.1%})",
            measured=g,
            manufactured_components=["gate_fn (CONFIRMED only) — the gate is IMPOSED,",
                                     "not discovered; purity gain is manufactured",
                                     "suppression = foregone trades (EV unmeasured)"],
            verdict="manufactured-fix",
        ))
        # T5: drift — CONFIRMs without being the pattern.
        ss, _, _, _ = stress_shapes(noise_std=4.0, seed=3)["drift"]
        tp5 = trigger_purity(ss)
        self.four_tests.append(FourTestProvenanceRecord(
            "T5", f"drift: {tp5['n_triggers']} triggers, purity {tp5['purity']:.1%} "
                  f"{tp5['counts']}",
            measured=tp5,
            manufactured_components=["audit CONFIRMs drift-swings",
                                     "no tradeable pattern in drift"],
            verdict="broken-assumption",
        ))
        return len(self.four_tests)

    def gating_provenance(self, series, w=45, delta=3.0, sustain=5,
                          smooth_w=21, horizon=90):
        """[SEVEN] The provenance of the SUPPRESSION: which triggers would
        survive gating, which would be suppressed, and why. Four left the
        fix undeployed; this registry records what deployment would do."""
        claims = detect_claims(series, w=w, delta=delta, sustain=sustain,
                               smooth_w=smooth_w)
        survived, suppressed = [], []
        for cl in claims:
            v, basis = audit_claim(cl, series, horizon=horizon)
            entry = (cl.true_time, cl.kind, v, basis[:50])
            if v == "CONFIRMED":
                survived.append(entry)
            else:
                suppressed.append(entry)
        return dict(
            n_total=len(claims),
            n_survived=len(survived),
            n_suppressed=len(suppressed),
            survived=survived,
            suppressed=suppressed,
            note=("suppression provenance: each suppressed trigger carries "
                  "its audit verdict as the reason; deployment would need "
                  "the EV cost of foregone trades (unmeasured)"),
        )


if __name__ == "__main__":
    print("### [FOUR] T1: trigger purity — does the 1 read the audit? ###")
    print("sine noise=4.0, delta=3.0: every detection tri-stated, purity measured.")
    s, _ = generate(noise_std=4.0, seed=3)
    tp_ = trigger_purity(s)
    print(f"  triggers emitted: {tp_['n_triggers']}")
    print(f"  verdict breakdown: {tp_['counts']}")
    print(f"  trigger purity (P(CONFIRMED|triggered)): {tp_['purity']:.1%}")
    print(f"  trigger hallucination rate: {tp_['hallucination_rate']:.1%}")
    print("  [FOUR] Every non-CONFIRMED trigger is a 1 the audit would not")
    print("  endorse — but the binary matrix emits it anyway. The audit is")
    print("  adjacent, not gating. NOTE: on clean sine this is vacuous (T1 was")
    print("  100% pure) — the gap bites on noise (T2), not on signal.")

    print("\n### [FOUR] T2: trigger purity on pure noise ###")
    for ns in (1.0, 4.0, 8.0):
        nz, _ = generate(noise_std=ns, seed=3, pure_noise=True)
        tp_ = trigger_purity(nz)
        print(f"  noise={ns}: triggers={tp_['n_triggers']:>3} "
              f"purity={tp_['purity']:.1%} hallucination={tp_['hallucination_rate']:.1%} "
              f"{tp_['counts']}")

    print("\n### [FOUR] T3: threshold manufacture — do the 1-positions move? ###")
    s, _ = generate(noise_std=4.0, seed=3)
    pos = threshold_manufacture(s)
    for d, peaks in pos.items():
        print(f"  delta={d}: {len(peaks)} peak-1s at {peaks[:6]}{'...' if len(peaks)>6 else ''}")
    # Jaccard between delta=1 and delta=8 peak sets
    s1, s8 = set(pos[1.0]), set(pos[8.0])
    # tolerance-matched: count peaks within 20 of each other
    matched = sum(1 for p in s1 if any(abs(p - q) <= 20 for q in s8))
    print(f"  [FOUR] delta=1 vs delta=8: {matched}/{len(s1)} peaks coincide (tol 20).")
    print(f"  On strong sine, positions are STABLE across thresholds — the manufacture")
    print(f"  thesis for *positions* fails here. The threshold governs *whether* the")
    print(f"  detector fires (sensitivity), not *where* on strong signals. Positions")
    print(f"  are grown in the signal; the firing decision is manufactured.")

    print("\n### [FOUR] T4: gated purity — what if the 1 read the audit? ###")
    s, _ = generate(noise_std=4.0, seed=3)
    g = gated_trigger_purity(s, gate_fn=lambda v: v == "CONFIRMED")
    print(f"  ungated: {g['n_ungated']} triggers at {g['ungated_purity']:.1%} purity")
    print(f"  gated (CONFIRMED only): {g['n_gated']} triggers at {g['gated_purity']:.1%} purity")
    print(f"  suppressed: {g['triggers_suppressed']} ({g['suppression_rate']:.1%})")
    print("  [FOUR] Gating on the audit costs triggers but buys purity.")
    print("  The binary matrix as built does NOT gate. This is the fix.")

    print("\n### [FOUR] T5: drift — trigger purity where no pattern exists ###")
    ss, _, _, _ = stress_shapes(noise_std=4.0, seed=3)["drift"]
    tp_ = trigger_purity(ss)
    print(f"  drift: triggers={tp_['n_triggers']} purity={tp_['purity']:.1%} {tp_['counts']}")
    print("  [FOUR] Reversed's R4 showed the audit CONFIRMs drift-swings.")
    print("  The trigger fires on them. But there is no tradeable pattern.")
    print("  Purity here measures audit-agreement, not market-truth.")

    print("\n### [FOUR] PROVENANCE CHAIN ###")
    print(f"  SIGNAL:    {PROV_SIGNAL}")
    print(f"  DETECTED:  {PROV_DETECTED}")
    print(f"  TRIGGERED: {PROV_TRIGGERED}")
    print("  [FOUR] The '1' is two manufactures from the market.")
    print("  The audit illuminates DETECTED. Nothing illuminates TRIGGERED.")

    print("\n### [FOUR] LOAD-BEARING ASSUMPTION (trigger) ###")
    print("  A detector CONFIRMED extremum is a tradeable market extremum.")
    print("  Reversed's R4 breaks it: drift-swings CONFIRM but are not the pattern.")
    print("  Shuffled's haircut prices it. This turn measures the gap: the")
    print("  trigger fires on audit verdicts, but the matrix does not read them.")
    print("  Until the 1 is gated on CONFIRMED, the hallucination rate above")
    print("  is the EXECUTION hallucination rate, not just the detection one.")

    print("\n### [FOUR] UNMEASURED (at the line) ###")
    print("- Real market data: synthetic only; trigger purity is generator-relative.")
    print("- The gated fix (T4) is demonstrated, not deployed — needs a live")
    print("  decision on whether CONFIRMED_STALE should also gate (it is")
    print("  confirmed-but-worthless; execution should suppress it).")
    print("- F/V on gated triggers: Shuffled's EV ledger on the gated set —")
    print("  not run here. The purity gain has an EV cost (suppressed triggers")
    print("  are foregone trades). Net EV of gating: unmeasured.")
    print("- Wall-clock: microseconds at this scale; symbolic only.")

    print("\n### [SEVEN] TRIGGER PROVENANCE REGISTRY (executed) ###")
    reg = TriggerProvenanceRegistry(
        stream_provenance="manufactured: synthetic sine (seed 3, noise 4.0)")
    s, _ = generate(noise_std=4.0, seed=3)
    n_t = reg.inscribe_triggers(s)
    print(f"  triggers inscribed: {n_t} (each: evidence + audit + discovery mark)")
    dc = reg.discovery_census()
    print(f"  discovery census: {dc['counts']}")
    print(f"    discovery rate: {dc['discovery_rate']:.1%}")
    print(f"    queried-pattern rate: {dc['queried_rate']:.1%}")
    print(f"  stream: {reg.stream_provenance}")
    print(f"  sample audit trails (first 3):")
    for t in reg.triggers[:3]:
        print(f"    {t.audit_trail()}")

    print("\n### [SEVEN] FOUR'S T1-T5 AS PROVENANCE RECORDS ###")
    n_f = reg.inscribe_four_tests()
    print(f"  tests inscribed: {n_f}")
    for ft in reg.four_tests:
        print(f"  [{ft.test_id}] {ft.claim_desc}")
        print(f"    manufactured: {ft.manufactured_components}")
        print(f"    verdict: {ft.verdict.upper()}")

    print("\n### [SEVEN] GATING PROVENANCE (the undeployed fix, recorded) ###")
    gp = reg.gating_provenance(s)
    print(f"  total: {gp['n_total']}, survived: {gp['n_survived']}, "
          f"suppressed: {gp['n_suppressed']}")
    print(f"  suppressed triggers (with reasons):")
    for tt, kind, v, basis in gp['suppressed'][:5]:
        print(f"    tt={tt} kind={kind}: {v} ({basis}...)")
    print(f"  note: {gp['note']}")

    print("\n### [SEVEN] UNMEASURED (at the exact line) ###")
    print("- Discovery-vs-queried on real data: threshold-stability is the only")
    print("  deploy-safe proxy; on real data there is no true_extrema to check")
    print("  against. Stability across thresholds != truth, only robustness.")
    print("- Trough (kind=0) stability: not computed (threshold_manufacture only")
    print("  tracks peaks); trough marks default to audit-verdict-only.")
    print("- EV cost of gating: Four's open item stands — suppressed triggers")
    print("  are foregone trades; Shuffled's EV ledger on the gated set unrun.")
    print("- Real market trigger provenance: synthetic only; on real data the")
    print("  SIGNAL layer stays grown-unobservable (no true_extrema array).")
