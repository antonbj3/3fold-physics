"""Line-edge roughness reality gap: a deterministic aerial-image forward model predicts ~0 stochastic LER, a
measured SEM line edge does not.

A deterministic partially coherent (SOCS) forward model is translation invariant along the line, so the contour it
produces has essentially NO stochastic line-edge roughness: its LER variance floor is ~0. A measured SEM line edge
carries substantial roughness from photon shot noise, resist acid reaction-diffusion and blur, plus the SEM's own
measurement noise. The deterministic model therefore under-predicts the roughness by the full measured amount, and
the certifiable floor is the measured stochastic roughness, not the model's smooth edge.

The comparison is statistical (LER is a stochastic edge property compared by magnitude on line/space patterns), and
the deterministic-vs-stochastic gap is pattern independent: a deterministic forward model has 0 stochastic LER for
any pattern. The measured side is reported both raw and noise-subtracted (the high-frequency plateau of the edge
displacement PSD is subtracted as measurement noise), so the gap is shown to survive removal of the SEM noise and
to be real resist roughness.

GATES: G1 both sides measured (SEM LER > 1 px after noise subtraction; SOCS contour LER < 0.2 px); G2 the
deterministic model under-predicts by more than 10x and the gap survives noise subtraction; G3 the certifiable
floor is the stochastic roughness.

I/O: the SEM side reads line images from data/smile-sem-ler/ at the repository root (Test*.tif); the model side
uses the pylitho SOCS forward model shipped with the LithoBench mask benchmark in data/lithobench-real-masks/
(needs torch). When either directory is absent the module prints a SYNTHETIC INPUT line and substitutes a
stand-in: a synthetic SEM line image whose edges carry a known correlated roughness plus white detector noise, and
a Gaussian optical low-pass in place of the SOCS aerial image. Both stand-ins exercise the same estimators.
Evidence -> artifacts/i_litho_synthetic_vs_real_contour_ler_reality_gap.json. Exit 0 on pass.
"""
import json
import os

import numpy as np
from scipy.ndimage import gaussian_filter1d

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SMILE = os.path.join(_REPO_ROOT, "data", "smile-sem-ler")
LITHO = os.path.join(_REPO_ROOT, "data", "lithobench-real-masks")
ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")


def synthetic_sem_image(h=220, w=320, pitch=40, cd=20, ler_px=2.0, blur=1.6, noise=7.0, seed=0):
    """Stand-in SEM line image: vertical lines whose edges carry a CORRELATED displacement of standard deviation
    ler_px (the resist roughness) on top of white detector noise (the measurement noise the PSD plateau removes)."""
    rng = np.random.default_rng(seed)
    y = np.arange(h)
    im = np.zeros((h, w))
    x = np.arange(w)[None, :]
    for x0 in range(30, w - 30, pitch):
        d = gaussian_filter1d(rng.standard_normal(h), 6.0)
        d = (d / d.std()) * ler_px                                   # correlated edge displacement
        left = x0 + d[:, None]
        right = x0 + cd + d[:, None]
        im += 120.0 * ((x > left) & (x < right))
    im = gaussian_filter1d(gaussian_filter1d(im, blur, axis=1), 0.6, axis=0) + 40.0
    im += noise * rng.standard_normal(im.shape)                      # white detector noise
    return np.clip(im, 0, 255)


def load_sem_image():
    """Measured SEM line image if the dataset is present, else the synthetic stand-in."""
    if os.path.isdir(SMILE):
        cand = sorted(f for f in os.listdir(SMILE) if f.lower().endswith((".tif", ".tiff")))
        if cand:
            from PIL import Image
            return np.array(Image.open(os.path.join(SMILE, cand[0])).convert("L")).astype(float), f"SEM-{cand[0]}"
    print(f"SYNTHETIC INPUT: no SEM line images in {SMILE}, using a synthetic line image with injected roughness")
    return synthetic_sem_image(), "SYNTHETIC-sem-line-image"


def sem_ler(im):
    """SEM line LER: local-band sub-pixel edges -> standard deviation + PSD noise-floor subtraction."""
    from scipy.signal import find_peaks, welch
    H, W = im.shape
    prof = gaussian_filter1d(im.mean(0), 2.0)
    pk, _ = find_peaks(np.abs(np.gradient(prof)), prominence=np.abs(np.gradient(prof)).std(), distance=15)
    devs = []
    for ex in pk:
        if ex - 8 < 0 or ex + 9 > W:
            continue
        pos = np.full(H, np.nan)
        for y in range(H):
            seg = gaussian_filter1d(im[y, ex - 8:ex + 9], 1.0)
            lo, hi = seg.min(), seg.max()
            if hi - lo < 8:
                continue
            t = lo + 0.5 * (hi - lo)
            c = np.where(np.diff((seg > t).astype(int)) != 0)[0]
            if len(c):
                i = c[0]
                pos[y] = (ex - 8 + i) + (t - seg[i]) / (seg[i + 1] - seg[i] + 1e-9)
        good = ~np.isnan(pos)
        if good.mean() > 0.8:
            devs.append(pos[good] - np.nanmean(pos))
    raw = float(np.mean([d.std() for d in devs]))                      # raw LER (px), includes detector noise
    # noise-subtracted (unbiased) LER via the PSD high-frequency plateau
    n = min(len(d) for d in devs)
    P = np.mean([welch(d[:n] - d[:n].mean(), nperseg=min(256, n), scaling="density")[1] for d in devs], axis=0)
    f = welch(devs[0][:n] - devs[0][:n].mean(), nperseg=min(256, n), scaling="density")[0]
    floor = np.median(P[f > 0.35 * f[-1]])
    var_true = max(np.trapezoid(P, f) - floor * f[-1], 0.0)
    return raw, float(np.sqrt(var_true)), len(devs)


def socs_ler():
    """Deterministic aerial-image contour LER: SOCS aerial image of a line/space mask -> threshold contour ->
    per-row edge -> standard deviation (expected ~0 because the forward model is deterministic)."""
    import sys
    cwd = os.getcwd()
    S = 256
    mask = np.zeros((S, S), np.float32)
    for x0 in range(16, S, 32):
        mask[:, x0:x0 + 8] = 1.0                                       # vertical line/space (8 px lines, 32 px pitch)
    if os.path.isdir(LITHO):
        try:
            os.chdir(LITHO); sys.path.insert(0, LITHO)
            import torch
            import pylitho.simple as ps
            sim = ps.LithoSim("./config/lithosimple.txt")
            aerial = sim.sim(torch.tensor(mask, dtype=torch.float32))[0].detach().numpy()
            prov = "pylitho-SOCS-24kernel"
        except Exception as e:
            aerial = gaussian_filter1d(mask, 4, axis=1); prov = f"SYNTHETIC-gaussian ({type(e).__name__})"
        finally:
            os.chdir(cwd)
    else:
        print(f"SYNTHETIC INPUT: no SOCS forward model at {LITHO}, using a Gaussian optical low-pass")
        aerial = gaussian_filter1d(mask, 4, axis=1); prov = "SYNTHETIC-gaussian-lowpass"
    # contour LER: for one edge column region, per-row sub-pixel threshold crossing
    prof = aerial.mean(0); pk = np.argmax(np.abs(np.gradient(gaussian_filter1d(prof, 2))))
    thr = 0.5 * (aerial.max() + aerial.min()); pos = []
    for y in range(aerial.shape[0]):
        seg = aerial[y, max(0, pk - 8):pk + 9]
        c = np.where(np.diff((seg > thr).astype(int)) != 0)[0]
        if len(c):
            i = c[0]; pos.append(i + (thr - seg[i]) / (seg[i + 1] - seg[i] + 1e-12))
    return (float(np.std(pos)) if len(pos) > 10 else 0.0), prov


def main():
    print("=" * 118)
    print("LER REALITY GAP: deterministic SOCS contour LER ~ 0 vs measured SEM LER — the model's variance floor")
    print("=" * 118)
    im, sem_prov = load_sem_image()
    raw_px, cert_px, n_edges = sem_ler(im)
    ler_socs, socs_prov = socs_ler()
    print(f"  SEM side ({sem_prov}, {n_edges} edges): raw LER={raw_px:.3f} px, noise-subtracted LER={cert_px:.3f} px")
    print(f"  model side ({socs_prov}): contour LER = {ler_socs:.4f} px (deterministic forward model => ~0 stochastic roughness)")
    ratio = cert_px / max(ler_socs, 1e-6)
    ratio_str = "unbounded (the model predicts ~0 LER)" if ler_socs < 1e-3 else f"x{ratio:.0f}"

    g1 = (cert_px > 1.0) and (ler_socs < 0.2)
    print(f"\n[G1] both sides measured: SEM noise-subtracted LER {cert_px:.2f} px > 1; SOCS contour LER {ler_socs:.3f} px ~ 0 -> {g1}")

    g2 = (ratio > 10) and (cert_px > 1.0)
    print(f"\n[G2] the DETERMINISTIC model UNDER-PREDICTS the roughness: {cert_px:.2f} px vs {ler_socs:.3f} px ({ratio_str});")
    print(f"     the gap SURVIVES subtraction of the detector-noise PSD plateau, so it is resist roughness the SOCS model omits -> {g2}")

    g3 = g1 and g2
    print(f"\n[G3] the certifiable floor is the stochastic roughness (photon shot noise + resist blur), not the model's ~0 -> {g3}")

    verdict = g1 and g2 and g3
    print("\n" + "=" * 118)
    print(f"VERDICT: {'PASS' if verdict else 'FAIL'} — G1(both measured)={g1} G2(model under-predicts)={g2} G3(certify the measured floor)={g3}")
    os.makedirs(ART, exist_ok=True)
    out = os.path.join(ART, "i_litho_synthetic_vs_real_contour_ler_reality_gap.json")
    print(f"EVIDENCE -> {out}")

    json.dump({
        "module": "i_litho_synthetic_vs_real_contour_ler_reality_gap",
        "provenance": {"sem": sem_prov, "model": socs_prov},
        "sem_ler_px": {"raw": raw_px, "noise_subtracted": cert_px, "n_edges": n_edges},
        "socs_contour_ler_px": ler_socs, "under_prediction_ratio": ratio,
        "gates": {"G1_both_measured": bool(g1), "G2_model_underpredicts": bool(g2), "G3_certify_stochastic_floor": bool(g3)},
        "claim": ("A deterministic partially coherent (SOCS) forward model produces contours with ~0 stochastic line-edge roughness "
                  "(%.4f px) while a SEM line edge carries substantial roughness (raw %.2f px, noise-subtracted %.2f px). The model "
                  "under-predicts the roughness by more than an order of magnitude and the gap survives subtraction of the detector "
                  "noise, so it is resist roughness (photon shot noise, acid reaction-diffusion, blur) that the deterministic model "
                  "omits: the certifiable floor is the stochastic roughness, not the model's smooth edge."
                  % (ler_socs, raw_px, cert_px)),
        "scope": ("Statistical LER comparison; the deterministic-vs-stochastic gap is pattern independent, and the two sides are "
                  "different line/space patterns of comparable CD, so the MAGNITUDE is the honest quantity, not a same-structure "
                  "pixel overlap. Without the datasets both sides run on stand-ins (a synthetic SEM image with injected correlated "
                  "roughness, a Gaussian optical low-pass) and the measured magnitudes are not reproduced."),
    }, open(out, "w"), indent=1)
    return verdict


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
