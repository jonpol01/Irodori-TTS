"""Parity + timing for the v4.1-Small-MF Core ML packages built by export_v4_coreml.py (macOS only).

1. Compiles each .mlpackage to .mlmodelc once (next to the packages, under compiled/).
2. Per compute-unit config: cold/warm load, per-package parity vs the fp32 PyTorch wrappers on real inputs.
3. End to end, caption only: PyTorch fp32 runtime path vs the Core ML host loop (the reference for the Swift
   runtime), same injected noise and frame count. Per-phase timings, latent / audio / log-spectral distance.
   WAVs (16-bit 48 kHz): torch_c{caption}_s{sentence}.wav and coreml_{units}_c*_s*.wav in <coreml_dir>/wav.

PyTorch runs first in this process; the Core ML side runs in child processes (see coreml_stage for why).

Run: <venv>/bin/python parity_v4_coreml.py <coreml_dir> [--units gpu,ane,cpu] [--wav <dir>] [--load-only]
"""
import argparse
import os
import time
import wave

import numpy as np
import torch

import coremltools as ct
from export_v4_coreml import (CAP_LEN, DEC_T, DIT_S, DIT_T, HOP, LATENT_DIM, NAMES, SR, TEXT_BUCKETS, DecoderCM,
                              DiTStepCM, DurationCM, EncoderCM, bucket, decoder_chunk_plan, load_runtime)

UNITS = {"gpu": ct.ComputeUnit.CPU_AND_GPU, "ane": ct.ComputeUnit.CPU_AND_NE, "cpu": ct.ComputeUnit.CPU_ONLY}
TEXTS = ["画像には、暗い色の物体（おそらくモニターの一部）と、茶色っぽい壁のような表面が写っています。",
         "画像には、茶色っぽい壁のような表面に黒いケーブルのようなものが横切っている様子が写っています。",
         "これは黒いモニターの一部です。具体的なブランド名は特定できません。"]
CAPS = ["落ち着いた自然な女性の声", "低く落ち着いた渋い男性の声"]
NUM_STEPS = 4


def rel_cos(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    rel = np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12)
    cos = float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
    return rel, cos


def lsd_db(a, b, n_fft=2048, hop=480, floor_db=80):
    """Mean log-spectral distance (dB), both spectra floored at (reference peak - floor_db), over frames within
    60 dB of the reference peak (otherwise near-silent bins dominate)."""
    def spec(x):
        n = 1 + (len(x) - n_fft) // hop
        fr = np.lib.stride_tricks.as_strided(x, (n, n_fft), (x.strides[0] * hop, x.strides[0]))
        return 20 * np.log10(np.abs(np.fft.rfft(fr * np.hanning(n_fft), axis=1)) + 1e-7)
    sa, sb = spec(np.asarray(a, np.float64)), spec(np.asarray(b, np.float64))
    fl = sb.max() - floor_db
    sa, sb = np.maximum(sa, fl), np.maximum(sb, fl)
    keep = sb.max(axis=1) > sb.max() - 60
    return float(np.sqrt(((sa - sb) ** 2).mean(axis=1))[keep].mean())


def write_wav(path, audio):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())


def compile_all(pkg_dir):
    out = os.path.join(pkg_dir, "compiled")
    os.makedirs(out, exist_ok=True)
    paths = {}
    for k, name in NAMES.items():
        dst = os.path.join(out, name.replace(".mlpackage", ".mlmodelc"))
        if not os.path.exists(dst):
            t0 = time.perf_counter()
            ct.models.utils.compile_model(os.path.join(pkg_dir, name), dst)
            print(f"[compile] {k}: {time.perf_counter() - t0:.1f}s", flush=True)
        paths[k] = dst
    return paths


class CM:
    """Loaded Core ML models for one compute-unit config; records first (cold) and second (warm) load time."""

    def __init__(self, paths, units):
        self.load = {}
        self.m = {}
        for k in ("duration", "decoder"):
            self.m[k] = self._load2(k, paths[k], units)
        for L in TEXT_BUCKETS:
            self.m[f"L{L}"] = self._load2(f"enc:L{L}", paths["encoder"], units, f"L{L}")
        for s in DIT_S:
            self.m[f"s{s}"] = self._load2(f"dit:s{s}", paths["dit"], units, f"s{s}")

    def _load2(self, key, path, units, fn=None):
        times = []
        for _ in range(2):
            t0 = time.perf_counter()
            m = ct.models.CompiledMLModel(path, compute_units=units, function_name=fn)
            times.append(time.perf_counter() - t0)
        self.load[key] = times
        return m

    def encoder(self, ids, n):
        o = self.m[f"L{ids.shape[1]}"].predict({"input_ids": ids.astype(np.int32), "length": np.array([n], np.int32)})
        return o["text_state"], o["caption_state"]

    def duration(self, ts, nt, cs, nc, feats):
        return float(self.m["duration"].predict({
            "text_state": ts.astype(np.float32), "text_len": np.array([nt], np.int32),
            "caption_state": cs.astype(np.float32), "caption_len": np.array([nc], np.int32),
            "duration_features": feats.astype(np.float32)})["log_frames"].ravel()[0])

    def dit(self, s, x, t, dt, n, ts, nt, cs, nc):
        return self.m[f"s{s}"].predict({
            "x_t": x.astype(np.float32), "t": np.array([t], np.float32), "delta_t": np.array([dt], np.float32),
            "latent_len": np.array([n], np.int32), "text_state": ts.astype(np.float32),
            "text_len": np.array([nt], np.int32), "caption_state": cs.astype(np.float32),
            "caption_len": np.array([nc], np.int32)})["v"]

    def decoder(self, z):
        return self.m["decoder"].predict({"z": z.astype(np.float32)})["audio"]


def pad_to(a, L, axis=1):
    w = [(0, 0)] * a.ndim
    w[axis] = (0, L - a.shape[axis])
    return np.pad(a, w)


# ── PyTorch fp32 reference (the runtime's synthesize() path, noise injected) ──

class TorchRef:
    def __init__(self, rt):
        from irodori_tts.duration import build_duration_features
        from irodori_tts.inference_runtime import SamplingRequest, find_flattening_point
        from irodori_tts.text_normalization import normalize_text

        self.rt, self.m = rt, rt.model
        self.bdf, self.req, self.flat, self.norm = build_duration_features, SamplingRequest, find_flattening_point, normalize_text

    def inputs(self, text, caption):
        rt = self.rt
        norm = self.norm(text).strip()
        ids, mask = rt.tokenizer.batch_encode([norm], max_length=rt.default_text_max_len)
        cids, cmask = rt.caption_tokenizer.batch_encode([caption], max_length=rt.default_caption_max_len)
        feats = self.bdf([norm], token_counts=mask.sum(dim=1), max_text_len=rt.default_text_max_len,
                         has_speaker=torch.zeros(1, dtype=torch.bool))
        return norm, ids, mask, cids, cmask, feats

    @torch.no_grad()
    def run(self, text, caption, noise_seed):
        rt, m = self.rt, self.m
        norm, ids, mask, cids, cmask, feats = self.inputs(text, caption)
        ref, refm = rt._load_reference_latent(req=self.req(text=text, caption=caption, no_ref=True),
                                              batch_size=1, messages=[])
        enc = m.encode_conditions(text_input_ids=ids, text_mask=mask, ref_latent=ref, ref_mask=refm,
                                  caption_input_ids=cids, caption_mask=cmask)
        ts, tm, ss, sm, cs, cm = enc
        logf = m.predict_duration_log_frames(
            text_state=ts, text_mask=tm, speaker_state=ss, speaker_mask=sm, caption_state=cs, caption_mask=cm,
            duration_features=feats, has_speaker=torch.zeros(1, dtype=torch.bool),
            has_caption=torch.ones(1, dtype=torch.bool))
        frames = frames_from_log(float(logf.item()))
        x = torch.randn(1, frames, LATENT_DIM, generator=torch.Generator().manual_seed(noise_seed))
        x0 = x.clone()
        sched = torch.linspace(1.0, 0.0, NUM_STEPS + 1)
        for i in range(NUM_STEPS):
            t, tn = sched[i], sched[i + 1]
            v = m.forward_with_encoded_conditions(x_t=x, t=t.expand(1), delta_t=(t - tn).expand(1), text_state=ts,
                                                  text_mask=tm, speaker_state=ss, speaker_mask=sm,
                                                  caption_state=cs, caption_mask=cm)
            x = x + v * (tn - t)
        audio = rt.codec.decode_latent(x)[0, 0].numpy()[: frames * HOP]
        return dict(frames=frames, log_frames=float(logf.item()), x0=x0.numpy(), z=x.numpy(), audio=audio,
                    keep=self.keep_samples(x[0], frames))

    def keep_samples(self, z, frames):  # runtime trim_tail defaults
        fp = self.flat(torch.as_tensor(z), window_size=20, std_threshold=0.05, mean_threshold=0.1)
        return min(frames * HOP, fp * HOP) if fp > 0 else frames * HOP


def frames_from_log(logf, min_s=0.5, max_s=30.0, scale=1.0):
    import math
    lo = max(1, math.ceil(min_s * SR / HOP))
    hi = max(1, math.floor(max_s * SR / HOP))
    return max(lo, min(hi, int(round(math.expm1(logf) * scale))))


# ── Core ML host loop (mirror this in Swift) ──

def coreml_tts(cm, ids_full, n, cids_full, nc, feats, x0, frames_override=None):
    """ids_full / cids_full: token ids incl. BOS, any padding length; n / nc: valid counts.
    x0: (1, frames, 32) initial noise. Returns (latent (1,frames,32), audio, timings, predicted frames)."""
    T = {}
    t0 = time.perf_counter()
    Lb = bucket(TEXT_BUCKETS, n)
    ts, _ = cm.encoder(pad_to(ids_full[:, :Lb], Lb), n)
    _, cs = cm.encoder(pad_to(cids_full[:, :CAP_LEN], CAP_LEN), nc)  # caption head of the same encoder
    T["encode"] = time.perf_counter() - t0
    t1 = time.perf_counter()
    logf = cm.duration(ts, n, cs, nc, feats)
    pred = frames_from_log(logf)
    T["duration"] = time.perf_counter() - t1
    frames = frames_override or pred
    S = bucket(DIT_S, frames)
    t2 = time.perf_counter()
    ts_d = pad_to(ts[:, :min(n, DIT_T)], DIT_T)
    x = pad_to(x0[:, :frames].astype(np.float32), S)
    sched = np.linspace(1.0, 0.0, NUM_STEPS + 1, dtype=np.float32)
    for i in range(NUM_STEPS):
        t, tn = float(sched[i]), float(sched[i + 1])
        v = cm.dit(S, x, t, t - tn, frames, ts_d, n, cs, nc)
        x = x + v * (tn - t)
    T["sampler"] = time.perf_counter() - t2
    z = x[:, :frames]
    t3 = time.perf_counter()
    zc = z.transpose(0, 2, 1)  # (1, 32, frames)
    parts = []
    for ws, w, ks, ke in decoder_chunk_plan(frames):
        win = pad_to(zc[:, :, ws:ws + w], w, axis=2)
        a = cm.decoder(win)[0, 0]
        parts.append(a[(ks - ws) * HOP:(ke - ws) * HOP])
    audio = np.concatenate(parts)[: frames * HOP]
    T["decode"] = time.perf_counter() - t3
    T["total"] = time.perf_counter() - t0
    return z, audio, T, pred, logf


# ── per-package parity ──

@torch.no_grad()
def package_refs(tref, wr, ref_latent):
    """[(name, kind, inputs, torch reference output)] on real inputs; plain arrays so they can be pickled."""
    cases = []
    i32 = lambda v: torch.tensor([v], dtype=torch.int32)  # noqa: E731
    norm, ids, mask, cids, cmask, feats = tref.inputs(TEXTS[0] + TEXTS[1] + TEXTS[2] + TEXTS[0], CAPS[1])
    nmax, nc = int(mask.sum()), int(cmask.sum())
    cpad = pad_to(cids.numpy()[:, :CAP_LEN], CAP_LEN)
    cs_ref = wr["enc"](torch.from_numpy(cpad).int(), i32(nc))[1]
    csn, f = cs_ref.numpy(), feats.numpy()
    for L in TEXT_BUCKETS:
        n = min(nmax, L - 3)
        x = pad_to(ids.numpy()[:, :L].copy(), L)
        x[0, n:] = 0
        r_t, r_c = wr["enc"](torch.from_numpy(x).int(), i32(n))
        cases.append((f"encoder L={L} text (n={n})", "enc_text", (x, n), r_t[0, :n].numpy()))
        cases.append((f"encoder L={L} caption head", "enc_cap", (x, n), r_c[0, :n].numpy()))
        r = float(wr["dur"](r_t, i32(n), cs_ref, i32(nc), feats).item())
        cases.append((f"duration L={L} log_frames", "dur", (r_t.numpy(), n, csn, nc, f), r))
    n = min(nmax, DIT_T)
    tpad = pad_to(ids.numpy()[:, :DIT_T].copy(), DIT_T)
    tpad[0, n:] = 0
    ts_ref = wr["enc"](torch.from_numpy(tpad).int(), i32(n))[0]
    g = torch.Generator().manual_seed(7)
    for s in DIT_S:
        x = torch.randn(1, s, LATENT_DIM, generator=g)
        nl = s - 5
        r = wr[f"s{s}"](x, torch.tensor([0.75]), torch.tensor([0.25]), i32(nl), ts_ref, i32(n), cs_ref, i32(nc))
        cases.append((f"dit s{s} v (t=0.75, len={nl})", "dit", (s, x.numpy(), nl, ts_ref.numpy(), n, csn, nc),
                      r[:, :nl].numpy()))
    for w in DEC_T:
        z = torch.from_numpy(pad_to(ref_latent[:, :w], w).transpose(0, 2, 1).copy())
        cases.append((f"decoder W={w} audio", "dec", (z.numpy(),), wr["dec"](z).numpy()))
    return cases


def package_parity(cm, cases):
    for name, kind, a, ref in cases:
        if kind == "enc_text":
            got = cm.encoder(*a)[0][0, :a[1]]
        elif kind == "enc_cap":
            got = cm.encoder(*a)[1][0, :a[1]]
        elif kind == "dur":
            got = cm.duration(*a)
            yield (f"{name} {got:.4f} vs {ref:.4f} (frames {frames_from_log(got)} vs {frames_from_log(ref)})",
                   abs(got - ref) / abs(ref), float("nan"))
            continue
        elif kind == "dit":
            s, x, nl, ts, n, cs, nc = a
            got = cm.dit(s, x, 0.75, 0.25, nl, ts, n, cs, nc)[:, :nl]
        else:
            got = cm.decoder(*a)
        yield (name, *rel_cos(got, ref))


def flattening_point(z, window=20, std_th=0.05, mean_th=0.1):
    """numpy port of inference_runtime.find_flattening_point (runtime trim_tail defaults); z (T, D)."""
    p = np.concatenate([z, np.zeros((window, z.shape[1]), z.dtype)])
    for i in range(len(z)):
        w = p[i:i + window]
        if w.std() < std_th and abs(w.mean()) < mean_th:
            return i
    return len(z)


def coreml_stage(paths, u, data, wav_dir, only, e2e):
    """Core ML work for one compute-unit config, in a process that runs no torch compute.

    only: run just the package-parity cases whose name contains this string ("" = all but the DiT cases), and
    skip the load report and end to end. Why the split: under CPU_ONLY / CPU_AND_NE, running the five DiT
    functions back to back after the other packages segfaulted inside Core ML (or in the next numpy call), while
    each DiT function alone, or the end-to-end loop (s192/s256) alone, ran clean. CPU_AND_GPU never crashed."""
    cm = CM(paths, UNITS[u])
    if not only:
        print(f"\n===== {u} ({UNITS[u].name}) =====", flush=True)
        for k, (c, w) in cm.load.items():
            print(f"[load {u}] {k:10s} first={c:.2f}s second={w:.2f}s", flush=True)
    cases = [c for c in data["pcases"] if (only in c[0] if only else c[1] != "dit")]
    for name, rel, cos in package_parity(cm, cases):
        print(f"[parity {u}] {name}: relL2={rel:.2e}" + ("" if np.isnan(cos) else f" cos={cos:.6f}"), flush=True)
    if not e2e:
        return
    tsum = {}
    for k, ((si, ci), R) in enumerate(data["e2e"].items()):
        z, audio, T, pred, logf = coreml_tts(cm, R["ids"], R["n"], R["cids"], R["nc"], R["feats"], R["x0"],
                                            frames_override=R["frames"])
        zr, _ = rel_cos(z, R["z"])
        ar, acos = rel_cos(audio, R["audio"])
        lsd = lsd_db(audio, R["audio"])
        fp = flattening_point(z[0])
        keep = min(R["frames"], fp) * HOP if fp > 0 else R["frames"] * HOP
        write_wav(os.path.join(wav_dir, f"coreml_{u}_c{ci}_s{si}.wav"), audio[:keep])
        print(f"[e2e {u}] c{ci} s{si}: frames ref={R['frames']} cm_pred={pred} (logf {logf:.4f} vs {R['log_frames']:.4f}) "
              f"S={bucket(DIT_S, R['frames'])} latent relL2={zr:.2e} audio relL2={ar:.2e} cos={acos:.4f} "
              f"LSD={lsd:.2f}dB wav={keep / SR:.2f}s (ref {R['keep'] / SR:.2f}s) | "
              + " ".join(f"{p}={v * 1000:.0f}ms" for p, v in T.items()), flush=True)
        if k > 0:
            for p, v in T.items():
                tsum[p] = tsum.get(p, 0.0) + v / (len(data["e2e"]) - 1)
    print(f"[timing {u}] steady mean (excl. first): " + " ".join(f"{p}={v * 1000:.0f}ms" for p, v in tsum.items()),
          flush=True)


def main():
    import pickle
    import subprocess
    import sys

    ap = argparse.ArgumentParser()
    ap.add_argument("coreml_dir")
    ap.add_argument("--units", default="gpu,ane,cpu")
    ap.add_argument("--wav", default=None)
    ap.add_argument("--load-only", action="store_true", help="only time model loads (run twice for cold/warm)")
    ap.add_argument("--stage", default=None, help=argparse.SUPPRESS)  # internal: pickled torch refs → Core ML stage
    ap.add_argument("--only", default="", help=argparse.SUPPRESS)  # internal: see coreml_stage
    a = ap.parse_args()
    wav_dir = a.wav or os.path.join(a.coreml_dir, "wav")
    os.makedirs(wav_dir, exist_ok=True)
    paths = compile_all(a.coreml_dir)
    if a.load_only:
        for u in a.units.split(","):
            cm = CM(paths, UNITS[u])
            print(f"[load {u}] " + " ".join(f"{k}={c:.2f}s" for k, (c, _) in cm.load.items()), flush=True)
        return
    if a.stage:
        with open(a.stage, "rb") as fh:
            data = pickle.load(fh)
        for u in a.units.split(","):
            coreml_stage(paths, u, data, wav_dir, a.only, e2e=not a.only)
        return

    rt = load_runtime()
    tref = TorchRef(rt)
    cases = [(si, ci) for ci in range(len(CAPS)) for si in range(len(TEXTS))]
    e2e = {}
    t0 = time.perf_counter()
    for si, ci in cases:
        R = tref.run(TEXTS[si], CAPS[ci], noise_seed=1000 + 10 * si + ci)
        write_wav(os.path.join(wav_dir, f"torch_c{ci}_s{si}.wav"), R["audio"][: R["keep"]])
        norm, ids, mask, cids, cmask, feats = tref.inputs(TEXTS[si], CAPS[ci])
        R.update(ids=ids.numpy(), n=int(mask.sum()), cids=cids.numpy(), nc=int(cmask.sum()), feats=feats.numpy())
        e2e[si, ci] = R
    print(f"[torch] 6 utterances fp32 CPU: {time.perf_counter() - t0:.1f}s", flush=True)
    m = rt.model
    wr = {"enc": EncoderCM(m).eval(), "dur": DurationCM(m).eval(), "dec": DecoderCM(rt.codec.model).eval()}
    for s in DIT_S:
        wr[f"s{s}"] = DiTStepCM(m, s).eval()
    stage = os.path.join(a.coreml_dir, "_parity_refs.pkl")
    with open(stage, "wb") as fh:
        pickle.dump({"pcases": package_refs(tref, wr, e2e[0, 0]["z"]), "e2e": e2e}, fh)
    del rt, tref, wr, m
    try:
        for u in a.units.split(","):  # fresh processes per config, so loads are cold and configs don't interact
            for only in [""] + [f"dit s{s} " for s in DIT_S]:
                r = subprocess.run([sys.executable, "-X", "faulthandler", os.path.abspath(__file__), a.coreml_dir,
                                    "--units", u, "--wav", wav_dir, "--stage", stage, "--only", only], check=False)
                if r.returncode:
                    print(f"[stage {u} {only or 'main'}] FAILED, exit code {r.returncode}", flush=True)
    finally:
        os.remove(stage)


if __name__ == "__main__":
    main()
