"""v4.1-Small-MF（MeanFlow）の ONNX エクスポート（端末 CPU 経路用）: encoder / duration predictor / 動的形状 DiT。
v3 との差分: テキスト/キャプションエンコーダが ModernBERT-ja(310M)+射影になり、DiT は delta_t（区間長）入力を持つ
MeanFlow パラメタ化（CFG 不要 → batch 1）。ref_state はアプリ側で bake 済み ref4_*.bin を与える。
最後に ONNX 3点 + 実コーデックで「まだ配車されていません。」を合成し whisper で読みを確認する（E2E ゲート）。
実行: <py311 venv>/bin/python export_v4_onnx.py"""
import glob
import os
import subprocess

import numpy as np
import onnxruntime as ort
import torch

from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey

torch.set_num_threads(6)
OUTDIR = "/tmp/v4_onnx"
os.makedirs(OUTDIR, exist_ok=True)
SNAP = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--Aratako--Irodori-TTS-v4.1-Small-MF/snapshots/*/"))[0]
ckpt = SNAP + "model.safetensors"
rt = InferenceRuntime.from_key(RuntimeKey(
    checkpoint=ckpt, model_device="cpu", codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim",
    model_precision="fp32", codec_device="cpu", codec_precision="fp32",
    codec_deterministic_encode=True, codec_deterministic_decode=True,
    compile_model=False, compile_dynamic=False))
m = rt.model.eval()
assert m.cfg.flow_parameterization == "meanflow" and m.delta_cond_module is not None
AUX = int(m.cfg.duration_aux_dim)
TD, SD = int(m.cfg.text_dim), int(m.cfg.speaker_dim)
print(f"duration_aux_dim={AUX} text_dim={TD} speaker_dim={SD} patch={m.cfg.latent_patch_size}")

# ── 1) encoder（ModernBERT 共有バックボーン + text/caption 射影 + norm）──
class Enc(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m
    def forward(self, text_ids, text_mask, caption_ids, caption_mask):
        bb = self.m.pretrained_text_backbone
        ts = self.m.text_norm(self.m.text_encoder(bb, text_ids, text_mask))
        cs = self.m.caption_norm(self.m.caption_encoder(bb, caption_ids, caption_mask))
        return ts, cs

enc = Enc(m).eval()
ti = torch.randint(8, 1000, (1, 17)); tmk = torch.ones(1, 17, dtype=torch.bool)
ci = torch.randint(8, 1000, (1, 9)); cmk = torch.ones(1, 9, dtype=torch.bool)
with torch.no_grad():
    ref_ts, ref_cs = enc(ti, tmk, ci, cmk)
enc_path = f"{OUTDIR}/irodori4_encoder.onnx"
def export_enc():
    torch.onnx.export(
        enc, (ti, tmk, ci, cmk), enc_path,
        input_names=["text_ids", "text_mask", "caption_ids", "caption_mask"],
        output_names=["text_state", "caption_state"],
        dynamic_axes={"text_ids": {1: "Tt"}, "text_mask": {1: "Tt"},
                      "caption_ids": {1: "Tc"}, "caption_mask": {1: "Tc"},
                      "text_state": {1: "Tt"}, "caption_state": {1: "Tc"}},
        opset_version=17, dynamo=False)
try:
    export_enc()
except Exception as e:  # sdpa がトレースできない環境向けフォールバック
    print("[encoder] sdpa export failed → eager:", str(e)[:200])
    m.pretrained_text_backbone.backbone.config._attn_implementation = "eager"
    export_enc()
s_enc = ort.InferenceSession(enc_path, providers=["CPUExecutionProvider"])
o = s_enc.run(None, {"text_ids": ti.numpy(), "text_mask": tmk.numpy(), "caption_ids": ci.numpy(), "caption_mask": cmk.numpy()})
c = np.corrcoef(o[0].ravel(), ref_ts.numpy().ravel())[0, 1]
print(f"[verify encoder onnx] corr={c:.6f} max|diff|={np.abs(o[0]-ref_ts.numpy()).max():.2e}")
assert c > 0.9999, "encoder ONNX parity failed"
# 長さ可変の確認（トレース長 17 と違う長さで壊れないこと）
o2 = s_enc.run(None, {"text_ids": torch.randint(8, 1000, (1, 41)).numpy(), "text_mask": np.ones((1, 41), bool),
                      "caption_ids": torch.randint(8, 1000, (1, 12)).numpy(), "caption_mask": np.ones((1, 12), bool)})
assert o2[0].shape == (1, 41, TD) and o2[1].shape == (1, 12, TD)
print("[export] encoder OK", os.path.getsize(enc_path) / 1e6, "MB")

# ── 2) duration predictor（v3 と同じ I/F）──
class Dur(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m
    def forward(self, text_state, text_mask, ss, sm, cs, cm, feats):
        ones = torch.ones(text_state.shape[0], dtype=torch.bool)
        return self.m.predict_duration_log_frames(
            text_state=text_state, text_mask=text_mask,
            speaker_state=ss, speaker_mask=sm,
            duration_features=feats, has_speaker=ones,
            caption_state=cs, caption_mask=cm, has_caption=ones)

dur = Dur(m).eval()
R = 51  # v4 は speaker_patch_size=4 → 8秒アンカーで 200/4+1(mean) = 51 トークン（ref4_soft.json と一致させる）
ss1 = torch.randn(1, R, SD); sm1 = torch.ones(1, R, dtype=torch.bool)
feats = torch.rand(1, AUX)
with torch.no_grad():
    ref_lf = dur(ref_ts, tmk, ss1, sm1, ref_cs, cmk, feats)
print("duration log_frames sample:", ref_lf.tolist())
dur_path = f"{OUTDIR}/irodori4_duration.onnx"
torch.onnx.export(
    dur, (ref_ts, tmk, ss1, sm1, ref_cs, cmk, feats), dur_path,
    input_names=["text_state", "text_mask", "speaker_state", "speaker_mask",
                 "caption_state", "caption_mask", "duration_features"],
    output_names=["log_frames"],
    dynamic_axes={"text_state": {1: "Tt"}, "text_mask": {1: "Tt"},
                  "caption_state": {1: "Tc"}, "caption_mask": {1: "Tc"}},
    opset_version=17, dynamo=False)
print("[export] duration OK")

# ── 3) DiT（動的形状, CPU fallback 用, batch 1, delta_t 付き）── RoPE 実数化（interleaved・厳密等価）
import irodori_tts.model as M

def pre_real(dim, end, theta=10000.0):
    fr = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    ang = torch.outer(torch.arange(end, dtype=torch.float32), fr)
    return torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1)

def apply_real(x, freqs):
    cos = freqs[..., 0][None, :, None, :]
    sin = freqs[..., 1][None, :, None, :]
    x0 = x[..., 0::2]; x1 = x[..., 1::2]
    return torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1).reshape(x.shape).type_as(x)

class Dit(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m
    def forward(self, x_t, t, delta_t, ts, tm, ss, sm, cs, cm, lm):
        return self.m.forward_with_encoded_conditions(
            x_t, t, ts, tm, ss, sm, cs, cm, latent_mask=lm, delta_t=delta_t)

dit = Dit(m).eval()
torch.manual_seed(0)
x = torch.randn(1, 100, 32); t1 = torch.full((1,), 0.75); d1 = torch.full((1,), 0.25)
lm1 = torch.ones(1, 100, dtype=torch.bool)
with torch.no_grad():
    v_ref = dit(x, t1, d1, ref_ts, tmk, ss1, sm1, ref_cs, cmk, lm1)
M.precompute_freqs_cis = pre_real
M.apply_rotary_emb = apply_real
for mod in m.modules():
    if hasattr(mod, "_freqs_cis_cache"):
        mod._freqs_cis_cache = torch.zeros(1, 1, 2)
with torch.no_grad():
    dit(torch.randn(1, 375, 32), t1, d1, ref_ts, tmk, ss1, sm1, ref_cs, cmk, torch.ones(1, 375, dtype=torch.bool))  # RoPE 表を最大長で温める
    v_real = dit(x, t1, d1, ref_ts, tmk, ss1, sm1, ref_cs, cmk, lm1)
c = np.corrcoef(v_real.numpy().ravel(), v_ref.numpy().ravel())[0, 1]
print(f"[verify real-RoPE dit] corr={c:.6f}")
assert c > 0.99999
dit_path = f"{OUTDIR}/irodori4_dit.onnx"
torch.onnx.export(
    dit, (x, t1, d1, ref_ts, tmk, ss1, sm1, ref_cs, cmk, lm1), dit_path,
    input_names=["x_t", "t", "delta_t", "text_state", "text_mask", "speaker_state", "speaker_mask",
                 "caption_state", "caption_mask", "latent_mask"],
    output_names=["v"],
    dynamic_axes={"x_t": {1: "S"}, "text_state": {1: "Tt"}, "text_mask": {1: "Tt"},
                  "caption_state": {1: "Tc"}, "caption_mask": {1: "Tc"}, "latent_mask": {1: "S"}, "v": {1: "S"}},
    opset_version=17, dynamo=False)
print("[export] dit onnx OK")
for f in sorted(glob.glob(f"{OUTDIR}/*.onnx")):
    print(f, round(os.path.getsize(f) / 1e6), "MB")

# ── 4) E2E ゲート: ONNX 3点 + bake 済み ref4_soft.bin + 実コーデックで合成 → whisper ──
from transformers import AutoTokenizer
from irodori_tts.duration import build_duration_features
from irodori_tts.text_normalization import normalize_text
tok = AutoTokenizer.from_pretrained(SNAP + "tokenizer")
ref_bin = os.path.expanduser("~/Models/irodori-android-v3/ref4_soft.bin")
ref = np.fromfile(ref_bin, dtype=np.float32).reshape(1, 51, SD) if os.path.exists(ref_bin) else None
s_dur = ort.InferenceSession(dur_path, providers=["CPUExecutionProvider"])
s_dit = ort.InferenceSession(dit_path, providers=["CPUExecutionProvider"])
CAPTION = "明るく元気ではきはきした若い女性の声"
for text in ["まだ配車されていません。", "お疲れ様です。次の配達先はさくら倉庫です。"]:
    norm = normalize_text(text)
    tids = np.array([[1] + tok.encode(norm, add_special_tokens=False)], dtype=np.int64)
    cids = np.array([[1] + tok.encode(CAPTION, add_special_tokens=False)], dtype=np.int64)
    ts_, cs_ = s_enc.run(None, {"text_ids": tids, "text_mask": np.ones(tids.shape, bool), "caption_ids": cids, "caption_mask": np.ones(cids.shape, bool)})
    ss_ = ref if ref is not None else np.zeros((1, 51, SD), np.float32)
    sm_ = np.ones((1, 51), bool) if ref is not None else np.zeros((1, 51), bool)
    feats_ = build_duration_features([norm], token_counts=[tids.shape[1] - 1], max_text_len=256, has_speaker=[ref is not None]).numpy().astype(np.float32)
    feed = {"text_state": ts_, "text_mask": np.ones(tids.shape, bool), "speaker_state": ss_, "speaker_mask": sm_,
            "caption_state": cs_, "caption_mask": np.ones(cids.shape, bool), "duration_features": feats_}
    feed = {k: v for k, v in feed.items() if k in [i.name for i in s_dur.get_inputs()]}
    lf = s_dur.run(None, feed)[0][0]
    S_ = int(round(np.expm1(lf) * 0.95))
    rng = np.random.default_rng(0)
    xt = rng.standard_normal((1, S_, 32)).astype(np.float32)
    sched = np.linspace(1.0, 0.0, 5, dtype=np.float32)
    for i in range(4):
        t_, nx = sched[i], sched[i + 1]
        v = s_dit.run(None, {"x_t": xt, "t": np.array([t_], np.float32), "delta_t": np.array([t_ - nx], np.float32),
                             "text_state": ts_, "text_mask": np.ones(tids.shape, bool), "speaker_state": ss_, "speaker_mask": sm_,
                             "caption_state": cs_, "caption_mask": np.ones(cids.shape, bool), "latent_mask": np.ones((1, S_), bool)})[0]
        xt = xt + v * (nx - t_)
    with torch.no_grad():
        audio = rt.codec.decode_latent(torch.from_numpy(xt)).cpu()[0]
    wav = f"{OUTDIR}/e2e_{tids.shape[1]}.wav"
    from scipy.io import wavfile
    wavfile.write(wav, 48000, (audio.reshape(-1).clamp(-1, 1).numpy() * 32767).astype(np.int16))
    r = subprocess.run(["uvx", "--from", "mlx-whisper", "mlx_whisper", wav, "--model", "mlx-community/whisper-large-v3-turbo",
                        "--language", "ja", "--output-dir", OUTDIR, "--output-format", "txt"], capture_output=True, text=True)
    hyp = open(wav.replace(".wav", ".txt"), encoding="utf-8").read().strip() if os.path.exists(wav.replace(".wav", ".txt")) else r.stderr[-200:]
    print(f"[e2e] text={text} tokens={tids.shape[1]} S={S_} ({S_/25:.2f}s) ref={'anchor' if ref is not None else 'none'} → whisper: {hyp}")
