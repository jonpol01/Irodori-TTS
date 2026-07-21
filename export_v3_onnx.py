"""v3 の ONNX エクスポート（端末 CPU 経路用）: encoder / duration predictor / 動的形状 DiT。
DiT には export_v3_dit.py と同じ RoPE 手術（実数化 + 重み並べ替え）を適用してから出力する。
ref_state はアプリ側で bake 済み .bin を与えるため、エンコーダは text/caption のみ。
実行: .venv/bin/python export_v3_onnx.py"""
import glob
import os

import numpy as np
import torch

from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey

torch.set_num_threads(6)
OUTDIR = "/tmp/v3_onnx"
os.makedirs(OUTDIR, exist_ok=True)

ckpt = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--Aratako--Irodori-TTS-600M-v3-VoiceDesign/snapshots/*/model.safetensors"))[0]
rt = InferenceRuntime.from_key(RuntimeKey(
    checkpoint=ckpt, model_device="cpu", codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim",
    model_precision="fp32", codec_device="cpu", codec_precision="fp32",
    codec_deterministic_encode=True, codec_deterministic_decode=True,
    compile_model=False, compile_dynamic=False))
m = rt.model.eval()
AUX = int(m.cfg.duration_aux_dim)
TD, SD = int(m.cfg.text_dim), int(m.cfg.speaker_dim)
print(f"duration_aux_dim={AUX} text_dim={TD} speaker_dim={SD}")

# ── RoPE 実数化（interleaved・厳密等価・重み変更なし）— ONNX は複素 dtype 非対応。
# エンコーダ（SelfAttention）も RoPE を使うため、全エクスポートの前にグローバル置換する。
# （half-split+並べ替えは GPU デレゲート都合の tflite 専用。ONNX は隣接ペアのままで良い）
import irodori_tts.model as M

_orig_pre = M.precompute_freqs_cis

def pre_real(dim, end, theta=10000.0):
    fr = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    ang = torch.outer(torch.arange(end, dtype=torch.float32), fr)
    return torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1)  # (end, dim/2, 2)

def apply_real(x, freqs):
    cos = freqs[..., 0][None, :, None, :]
    sin = freqs[..., 1][None, :, None, :]
    x0 = x[..., 0::2]; x1 = x[..., 1::2]
    o0 = x0 * cos - x1 * sin
    o1 = x0 * sin + x1 * cos
    return torch.stack([o0, o1], dim=-1).reshape(x.shape).type_as(x)

# 実数化前の参照出力を取っておく（等価性検証用）
class _EncRef(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m
    def forward(self, ti, tmk, ci, cmk):
        return (self.m.text_norm(self.m.text_encoder(ti, tmk)),
                self.m.caption_norm(self.m.caption_encoder(ci, cmk)))

_ti = torch.randint(2, 1000, (1, 17)); _tm = torch.ones(1, 17, dtype=torch.bool)
_ci = torch.randint(2, 1000, (1, 9)); _cm = torch.ones(1, 9, dtype=torch.bool)
with torch.no_grad():
    _enc_ref = _EncRef(m)(_ti, _tm, _ci, _cm)

M.precompute_freqs_cis = pre_real
M.apply_rotary_emb = apply_real
for mod in m.modules():
    if hasattr(mod, "_freqs_cis_cache"):
        mod._freqs_cis_cache = torch.zeros(1, 1, 2)
# ⚠️ RoPE 表はトレース時の系列長で定数化され、実行時にその長さまで Slice される。
# 短い入力でトレースすると長い実行時入力が broadcast エラーで壊れる（実機で caption 10 トークン > トレース 9 で発症）。
# → 最大長でキャッシュを温めてから短いサンプルでトレースする（表=最大長の定数、Slice が実行時長を処理）。
with torch.no_grad():
    _long_ids = torch.randint(2, 1000, (1, 256))
    _long_mask = torch.ones(1, 256, dtype=torch.bool)
    _EncRef(m)(_long_ids, _long_mask, _long_ids, _long_mask)
with torch.no_grad():
    _enc_new = _EncRef(m)(_ti, _tm, _ci, _cm)
_c = np.corrcoef(_enc_new[0].numpy().ravel(), _enc_ref[0].numpy().ravel())[0, 1]
print(f"[verify real-RoPE encoder] corr={_c:.6f}")
assert _c > 0.99999, "real-RoPE substitution broke the encoder!"

# ── 1) encoder（text/caption）──
class Enc(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m
    def forward(self, text_ids, text_mask, caption_ids, caption_mask):
        ts = self.m.text_norm(self.m.text_encoder(text_ids, text_mask))
        cs = self.m.caption_norm(self.m.caption_encoder(caption_ids, caption_mask))
        return ts, cs

enc = Enc(m).eval()
ti = torch.randint(2, 1000, (1, 17)); tmk = torch.ones(1, 17, dtype=torch.bool)
ci = torch.randint(2, 1000, (1, 9)); cmk = torch.ones(1, 9, dtype=torch.bool)
with torch.no_grad():
    ref_ts, ref_cs = enc(ti, tmk, ci, cmk)
torch.onnx.export(
    enc, (ti, tmk, ci, cmk), f"{OUTDIR}/irodori3_encoder.onnx",
    input_names=["text_ids", "text_mask", "caption_ids", "caption_mask"],
    output_names=["text_state", "caption_state"],
    dynamic_axes={"text_ids": {1: "Tt"}, "text_mask": {1: "Tt"},
                  "caption_ids": {1: "Tc"}, "caption_mask": {1: "Tc"},
                  "text_state": {1: "Tt"}, "caption_state": {1: "Tc"}},
    opset_version=17, dynamo=False)
print("[export] encoder OK")

# ── 2) duration predictor ──
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
ss1 = torch.randn(1, 201, SD); sm1 = torch.ones(1, 201, dtype=torch.bool)
feats = torch.rand(1, AUX)
with torch.no_grad():
    ref_lf = dur(ref_ts, tmk, ss1, sm1, ref_cs, cmk, feats)
print("duration log_frames sample:", ref_lf.tolist())
torch.onnx.export(
    dur, (ref_ts, tmk, ss1, sm1, ref_cs, cmk, feats), f"{OUTDIR}/irodori3_duration.onnx",
    input_names=["text_state", "text_mask", "speaker_state", "speaker_mask",
                 "caption_state", "caption_mask", "duration_features"],
    output_names=["log_frames"],
    dynamic_axes={"text_state": {1: "Tt"}, "text_mask": {1: "Tt"},
                  "caption_state": {1: "Tc"}, "caption_mask": {1: "Tc"}},
    opset_version=17, dynamo=False)
print("[export] duration OK")

# ── 3) DiT（動的形状, CPU fallback 用）— 実数化は冒頭で適用済み（重み無変更）──
class Dit(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m
    def forward(self, x_t, t, ts, tm, ss, sm, cs, cm, lm):
        return self.m.forward_with_encoded_conditions(
            x_t, t, ts, tm, ss, sm, cs, cm, latent_mask=lm)

dit = Dit(m).eval()
B, S = 4, 100
torch.manual_seed(0)
x = torch.randn(B, S, 32); t4 = torch.full((B,), 0.7)
ts4 = torch.randn(B, 17, TD); tm4 = torch.ones(B, 17, dtype=torch.bool)
ss4 = torch.randn(B, 201, SD); sm4 = torch.ones(B, 201, dtype=torch.bool)
cs4 = torch.randn(B, 9, TD); cm4 = torch.ones(B, 9, dtype=torch.bool)
lm4 = torch.ones(B, S, dtype=torch.bool)
# DiT の RoPE 表も最大潜在長（15秒=375）で温めてからトレースする（上の encoder と同じ罠対策）
with torch.no_grad():
    dit(torch.randn(1, 375, 32), torch.full((1,), 0.7),
        torch.randn(1, 17, TD), torch.ones(1, 17, dtype=torch.bool),
        torch.randn(1, 201, SD), torch.ones(1, 201, dtype=torch.bool),
        torch.randn(1, 9, TD), torch.ones(1, 9, dtype=torch.bool),
        torch.ones(1, 375, dtype=torch.bool))

torch.onnx.export(
    dit, (x, t4, ts4, tm4, ss4, sm4, cs4, cm4, lm4), f"{OUTDIR}/irodori3_dit.onnx",
    input_names=["x_t", "t", "text_state", "text_mask", "speaker_state", "speaker_mask",
                 "caption_state", "caption_mask", "latent_mask"],
    output_names=["v"],
    dynamic_axes={"x_t": {0: "B", 1: "S"}, "t": {0: "B"},
                  "text_state": {0: "B", 1: "Tt"}, "text_mask": {0: "B", 1: "Tt"},
                  "speaker_state": {0: "B"}, "speaker_mask": {0: "B"},
                  "caption_state": {0: "B", 1: "Tc"}, "caption_mask": {0: "B", 1: "Tc"},
                  "latent_mask": {0: "B", 1: "S"}, "v": {0: "B", 1: "S"}},
    opset_version=17, dynamo=False)
print("[export] dit onnx OK")
for f in sorted(glob.glob(f"{OUTDIR}/*.onnx")):
    print(f, os.path.getsize(f) / 1e6, "MB")
