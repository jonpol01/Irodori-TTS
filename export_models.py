"""Encoder(text+caption) と DiT-step を ONNX 化し、PyTorch と parity 検証（Android 移植 Phase A）。"""
import glob
import os
import numpy as np
import torch
import torch.nn as nn
from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest

torch.set_num_threads(4)
ckpt = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--Aratako--Irodori-TTS-500M-v2-VoiceDesign/snapshots/*/model.safetensors"))[0]
rt = InferenceRuntime.from_key(RuntimeKey(
    checkpoint=ckpt, model_device="cpu", codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim",
    model_precision="fp32", codec_device="cpu", codec_precision="fp32",
    codec_deterministic_encode=True, codec_deterministic_decode=True,
    compile_model=False, compile_dynamic=False))
m = rt.model.eval()
tok = rt.tokenizer
ctok = rt.caption_tokenizer or rt.tokenizer

tids, tmask = tok.batch_encode(["了解しました。到着したらお知らせください。"], 64)
cids, cmask = ctok.batch_encode(["柔らかく温かみのある若い女性の声"], 64)
print(f"[shapes] text_ids={tuple(tids.shape)} caption_ids={tuple(cids.shape)}")


class Enc(nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, text_ids, text_mask, caption_ids, caption_mask):
        ts, tm, ss, sm, cs, cm = self.m.encode_conditions(text_ids, text_mask, None, None, caption_ids, caption_mask)
        return ts, cs


class Dit(nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, x_t, t, text_state, text_mask, caption_state, caption_mask):
        return self.m.forward_with_encoded_conditions(x_t, t, text_state, text_mask, None, None, caption_state, caption_mask)


enc = Enc(m).eval()
dit = Dit(m).eval()

with torch.no_grad():
    ts_ref, cs_ref = enc(tids, tmask, cids, cmask)
print(f"[enc] text_state={tuple(ts_ref.shape)} caption_state={tuple(cs_ref.shape)}")
pd = m.cfg.latent_dim * m.cfg.latent_patch_size
S = 90
xt = torch.randn(1, S, pd); t = torch.tensor([0.5])
with torch.no_grad():
    v_ref = dit(xt, t, ts_ref, tmask, cs_ref, cmask)
print(f"[dit] x_t={tuple(xt.shape)} -> v={tuple(v_ref.shape)}")

# --- RoPE を完全実数化（ONNX は複素数非対応）。freqs_cis を複素→実(端,dim/2,2)[cos,sin] に置換。
import irodori_tts.model as M  # noqa: E402
_orig_pre = M.precompute_freqs_cis
_orig_apply = M.apply_rotary_emb


def precompute_freqs_real(dim, end, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    ang = torch.outer(torch.arange(end, dtype=torch.float32), freqs)  # (end, dim/2)
    return torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1)  # (end, dim/2, 2)


def apply_rotary_emb_real(x, freqs):
    cos = freqs[..., 0][None, :, None, :]  # (1,S,1,dim/2)
    sin = freqs[..., 1][None, :, None, :]
    xf = x.float()
    x0 = xf[..., 0::2]
    x1 = xf[..., 1::2]
    o0 = x0 * cos - x1 * sin
    o1 = x0 * sin + x1 * cos
    out = torch.stack([o0, o1], dim=-1).reshape_as(x)
    return out.type_as(x)


# 元の複素実装と数値一致を確認
with torch.no_grad():
    _x = torch.randn(1, 40, 8, 32)
    _out_c = _orig_apply(_x, _orig_pre(32, 40))
    _out_r = apply_rotary_emb_real(_x, precompute_freqs_real(32, 40))
    print(f"[rope] complex-vs-real max|diff|={(_out_c - _out_r).abs().max().item():.2e}")
M.precompute_freqs_cis = precompute_freqs_real
M.apply_rotary_emb = apply_rotary_emb_real
# キャッシュ済みの複素 freqs をクリアして実数で再計算させる
for _mod in m.modules():
    if hasattr(_mod, "_freqs_cis_cache"):
        _mod._freqs_cis_cache = torch.zeros(1, 1, 2)

# ---- export encoder ----
torch.onnx.export(
    enc, (tids, tmask, cids, cmask), "/tmp/irodori_encoder.onnx",
    input_names=["text_ids", "text_mask", "caption_ids", "caption_mask"],
    output_names=["text_state", "caption_state"],
    dynamic_axes={"text_ids": {1: "Tt"}, "text_mask": {1: "Tt"}, "caption_ids": {1: "Tc"}, "caption_mask": {1: "Tc"},
                  "text_state": {1: "Tt"}, "caption_state": {1: "Tc"}},
    opset_version=17, dynamo=False)
print("[onnx] encoder exported")

# ---- export dit step ----
torch.onnx.export(
    dit, (xt, t, ts_ref, tmask, cs_ref, cmask), "/tmp/irodori_dit.onnx",
    input_names=["x_t", "t", "text_state", "text_mask", "caption_state", "caption_mask"],
    output_names=["v"],
    dynamic_axes={"x_t": {1: "S"}, "text_state": {1: "Tt"}, "text_mask": {1: "Tt"},
                  "caption_state": {1: "Tc"}, "caption_mask": {1: "Tc"}, "v": {1: "S"}},
    opset_version=17, dynamo=False)
print("[onnx] dit exported")

# ---- parity ----
import onnxruntime as ort  # noqa
so = ort.SessionOptions(); so.intra_op_num_threads = 4
es = ort.InferenceSession("/tmp/irodori_encoder.onnx", so, providers=["CPUExecutionProvider"])
ds = ort.InferenceSession("/tmp/irodori_dit.onnx", so, providers=["CPUExecutionProvider"])

eo = es.run(None, {"text_ids": tids.numpy(), "text_mask": tmask.numpy(), "caption_ids": cids.numpy(), "caption_mask": cmask.numpy()})
print(f"[parity enc] text_state diff={np.abs(eo[0]-ts_ref.numpy()).max():.2e}  caption_state diff={np.abs(eo[1]-cs_ref.numpy()).max():.2e}")
do = ds.run(None, {"x_t": xt.numpy(), "t": t.numpy(), "text_state": ts_ref.numpy(), "text_mask": tmask.numpy(), "caption_state": cs_ref.numpy(), "caption_mask": cmask.numpy()})
print(f"[parity dit] v diff={np.abs(do[0]-v_ref.numpy()).max():.2e}")
