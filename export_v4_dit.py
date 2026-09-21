"""v4.1-Small-MF DiT の TPU バケット変換: batch-1（MeanFlow は CFG 不要）・delta_t 入力・latent_mask 付き静的形状 tflite。
入力順: x_t, t, delta_t, text_state, text_mask, speaker_state, speaker_mask, caption_state, caption_mask, latent_mask。
検証ゲート（v3 と同じ）: RoPE 手術等価 / pad 不変 / tflite CPU パリティ すべて corr>0.9999。
実行: BUCKET_S=96 <py311 venv>/bin/python export_v4_dit.py"""
import glob
import os

import numpy as np
import torch

from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey

torch.set_num_threads(6)
S_LAT = int(os.environ.get("BUCKET_S", "96"))
OUT = f"/tmp/irodori4_dit_S{S_LAT}.tflite"
ckpt = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--Aratako--Irodori-TTS-v4.1-Small-MF/snapshots/*/model.safetensors"))[0]
rt = InferenceRuntime.from_key(RuntimeKey(
    checkpoint=ckpt, model_device="cpu", codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim",
    model_precision="fp32", codec_device="cpu", codec_precision="fp32",
    codec_deterministic_encode=True, codec_deterministic_decode=True,
    compile_model=False, compile_dynamic=False))
m = rt.model.eval()
assert m.cfg.flow_parameterization == "meanflow"
PATCH = int(getattr(m.cfg, "latent_patch_size", 1))
S = S_LAT // PATCH
DIM = 32 * PATCH
print(f"latent_patch_size={PATCH} → model seq S={S}, dim={DIM}")

class Dit(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m
    def forward(self, x_t, t, dt, ts, tm, ss, sm, cs, cm, lm):
        return self.m.forward_with_encoded_conditions(
            x_t, t, ts, tm, ss, sm, cs, cm, latent_mask=lm, delta_t=dt)

dit = Dit(m).eval()
import irodori_tts.model as M

def pre_half(dim, end, theta=10000.0):
    fr = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    ang = torch.outer(torch.arange(end, dtype=torch.float32), fr)
    return torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1)

def apply_half(xx, freqs):
    cos = freqs[..., 0][None, :, None, :]
    sin = freqs[..., 1][None, :, None, :]
    half = xx.shape[-1] // 2
    x1 = xx[..., :half]; x2 = xx[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

def permute_attention(attn):
    H, Dh = attn.heads, attn.head_dim
    perm_within = torch.cat([torch.arange(0, Dh, 2), torch.arange(1, Dh, 2)])
    perm_full = torch.cat([h * Dh + perm_within for h in range(H)])
    with torch.no_grad():
        for lin in (attn.wq, attn.wk, attn.wk_text):
            lin.weight.copy_(lin.weight[perm_full])
        if getattr(attn, "has_speaker_condition", False):
            attn.wk_speaker.weight.copy_(attn.wk_speaker.weight[perm_full])
        if getattr(attn, "has_caption_condition", False):
            attn.wk_caption.weight.copy_(attn.wk_caption.weight[perm_full])
        for nrm in (attn.q_norm, attn.k_norm):
            w = nrm.weight
            if w.dim() == 2:
                nrm.weight.copy_(w[:, perm_within])
            else:
                nrm.weight.copy_(w.view(H, Dh)[:, perm_within].reshape(-1))

B, Tt, Tc, R = 1, 64, 64, 51  # v4 は speaker_patch_size=4 → 8秒アンカーで 200/4+1(mean) = 51 トークン
TD, SD = int(m.cfg.text_dim), int(m.cfg.speaker_dim)
torch.manual_seed(0)
T_REAL = S * 2 // 3
x = torch.randn(B, S, DIM)
t = torch.full((B,), 0.75); dt = torch.full((B,), 0.25)
ts = torch.randn(B, Tt, TD); cs = torch.randn(B, Tc, TD)
ss = torch.randn(B, R, SD)
tt_real, tc_real = 17, 9
tm = torch.zeros(B, Tt, dtype=torch.bool); tm[:, :tt_real] = True
sm = torch.ones(B, R, dtype=torch.bool)
cm = torch.zeros(B, Tc, dtype=torch.bool); cm[:, :tc_real] = True
lm_pad = torch.zeros(B, S, dtype=torch.bool); lm_pad[:, :T_REAL] = True

with torch.no_grad():
    v_ref = dit(x, t, dt, ts, tm, ss, sm, cs, cm, lm_pad)
M.precompute_freqs_cis = pre_half
M.apply_rotary_emb = apply_half
for mod in m.modules():
    if hasattr(mod, "_freqs_cis_cache"):
        mod._freqs_cis_cache = torch.zeros(1, 1, 2)
for blk in m.blocks:
    permute_attention(blk.attention)
with torch.no_grad():
    v_perm = dit(x, t, dt, ts, tm, ss, sm, cs, cm, lm_pad)
perm_corr = np.corrcoef(v_perm.numpy().ravel(), v_ref.numpy().ravel())[0, 1]
print(f"[verify perm] corr={perm_corr:.6f} max|diff|={(v_perm - v_ref).abs().max().item():.2e}")
assert perm_corr > 0.9999, "RoPE surgery broke the model!"
with torch.no_grad():
    v_exact = dit(x[:, :T_REAL].contiguous(), t, dt, ts, tm, ss, sm, cs, cm, torch.ones(B, T_REAL, dtype=torch.bool))
    v_pad = dit(x, t, dt, ts, tm, ss, sm, cs, cm, lm_pad)
pi_corr = np.corrcoef(v_pad[:, :T_REAL].numpy().ravel(), v_exact.numpy().ravel())[0, 1]
print(f"[verify pad-invariance] T={T_REAL}→S={S}: corr={pi_corr:.6f} max|diff|={(v_pad[:, :T_REAL] - v_exact).abs().max().item():.2e}")
assert pi_corr > 0.9999, "pad contaminates real positions!"
with torch.no_grad():
    dit(x, t, dt, ts, tm, ss, sm, cs, cm, lm_pad)
import litert_torch as lt
edge = lt.convert(dit, (x, t, dt, ts, tm, ss, sm, cs, cm, lm_pad))
out = edge(x, t, dt, ts, tm, ss, sm, cs, cm, lm_pad)
out = out[0] if isinstance(out, (list, tuple)) else out
tcorr = np.corrcoef(np.asarray(out).ravel(), v_pad.numpy().ravel())[0, 1]
print(f"[parity tflite-cpu] corr={tcorr:.6f}")
assert tcorr > 0.9999
edge.export(OUT)
print(f"[export] {OUT} {os.path.getsize(OUT) / 1e9:.2f} GB")
