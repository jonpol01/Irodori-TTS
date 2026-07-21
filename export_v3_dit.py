"""v3 DiT の TPU バケット変換: batch-4 CFG・latent_mask 付き静的形状 tflite。
v3 は half-RoPE ネイティブなので v2 で必要だった重み並べ替え手術は不要。
行構成は independent CFG の [cond, text-uncond, speaker-uncond, caption-uncond]。
BUCKET_S は潜在フレーム数 (150/225)。latent_patch_size>1 の場合はモデル入力側で
S_model = BUCKET_S/patch, dim = 32*patch に変換される。
実行: BUCKET_S=150 .venv/bin/python export_v3_dit.py"""
import glob
import os

import numpy as np
import torch

from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey

torch.set_num_threads(6)
S_LAT = int(os.environ.get("BUCKET_S", "150"))
OUT = f"/tmp/irodori3_dit_S{S_LAT}.tflite"

ckpt = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--Aratako--Irodori-TTS-600M-v3-VoiceDesign/snapshots/*/model.safetensors"))[0]
rt = InferenceRuntime.from_key(RuntimeKey(
    checkpoint=ckpt, model_device="cpu", codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim",
    model_precision="fp32", codec_device="cpu", codec_precision="fp32",
    codec_deterministic_encode=True, codec_deterministic_decode=True,
    compile_model=False, compile_dynamic=False))
m = rt.model.eval()
PATCH = int(getattr(m.cfg, "latent_patch_size", 1))
S = S_LAT // PATCH
DIM = 32 * PATCH
print(f"latent_patch_size={PATCH} → model seq S={S}, dim={DIM}")

class Dit(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m
    def forward(self, x_t, t, ts, tm, ss, sm, cs, cm, lm):
        return self.m.forward_with_encoded_conditions(
            x_t, t, ts, tm, ss, sm, cs, cm, latent_mask=lm)

dit = Dit(m).eval()

# ── v2 で実証済みの RoPE 手術（v3 も同一実装なのでそのまま適用）──
# 複素 RoPE は torch.export 不可 → 実数 half-split に置換し、q/k 系重みを per-head 並べ替えで
# 数学的等価に保つ（(Pq)·(Pk)=q·k）。v3 は speaker 枝の wk_speaker も対象に加える。
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

B, Tt, Tc, R = 4, 64, 64, 201
TD, SD = int(m.cfg.text_dim), int(m.cfg.speaker_dim)
torch.manual_seed(0)
T_REAL = S * 2 // 3
x = torch.randn(B, S, DIM)
t = torch.full((B,), 0.7)
ts = torch.randn(B, Tt, TD); cs = torch.randn(B, Tc, TD)
ss = torch.randn(B, R, SD)
# 行構成: [cond, text-unc, speaker-unc, caption-unc] — uncond 行は状態ゼロ + マスク False
tt_real, tc_real = 17, 9
ts[1] = 0; ss[2] = 0; cs[3] = 0
tm = torch.zeros(B, Tt, dtype=torch.bool); tm[:, :tt_real] = True; tm[1] = False
sm = torch.ones(B, R, dtype=torch.bool); sm[2] = False
cm = torch.zeros(B, Tc, dtype=torch.bool); cm[:, :tc_real] = True; cm[3] = False
lm_pad = torch.zeros(B, S, dtype=torch.bool); lm_pad[:, :T_REAL] = True

# 手術前の参照出力（複素 RoPE のまま）
with torch.no_grad():
    v_ref = dit(x, t, ts, tm, ss, sm, cs, cm, lm_pad)

# RoPE を実数 half-split に置換 + 重み並べ替え → 等価性検証
M.precompute_freqs_cis = pre_half
M.apply_rotary_emb = apply_half
for mod in m.modules():
    if hasattr(mod, "_freqs_cis_cache"):
        mod._freqs_cis_cache = torch.zeros(1, 1, 2)
for blk in m.blocks:
    permute_attention(blk.attention)
with torch.no_grad():
    v_perm = dit(x, t, ts, tm, ss, sm, cs, cm, lm_pad)
perm_corr = np.corrcoef(v_perm.numpy().ravel(), v_ref.numpy().ravel())[0, 1]
print(f"[verify perm] corr={perm_corr:.6f} max|diff|={(v_perm - v_ref).abs().max().item():.2e}")
assert perm_corr > 0.9999, "RoPE surgery broke the model!"

# パッド不変性: 実長 exact 実行 vs バケット+マスク実行
with torch.no_grad():
    v_exact = dit(x[:, :T_REAL].contiguous(), t, ts, tm, ss, sm, cs, cm,
                  torch.ones(B, T_REAL, dtype=torch.bool))
    v_pad = dit(x, t, ts, tm, ss, sm, cs, cm, lm_pad)
pi_corr = np.corrcoef(v_pad[:, :T_REAL].numpy().ravel(), v_exact.numpy().ravel())[0, 1]
print(f"[verify pad-invariance] T={T_REAL}→S={S}: corr={pi_corr:.6f} "
      f"max|diff|={(v_pad[:, :T_REAL] - v_exact).abs().max().item():.2e}")
assert pi_corr > 0.9999, "pad contaminates real positions!"

with torch.no_grad():
    dit(x, t, ts, tm, ss, sm, cs, cm, lm_pad)  # キャッシュ温め（RoPE 表等を定数化）

import litert_torch as lt
edge = lt.convert(dit, (x, t, ts, tm, ss, sm, cs, cm, lm_pad))
out = edge(x, t, ts, tm, ss, sm, cs, cm, lm_pad)
out = out[0] if isinstance(out, (list, tuple)) else out
tcorr = np.corrcoef(np.asarray(out).ravel(), v_pad.numpy().ravel())[0, 1]
print(f"[parity tflite-cpu] corr={tcorr:.6f}")
edge.export(OUT)
print(f"[export] {OUT} {os.path.getsize(OUT) / 1e9:.2f} GB")
