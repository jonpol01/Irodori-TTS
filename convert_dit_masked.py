"""TPU バケット用 DiT 変換: perm(half-split RoPE) + latent_mask 入力付きで静的 S へ。
latent_mask はモデル既存の self_mask 経路（各ブロックの JointAttention で pad キーを除外）
なのでパッド位置が実位置を汚染しない。S は環境変数 BUCKET_S (150/225)。
実行: BUCKET_S=150 .venv/bin/python convert_dit_masked.py"""
import glob, os
import numpy as np, torch
from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey

torch.set_num_threads(6)
S = int(os.environ.get("BUCKET_S", "150"))
T_REAL = int(os.environ.get("T_REAL", str(int(S * 2 / 3))))  # パッド不変性チェック用の実長
OUT = f"/tmp/irodori_dit_g5_S{S}.tflite"

ckpt = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--Aratako--Irodori-TTS-500M-v2-VoiceDesign/snapshots/*/model.safetensors"))[0]
rt = InferenceRuntime.from_key(RuntimeKey(
    checkpoint=ckpt, model_device="cpu", codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim",
    model_precision="fp32", codec_device="cpu", codec_precision="fp32",
    codec_deterministic_encode=True, codec_deterministic_decode=True,
    compile_model=False, compile_dynamic=False))
m = rt.model.eval()

class Dit(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m
    def forward(self, x_t, t, ts, tm, cs, cm, lm):
        return self.m.forward_with_encoded_conditions(
            x_t, t, ts, tm, None, None, cs, cm, latent_mask=lm)

dit = Dit(m).eval()
B, Tt, Tc, D = 3, 64, 64, 512
torch.manual_seed(0)
x = torch.randn(B, S, 32); t = torch.tensor([0.7, 0.7, 0.7])
ts = torch.randn(B, Tt, D); tm = torch.ones(B, Tt, dtype=torch.bool)
cs = torch.randn(B, Tc, D); cm = torch.ones(B, Tc, dtype=torch.bool)
ts[1] = 0; tm[1] = False; cs[2] = 0; cm[2] = False
lm_full = torch.ones(B, S, dtype=torch.bool)

# 参照（元の複素 RoPE・全 true マスク）
with torch.no_grad():
    v_ref = dit(x, t, ts, tm, cs, cm, lm_full)

# ── half-split RoPE パッチ（convert_dit_perm.py と同一）──
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

M.precompute_freqs_cis = pre_half
M.apply_rotary_emb = apply_half
for mod in m.modules():
    if hasattr(mod, "_freqs_cis_cache"):
        mod._freqs_cis_cache = torch.zeros(1, 1, 2)

def permute_attention(attn):
    H, Dh = attn.heads, attn.head_dim
    perm_within = torch.cat([torch.arange(0, Dh, 2), torch.arange(1, Dh, 2)])
    perm_full = torch.cat([h * Dh + perm_within for h in range(H)])
    with torch.no_grad():
        for lin in (attn.wq, attn.wk, attn.wk_text):
            lin.weight.copy_(lin.weight[perm_full])
        if getattr(attn, "has_caption_condition", False):
            attn.wk_caption.weight.copy_(attn.wk_caption.weight[perm_full])
        for nrm in (attn.q_norm, attn.k_norm):
            w = nrm.weight
            if w.dim() == 2:
                nrm.weight.copy_(w[:, perm_within])
            else:
                nrm.weight.copy_(w.view(H, Dh)[:, perm_within].reshape(-1))

for blk in m.blocks:
    permute_attention(blk.attention)

with torch.no_grad():
    v_perm = dit(x, t, ts, tm, cs, cm, lm_full)
corr = np.corrcoef(v_perm.numpy().ravel(), v_ref.numpy().ravel())[0, 1]
print(f"[verify perm] corr={corr:.6f} max|diff|={(v_perm - v_ref).abs().max().item():.2e}")
assert corr > 0.9999, "permutation broke the model!"

# ── パッド不変性: 実長 T_REAL を exact 実行 vs S へパッド+マスク実行 ──
x_real = x[:, :T_REAL].contiguous()
lm_real = torch.ones(B, T_REAL, dtype=torch.bool)
x_pad = torch.cat([x_real, torch.randn(B, S - T_REAL, 32)], dim=1)  # パッド値は何でも良い（キー除外）
lm_pad = torch.cat([lm_real, torch.zeros(B, S - T_REAL, dtype=torch.bool)], dim=1)
with torch.no_grad():
    v_exact = dit(x_real, t, ts, tm, cs, cm, lm_real)
    v_padded = dit(x_pad, t, ts, tm, cs, cm, lm_pad)
pi_corr = np.corrcoef(v_padded[:, :T_REAL].numpy().ravel(), v_exact.numpy().ravel())[0, 1]
pi_diff = (v_padded[:, :T_REAL] - v_exact).abs().max().item()
print(f"[verify pad-invariance] T={T_REAL}→S={S}: corr={pi_corr:.6f} max|diff|={pi_diff:.2e}")
assert pi_corr > 0.9999, "pad contaminates real positions — mask path broken!"

with torch.no_grad():
    dit(x_pad, t, ts, tm, cs, cm, lm_pad)  # キャッシュ温め（S サイズの表を定数化）

import litert_torch as lt
edge = lt.convert(dit, (x_pad, t, ts, tm, cs, cm, lm_pad))
out = edge(x_pad, t, ts, tm, cs, cm, lm_pad)
out = out[0] if isinstance(out, (list, tuple)) else out
tcorr = np.corrcoef(np.asarray(out).ravel(), v_padded.numpy().ravel())[0, 1]
print(f"[parity tflite-cpu] corr={tcorr:.6f}")
edge.export(OUT)
print(f"[export] {OUT} {os.path.getsize(OUT)/1e9:.2f} GB")
