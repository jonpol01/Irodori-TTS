"""DiT の RoPE を interleaved → half-split へ重み並べ替えで変換（数学的に等価）。
R 行列 matmul（=BROADCAST_TO の元）を排除し、グラフを 100% GPU 化する。"""
import glob, os
import numpy as np, torch
from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey

torch.set_num_threads(6)
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
    def forward(self, x_t, t, ts, tm, cs, cm):
        return self.m.forward_with_encoded_conditions(x_t, t, ts, tm, None, None, cs, cm)

dit = Dit(m).eval()
B, S, Tt, Tc, D = 3, 150, 64, 64, 512
torch.manual_seed(0)
x = torch.randn(B, S, 32); t = torch.tensor([0.7, 0.7, 0.7])
ts = torch.randn(B, Tt, D); tm = torch.ones(B, Tt, dtype=torch.bool)
cs = torch.randn(B, Tc, D); cm = torch.ones(B, Tc, dtype=torch.bool)
ts[1] = 0; tm[1] = False; cs[2] = 0; cm[2] = False

# 参照（元の複素 RoPE 実装のまま）
with torch.no_grad():
    v_ref = dit(x, t, ts, tm, cs, cm)

# ── half-split RoPE（4D演算のみ・R行列なし）──
import irodori_tts.model as M
def pre_half(dim, end, theta=10000.0):
    fr = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    ang = torch.outer(torch.arange(end, dtype=torch.float32), fr)  # (end, dim/2)
    return torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1)   # (end, dim/2, 2)

def apply_half(xx, freqs):
    cos = freqs[..., 0][None, :, None, :]  # (1,S,1,D/2)
    sin = freqs[..., 1][None, :, None, :]
    half = xx.shape[-1] // 2
    x1 = xx[..., :half]; x2 = xx[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

M.precompute_freqs_cis = pre_half
M.apply_rotary_emb = apply_half
for mod in m.modules():
    if hasattr(mod, "_freqs_cis_cache"):
        mod._freqs_cis_cache = torch.zeros(1, 1, 2)

# ── 重み並べ替え: interleaved(q0,q1 が隣接ペア) → half-split(前半=偶数, 後半=奇数) ──
# q'=P q, k'=P k で (Pq)·(Pk)=q·k（注意スコア不変）。RoPE の回転対は
# interleaved の (2i, 2i+1) が half-split の (i, D/2+i) に一致する。
def permute_attention(attn):
    H, Dh = attn.heads, attn.head_dim
    perm_within = torch.cat([torch.arange(0, Dh, 2), torch.arange(1, Dh, 2)])  # [0,2,..,1,3,..]
    perm_full = torch.cat([h * Dh + perm_within for h in range(H)])
    with torch.no_grad():
        for lin in (attn.wq, attn.wk, attn.wk_text):
            lin.weight.copy_(lin.weight[perm_full])
        if getattr(attn, "has_caption_condition", False):
            attn.wk_caption.weight.copy_(attn.wk_caption.weight[perm_full])
        # q_norm/k_norm の重み (H, Dh) は head 内次元を同順で並べ替え
        for nrm in (attn.q_norm, attn.k_norm):
            w = nrm.weight
            if w.dim() == 2:
                nrm.weight.copy_(w[:, perm_within])
            else:  # (H*Dh,) フラットの場合
                nrm.weight.copy_(w.view(H, Dh)[:, perm_within].reshape(-1))

count = 0
for blk in m.blocks:
    permute_attention(blk.attention)
    count += 1
print(f"[perm] permuted {count} blocks' q/k spaces to half-split layout")

with torch.no_grad():
    v_perm = dit(x, t, ts, tm, cs, cm)
diff = (v_perm - v_ref).abs().max().item()
corr = np.corrcoef(v_perm.numpy().ravel(), v_ref.numpy().ravel())[0, 1]
print(f"[verify eager] permuted-vs-original: corr={corr:.6f} max|diff|={diff:.2e}")
assert corr > 0.9999, "permutation broke the model!"

with torch.no_grad():
    dit(x, t, ts, tm, cs, cm)  # キャッシュ温め（表を定数化）

import litert_torch as lt
edge = lt.convert(dit, (x, t, ts, tm, cs, cm))
out = edge(x, t, ts, tm, cs, cm)
out = out[0] if isinstance(out, (list, tuple)) else out
tcorr = np.corrcoef(np.asarray(out).ravel(), v_ref.numpy().ravel())[0, 1]
print(f"[parity tflite-cpu] corr={tcorr:.6f}")
edge.export("/tmp/irodori_dit_perm.tflite")
print(f"[export] /tmp/irodori_dit_perm.tflite {os.path.getsize('/tmp/irodori_dit_perm.tflite')/1e9:.2f} GB")
