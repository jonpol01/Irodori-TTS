"""DiT を先頭 N ブロックに切り詰めて .tflite 化（fp16 破綻レイヤの二分探索用）。N は環境変数。"""
import glob, os, sys
import torch
from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey

N = int(os.environ["TRUNC_N"])
torch.set_num_threads(6)
ckpt = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--Aratako--Irodori-TTS-500M-v2-VoiceDesign/snapshots/*/model.safetensors"))[0]
rt = InferenceRuntime.from_key(RuntimeKey(
    checkpoint=ckpt, model_device="cpu", codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim",
    model_precision="fp32", codec_device="cpu", codec_precision="fp32",
    codec_deterministic_encode=True, codec_deterministic_decode=True,
    compile_model=False, compile_dynamic=False))
m = rt.model.eval()

import irodori_tts.model as M
def pre_gpu(dim, end, theta=10000.0):
    fr = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    ang = torch.outer(torch.arange(end, dtype=torch.float32), fr)
    cos = torch.repeat_interleave(torch.cos(ang), 2, dim=-1)
    sin = torch.repeat_interleave(torch.sin(ang), 2, dim=-1)
    return torch.stack([cos, sin], dim=-1)
_R_cache = {}
def _R(dim, dtype):
    key = (dim, dtype)
    if key not in _R_cache:
        R = torch.zeros(dim, dim, dtype=dtype)
        for i in range(0, dim, 2):
            R[i + 1, i] = -1.0; R[i, i + 1] = 1.0
        _R_cache[key] = R
    return _R_cache[key]
def apply_gpu(x, freqs):
    cos = freqs[..., 0][None, :, None, :]; sin = freqs[..., 1][None, :, None, :]
    xr = torch.nn.functional.linear(x, _R(x.shape[-1], x.dtype).t())
    return x * cos + xr * sin
M.precompute_freqs_cis = pre_gpu
M.apply_rotary_emb = apply_gpu
for mod in m.modules():
    if hasattr(mod, "_freqs_cis_cache"):
        mod._freqs_cis_cache = torch.zeros(1, 1, 2)

# ブロック切り詰め
full = len(m.blocks)
m.blocks = m.blocks[:N]
print(f"[trunc] blocks {full} -> {N}")

class Dit(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m
    def forward(self, x_t, t, ts, tm, cs, cm):
        return self.m.forward_with_encoded_conditions(x_t, t, ts, tm, None, None, cs, cm)

dit = Dit(m).eval()
B, S, Tt, Tc, D = 3, 150, 64, 64, 512
x = torch.randn(B, S, 32); t = torch.tensor([0.7, 0.7, 0.7])
ts = torch.randn(B, Tt, D); tm = torch.ones(B, Tt, dtype=torch.bool)
cs = torch.randn(B, Tc, D); cm = torch.ones(B, Tc, dtype=torch.bool)
ts[1] = 0; tm[1] = False; cs[2] = 0; cm[2] = False
with torch.no_grad():
    dit(x, t, ts, tm, cs, cm)  # キャッシュを実テーブルで温める

import litert_torch as lt
edge = lt.convert(dit, (x, t, ts, tm, cs, cm))
out_path = f"/tmp/dit_trunc_{N}.tflite"
edge.export(out_path)
print(f"[export] {out_path}  {os.path.getsize(out_path)/1e6:.0f} MB")
