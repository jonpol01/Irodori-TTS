"""DACVAE デコーダを ONNX 化して parity と CPU 速度を測る（Android 移植の最重要ボトルネック検証）。"""
import time
import numpy as np
import torch
from irodori_tts.codec import DACVAECodec

torch.set_num_threads(4)  # Pixel の大コア相当を模擬
codec = DACVAECodec.load(device="cpu", dtype=torch.float32, deterministic_decode=True)
print(f"[codec] sr={codec.sample_rate} latent_dim={codec.latent_dim}")

# 4秒の音声を encode して現実的な latent 長 T を得る
dummy = torch.randn(1, 1, codec.sample_rate * 4)
with torch.inference_mode():
    lat = codec.encode_waveform(dummy, codec.sample_rate)  # (B, T, D)
T = lat.shape[1]
print(f"[shape] 4s audio -> latent T={T}  ({T / 4:.1f} frames/sec)")
z = lat.transpose(1, 2).contiguous().float()  # (B, D, T)


class Dec(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, z):
        return self.m.decode(z)  # (B, 1, samples)


dec = Dec(codec.model).eval()

# torch baseline（parity 基準 + 速度）
with torch.inference_mode():
    for _ in range(2):
        a_torch = dec(z)  # warmup
    t0 = time.time()
    a_torch = dec(z)
    t_torch = time.time() - t0
print(f"[torch] decode {t_torch * 1000:.0f} ms -> audio {tuple(a_torch.shape)} ({a_torch.shape[-1] / codec.sample_rate:.2f}s)")

# ONNX export
try:
    torch.onnx.export(
        dec, (z,), "/tmp/dacvae_decoder.onnx",
        input_names=["z"], output_names=["audio"],
        dynamic_axes={"z": {2: "T"}, "audio": {2: "S"}},
        opset_version=17, do_constant_folding=True,
    )
    print("[onnx] exported /tmp/dacvae_decoder.onnx")
except Exception as e:
    print(f"[onnx] EXPORT FAILED: {type(e).__name__}: {str(e)[:300]}")
    raise

# ONNX parity + speed
import onnxruntime as ort  # noqa: E402
so = ort.SessionOptions()
so.intra_op_num_threads = 4
sess = ort.InferenceSession("/tmp/dacvae_decoder.onnx", so, providers=["CPUExecutionProvider"])
zn = z.numpy()
for _ in range(2):
    a_onnx = sess.run(None, {"z": zn})[0]  # warmup
t0 = time.time()
a_onnx = sess.run(None, {"z": zn})[0]
t_onnx = time.time() - t0
diff = float(np.abs(a_onnx - a_torch.numpy()).max())
print(f"[onnx-fp32] decode {t_onnx * 1000:.0f} ms  max|diff|={diff:.2e}")
