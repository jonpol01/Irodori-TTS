"""3つの ONNX を int8 動的量子化し、サイズ・速度・parity を測る（Android 実機規模の指標）。"""
import glob
import os
import time
import numpy as np
import torch
import soundfile as sf
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest

torch.set_num_threads(4)

# --- int8 動的量子化（encoder/dit のみ。decoder は weight_norm conv のため fp32 のまま）---
q_names = ["irodori_encoder", "irodori_dit"]
print("=== int8 quantization ===")
for n in q_names:
    quantize_dynamic(f"/tmp/{n}.onnx", f"/tmp/{n}_int8.onnx", weight_type=QuantType.QInt8)
    fp = os.path.getsize(f"/tmp/{n}.onnx") / 1e6
    q = os.path.getsize(f"/tmp/{n}_int8.onnx") / 1e6
    print(f"  {n:18s} fp32={fp:6.1f}MB -> int8={q:6.1f}MB")
print(f"  dacvae_decoder     fp32={os.path.getsize('/tmp/dacvae_decoder.onnx') / 1e6:6.1f}MB (kept fp32)")

ckpt = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--Aratako--Irodori-TTS-500M-v2-VoiceDesign/snapshots/*/model.safetensors"))[0]
rt = InferenceRuntime.from_key(RuntimeKey(
    checkpoint=ckpt, model_device="cpu", codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim",
    model_precision="fp32", codec_device="cpu", codec_precision="fp32",
    codec_deterministic_encode=True, codec_deterministic_decode=True,
    compile_model=False, compile_dynamic=False))
m = rt.model; codec = rt.codec
T = "お疲れ様です！次の配達先はさくら倉庫です。安全運転でいきましょう！"
C = "柔らかく温かみのある若い女性の声"
req = lambda: SamplingRequest(text=T, caption=C, no_ref=True, num_steps=8, seconds=6.0, cfg_scale_text=3.0, cfg_scale_caption=3.0, cfg_guidance_mode="independent", decode_mode="batch", context_kv_cache=False, t_schedule_mode="sway", seed=0)
ref = rt.synthesize(req()); ref_audio = ref.audios[0].squeeze().float().numpy()

so = ort.SessionOptions(); so.intra_op_num_threads = 4
sess = {n: ort.InferenceSession(f"/tmp/{n}_int8.onnx", so, providers=["CPUExecutionProvider"]) for n in q_names}
sess["dacvae_decoder"] = ort.InferenceSession("/tmp/dacvae_decoder.onnx", so, providers=["CPUExecutionProvider"])


def onnx_encode(text_input_ids, text_mask, ref_latent, ref_mask, caption_input_ids=None, caption_mask=None, **kw):
    o = sess["irodori_encoder"].run(None, {"text_ids": text_input_ids.cpu().numpy().astype(np.int64), "text_mask": text_mask.cpu().numpy(), "caption_ids": caption_input_ids.cpu().numpy().astype(np.int64), "caption_mask": caption_mask.cpu().numpy()})
    return torch.from_numpy(o[0]), text_mask, None, None, torch.from_numpy(o[1]), caption_mask


def onnx_dit(x_t, t, text_state, text_mask, speaker_state, speaker_mask, caption_state=None, caption_mask=None, latent_mask=None, context_kv_cache=None):
    outs = []
    for b in range(x_t.shape[0]):
        o = sess["irodori_dit"].run(None, {"x_t": x_t[b:b + 1].cpu().numpy(), "t": t[b:b + 1].cpu().numpy(), "text_state": text_state[b:b + 1].cpu().numpy(), "text_mask": text_mask[b:b + 1].cpu().numpy(), "caption_state": caption_state[b:b + 1].cpu().numpy(), "caption_mask": caption_mask[b:b + 1].cpu().numpy()})
        outs.append(torch.from_numpy(o[0]))
    return torch.cat(outs, 0)


def onnx_decode(latent):
    return torch.from_numpy(sess["dacvae_decoder"].run(None, {"z": latent.transpose(1, 2).contiguous().float().cpu().numpy()})[0])


m.encode_conditions = onnx_encode
m.forward_with_encoded_conditions = onnx_dit
codec.decode_latent = onnx_decode

rt.synthesize(req())  # warm
t0 = time.time(); out = rt.synthesize(req()); wall = time.time() - t0
a = out.audios[0].squeeze().float().numpy()
n = min(len(a), len(ref_audio))
print("=== int8 ONNX pipeline (Mac CPU 4thr) ===")
print(f"  wall={wall:.2f}s  audio={len(a) / 48000:.2f}s  RTF={wall / (len(a) / 48000):.2f}  corr_vs_fp32={np.corrcoef(a[:n], ref_audio[:n])[0, 1]:.5f}")
sf.write(os.path.expanduser("~/Desktop/irodori_onnx_int8.wav"), a, 48000)
print("  saved ~/Desktop/irodori_onnx_int8.wav")
