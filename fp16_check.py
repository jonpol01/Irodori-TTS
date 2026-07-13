"""ONNX 3モデルで実パイプラインを駆動（sampler は runtime を流用）→ PyTorch と最終音声を parity 検証。
Phase A の締め: ONNX だけで cute voice を生成できることを証明する。"""
import glob
import os
import numpy as np
import torch
import soundfile as sf
import onnxruntime as ort
from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest

torch.set_num_threads(4)
ckpt = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--Aratako--Irodori-TTS-500M-v2-VoiceDesign/snapshots/*/model.safetensors"))[0]
rt = InferenceRuntime.from_key(RuntimeKey(
    checkpoint=ckpt, model_device="cpu", codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim",
    model_precision="fp32", codec_device="cpu", codec_precision="fp32",
    codec_deterministic_encode=True, codec_deterministic_decode=True,
    compile_model=False, compile_dynamic=False))
m = rt.model
codec = rt.codec
T = "お疲れ様です！次の配達先はさくら倉庫です。安全運転でいきましょう！"
C = "柔らかく温かみのある若い女性の声"


def req():
    return SamplingRequest(text=T, caption=C, no_ref=True, num_steps=8, seconds=6.0,
                           cfg_scale_text=3.0, cfg_scale_caption=3.0, cfg_guidance_mode="independent",
                           decode_mode="batch", context_kv_cache=False, t_schedule_mode="sway", seed=0)


# --- pure-PyTorch reference (before patching) ---
ref = rt.synthesize(req())
ref_audio = ref.audios[0].squeeze().float().numpy()

# --- ONNX sessions ---
so = ort.SessionOptions(); so.intra_op_num_threads = 4
enc_s = ort.InferenceSession("/tmp/irodori_encoder_fp16.onnx", so, providers=["CPUExecutionProvider"])
dit_s = ort.InferenceSession("/tmp/irodori_dit_fp16.onnx", so, providers=["CPUExecutionProvider"])
dec_s = ort.InferenceSession("/tmp/dacvae_decoder_fp16.onnx", so, providers=["CPUExecutionProvider"])


def onnx_encode(text_input_ids, text_mask, ref_latent, ref_mask, caption_input_ids=None, caption_mask=None, **kw):
    o = enc_s.run(None, {
        "text_ids": text_input_ids.cpu().numpy().astype(np.int64), "text_mask": text_mask.cpu().numpy(),
        "caption_ids": caption_input_ids.cpu().numpy().astype(np.int64), "caption_mask": caption_mask.cpu().numpy()})
    return torch.from_numpy(o[0]), text_mask, None, None, torch.from_numpy(o[1]), caption_mask


def onnx_dit(x_t, t, text_state, text_mask, speaker_state, speaker_mask,
             caption_state=None, caption_mask=None, latent_mask=None, context_kv_cache=None):
    outs = []
    for b in range(x_t.shape[0]):  # バッチ毎に（CFG variant がバッチ化されても安全）
        o = dit_s.run(None, {
            "x_t": x_t[b:b + 1].cpu().numpy(), "t": t[b:b + 1].cpu().numpy(),
            "text_state": text_state[b:b + 1].cpu().numpy(), "text_mask": text_mask[b:b + 1].cpu().numpy(),
            "caption_state": caption_state[b:b + 1].cpu().numpy(), "caption_mask": caption_mask[b:b + 1].cpu().numpy()})
        outs.append(torch.from_numpy(o[0]))
    return torch.cat(outs, 0)


def onnx_decode(latent):
    z = latent.transpose(1, 2).contiguous().float().cpu().numpy()
    return torch.from_numpy(dec_s.run(None, {"z": z})[0])


m.encode_conditions = onnx_encode
m.forward_with_encoded_conditions = onnx_dit
codec.decode_latent = onnx_decode

# --- run the SAME request through the ONNX-backed pipeline ---
out = rt.synthesize(req())
onnx_audio = out.audios[0].squeeze().float().numpy()

n = min(len(ref_audio), len(onnx_audio))
diff = float(np.abs(ref_audio[:n] - onnx_audio[:n]).max())
corr = float(np.corrcoef(ref_audio[:n], onnx_audio[:n])[0, 1])
print(f"[E2E] pytorch={len(ref_audio)} onnx={len(onnx_audio)} samples  max|diff|={diff:.2e}  corr={corr:.6f}")
p = os.path.expanduser("~/Desktop/irodori_onnx_fp16.wav")
sf.write(p, onnx_audio, 48000)
print(f"saved {p}  ({len(onnx_audio) / 48000:.2f}s)")
