"""ランタイム非依存の純 numpy+ONNX サンプラ。これが Kotlin 移植の逐語仕様になる。
DiT は batch=1 でエクスポート済みなので CFG3変種は3回に分けて呼ぶ。"""
import glob
import os
import numpy as np
import torch
import soundfile as sf
import onnxruntime as ort
from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest
from irodori_tts.text_normalization import normalize_text

torch.set_num_threads(4)
ckpt = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--Aratako--Irodori-TTS-500M-v2-VoiceDesign/snapshots/*/model.safetensors"))[0]
rt = InferenceRuntime.from_key(RuntimeKey(
    checkpoint=ckpt, model_device="cpu", codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim",
    model_precision="fp32", codec_device="cpu", codec_precision="fp32",
    codec_deterministic_encode=True, codec_deterministic_decode=True, compile_model=False, compile_dynamic=False))
tok, ctok = rt.tokenizer, (rt.caption_tokenizer or rt.tokenizer)
TEXT = "お疲れ様です！次の配達先はさくら倉庫です。安全運転でいきましょう！"
CAP = "柔らかく温かみのある若い女性の声"
NUM_STEPS, SECONDS, CFG_T, CFG_C, CFG_MIN, CFG_MAX, SWAY = 8, 6.0, 3.0, 3.0, 0.5, 1.0, -1.0

# --- reference (runtime, seed=0) ---
ref = rt.synthesize(SamplingRequest(text=TEXT, caption=CAP, no_ref=True, num_steps=NUM_STEPS, seconds=SECONDS,
                    cfg_scale_text=CFG_T, cfg_scale_caption=CFG_C, cfg_guidance_mode="independent",
                    decode_mode="batch", context_kv_cache=False, t_schedule_mode="sway", seed=0))
ref_audio = ref.audios[0].squeeze().float().numpy()

# --- standalone pipeline ---
so = ort.SessionOptions(); so.intra_op_num_threads = 4
enc = ort.InferenceSession("/tmp/irodori_encoder.onnx", so, providers=["CPUExecutionProvider"])
dit = ort.InferenceSession("/tmp/irodori_dit.onnx", so, providers=["CPUExecutionProvider"])
dec = ort.InferenceSession("/tmp/dacvae_decoder.onnx", so, providers=["CPUExecutionProvider"])

ti, tm = tok.batch_encode([normalize_text(TEXT)], 64)  # 正規化してからトークナイズ（必須）
ci, cm = ctok.batch_encode([normalize_text(CAP)], 64)
ti, tm, ci, cm = ti.numpy().astype(np.int64), tm.numpy(), ci.numpy().astype(np.int64), cm.numpy()
text_state, caption_state = enc.run(None, {"text_ids": ti, "text_mask": tm, "caption_ids": ci, "caption_mask": cm})
t_unc = np.zeros_like(text_state); tm_unc = np.zeros_like(tm)
c_unc = np.zeros_like(caption_state); cm_unc = np.zeros_like(cm)

S = round(SECONDS * 25)  # 25Hz latent (hop 1920 @48k)
g = torch.Generator().manual_seed(0)
x = torch.randn((1, S, 32), generator=g).numpy().astype(np.float32)  # runtime seed=0 と一致させる
u = np.linspace(0.0, 1.0, NUM_STEPS + 1)
u = np.clip(u + SWAY * (np.cos(0.5 * np.pi * u) + u - 1.0), 0.0, 1.0)
t_sched = (1.0 - u) * 0.999


def dit_run(xx, t, ts, tmk, cs, cmk):
    return dit.run(None, {"x_t": xx, "t": np.array([t], np.float32), "text_state": ts, "text_mask": tmk,
                          "caption_state": cs, "caption_mask": cmk})[0]


for i in range(NUM_STEPS):
    t, t_next = float(t_sched[i]), float(t_sched[i + 1])
    if CFG_MIN <= t <= CFG_MAX:
        v_cond = dit_run(x, t, text_state, tm, caption_state, cm)
        v_tu = dit_run(x, t, t_unc, tm_unc, caption_state, cm)         # text uncond
        v_cu = dit_run(x, t, text_state, tm, c_unc, cm_unc)            # caption uncond
        v = v_cond + CFG_T * (v_cond - v_tu) + CFG_C * (v_cond - v_cu)
    else:
        v = dit_run(x, t, text_state, tm, caption_state, cm)
    x = x + v * (t_next - t)

audio = dec.run(None, {"z": np.transpose(x, (0, 2, 1)).astype(np.float32)})[0].squeeze()
n = min(len(audio), len(ref_audio))
print(f"[standalone] samples={len(audio)} vs ref={len(ref_audio)}  max|diff|={np.abs(audio[:n] - ref_audio[:n]).max():.2e}  corr={np.corrcoef(audio[:n], ref_audio[:n])[0, 1]:.6f}")
sf.write(os.path.expanduser("~/Desktop/irodori_standalone.wav"), audio, 48000)
print("saved ~/Desktop/irodori_standalone.wav")
