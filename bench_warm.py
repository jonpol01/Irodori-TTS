"""Irodori-500M-v2-VoiceDesign を CPU で warm 計測（Android 移植の実効レイテンシ検証）。
初回は cold（重み初期化等）で遅いので、warmup 後の定常値を見る。"""
import glob
import os
import time
import torch
from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest

torch.set_num_threads(4)  # Pixel 大コア相当
ckpt = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--Aratako--Irodori-TTS-500M-v2-VoiceDesign/snapshots/*/model.safetensors"))[0]
print("ckpt:", ckpt.split("snapshots/")[1][:60], flush=True)

rt = InferenceRuntime.from_key(RuntimeKey(
    checkpoint=ckpt, model_device="cpu", codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim",
    model_precision="fp32", codec_device="cpu", codec_precision="fp32",
    codec_deterministic_encode=True, codec_deterministic_decode=True,
    compile_model=False, compile_dynamic=False))

TEXT = "お疲れ様です。次の配達先はさくら倉庫です。安全運転を心がけてください。"
CAP = "柔らかく温かみのある若い女性の声"


def synth(steps):
    t0 = time.time()
    r = rt.synthesize(SamplingRequest(
        text=TEXT, caption=CAP, no_ref=True, num_steps=steps,
        t_schedule_mode="sway", cfg_guidance_mode="independent"))
    return time.time() - t0, r


print("=== WARMUP (cold, ignore) ===", flush=True)
synth(8)
synth(8)
print("=== WARM measurements ===", flush=True)
for steps in [8, 6, 4, 2]:
    dt, r = synth(steps)
    audio = r.audios[0] if hasattr(r, "audios") else None
    dur = (audio.shape[-1] / 48000) if audio is not None else 0
    print(f">>> num_steps={steps}: total_wall={dt:.2f}s  audio={dur:.2f}s  RTF={dt / max(dur, 0.01):.1f}", flush=True)
