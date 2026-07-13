"""短い配達返答で warm 計測。CFG パス数削減が最大レバーなので変えて測る。"""
import glob
import os
import time
import torch
from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest

torch.set_num_threads(4)
ckpt = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--Aratako--Irodori-TTS-500M-v2-VoiceDesign/snapshots/*/model.safetensors"))[0]
rt = InferenceRuntime.from_key(RuntimeKey(
    checkpoint=ckpt, model_device="cpu", codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim",
    model_precision="fp32", codec_device="cpu", codec_precision="fp32",
    codec_deterministic_encode=True, codec_deterministic_decode=True,
    compile_model=False, compile_dynamic=False))

TEXT = "了解しました。到着したらお知らせください。"  # 現実的に短い返答（~3s）
CAP = "柔らかく温かみのある若い女性の声"


def synth(steps, cfg_text, cfg_cap):
    t0 = time.time()
    r = rt.synthesize(SamplingRequest(
        text=TEXT, caption=CAP, no_ref=True, num_steps=steps, t_schedule_mode="sway",
        cfg_scale_text=cfg_text, cfg_scale_caption=cfg_cap, cfg_guidance_mode="independent"))
    dt = time.time() - t0
    a = r.audios[0] if hasattr(r, "audios") else None
    dur = (a.shape[-1] / 48000) if a is not None else 0
    return dt, dur


print("warmup...", flush=True)
synth(6, 3.0, 3.0)
synth(6, 3.0, 3.0)
print("=== short reply, warm (Mac CPU 4thr, fp32) ===", flush=True)
configs = [
    (6, 3.0, 3.0, "6步 CFG=text+caption(3pass)"),
    (6, 3.0, 0.0, "6步 CFG=text only(2pass)"),
    (6, 0.0, 0.0, "6步 CFG=none(1pass)"),
    (4, 3.0, 0.0, "4步 CFG=text only(2pass)"),
    (4, 0.0, 0.0, "4步 CFG=none(1pass)"),
]
for steps, ct, cc, label in configs:
    dt, dur = synth(steps, ct, cc)
    print(f">>> {label:32s}  wall={dt:5.2f}s  audio={dur:.2f}s  RTF={dt / max(dur, 0.01):.2f}", flush=True)
