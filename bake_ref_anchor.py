"""声アンカーの ref_state 事前計算: アンカー wav → DACVAE encode → patch →
speaker_encoder → mean-token 前置 → f32 .bin 保存。アプリはこれを DiT の
speaker_state/speaker_mask 入力としてそのまま与える（端末に ref エンコーダ不要）。
全アンカーを同一長 (--seconds) に切り揃えて R を固定し、静的形状エクスポートに合わせる。
usage: .venv/bin/python bake_ref_anchor.py <anchor.wav> <out_prefix> [--seconds 8.0]
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
import torchaudio

from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey

parser = argparse.ArgumentParser()
parser.add_argument("wav")
parser.add_argument("out_prefix")
parser.add_argument("--seconds", type=float, default=8.0)
args = parser.parse_args()

torch.set_num_threads(6)
ckpt = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--Aratako--Irodori-TTS-600M-v3-VoiceDesign/snapshots/*/model.safetensors"))[0]
rt = InferenceRuntime.from_key(RuntimeKey(
    checkpoint=ckpt, model_device="cpu", codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim",
    model_precision="fp32", codec_device="cpu", codec_precision="fp32",
    codec_deterministic_encode=True, codec_deterministic_decode=True,
    compile_model=False, compile_dynamic=False))
m = rt.model.eval()
print("model cfg: speaker_dim=", m.cfg.speaker_dim, "speaker_patch_size=", m.cfg.speaker_patch_size,
      "text_dim=", m.cfg.text_dim, "caption_dim=", m.cfg.caption_dim,
      "model_dim=", m.cfg.model_dim, "layers=", m.cfg.num_layers)

# wav → 固定長に切り揃え → DACVAE encode → ref latent (1, T_ref, 32)
wav, sr = torchaudio.load(args.wav)
wav = wav.mean(dim=0, keepdim=True)
target = int(args.seconds * sr)
if wav.shape[1] >= target:
    wav = wav[:, :target]
else:
    wav = torch.nn.functional.pad(wav, (0, target - wav.shape[1]))
print(f"anchor audio: {wav.shape[1] / sr:.2f}s @ {sr}Hz")

with torch.no_grad():
    # ランタイムと同じ参照エンコード（encode_conditions が内部で speaker_patch_size にパッチする）。
    # ⚠️ normalize_db=-16 はランタイム既定値 — これを外すと参照条件付けが学習分布より強くなり、
    # 合成音声の冒頭にアンカー音声の「言葉」が滲む（実機で発症: 返答の頭に変な単語）。
    ref_latent = rt.codec.encode_waveform(wav.unsqueeze(0), sample_rate=int(sr), normalize_db=-16.0, ensure_max=True).cpu()
    print("ref_latent:", tuple(ref_latent.shape))
    T_ref = ref_latent.shape[1]
    ref_mask = torch.ones(1, T_ref, dtype=torch.bool)
    # encode_conditions で ref_state を得る（text/caption はダミー、ref だけ取り出す）
    dummy_ids = torch.tensor([[1]], dtype=torch.long)
    dummy_mask = torch.ones(1, 1, dtype=torch.bool)
    _, _, ref_state, ref_state_mask, _, _ = m.encode_conditions(
        text_input_ids=dummy_ids, text_mask=dummy_mask,
        ref_latent=ref_latent, ref_mask=ref_mask,
        caption_input_ids=dummy_ids, caption_mask=dummy_mask,
    )
print("ref_state:", tuple(ref_state.shape), "ref_mask:", tuple(ref_state_mask.shape))

ref_np = ref_state.float().numpy()
ref_np.tofile(args.out_prefix + ".bin")
meta = {"shape": list(ref_state.shape), "mask_true": int(ref_state_mask.sum().item()),
        "seconds": args.seconds, "source": os.path.basename(args.wav)}
with open(args.out_prefix + ".json", "w") as f:
    json.dump(meta, f, ensure_ascii=False, indent=1)
print("saved:", args.out_prefix + ".bin", ref_np.nbytes, "bytes |", meta)
