"""v4.1-Small-MF (MeanFlow) → Core ML packages for iOS 18+ (caption-only voice design, no reference audio).

Packages (written to $COREML_OUT, default ./coreml_v4):
  irodori4_encoder.mlpackage   ModernBERT-ja + text/caption projectors.  input_ids (1,L) int32, length (1,) int32
                               → text_state, caption_state (1,L,512).  Multifunction, one static function "L{L}"
                               per L ∈ TEXT_BUCKETS (the enumerated-shape build failed every predict).
  irodori4_duration.mlpackage  text_state (1,L,512), text_len, caption_state (1,CAP_LEN,512), caption_len,
                               duration_features (1,14) → log_frames (1,).
  irodori4_dit.mlpackage       one MeanFlow step: x_t (1,S,32), t, delta_t, latent_len, text_state (1,DIT_T,512),
                               text_len, caption_state (1,CAP_LEN,512), caption_len → v (1,S,32).
                               Multifunction package, one static-shape function "s{S}" per S ∈ DIT_S, shared
                               weights (enumerated shapes made coremltools' symbolic shape pass blow up).
  irodori4_decoder.mlpackage   DACVAE decoder chunk: z (1,32,W) → audio (1,1,W*1920).  W ∈ DEC_T, chunked with
                               DEC_OVERLAP frames of context (receptive field is ±7.1 latent frames).

Masks are built in-graph from lengths so only one input per package varies in shape. The speaker segment is
dropped: with no reference its keys are fully masked, so removing them is exact. The host loop
(CoreMLPipeline below) is the reference for the Swift runtime.

Run: COREML_OUT=<dir> <venv>/bin/python export_v4_coreml.py [enc] [dur] [dit] [dec]   (default: all)
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(8)
OUT = os.environ.get("COREML_OUT", os.path.abspath("coreml_v4"))
TEXT_BUCKETS = (32, 64, 128, 256)  # tokens incl. BOS; 256 = runtime max_text_len
CAP_LEN = 64
DIT_S = (64, 128, 192, 256, 384)  # latent frames (25 Hz): 2.6 / 5.1 / 7.7 / 10.2 / 15.4 s
DIT_T = 128  # a 15 s utterance is ~90 tokens, so 128 covers every DiT bucket
DEC_T = (64, 128)
DEC_OVERLAP = 8
LATENT_DIM, HOP, SR = 32, 1920, 48000
NEG = -1e4  # additive mask; fp16-safe
NAMES = {k: f"irodori4_{k}.mlpackage" for k in ("encoder", "duration", "dit", "decoder")}


def load_runtime():
    from huggingface_hub import snapshot_download

    from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey

    snap = snapshot_download("Aratako/Irodori-TTS-v4.1-Small-MF")
    rt = InferenceRuntime.from_key(RuntimeKey(
        checkpoint=os.path.join(snap, "model.safetensors"), model_device="cpu",
        codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim", model_precision="fp32", codec_device="cpu",
        codec_precision="fp32", codec_deterministic_encode=True, codec_deterministic_decode=True,
        compile_model=False, compile_dynamic=False))
    rt.model.eval()
    assert rt.model.cfg.flow_parameterization == "meanflow"
    return rt


def safe_rms(x, weight, eps):
    # Max-abs prescale keeps x*x inside fp16 range; algebraically identical to x*rsqrt(mean(x^2)+eps).
    s = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    xs = x / s
    return xs * torch.rsqrt((xs * xs).mean(dim=-1, keepdim=True) + eps / (s * s)) * weight


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class EncoderCM(nn.Module):
    """ModernBERT forward rewritten with explicit additive masks (HF's mask builders do not trace)."""

    def __init__(self, m, max_len=max(TEXT_BUCKETS), L=None):
        super().__init__()
        self.m = m
        self.L = L  # static length for conversion: coremltools mis-casts dims read back from a fixed shape
        bb = m.pretrained_text_backbone.backbone
        cfg = bb.config
        self.bb = bb
        self.heads = cfg.num_attention_heads
        self.hd = cfg.hidden_size // self.heads
        self.window = int(cfg.sliding_window)
        self.types = list(cfg.layer_types)
        pos = torch.arange(max_len, dtype=torch.float32)
        for lt in set(self.types):
            inv = getattr(bb.rotary_emb, f"{lt}_inv_freq").float()
            sc = float(getattr(bb.rotary_emb, f"{lt}_attention_scaling"))
            f = torch.outer(pos, inv)
            e = torch.cat([f, f], dim=-1)
            self.register_buffer(f"cos_{lt}", e.cos() * sc)
            self.register_buffer(f"sin_{lt}", e.sin() * sc)

    @staticmethod
    def _proj(p, norm, state, mask):
        y = p.projector(state)
        if p.projector_type == "residual_mlp":
            r = safe_rms(state, p.residual_norm.weight, p.residual_norm.eps)
            y = y + p.residual_down(F.silu(p.residual_up(r)))
        return safe_rms(y * mask, norm.weight, norm.eps)

    def forward(self, input_ids, length):
        L = self.L or input_ids.shape[1]
        idx = torch.arange(L, dtype=torch.int32)
        valid = idx < length
        full_bias = torch.where(valid, 0.0, NEG)[None, None, None, :]
        band = (idx[:, None] - idx[None, :]).abs() <= self.window
        local_bias = full_bias + torch.where(band, 0.0, NEG)[None, None]
        h = self.bb.embeddings.norm(self.bb.embeddings.tok_embeddings(input_ids))
        for layer, lt in zip(self.bb.layers, self.types):
            cos = getattr(self, f"cos_{lt}")[:L]
            sin = getattr(self, f"sin_{lt}")[:L]
            qkv = layer.attn.Wqkv(layer.attn_norm(h)).view(1, L, 3, self.heads, self.hd)
            q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))
            q = q * cos + _rotate_half(q) * sin
            k = k * cos + _rotate_half(k) * sin
            a = F.scaled_dot_product_attention(
                q, k, v, attn_mask=local_bias if lt == "sliding_attention" else full_bias)
            h = h + layer.attn.Wo(a.transpose(1, 2).reshape(1, L, -1))
            h = h + layer.mlp(layer.mlp_norm(h))
        mask = valid[None, :, None].to(h.dtype)
        h = self.bb.final_norm(h) * mask
        m = self.m
        return (self._proj(m.text_encoder, m.text_norm, h, mask),
                self._proj(m.caption_encoder, m.caption_norm, h, mask))


class DurationCM(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, text_state, text_len, caption_state, caption_len, duration_features):
        tm = (torch.arange(text_state.shape[1], dtype=torch.int32) < text_len)[None]
        cm = (torch.arange(caption_state.shape[1], dtype=torch.int32) < caption_len)[None]
        # has_speaker=False selects the learned null speaker, same as the runtime's no_ref path.
        return self.m.predict_duration_log_frames(
            text_state=text_state, text_mask=tm, speaker_state=None, speaker_mask=None,
            duration_features=duration_features, has_speaker=torch.zeros(1, dtype=torch.bool),
            caption_state=caption_state, caption_mask=cm, has_caption=torch.ones(1, dtype=torch.bool))


class DiTStepCM(nn.Module):
    """forward_with_encoded_conditions without the (fully masked) speaker segment, real-valued RoPE."""

    def __init__(self, m, S, T=DIT_T):
        super().__init__()
        self.m = m
        self.S, self.T = S, T  # static: coremltools mis-casts dims read back from a fixed shape
        self.heads = m.cfg.num_heads
        self.hd = m.cfg.model_dim // self.heads
        fr = 1.0 / (10000.0 ** (torch.arange(0, self.hd, 2, dtype=torch.float32) / self.hd))
        ang = torch.outer(torch.arange(S, dtype=torch.float32), fr)
        self.register_buffer("cos", ang.cos())
        self.register_buffer("sin", ang.sin())

    def _rope(self, x, S):  # interleaved pairs, as apply_rotary_emb's view_as_complex
        cos = self.cos[None, :, None, :]
        sin = self.sin[None, :, None, :]
        xr = x.reshape(1, S, self.heads // 2, self.hd // 2, 2)  # RoPE covers the first half of the heads
        x0, x1 = xr[..., 0], xr[..., 1]
        return torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1).reshape(1, S, self.heads // 2, self.hd)

    @staticmethod
    def _adaln(mod, x, cond):
        shift, scale, gate = cond.chunk(3, dim=-1)
        shift = mod.shift_up(mod.shift_down(F.silu(shift))) + shift
        scale = mod.scale_up(mod.scale_down(F.silu(scale))) + scale
        gate = mod.gate_up(mod.gate_down(F.silu(gate))) + gate
        return safe_rms(x, 1.0, mod.eps) * (1.0 + scale) + shift, torch.tanh(gate)

    def forward(self, x_t, t, delta_t, latent_len, text_state, text_len, caption_state, caption_len):
        from irodori_tts.model import get_timestep_embedding

        m, H, D = self.m, self.heads, self.hd
        S, Lt, Lc = self.S, self.T, CAP_LEN
        valid = torch.cat([
            torch.arange(S, dtype=torch.int32) < latent_len,
            torch.arange(Lt, dtype=torch.int32) < text_len,
            torch.arange(Lc, dtype=torch.int32) < caption_len])
        bias = torch.where(valid, 0.0, NEG)[None, None, None, :]
        ted = m.cfg.timestep_embed_dim
        cond = m.cond_module(get_timestep_embedding(t, ted)) + m.delta_cond_module(get_timestep_embedding(delta_t, ted))
        cond = cond[:, None, :]
        x = m.in_proj(x_t)
        half = H // 2
        for blk in m.blocks:
            a = blk.attention
            h, g = self._adaln(blk.attention_adaln, x, cond)
            q = safe_rms(a.wq(h).view(1, S, H, D), a.q_norm.weight, a.q_norm.eps)
            k = safe_rms(a.wk(h).view(1, S, H, D), a.k_norm.weight, a.k_norm.eps)
            v = a.wv(h).view(1, S, H, D)
            q = torch.cat([self._rope(q[:, :, :half], S), q[:, :, half:]], dim=2)
            k = torch.cat([self._rope(k[:, :, :half], S), k[:, :, half:]], dim=2)
            kt = safe_rms(a.wk_text(text_state).view(1, Lt, H, D), a.k_norm.weight, a.k_norm.eps)
            vt = a.wv_text(text_state).view(1, Lt, H, D)
            kc = safe_rms(a.wk_caption(caption_state).view(1, Lc, H, D), a.k_norm.weight, a.k_norm.eps)
            vc = a.wv_caption(caption_state).view(1, Lc, H, D)
            K = torch.cat([k, kt, kc], dim=1).transpose(1, 2)
            V = torch.cat([v, vt, vc], dim=1).transpose(1, 2)
            y = F.scaled_dot_product_attention(q.transpose(1, 2), K, V, attn_mask=bias)
            y = y.transpose(1, 2).reshape(1, S, H * D) * torch.sigmoid(a.gate(h))
            x = x + g * a.wo(y)
            h, g = self._adaln(blk.mlp_adaln, x, cond)
            x = x + g * blk.mlp(h)
        return m.out_proj(safe_rms(x, m.out_norm.weight, m.out_norm.eps))


def _snake_forward(self, x):  # replaces the TorchScript snake(), which torch.jit.trace cannot inline
    a = self.alpha
    return x + (a + 1e-9).reciprocal() * torch.sin(a * x).pow(2)


class DecoderCM(nn.Module):
    """quantizer.out_proj → decoder.model → watermark passthrough (Irodori runs the codec with alpha=0)."""

    def __init__(self, codec_model):
        super().__init__()
        import dacvae.nn.layers as L

        L.Snake1d.forward = _snake_forward
        for mod in codec_model.modules():
            if hasattr(mod, "weight_g"):
                torch.nn.utils.remove_weight_norm(mod)
        self.out_proj = codec_model.quantizer.out_proj
        self.layers = codec_model.decoder.model
        self.tail = codec_model.decoder.wm_model.encoder_block.pre[:-1]

    def forward(self, z):
        x = self.out_proj(z)
        for layer in self.layers:
            x = layer(x)
        return self.tail(x)


# ── host-side logic shared by the parity script (mirror this in Swift) ──

def bucket(buckets, n):
    for b in buckets:
        if n <= b:
            return b
    raise ValueError(f"length {n} exceeds largest bucket {buckets[-1]}")


def decoder_chunk_plan(S, widths=DEC_T, overlap=DEC_OVERLAP):
    """[(win_start, width, keep_start, keep_end)] in latent frames. Kept frames always see >= overlap frames
    of real context (or the true utterance edge), so chunked output matches a full decode."""
    plan, a = [], 0
    while a < S:
        ws = a - min(overlap, a)
        w = next((w for w in widths if ws + w >= S), widths[-1])
        if ws + w >= S:  # final chunk: end the window at S when possible (no right padding)
            ws = max(0, S - w)
            plan.append((ws, w, a, S))
            break
        plan.append((ws, w, a, ws + w - overlap))
        a = ws + w - overlap
    return plan


def _ct():
    import coremltools as ct
    return ct


def _convert(traced, inputs, outputs, name, precision=None):
    ct = _ct()
    mlm = ct.convert(
        traced, inputs=inputs, outputs=outputs, convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=precision or ct.precision.FLOAT16, compute_units=ct.ComputeUnit.ALL)
    path = os.path.join(OUT, NAMES[name])
    mlm.save(path)
    print(f"[export] {path}", flush=True)
    return mlm


def _cmp(name, a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    rel = np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12)
    cos = float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
    print(f"[verify {name}] relL2={rel:.2e} cos={cos:.6f}", flush=True)
    return rel, cos


def export_encoder(rt):
    ct = _ct()
    m = rt.model
    enc = EncoderCM(m).eval()
    # wrapper ≡ model.encode_conditions (text padded to 256 like the runtime; caption at its own length)
    ids = torch.randint(8, 30000, (1, 200))
    ids[0, 0] = 1
    n = 150
    pad = torch.zeros(1, 256, dtype=torch.long)
    pad[0, :n] = ids[0, :n]
    pm = torch.zeros(1, 256, dtype=torch.bool)
    pm[0, :n] = True
    with torch.no_grad():
        ref_ts = m.text_norm(m.text_encoder(m.pretrained_text_backbone, pad, pm))[0, :n]
        ref_cs = m.caption_norm(m.caption_encoder(m.pretrained_text_backbone, pad, pm))[0, :n]
        ts, cs = enc(pad.int(), torch.tensor([n], dtype=torch.int32))
    _cmp("encoder wrapper text (n=150 > window)", ts[0, :n], ref_ts)
    _cmp("encoder wrapper caption", cs[0, :n], ref_cs)
    # Static multifunction like the DiT: the EnumeratedShapes build converted, but every predict failed with
    # error -7 on CPU, GPU and ANE alike.
    fns = []
    for L in TEXT_BUCKETS:
        ex = (torch.ones(1, L, dtype=torch.int32), torch.tensor([L - 3], dtype=torch.int32))
        with torch.no_grad():
            tr = torch.jit.trace(EncoderCM(m, L=L).eval(), ex, check_trace=False)
        mlm = ct.convert(
            tr, inputs=[ct.TensorType("input_ids", shape=(1, L), dtype=np.int32),
                        ct.TensorType("length", shape=(1,), dtype=np.int32)],
            outputs=[ct.TensorType("text_state"), ct.TensorType("caption_state")], convert_to="mlprogram",
            minimum_deployment_target=ct.target.iOS18, compute_precision=ct.precision.FLOAT16)
        fns.append((f"L{L}", mlm))
    _save_multifunction(fns, "encoder", "L64")


def _save_multifunction(fns, name, default):
    """[(function_name, MLModel)] → one multifunction package with deduplicated weights."""
    import shutil

    ct = _ct()
    desc = ct.utils.MultiFunctionDescriptor()
    tmp = []
    for fn, mlm in fns:
        p = os.path.join(OUT, f"_tmp_{name}_{fn}.mlpackage")
        mlm.save(p)
        tmp.append(p)
        desc.add_function(p, src_function_name="main", target_function_name=fn)
    desc.default_function_name = default
    path = os.path.join(OUT, NAMES[name])
    ct.utils.save_multifunction(desc, path)
    for p in tmp:
        shutil.rmtree(p)
    print(f"[export] {path}", flush=True)


def export_duration(rt):
    ct = _ct()
    m = rt.model
    dur = DurationCM(m).eval()
    TD = int(m.cfg.text_dim)
    ts = torch.randn(1, 64, TD)
    cs = torch.randn(1, CAP_LEN, TD)
    feats = torch.rand(1, int(m.cfg.duration_aux_dim))
    nt, nc = 37, 11
    with torch.no_grad():
        got = dur(ts, torch.tensor([nt], dtype=torch.int32), cs, torch.tensor([nc], dtype=torch.int32), feats)
        # runtime no_ref path: 2-token zero-masked speaker state, has_speaker=False, unpadded text/caption
        sp = torch.zeros(1, 2, int(m.cfg.speaker_dim))
        ref = m.predict_duration_log_frames(
            text_state=ts[:, :nt], text_mask=torch.ones(1, nt, dtype=torch.bool), speaker_state=sp,
            speaker_mask=torch.zeros(1, 2, dtype=torch.bool), duration_features=feats,
            has_speaker=torch.zeros(1, dtype=torch.bool), caption_state=cs[:, :nc],
            caption_mask=torch.ones(1, nc, dtype=torch.bool), has_caption=torch.ones(1, dtype=torch.bool))
    print(f"[verify duration wrapper] {got.item():.6f} vs {ref.item():.6f}")
    assert abs(got.item() - ref.item()) < 1e-4
    ex = (ts, torch.tensor([nt], dtype=torch.int32), cs, torch.tensor([nc], dtype=torch.int32), feats)
    with torch.no_grad():
        tr = torch.jit.trace(dur, ex, check_trace=False)
    _convert(tr, [ct.TensorType("text_state", shape=ct.EnumeratedShapes([(1, b, TD) for b in TEXT_BUCKETS], default=(1, 64, TD))),
                  ct.TensorType("text_len", shape=(1,), dtype=np.int32),
                  ct.TensorType("caption_state", shape=(1, CAP_LEN, TD)),
                  ct.TensorType("caption_len", shape=(1,), dtype=np.int32),
                  ct.TensorType("duration_features", shape=(1, int(m.cfg.duration_aux_dim)))],
             [ct.TensorType("log_frames")], "duration")


def export_dit(rt):
    import shutil

    ct = _ct()
    m = rt.model
    TD, SD = int(m.cfg.text_dim), int(m.cfg.speaker_dim)
    torch.manual_seed(0)
    S, nS, nt, nc = 128, 101, 37, 11
    x = torch.randn(1, S, LATENT_DIM)
    t = torch.full((1,), 0.75)
    dt = torch.full((1,), 0.25)
    ts = torch.randn(1, DIT_T, TD)
    cs = torch.randn(1, CAP_LEN, TD)
    i32 = lambda v: torch.tensor([v], dtype=torch.int32)  # noqa: E731
    with torch.no_grad():
        got = DiTStepCM(m, S)(x, t, dt, i32(nS), ts, i32(nt), cs, i32(nc))[:, :nS]
        # original model, unpadded, with the runtime's no_ref speaker state (2 masked tokens)
        ref = m.forward_with_encoded_conditions(
            x[:, :nS], t, ts[:, :nt], torch.ones(1, nt, dtype=torch.bool), torch.zeros(1, 2, SD),
            torch.zeros(1, 2, dtype=torch.bool), cs[:, :nc], torch.ones(1, nc, dtype=torch.bool), delta_t=dt)
    rel, _ = _cmp("dit wrapper (padded S/T/caption vs unpadded original)", got, ref)
    assert rel < 1e-4
    m.pretrained_text_backbone = None  # not used by the step; frees ~1.3 GB before conversion
    m.duration_predictor = None
    rt.codec = None
    desc = ct.utils.MultiFunctionDescriptor()
    tmp = []
    for s in DIT_S:
        ex = (torch.randn(1, s, LATENT_DIM), t, dt, i32(s - 7), ts, i32(nt), cs, i32(nc))
        with torch.no_grad():
            tr = torch.jit.trace(DiTStepCM(m, s).eval(), ex, check_trace=False)
        mlm = ct.convert(
            tr, inputs=[ct.TensorType("x_t", shape=(1, s, LATENT_DIM)), ct.TensorType("t", shape=(1,)),
                        ct.TensorType("delta_t", shape=(1,)), ct.TensorType("latent_len", shape=(1,), dtype=np.int32),
                        ct.TensorType("text_state", shape=(1, DIT_T, TD)), ct.TensorType("text_len", shape=(1,), dtype=np.int32),
                        ct.TensorType("caption_state", shape=(1, CAP_LEN, TD)), ct.TensorType("caption_len", shape=(1,), dtype=np.int32)],
            outputs=[ct.TensorType("v")], convert_to="mlprogram", minimum_deployment_target=ct.target.iOS18,
            compute_precision=ct.precision.FLOAT16)
        p = os.path.join(OUT, f"_tmp_dit_s{s}.mlpackage")
        mlm.save(p)
        del mlm, tr
        tmp.append(p)
        desc.add_function(p, src_function_name="main", target_function_name=f"s{s}")
        print(f"[export] dit function s{s}", flush=True)
    desc.default_function_name = f"s{DIT_S[1]}"
    path = os.path.join(OUT, NAMES["dit"])
    ct.utils.save_multifunction(desc, path)
    for p in tmp:
        shutil.rmtree(p)
    print(f"[export] {path}", flush=True)


def export_decoder(rt):
    ct = _ct()
    torch.manual_seed(0)
    z = torch.randn(1, LATENT_DIM, 64) * 0.8
    with torch.no_grad():
        ref = rt.codec.decode_latent(z.transpose(1, 2))  # before the snake patch / weight-norm removal
    dec = DecoderCM(rt.codec.model).eval()
    with torch.no_grad():
        got = dec(z)
    rel, _ = _cmp("decoder wrapper", got, ref)
    assert rel < 1e-5
    with torch.no_grad():
        tr = torch.jit.trace(dec, (z,), check_trace=False)
    _convert(tr, [ct.TensorType("z", shape=ct.EnumeratedShapes([(1, LATENT_DIM, w) for w in DEC_T], default=(1, LATENT_DIM, 64)))],
             [ct.TensorType("audio")], "decoder")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    which = sys.argv[1:] or ["enc", "dur", "dit", "dec"]
    rt = load_runtime()
    for w in which:
        {"enc": export_encoder, "dur": export_duration, "dit": export_dit, "dec": export_decoder}[w](rt)
