import os, onnx
from onnxconverter_common import float16
for n in ["irodori_encoder", "irodori_dit", "dacvae_decoder"]:
    mdl = onnx.load(f"/tmp/{n}.onnx")
    # keep_io_types so inputs/outputs stay fp32 (masks stay bool/int) — internal compute fp16
    m16 = float16.convert_float_to_float16(mdl, keep_io_types=True, disable_shape_infer=True)
    onnx.save(m16, f"/tmp/{n}_fp16.onnx")
    print(f"  {n:18s} fp32={os.path.getsize(f'/tmp/{n}.onnx')/1e6:6.1f}MB -> fp16={os.path.getsize(f'/tmp/{n}_fp16.onnx')/1e6:6.1f}MB")
