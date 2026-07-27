---
name: paddle-gpu-env
description: "PaddlePaddle GPU setup for the AST/layout project (CUDA wheel, device, known caveats)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 317792c1-ffe7-4d8b-863f-33cc59879d12
---

The AST project (`C:\Users\faree\Documents\Masters\AST`) runs PP-DocLayoutV3 +
PP-OCRv5 layout/OCR on GPU. As of 2026-06-10:

- GPU: NVIDIA RTX 4070 Ti (12 GB), compute 8.9; driver CUDA 13.1.
- Installed `paddlepaddle-gpu==3.3.1` from the **cu126** index
  (`https://www.paddlepaddle.org.cn/packages/stable/cu126/`), replacing the
  CPU-only `paddlepaddle`. paddleocr 3.6.0, paddlex 3.6.1.
- Device selection is `device="auto"` (`layout/detector.py:_resolve_device`) →
  GPU when `paddle.is_compiled_with_cuda()`, else CPU.
- **CPU caveat:** when running on CPU, predictors MUST pass `enable_mkldnn=False`
  — Paddle 3.x's OneDNN backend fails under the PIR executor
  ("ConvertPirAttribute2RuntimeAttribute not support
  pir::ArrayAttribute<pir::DoubleAttribute>"). The flag is a no-op on GPU.
- **Benign warning:** Paddle built against cuDNN 9.9 but machine has cuDNN 9.5
  ("may cause serious incompatible bug"). Inference verified correct despite it;
  only revisit (align cuDNN) if recognition crashes.
- The CUDA wheels are multi-GB; the first install attempt failed with "No space
  left on device" — needs ~5 GB free on C:.

See [[reading-order-xycut]] for the layout pipeline's reading-order stage.
