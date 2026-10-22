"""T1: MiniLM fine-tuned as a difficulty regressor, exported to ONNX and quantized to INT8.

Everything here is torch-free except ``torch_impl`` (imported lazily, exercised by the ``ml`` tests
inside the training container). The training loop, batching, metrics, artifact writing and the
quantization decision are plain Python/numpy and are tested with fakes in CI.
"""
