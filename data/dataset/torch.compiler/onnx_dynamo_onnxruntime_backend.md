ONNX Backend for TorchDynamo 
============================================================================================

For a quick overview of `torch.compiler`  , see [torch.compiler](torch.compiler.html#torch-compiler-overview)  . 

Warning 

The ONNX backend for torch.compile is a rapidly evolving beta technology.

torch.onnx. is_onnxrt_backend_supported ( ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/onnx/_internal/onnxruntime.py#L47) 
:   Returns `True`  if ONNX Runtime dependencies are installed and usable
to support TorchDynamo backend integration; `False`  otherwise. 

Example: 

```
# xdoctest: +REQUIRES(env:TORCH_DOCTEST_ONNX)
>>> import torch
>>> if torch.onnx.is_onnxrt_backend_supported():
...     @torch.compile(backend="onnxrt")
...     def f(x):
...             return x * x
...     print(f(torch.randn(10)))
... else:
...     print("pip install onnx onnxscript onnxruntime")
...

```

Return type
:   [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")

