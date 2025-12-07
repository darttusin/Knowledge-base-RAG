torch.use_deterministic_algorithms 
==========================================================================================================

torch. use_deterministic_algorithms ( *mode*  , *** , *warn_only = False* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/__init__.py#L1320) 
:   Sets whether PyTorch operations must use “deterministic”
algorithms. That is, algorithms which, given the same input, and when
run on the same software and hardware, always produce the same output.
When enabled, operations will use deterministic algorithms when available,
and if only nondeterministic algorithms are available they will throw a [`RuntimeError`](https://docs.python.org/3/library/exceptions.html#RuntimeError "(in Python v3.13)")  when called. 

Note 

This setting alone is not always enough to make an application
reproducible. Refer to [Reproducibility](../notes/randomness.html#reproducibility)  for more information.

Note 

[`torch.set_deterministic_debug_mode()`](torch.set_deterministic_debug_mode.html#torch.set_deterministic_debug_mode "torch.set_deterministic_debug_mode")  offers an alternative
interface for this feature.

The following normally-nondeterministic operations will act
deterministically when `mode=True`  : 

> * [`torch.nn.Conv1d`](torch.nn.Conv1d.html#torch.nn.Conv1d "torch.nn.Conv1d")  when called on CUDA tensor
> * [`torch.nn.Conv2d`](torch.nn.Conv2d.html#torch.nn.Conv2d "torch.nn.Conv2d")  when called on CUDA tensor
> * [`torch.nn.Conv3d`](torch.nn.Conv3d.html#torch.nn.Conv3d "torch.nn.Conv3d")  when called on CUDA tensor
> * [`torch.nn.ConvTranspose1d`](torch.nn.ConvTranspose1d.html#torch.nn.ConvTranspose1d "torch.nn.ConvTranspose1d")  when called on CUDA tensor
> * [`torch.nn.ConvTranspose2d`](torch.nn.ConvTranspose2d.html#torch.nn.ConvTranspose2d "torch.nn.ConvTranspose2d")  when called on CUDA tensor
> * [`torch.nn.ConvTranspose3d`](torch.nn.ConvTranspose3d.html#torch.nn.ConvTranspose3d "torch.nn.ConvTranspose3d")  when called on CUDA tensor
> * [`torch.nn.ReplicationPad1d`](torch.nn.ReplicationPad1d.html#torch.nn.ReplicationPad1d "torch.nn.ReplicationPad1d")  when attempting to differentiate a CUDA tensor
> * [`torch.nn.ReplicationPad2d`](torch.nn.ReplicationPad2d.html#torch.nn.ReplicationPad2d "torch.nn.ReplicationPad2d")  when attempting to differentiate a CUDA tensor
> * [`torch.nn.ReplicationPad3d`](torch.nn.ReplicationPad3d.html#torch.nn.ReplicationPad3d "torch.nn.ReplicationPad3d")  when attempting to differentiate a CUDA tensor
> * [`torch.bmm()`](torch.bmm.html#torch.bmm "torch.bmm")  when called on sparse-dense CUDA tensors
> * `torch.Tensor.__getitem__()`  when attempting to differentiate a CPU tensor
> and the index is a list of tensors
> * [`torch.Tensor.index_put()`](torch.Tensor.index_put.html#torch.Tensor.index_put "torch.Tensor.index_put")  with `accumulate=False`
> * [`torch.Tensor.index_put()`](torch.Tensor.index_put.html#torch.Tensor.index_put "torch.Tensor.index_put")  with `accumulate=True`  when called on a CPU
> tensor
> * [`torch.Tensor.put_()`](torch.Tensor.put_.html#torch.Tensor.put_ "torch.Tensor.put_")  with `accumulate=True`  when called on a CPU
> tensor
> * [`torch.Tensor.scatter_add_()`](torch.Tensor.scatter_add_.html#torch.Tensor.scatter_add_ "torch.Tensor.scatter_add_")  when called on a CUDA tensor
> * [`torch.gather()`](torch.gather.html#torch.gather "torch.gather")  when called on a CUDA tensor that requires grad
> * [`torch.index_add()`](torch.index_add.html#torch.index_add "torch.index_add")  when called on CUDA tensor
> * [`torch.index_select()`](torch.index_select.html#torch.index_select "torch.index_select")  when attempting to differentiate a CUDA tensor
> * [`torch.repeat_interleave()`](torch.repeat_interleave.html#torch.repeat_interleave "torch.repeat_interleave")  when attempting to differentiate a CUDA tensor
> * [`torch.Tensor.index_copy()`](torch.Tensor.index_copy.html#torch.Tensor.index_copy "torch.Tensor.index_copy")  when called on a CPU or CUDA tensor
> * [`torch.Tensor.scatter()`](torch.Tensor.scatter.html#torch.Tensor.scatter "torch.Tensor.scatter")  when *src* type is Tensor and called on CUDA tensor
> * [`torch.Tensor.scatter_reduce()`](torch.Tensor.scatter_reduce.html#torch.Tensor.scatter_reduce "torch.Tensor.scatter_reduce")  when `reduce='sum'`  or `reduce='mean'`  and called on CUDA tensor

The following normally-nondeterministic operations will throw a [`RuntimeError`](https://docs.python.org/3/library/exceptions.html#RuntimeError "(in Python v3.13)")  when `mode=True`  : 

> * [`torch.nn.AvgPool3d`](torch.nn.AvgPool3d.html#torch.nn.AvgPool3d "torch.nn.AvgPool3d")  when attempting to differentiate a CUDA tensor
> * [`torch.nn.AdaptiveAvgPool2d`](torch.nn.AdaptiveAvgPool2d.html#torch.nn.AdaptiveAvgPool2d "torch.nn.AdaptiveAvgPool2d")  when attempting to differentiate a CUDA tensor
> * [`torch.nn.AdaptiveAvgPool3d`](torch.nn.AdaptiveAvgPool3d.html#torch.nn.AdaptiveAvgPool3d "torch.nn.AdaptiveAvgPool3d")  when attempting to differentiate a CUDA tensor
> * [`torch.nn.MaxPool3d`](torch.nn.MaxPool3d.html#torch.nn.MaxPool3d "torch.nn.MaxPool3d")  when attempting to differentiate a CUDA tensor
> * [`torch.nn.AdaptiveMaxPool2d`](torch.nn.AdaptiveMaxPool2d.html#torch.nn.AdaptiveMaxPool2d "torch.nn.AdaptiveMaxPool2d")  when attempting to differentiate a CUDA tensor
> * [`torch.nn.FractionalMaxPool2d`](torch.nn.FractionalMaxPool2d.html#torch.nn.FractionalMaxPool2d "torch.nn.FractionalMaxPool2d")  when attempting to differentiate a CUDA tensor
> * [`torch.nn.FractionalMaxPool3d`](torch.nn.FractionalMaxPool3d.html#torch.nn.FractionalMaxPool3d "torch.nn.FractionalMaxPool3d")  when attempting to differentiate a CUDA tensor
> * [`torch.nn.MaxUnpool1d`](torch.nn.MaxUnpool1d.html#torch.nn.MaxUnpool1d "torch.nn.MaxUnpool1d")
> * [`torch.nn.MaxUnpool2d`](torch.nn.MaxUnpool2d.html#torch.nn.MaxUnpool2d "torch.nn.MaxUnpool2d")
> * [`torch.nn.MaxUnpool3d`](torch.nn.MaxUnpool3d.html#torch.nn.MaxUnpool3d "torch.nn.MaxUnpool3d")
> * [`torch.nn.functional.interpolate()`](torch.nn.functional.interpolate.html#torch.nn.functional.interpolate "torch.nn.functional.interpolate")  when attempting to differentiate a CUDA tensor
> and one of the following modes is used:
> 
> + `linear`
> + `bilinear`
> + `bicubic`
> + `trilinear`
> * [`torch.nn.ReflectionPad1d`](torch.nn.ReflectionPad1d.html#torch.nn.ReflectionPad1d "torch.nn.ReflectionPad1d")  when attempting to differentiate a CUDA tensor
> * [`torch.nn.ReflectionPad2d`](torch.nn.ReflectionPad2d.html#torch.nn.ReflectionPad2d "torch.nn.ReflectionPad2d")  when attempting to differentiate a CUDA tensor
> * [`torch.nn.ReflectionPad3d`](torch.nn.ReflectionPad3d.html#torch.nn.ReflectionPad3d "torch.nn.ReflectionPad3d")  when attempting to differentiate a CUDA tensor
> * [`torch.nn.NLLLoss`](torch.nn.NLLLoss.html#torch.nn.NLLLoss "torch.nn.NLLLoss")  when called on a CUDA tensor
> * [`torch.nn.CTCLoss`](torch.nn.CTCLoss.html#torch.nn.CTCLoss "torch.nn.CTCLoss")  when attempting to differentiate a CUDA tensor
> * [`torch.nn.EmbeddingBag`](torch.nn.EmbeddingBag.html#torch.nn.EmbeddingBag "torch.nn.EmbeddingBag")  when attempting to differentiate a CUDA tensor when `mode='max'`
> * [`torch.Tensor.put_()`](torch.Tensor.put_.html#torch.Tensor.put_ "torch.Tensor.put_")  when `accumulate=False`
> * [`torch.Tensor.put_()`](torch.Tensor.put_.html#torch.Tensor.put_ "torch.Tensor.put_")  when `accumulate=True`  and called on a CUDA tensor
> * [`torch.histc()`](torch.histc.html#torch.histc "torch.histc")  when called on a CUDA tensor
> * [`torch.bincount()`](torch.bincount.html#torch.bincount "torch.bincount")  when called on a CUDA tensor and `weights`  tensor is given
> * [`torch.kthvalue()`](torch.kthvalue.html#torch.kthvalue "torch.kthvalue")  with called on a CUDA tensor
> * [`torch.median()`](torch.median.html#torch.median "torch.median")  with indices output when called on a CUDA tensor
> * [`torch.nn.functional.grid_sample()`](torch.nn.functional.grid_sample.html#torch.nn.functional.grid_sample "torch.nn.functional.grid_sample")  when attempting to differentiate a CUDA tensor
> * [`torch.cumsum()`](torch.cumsum.html#torch.cumsum "torch.cumsum")  when called on a CUDA tensor when dtype is floating point or complex
> * [`torch.Tensor.scatter_reduce()`](torch.Tensor.scatter_reduce.html#torch.Tensor.scatter_reduce "torch.Tensor.scatter_reduce")  when `reduce='prod'`  and called on CUDA tensor
> * [`torch.Tensor.resize_()`](torch.Tensor.resize_.html#torch.Tensor.resize_ "torch.Tensor.resize_")  when called with a quantized tensor

In addition, several operations fill uninitialized memory when this setting
is turned on and when [`torch.utils.deterministic.fill_uninitialized_memory`](../deterministic.html#torch.utils.deterministic.fill_uninitialized_memory "torch.utils.deterministic.fill_uninitialized_memory")  is turned on.
See the documentation for that attribute for more information. 

A handful of CUDA operations are nondeterministic if the CUDA version is
10.2 or greater, unless the environment variable `CUBLAS_WORKSPACE_CONFIG=:4096:8`  or `CUBLAS_WORKSPACE_CONFIG=:16:8`  is set. See the CUDA documentation for more
details: [https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility](https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility)  If one of these environment variable configurations is not set, a [`RuntimeError`](https://docs.python.org/3/library/exceptions.html#RuntimeError "(in Python v3.13)")  will be raised from these operations when called with CUDA tensors: 

> * [`torch.mm()`](torch.mm.html#torch.mm "torch.mm")
> * [`torch.mv()`](torch.mv.html#torch.mv "torch.mv")
> * [`torch.bmm()`](torch.bmm.html#torch.bmm "torch.bmm")

Note that deterministic operations tend to have worse performance than
nondeterministic operations. 

Note 

This flag does not detect or prevent nondeterministic behavior caused
by calling an inplace operation on a tensor with an internal memory
overlap or by giving such a tensor as the `out`  argument for an
operation. In these cases, multiple writes of different data may target
a single memory location, and the order of writes is not guaranteed.

Parameters
: **mode** ( [`bool`](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – If True, makes potentially nondeterministic
operations switch to a deterministic algorithm or throw a runtime
error. If False, allows nondeterministic operations.

Keyword Arguments
: **warn_only** ( [`bool`](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  , optional) – If True, operations that do not
have a deterministic implementation will throw a warning instead of
an error. Default: `False`

Example: 

```
>>> torch.use_deterministic_algorithms(True)

# Forward mode nondeterministic error
>>> torch.randn(10, device='cuda').kthvalue(1)
...
RuntimeError: kthvalue CUDA does not have a deterministic implementation...

# Backward mode nondeterministic error
>>> torch.nn.AvgPool3d(1)(torch.randn(3, 4, 5, 6, requires_grad=True).cuda()).sum().backward()
...
RuntimeError: avg_pool3d_backward_cuda does not have a deterministic implementation...

```

