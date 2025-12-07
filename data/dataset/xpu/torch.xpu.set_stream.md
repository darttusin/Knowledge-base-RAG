torch.xpu.set_stream 
=============================================================================

torch.xpu. set_stream ( *stream* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/xpu/__init__.py#L349) 
:   Set the current stream.This is a wrapper API to set the stream.
:   Usage of this function is discouraged in favor of the `stream`  context manager.

Parameters
: **stream** ( [*Stream*](torch.xpu.Stream.html#torch.xpu.Stream "torch.xpu.Stream")  ) – selected stream. This function is a no-op
if this argument is `None`  .

