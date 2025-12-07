torch.cuda.set_stream 
===============================================================================

torch.cuda. set_stream ( *stream* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/cuda/__init__.py#L720) 
:   Set the current stream.This is a wrapper API to set the stream.
:   Usage of this function is discouraged in favor of the `stream`  context manager.

Parameters
: **stream** ( [*Stream*](torch.cuda.Stream.html#torch.cuda.Stream "torch.cuda.Stream")  ) – selected stream. This function is a no-op
if this argument is `None`  .

