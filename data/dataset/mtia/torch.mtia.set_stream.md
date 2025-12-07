torch.mtia.set_stream 
===============================================================================

torch.mtia. set_stream ( *stream* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/mtia/__init__.py#L223) 
:   Set the current stream.This is a wrapper API to set the stream.
:   Usage of this function is discouraged in favor of the `stream`  context manager.

Parameters
: **stream** ( [*Stream*](torch.mtia.Stream.html#torch.mtia.Stream "torch.mtia.Stream")  ) – selected stream. This function is a no-op
if this argument is `None`  .

