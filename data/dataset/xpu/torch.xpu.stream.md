torch.xpu.stream 
====================================================================

torch.xpu. stream ( *stream* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/xpu/__init__.py#L326) 
:   Wrap around the Context-manager StreamContext that selects a given stream. 

Parameters
: **stream** ( [*Stream*](torch.xpu.Stream.html#torch.xpu.Stream "torch.xpu.Stream")  ) – selected stream. This manager is a no-op if it’s `None`  .

Return type
:   [*StreamContext*](torch.xpu.StreamContext.html#torch.xpu.StreamContext "torch.xpu.StreamContext")

