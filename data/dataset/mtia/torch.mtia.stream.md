torch.mtia.stream 
======================================================================

torch.mtia. stream ( *stream* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/mtia/__init__.py#L338) 
:   Wrap around the Context-manager StreamContext that selects a given stream. 

Parameters
: **stream** ( [*Stream*](torch.mtia.Stream.html#torch.mtia.Stream "torch.mtia.Stream")  ) – selected stream. This manager is a no-op if it’s `None`  .

Return type
:   [*StreamContext*](torch.mtia.StreamContext.html#torch.mtia.StreamContext "torch.mtia.StreamContext")

Note 

In eager mode stream is of type Stream class while in JIT it doesn’t support torch.mtia.stream

