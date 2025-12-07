StreamContext 
==============================================================

*class* torch.mtia. StreamContext ( *stream* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/mtia/__init__.py#L280) 
:   Context-manager that selects a given stream. 

All MTIA kernels queued within its context will be enqueued on a selected
stream. 

Parameters
: **Stream** ( [*Stream*](torch.mtia.Stream.html#torch.mtia.Stream "torch.mtia.Stream")  ) – selected stream. This manager is a no-op if it’s `None`  .

Note 

Streams are per-device.

