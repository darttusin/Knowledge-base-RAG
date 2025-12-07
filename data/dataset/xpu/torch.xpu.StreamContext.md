StreamContext 
==============================================================

*class* torch.xpu. StreamContext ( *stream* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/xpu/__init__.py#L284) 
:   Context-manager that selects a given stream. 

All XPU kernels queued within its context will be enqueued on a selected
stream. 

Parameters
: **Stream** ( [*Stream*](torch.xpu.Stream.html#torch.xpu.Stream "torch.xpu.Stream")  ) – selected stream. This manager is a no-op if it’s `None`  .

Note 

Streams are per-device.

