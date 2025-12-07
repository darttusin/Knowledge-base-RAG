device_of 
=======================================================

*class* torch.xpu. device_of ( *obj* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/xpu/__init__.py#L180) 
:   Context-manager that changes the current device to that of given object. 

You can use both tensors and storages as arguments. If a given object is
not allocated on a XPU, this is a no-op. 

Parameters
: **obj** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *or* *Storage*  ) – object allocated on the selected device.

