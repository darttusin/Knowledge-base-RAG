torch.accelerator.set_device_idx 
======================================================================================================

torch.accelerator. set_device_idx ( *device*  , */* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/accelerator/__init__.py#L123) 
:   Set the current device index to a given device. 

Parameters
: **device** ( [`torch.device`](../tensor_attributes.html#torch.device "torch.device")  , str, int) – a given device that must match the current [accelerator](../torch.html#accelerators)  device type.

Note 

This function is a no-op if this device index is negative.

