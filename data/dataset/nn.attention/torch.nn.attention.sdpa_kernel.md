torch.nn.attention.sdpa_kernel 
=================================================================================================

torch.nn.attention. sdpa_kernel ( *backends*  , *set_priority = False* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/attention/__init__.py#L104) 
:   Context manager to select which backend to use for scaled dot product attention. 

Warning 

This function is beta and subject to change.

Parameters
:   * **backends** ( *Union* *[* *List* *[* [*SDPBackend*](torch.nn.attention.SDPBackend.html#torch.nn.attention.SDPBackend "torch.nn.attention.SDPBackend") *]* *,* [*SDPBackend*](torch.nn.attention.SDPBackend.html#torch.nn.attention.SDPBackend "torch.nn.attention.SDPBackend") *]*  ) – A backend or list of backends for scaled dot product attention.
* **set_priority_order** ( *python:bool=False*  ) – Whether the ordering of the backends is interpreted as their priority order.

Example: 

```
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend, sdpa_kernel

# Only enable flash attention backend
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    scaled_dot_product_attention(...)

# Enable the Math or Efficient attention backends
with sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
    scaled_dot_product_attention(...)

```

This context manager can be used to select which backend to use for scaled dot product attention.
Upon exiting the context manager, the previous state of the flags will be restored, enabling all backends.

