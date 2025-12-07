SDPBackend 
========================================================

*class* torch.nn.attention. SDPBackend 
:   An enum-like class that contains the different backends for scaled dot product attention.
This backend class is designed to be used with the sdpa_kernel context manager. 

The following Enums are available:
:   * ERROR: An error occurred when trying to determine the backend.
* MATH: The math backend for scaled dot product attention.
* FLASH_ATTENTION: The flash attention backend for scaled dot product attention.
* EFFICIENT_ATTENTION: The efficient attention backend for scaled dot product attention.
* CUDNN_ATTENTION: The cuDNN backend for scaled dot product attention.

See [`torch.nn.attention.sdpa_kernel()`](torch.nn.attention.sdpa_kernel.html#torch.nn.attention.sdpa_kernel "torch.nn.attention.sdpa_kernel")  for more details. 

Warning 

This class is in beta and subject to change.

*property* name 
:

