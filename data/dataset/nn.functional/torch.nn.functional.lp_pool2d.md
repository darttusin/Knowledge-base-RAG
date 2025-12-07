torch.nn.functional.lp_pool2d 
===============================================================================================

torch.nn.functional. lp_pool2d ( *input*  , *norm_type*  , *kernel_size*  , *stride = None*  , *ceil_mode = False* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/functional.py#L1111) 
:   Apply a 2D power-average pooling over an input signal composed of several input planes. 

If the sum of all inputs to the power of *p* is
zero, the gradient is set to zero as well. 

See [`LPPool2d`](torch.nn.LPPool2d.html#torch.nn.LPPool2d "torch.nn.LPPool2d")  for details. 

Return type
:   [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")

