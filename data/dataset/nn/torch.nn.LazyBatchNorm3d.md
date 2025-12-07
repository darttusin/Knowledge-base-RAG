LazyBatchNorm3d 
==================================================================

*class* torch.nn. LazyBatchNorm3d ( *eps = 1e-05*  , *momentum = 0.1*  , *affine = True*  , *track_running_stats = True*  , *device = None*  , *dtype = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/batchnorm.py#L566) 
:   A [`torch.nn.BatchNorm3d`](torch.nn.BatchNorm3d.html#torch.nn.BatchNorm3d "torch.nn.BatchNorm3d")  module with lazy initialization. 

Lazy initialization is done for the `num_features`  argument of the [`BatchNorm3d`](torch.nn.BatchNorm3d.html#torch.nn.BatchNorm3d "torch.nn.BatchNorm3d")  that is inferred
from the `input.size(1)`  .
The attributes that will be lazily initialized are *weight* , *bias* , *running_mean* and *running_var* . 

Check the [`torch.nn.modules.lazy.LazyModuleMixin`](torch.nn.modules.lazy.LazyModuleMixin.html#torch.nn.modules.lazy.LazyModuleMixin "torch.nn.modules.lazy.LazyModuleMixin")  for further documentation
on lazy modules and their limitations. 

Parameters
:   * **eps** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ) – a value added to the denominator for numerical stability.
Default: 1e-5
* **momentum** ( [*Optional*](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)") *[* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *]*  ) – the value used for the running_mean and running_var
computation. Can be set to `None`  for cumulative moving average
(i.e. simple average). Default: 0.1
* **affine** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – a boolean value that when set to `True`  , this module has
learnable affine parameters. Default: `True`
* **track_running_stats** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – a boolean value that when set to `True`  , this
module tracks the running mean and variance, and when set to `False`  ,
this module does not track such statistics, and initializes statistics
buffers `running_mean`  and `running_var`  as `None`  .
When these buffers are `None`  , this module always uses batch statistics.
in both training and eval modes. Default: `True`

cls_to_become [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/batchnorm.py#L489) 
:   alias of [`BatchNorm3d`](torch.nn.BatchNorm3d.html#torch.nn.BatchNorm3d "torch.nn.modules.batchnorm.BatchNorm3d")

