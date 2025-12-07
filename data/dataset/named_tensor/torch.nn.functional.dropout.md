torch.nn.functional.dropout 
==========================================================================================

torch.nn.functional. dropout ( *input*  , *p = 0.5*  , *training = True*  , *inplace = False* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/functional.py#L1398) 
:   During training, randomly zeroes some elements of the input tensor with probability `p`  . 

Uses samples from a Bernoulli distribution. 

See [`Dropout`](torch.nn.Dropout.html#torch.nn.Dropout "torch.nn.Dropout")  for details. 

Parameters
:   * **p** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  ) – probability of an element to be zeroed. Default: 0.5
* **training** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – apply dropout if is `True`  . Default: `True`
* **inplace** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – If set to `True`  , will do this operation in-place. Default: `False`

Return type
:   [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")

