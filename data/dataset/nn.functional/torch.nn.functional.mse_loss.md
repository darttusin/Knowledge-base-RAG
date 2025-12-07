torch.nn.functional.mse_loss 
=============================================================================================

torch.nn.functional. mse_loss ( *input*  , *target*  , *size_average = None*  , *reduce = None*  , *reduction = 'mean'*  , *weight = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/functional.py#L3819) 
:   Compute the element-wise mean squared error, with optional weighting. 

See [`MSELoss`](torch.nn.MSELoss.html#torch.nn.MSELoss "torch.nn.MSELoss")  for details. 

Parameters
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – Predicted values.
* **target** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – Ground truth values.
* **size_average** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – Deprecated (see `reduction`  ).
* **reduce** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – Deprecated (see `reduction`  ).
* **reduction** ( [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *,* *optional*  ) – Specifies the reduction to apply to the output:
‘none’ | ‘mean’ | ‘sum’. ‘mean’: the mean of the output is taken.
‘sum’: the output will be summed. ‘none’: no reduction will be applied.
Default: ‘mean’.
* **weight** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *,* *optional*  ) – Weights for each sample. Default: None.

Returns
:   Mean Squared Error loss (optionally weighted).

Return type
:   [Tensor](../tensors.html#torch.Tensor "torch.Tensor")

