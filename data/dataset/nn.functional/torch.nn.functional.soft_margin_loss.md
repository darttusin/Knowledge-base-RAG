torch.nn.functional.soft_margin_loss 
==============================================================================================================

torch.nn.functional. soft_margin_loss ( *input*  , *target*  , *size_average = None*  , *reduce = None*  , *reduction = 'mean'* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/functional.py#L4030) 
:   Compute the soft margin loss. 

See [`SoftMarginLoss`](torch.nn.SoftMarginLoss.html#torch.nn.SoftMarginLoss "torch.nn.SoftMarginLoss")  for details. 

Parameters
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – Predicted values.
* **target** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – Ground truth values.
* **size_average** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – Deprecated (see `reduction`  ).
* **reduce** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – Deprecated (see `reduction`  ).
* **reduction** ( [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *,* *optional*  ) – Specifies the reduction to apply to the output:
‘none’ | ‘mean’ | ‘sum’. ‘mean’: the mean of the output is taken.
‘sum’: the output will be summed. ‘none’: no reduction will be applied.
Default: ‘mean’.

Returns
:   Soft margin loss.

Return type
:   [Tensor](../tensors.html#torch.Tensor "torch.Tensor")

