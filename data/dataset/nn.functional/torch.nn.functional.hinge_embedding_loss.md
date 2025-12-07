torch.nn.functional.hinge_embedding_loss 
======================================================================================================================

torch.nn.functional. hinge_embedding_loss ( *input*  , *target*  , *margin = 1.0*  , *size_average = None*  , *reduce = None*  , *reduction = 'mean'* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/functional.py#L3945) 
:   Compute the hinge embedding loss. 

See [`HingeEmbeddingLoss`](torch.nn.HingeEmbeddingLoss.html#torch.nn.HingeEmbeddingLoss "torch.nn.HingeEmbeddingLoss")  for details. 

Parameters
:   * **input** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – Predicted values.
* **target** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  ) – Ground truth values.
* **margin** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *,* *optional*  ) – Margin for hinge loss. Has a default value of 1.
* **size_average** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – Deprecated (see `reduction`  ).
* **reduce** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – Deprecated (see `reduction`  ).
* **reduction** ( [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *,* *optional*  ) – Specifies the reduction to apply to the output:
‘none’ | ‘mean’ | ‘sum’. ‘mean’: the mean of the output is taken.
‘sum’: the output will be summed. ‘none’: no reduction will be applied.
Default: ‘mean’.

Returns
:   Hinge embedding loss.

Return type
:   [Tensor](../tensors.html#torch.Tensor "torch.Tensor")

