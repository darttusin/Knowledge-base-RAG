torch.Tensor.sparse_resize_and_clear_ 
=================================================================================================================

Tensor. sparse_resize_and_clear_ ( *size*  , *sparse_dim*  , *dense_dim* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Removes all specified elements from a [sparse tensor](../sparse.html#sparse-docs) `self`  and resizes `self`  to the desired
size and the number of sparse and dense dimensions. 

Parameters
:   * **size** ( [*torch.Size*](../size.html#torch.Size "torch.Size")  ) – the desired size.
* **sparse_dim** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – the number of sparse dimensions
* **dense_dim** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – the number of dense dimensions

