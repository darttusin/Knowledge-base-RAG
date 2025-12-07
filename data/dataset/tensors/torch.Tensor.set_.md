torch.Tensor.set_ 
======================================================================

Tensor. set_ ( *source = None*  , *storage_offset = 0*  , *size = None*  , *stride = None* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Sets the underlying storage, size, and strides. If `source`  is a tensor, `self`  tensor will share the same storage and have the same size and
strides as `source`  . Changes to elements in one tensor will be reflected
in the other. 

If `source`  is a `Storage`  , the method sets the underlying
storage, offset, size, and stride. 

Parameters
:   * **source** ( [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor") *or* *Storage*  ) – the tensor or storage to use
* **storage_offset** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* *optional*  ) – the offset in the storage
* **size** ( [*torch.Size*](../size.html#torch.Size "torch.Size") *,* *optional*  ) – the desired size. Defaults to the size of the source.
* **stride** ( [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)") *,* *optional*  ) – the desired stride. Defaults to C-contiguous strides.

