torch.Tensor.storage 
============================================================================

Tensor. storage ( ) → [torch.TypedStorage](../storage.html#torch.TypedStorage "torch.TypedStorage") [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_tensor.py#L288) 
:   Returns the underlying [`TypedStorage`](../storage.html#torch.TypedStorage "torch.TypedStorage")  . 

Warning 

[`TypedStorage`](../storage.html#torch.TypedStorage "torch.TypedStorage")  is deprecated. It will be removed in the future, and [`UntypedStorage`](../storage.html#torch.UntypedStorage "torch.UntypedStorage")  will be the only storage class. To access the [`UntypedStorage`](../storage.html#torch.UntypedStorage "torch.UntypedStorage")  directly, use [`Tensor.untyped_storage()`](torch.Tensor.untyped_storage.html#torch.Tensor.untyped_storage "torch.Tensor.untyped_storage")  .

