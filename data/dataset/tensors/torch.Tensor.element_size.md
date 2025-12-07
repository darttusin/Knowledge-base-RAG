torch.Tensor.element_size 
=======================================================================================

Tensor. element_size ( ) → [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") 
:   Returns the size in bytes of an individual element. 

Example: 

```
>>> torch.tensor([]).element_size()
4
>>> torch.tensor([], dtype=torch.uint8).element_size()
1

```

