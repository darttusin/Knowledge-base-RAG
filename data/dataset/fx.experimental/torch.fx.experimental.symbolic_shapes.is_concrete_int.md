torch.fx.experimental.symbolic_shapes.is_concrete_int 
=================================================================================================================================================

torch.fx.experimental.symbolic_shapes. is_concrete_int ( *a* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/experimental/symbolic_shapes.py#L379) 
:   Utility to check if underlying object
in SymInt is concrete value. Also returns
true if integer is passed in. 

Parameters
: **a** ( [*SymInt*](../torch.html#torch.SymInt "torch.SymInt") *or* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – Object to test if it int

Return type
:   [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")

