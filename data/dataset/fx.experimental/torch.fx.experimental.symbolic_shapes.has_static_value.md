torch.fx.experimental.symbolic_shapes.has_static_value 
===================================================================================================================================================

torch.fx.experimental.symbolic_shapes. has_static_value ( *a* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/experimental/symbolic_shapes.py#L440) 
:   User-code friendly utility to check if a value is static or dynamic.
Returns true if given a constant, or a symbolic expression with a fixed value. 

Parameters
: **a** ( *Union* *[* [*SymBool*](../torch.html#torch.SymBool "torch.SymBool") *,* [*SymFloat*](../torch.html#torch.SymFloat "torch.SymFloat") *,* [*SymInt*](../torch.html#torch.SymInt "torch.SymInt") *,* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *,* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *]*  ) – Object to test

Return type
:   [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")

