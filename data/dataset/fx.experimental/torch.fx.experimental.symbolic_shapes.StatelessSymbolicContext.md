StatelessSymbolicContext 
====================================================================================

*class* torch.fx.experimental.symbolic_shapes. StatelessSymbolicContext ( *dynamic_sizes*  , *dynamic_strides = None*  , *constraint_sizes = None*  , *constraint_strides = None*  , *specialize_on = None*  , *view_base_context = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/experimental/symbolic_shapes.py#L2040) 
:   Create symbols in `create_symbolic_sizes_strides_storage_offset`  via
a symbolic_context determination as given by `DimDynamic`  and `DimConstraint`  .
This will cause fresh symbols to be allocated

