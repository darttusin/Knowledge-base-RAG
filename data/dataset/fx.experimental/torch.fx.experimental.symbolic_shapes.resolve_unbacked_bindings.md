torch.fx.experimental.symbolic_shapes.resolve_unbacked_bindings 
=====================================================================================================================================================================

torch.fx.experimental.symbolic_shapes. resolve_unbacked_bindings ( *shape_env*  , *bindings* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/experimental/symbolic_shapes.py#L508) 
:   When we do fake tensor prop, we oftentimes will allocate new unbacked symints.
We then run proxy tensor mode, which populates node.meta[“unbacked_bindings”]
with these new symints. To ensure consistency we use PropagateUnbackedSymInts
to rename unbacked bindings to their old ones. But all of the node metas are
still using the old bindings from before the renaming. This function helps to
post facto apply any renamings discovered in the PropogateUnbackedSymInts pass. 

Return type
:   [*Optional*](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)")  [ [dict](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.13)")  [sympy.core.symbol.Symbol, [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)")  [torch.utils._pytree.KeyEntry, …]]]

