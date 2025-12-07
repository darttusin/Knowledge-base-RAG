StatefulSymbolicContext 
==================================================================================

*class* torch.fx.experimental.symbolic_shapes. StatefulSymbolicContext ( *dynamic_sizes*  , *dynamic_strides = None*  , *constraint_sizes = None*  , *constraint_strides = None*  , *specialize_on = None*  , *view_base_context = None*  , *tensor_source = None*  , *shape_env_to_source_to_symbol_cache = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/experimental/symbolic_shapes.py#L2108) 
:   Create symbols in `create_symbolic_sizes_strides_storage_offset`  via
a symbolic_context determination as given by a cache of Source:Symbol. A cache hit
will reuse a stored symbol, and a cache miss will write to this cache. 

This behaves like StatelessSymbolicContext, except the cache supersedes the
other values - dynamic_sizes and constraint_sizes will not be read if we cache
hit. 

It is the cache owner’s responsibility to maintain the lifecycle of the cache
with respect to different shape_envs, clearing, etc.

