SubclassSymbolicContext 
==================================================================================

*class* torch.fx.experimental.symbolic_shapes. SubclassSymbolicContext ( *dynamic_sizes*  , *dynamic_strides = None*  , *constraint_sizes = None*  , *constraint_strides = None*  , *specialize_on = None*  , *view_base_context = None*  , *tensor_source = None*  , *shape_env_to_source_to_symbol_cache = None*  , *inner_contexts = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/experimental/symbolic_shapes.py#L2143) 
:   The correct symbolic context for a given inner tensor of a traceable tensor subclass
may differ from that of the outer symbolic context. This structure allows for this
flexibility, with inner symbolic contexts mapped via attr -> symbolic context.

