torch.fx.experimental.proxy_tensor.make_fx 
==========================================================================================================================

torch.fx.experimental.proxy_tensor. make_fx ( *f*  , *decomposition_table = None*  , *tracing_mode = 'real'*  , *_allow_non_fake_inputs = False*  , *** , *pre_dispatch = False*  , *record_module_stack = False*  , *_allow_fake_constant = False*  , *_error_on_data_dependent_ops = True*  , *stack_trace = False* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/experimental/proxy_tensor.py#L2281) 
:   Given a function f, return a new function which when executed with valid
arguments to f, returns an FX GraphModule representing the set of operations that
were executed during the course of execution. 

If stack_trace is True, the stack_trace will be preserved on node.meta[“stack_trace”] 

Return type
:   Callable[…, [GraphModule](../fx.html#torch.fx.GraphModule "torch.fx.GraphModule")  ]

