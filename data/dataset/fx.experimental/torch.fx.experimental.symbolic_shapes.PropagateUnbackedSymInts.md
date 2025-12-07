PropagateUnbackedSymInts 
====================================================================================

*class* torch.fx.experimental.symbolic_shapes. PropagateUnbackedSymInts ( *module*  , *garbage_collect_values = True*  , *graph = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/experimental/symbolic_shapes.py#L7851) 
:   boxed_run ( *args_list* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/interpreter.py#L202) 
:   Run *module* via interpretation and return the result. This uses the “boxed”
calling convention, where you pass a list of arguments, which will be cleared
by the interpreter. This ensures that input tensors are promptly deallocated. 

Note 

Backwards-compatibility for this API is guaranteed.

call_function ( *target*  , *args*  , *kwargs* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/interpreter.py#L302) 
:   Execute a `call_function`  node and return the result. 

Parameters
:   * **target** ( *Target*  ) – The call target for this node. See [Node](https://localhost:8000/docs/main/fx.html#torch.fx.Node)  for
details on semantics
* **args** ( *Tuple*  ) – Tuple of positional args for this invocation
* **kwargs** ( *Dict*  ) – Dict of keyword arguments for this invocation

Return type
:   [*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.13)")

Return
:   Any: The value returned by the function invocation

Note 

Backwards-compatibility for this API is guaranteed.

call_method ( *target*  , *args*  , *kwargs* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/interpreter.py#L324) 
:   Execute a `call_method`  node and return the result. 

Parameters
:   * **target** ( *Target*  ) – The call target for this node. See [Node](https://localhost:8000/docs/main/fx.html#torch.fx.Node)  for
details on semantics
* **args** ( *Tuple*  ) – Tuple of positional args for this invocation
* **kwargs** ( *Dict*  ) – Dict of keyword arguments for this invocation

Return type
:   [*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.13)")

Return
:   Any: The value returned by the method invocation

Note 

Backwards-compatibility for this API is guaranteed.

call_module ( *target*  , *args*  , *kwargs* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/interpreter.py#L348) 
:   Execute a `call_module`  node and return the result. 

Parameters
:   * **target** ( *Target*  ) – The call target for this node. See [Node](https://localhost:8000/docs/main/fx.html#torch.fx.Node)  for
details on semantics
* **args** ( *Tuple*  ) – Tuple of positional args for this invocation
* **kwargs** ( *Dict*  ) – Dict of keyword arguments for this invocation

Return type
:   [*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.13)")

Return
:   Any: The value returned by the module invocation

Note 

Backwards-compatibility for this API is guaranteed.

fetch_args_kwargs_from_env ( *n* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/interpreter.py#L415) 
:   Fetch the concrete values of `args`  and `kwargs`  of node `n`  from the current execution environment. 

Parameters
: **n** ( [*Node*](../fx.html#torch.fx.Node "torch.fx.Node")  ) – The node for which `args`  and `kwargs`  should be fetched.

Returns
:   `args`  and `kwargs`  with concrete values for `n`  .

Return type
:   Tuple[Tuple, Dict]

Note 

Backwards-compatibility for this API is guaranteed.

fetch_attr ( *target* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/interpreter.py#L394) 
:   Fetch an attribute from the `Module`  hierarchy of `self.module`  . 

Parameters
: **target** ( [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")  ) – The fully-qualified name of the attribute to fetch

Returns
:   The value of the attribute.

Return type
:   Any

Note 

Backwards-compatibility for this API is guaranteed.

get_attr ( *target*  , *args*  , *kwargs* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/interpreter.py#L281) 
:   Execute a `get_attr`  node. Will retrieve an attribute
value from the `Module`  hierarchy of `self.module`  . 

Parameters
:   * **target** ( *Target*  ) – The call target for this node. See [Node](https://localhost:8000/docs/main/fx.html#torch.fx.Node)  for
details on semantics
* **args** ( *Tuple*  ) – Tuple of positional args for this invocation
* **kwargs** ( *Dict*  ) – Dict of keyword arguments for this invocation

Returns
:   The value of the attribute that was retrieved

Return type
:   Any

Note 

Backwards-compatibility for this API is guaranteed.

map_nodes_to_values ( *args*  , *n* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/interpreter.py#L433) 
:   Recursively descend through `args`  and look up the concrete value
for each `Node`  in the current execution environment. 

Parameters
:   * **args** ( *Argument*  ) – Data structure within which to look up concrete values
* **n** ( [*Node*](../fx.html#torch.fx.Node "torch.fx.Node")  ) – Node to which `args`  belongs. This is only used for error reporting.

Return type
:   [*Optional*](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)")  [ [*Union*](https://docs.python.org/3/library/typing.html#typing.Union "(in Python v3.13)")  [ [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)")  [ [*Union*](https://docs.python.org/3/library/typing.html#typing.Union "(in Python v3.13)")  [ [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)")  [ForwardRef(‘Argument’), …], [collections.abc.Sequence](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence "(in Python v3.13)")  [ForwardRef(‘Argument’)], [collections.abc.Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping "(in Python v3.13)")  [ [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")  , ForwardRef(‘Argument’)], [slice](https://docs.python.org/3/library/functions.html#slice "(in Python v3.13)")  , [range](https://docs.python.org/3/library/stdtypes.html#range "(in Python v3.13)")  , [torch.fx.node.Node](../fx.html#torch.fx.Node "torch.fx.node.Node")  , [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")  , [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  , [float](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  , [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  , [complex](https://docs.python.org/3/library/functions.html#complex "(in Python v3.13)")  , [torch.dtype](../tensor_attributes.html#torch.dtype "torch.dtype")  , [torch.Tensor](../tensors.html#torch.Tensor "torch.Tensor")  , [torch.device](../tensor_attributes.html#torch.device "torch.device")  , [torch.memory_format](../tensor_attributes.html#torch.memory_format "torch.memory_format")  , [torch.layout](../tensor_attributes.html#torch.layout "torch.layout")  , torch._ops.OpOverload, [torch.SymInt](../torch.html#torch.SymInt "torch.SymInt")  , [torch.SymBool](../torch.html#torch.SymBool "torch.SymBool")  , [torch.SymFloat](../torch.html#torch.SymFloat "torch.SymFloat")  , NoneType], …], [*Sequence*](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence "(in Python v3.13)")  [ [*Optional*](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)")  [ [*Union*](https://docs.python.org/3/library/typing.html#typing.Union "(in Python v3.13)")  [ [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)")  [ForwardRef(‘Argument’), …], [*Sequence*](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence "(in Python v3.13)")  [Argument], [*Mapping*](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping "(in Python v3.13)")  [ [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")  , Argument], [slice](https://docs.python.org/3/library/functions.html#slice "(in Python v3.13)")  , [range](https://docs.python.org/3/library/stdtypes.html#range "(in Python v3.13)")  , [*Node*](../fx.html#torch.fx.Node "torch.fx.node.Node")  , [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")  , [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  , [float](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  , [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  , [complex](https://docs.python.org/3/library/functions.html#complex "(in Python v3.13)")  , [*dtype*](../tensor_attributes.html#torch.dtype "torch.dtype")  , [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  , [*device*](../tensor_attributes.html#torch.device "torch.device")  , [*memory_format*](../tensor_attributes.html#torch.memory_format "torch.memory_format")  , [*layout*](../tensor_attributes.html#torch.layout "torch.layout")  , *OpOverload*  , [*SymInt*](../torch.html#torch.SymInt "torch.SymInt")  , [*SymBool*](../torch.html#torch.SymBool "torch.SymBool")  , [*SymFloat*](../torch.html#torch.SymFloat "torch.SymFloat")  ]]], [*Mapping*](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping "(in Python v3.13)")  [ [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")  , [*Optional*](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)")  [ [*Union*](https://docs.python.org/3/library/typing.html#typing.Union "(in Python v3.13)")  [ [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)")  [ForwardRef(‘Argument’), …], [*Sequence*](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence "(in Python v3.13)")  [Argument], [*Mapping*](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping "(in Python v3.13)")  [ [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")  , Argument], [slice](https://docs.python.org/3/library/functions.html#slice "(in Python v3.13)")  , [range](https://docs.python.org/3/library/stdtypes.html#range "(in Python v3.13)")  , [*Node*](../fx.html#torch.fx.Node "torch.fx.node.Node")  , [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")  , [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  , [float](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  , [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  , [complex](https://docs.python.org/3/library/functions.html#complex "(in Python v3.13)")  , [*dtype*](../tensor_attributes.html#torch.dtype "torch.dtype")  , [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  , [*device*](../tensor_attributes.html#torch.device "torch.device")  , [*memory_format*](../tensor_attributes.html#torch.memory_format "torch.memory_format")  , [*layout*](../tensor_attributes.html#torch.layout "torch.layout")  , *OpOverload*  , [*SymInt*](../torch.html#torch.SymInt "torch.SymInt")  , [*SymBool*](../torch.html#torch.SymBool "torch.SymBool")  , [*SymFloat*](../torch.html#torch.SymFloat "torch.SymFloat")  ]]], [slice](https://docs.python.org/3/library/functions.html#slice "(in Python v3.13)")  , [range](https://docs.python.org/3/library/stdtypes.html#range "(in Python v3.13)")  , [*Node*](../fx.html#torch.fx.Node "torch.fx.node.Node")  , [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")  , [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  , [float](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  , [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  , [complex](https://docs.python.org/3/library/functions.html#complex "(in Python v3.13)")  , [*dtype*](../tensor_attributes.html#torch.dtype "torch.dtype")  , [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  , [*device*](../tensor_attributes.html#torch.device "torch.device")  , [*memory_format*](../tensor_attributes.html#torch.memory_format "torch.memory_format")  , [*layout*](../tensor_attributes.html#torch.layout "torch.layout")  , *OpOverload*  , [*SymInt*](../torch.html#torch.SymInt "torch.SymInt")  , [*SymBool*](../torch.html#torch.SymBool "torch.SymBool")  , [*SymFloat*](../torch.html#torch.SymFloat "torch.SymFloat")  ]]

Note 

Backwards-compatibility for this API is guaranteed.

output ( *target*  , *args*  , *kwargs* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/interpreter.py#L373) 
:   Execute an `output`  node. This really just retrieves
the value referenced by the `output`  node and returns it. 

Parameters
:   * **target** ( *Target*  ) – The call target for this node. See [Node](https://localhost:8000/docs/main/fx.html#torch.fx.Node)  for
details on semantics
* **args** ( *Tuple*  ) – Tuple of positional args for this invocation
* **kwargs** ( *Dict*  ) – Dict of keyword arguments for this invocation

Returns
:   The return value referenced by the output node

Return type
:   Any

Note 

Backwards-compatibility for this API is guaranteed.

placeholder ( *target*  , *args*  , *kwargs* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/interpreter.py#L245) 
:   Execute a `placeholder`  node. Note that this is stateful: `Interpreter`  maintains an internal iterator over
arguments passed to `run`  and this method returns
next() on that iterator. 

Parameters
:   * **target** ( *Target*  ) – The call target for this node. See [Node](https://localhost:8000/docs/main/fx.html#torch.fx.Node)  for
details on semantics
* **args** ( *Tuple*  ) – Tuple of positional args for this invocation
* **kwargs** ( *Dict*  ) – Dict of keyword arguments for this invocation

Returns
:   The argument value that was retrieved.

Return type
:   Any

Note 

Backwards-compatibility for this API is guaranteed.

run ( ** args*  , *initial_env = None*  , *enable_io_processing = True* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/interpreter.py#L123) 
:   Run *module* via interpretation and return the result. 

Parameters
:   * ***args** – The arguments to the Module to run, in positional order
* **initial_env** ( *Optional* *[* *Dict* *[* [*Node*](../fx.html#torch.fx.Node "torch.fx.Node") *,* *Any* *]* *]*  ) – An optional starting environment for execution.
This is a dict mapping *Node* to any value. This can be used, for example, to
pre-populate results for certain *Nodes* so as to do only partial evaluation within
the interpreter.
* **enable_io_processing** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ) – If true, we process the inputs and outputs with graph’s process_inputs and
process_outputs function first before using them.

Returns
:   The value returned from executing the Module

Return type
:   Any

Note 

Backwards-compatibility for this API is guaranteed.

run_node ( *n* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/fx/experimental/symbolic_shapes.py#L7852) 
:   Run an FX node, propagating unbacked Symbol bindings to the new fake tensor 

Return type
:   [*Union*](https://docs.python.org/3/library/typing.html#typing.Union "(in Python v3.13)")  [ [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")  , [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)")  [ [torch.Tensor](../tensors.html#torch.Tensor "torch.Tensor")  , …]]

