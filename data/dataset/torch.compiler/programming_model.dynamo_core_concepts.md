Dynamo Core Concepts 
============================================================================

**Summary:** 

* Dynamo, `torch.compile`  ’s frontend, performs **tracing** to capture the semantics of a Python function
(and its nested function calls) into a linear sequence of operations (the “(FX) graph”),
residual bytecode, and “guards” (a list of conditions under which the graph and bytecode are valid).
* Unsupported Python features lead to **graph breaks** , where Dynamo compiles a partial graph acquired from tracing,
then runs the unsupported code, then resumes tracing.
* Graph breaks may lead to slowness in torch.compile and prevent backend optimization opportunities.
If you’re not seeing the performance you expect, then check for graph breaks.

Dynamo Tracing 
----------------------------------------------------------------

`torch.compile`  ’s frontend (Dynamo) is a custom Python bytecode interpreter designed to allow graph compilation
in PyTorch programs while retaining the full flexibility of Python. Given a function to be compiled, Dynamo
interprets Python bytecode to extract sequences of PyTorch operations into 1 or more FX graphs that may be further optimized by a backend. 

![Summary diagram of Dynamo](../_images/dynamo_summary_diagram.png)

For example, for the function `f`  in the above diagram, Dynamo produces: 

* a single **FX graph** that takes in the original input plus some additional inputs required by the function.
* **Python bytecode** that can be used as a drop-in replacement for `f`  . In our example, the bytecode retrieves
the additional inputs and passes it to the graph and also contains unoptimizable Python side effects (the list append)
* **guards** that specify the conditions under which the graph and bytecode are valid. Unless otherwise specified,
the graph produced by Dynamo specializes on the shapes of input Tensors.

Graph Breaks 
------------------------------------------------------------

Dynamo traces your code and attempts to capture your PyTorch code into a single computation graph of PyTorch
operators (FX graph). However, this is not always possible. When encountering code that can’t be traced, a “ **graph break** ” occurs.
In the default `torch.compile`  settings, a graph break involves compiling the FX graph that has been determined so far,
running the unsupported code in regular Python, then resuming tracing after the unsupported code with a new FX graph. 

Graph breaks are a feature that allows Dynamo to run over arbitrary Python code and carve out functional subgraphs that can each be individually optimized. 

However, it is possible for graph breaks to lead to unexpected slowness in `torch.compile`  .
If you’re not getting the speedups you expect, we recommend checking for graph breaks and removing them. 

Graph breaks may occur on things like: 

* Data-dependent if-statements
* Many Python built-in functions
* C functions

Below is an example of a graph break due to calling an unsupported operation `torch.save`  : 

```
@torch.compile
def f(x):
   y = x ** 2  / 2
   torch.save(y, "foo.pt")  # torch.save is an unsupported operation
   z = y ** 3 / 6
   return z

x = torch.randn(3)
print(f(x))

```

```
tensor([2.3534, 3.7211, 0.0269])

```

```
Graph break in user code at /tmp/ipykernel_231194/215272159.py:4
Graph Break Reason: Attempted to call function marked as skipped
  Explanation: Dynamo developers have intentionally marked that the function `save` in file `/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/serialization.py` should not be traced.
  Hint: Avoid calling the function `save`.
  Hint: Apply `@torch._dynamo.dont_skip_tracing` to the function `save` to force tracing into the function. More graph breaks may occur as a result of attempting to trace into the function.
  Hint: Please file an issue to PyTorch.

  Developer debug context: module: torch.serialization, qualname: save, skip reason: <missing reason>

User code traceback:
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/ipykernel_launcher.py", line 18, in <module>
    app.launch_new_instance()
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/traitlets/config/application.py", line 1075, in launch_instance
    app.start()
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/ipykernel/kernelapp.py", line 739, in start
    self.io_loop.start()
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/tornado/platform/asyncio.py", line 211, in start
    self.asyncio_loop.run_forever()
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/asyncio/base_events.py", line 641, in run_forever
    self._run_once()
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/asyncio/base_events.py", line 1986, in _run_once
    handle._run()
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/asyncio/events.py", line 88, in _run
    self._context.run(self._callback, *self._args)
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/ipykernel/kernelbase.py", line 519, in dispatch_queue
    await self.process_one()
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/ipykernel/kernelbase.py", line 508, in process_one
    await dispatch(*args)
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/ipykernel/kernelbase.py", line 400, in dispatch_shell
    await result
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/ipykernel/ipkernel.py", line 368, in execute_request
    await super().execute_request(stream, ident, parent)
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/ipykernel/kernelbase.py", line 767, in execute_request
    reply_content = await reply_content
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/ipykernel/ipkernel.py", line 455, in do_execute
    res = shell.run_cell(
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/ipykernel/zmqshell.py", line 577, in run_cell
    return super().run_cell(*args, **kwargs)
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/IPython/core/interactiveshell.py", line 3006, in run_cell
    result = self._run_cell(
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/IPython/core/interactiveshell.py", line 3061, in _run_cell
    result = runner(coro)
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/IPython/core/async_helpers.py", line 129, in _pseudo_sync_runner
    coro.send(None)
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/IPython/core/interactiveshell.py", line 3266, in run_cell_async
    has_raised = await self.run_ast_nodes(code_ast.body, cell_name,
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/IPython/core/interactiveshell.py", line 3445, in run_ast_nodes
    if await self.run_code(code, result, async_=asy):
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/IPython/core/interactiveshell.py", line 3505, in run_code
    exec(code_obj, self.user_global_ns, self.user_ns)
  File "/tmp/ipykernel_231194/215272159.py", line 9, in <module>
    print(f(x))
  File "/tmp/ipykernel_231194/215272159.py", line 4, in f
    torch.save(y, "foo.pt")  # torch.save is an unsupported operation

```

The semantics of `torch.compile(f)(x)`  are roughly this: 

```
def compiled_f_semantics(x):
   y = torch.compile(g, fullgraph=True)(x)
   torch.save(y, "foo.pt")
   z = torch.compile(h, fullgraph=True)(x)
   return z

def g(x):
    return x ** 2  / 2

def h(x):
    return y ** 3 / 6

```

Guards 
------------------------------------------------

`torch.compile`  makes some assumptions about runtime values as we trace through code. During tracing, we generate “guards”,
which are runtime checks for these assumptions. Guards are run in future calls to the compiled function to determine if we
can reuse previously compiled code. Examples of runtime checks are constant values, types, and object IDs. 

Below is an example of generated guards. The `TENSOR_MATCH`  guard checks for the input’s type, device, dtype, shape, etc. 

```
@torch.compile
def fn(x):
    return x + 1

print(fn(torch.ones(3, 3)))

```

```
tensor([[2., 2., 2.],
        [2., 2., 2.],
        [2., 2., 2.]])

```

```
GUARDS:

TREE_GUARD_MANAGER:
+- RootGuardManager
| +- LAMBDA_GUARD: torch._functorch.aot_autograd.utils.top_saved_tensors_hooks ids == None  # _dynamo/output_graph.py:633 in init_ambient_guards
| +- DEFAULT_DEVICE: utils_device.CURRENT_DEVICE == None                           # _dynamo/output_graph.py:621 in init_ambient_guards
| +- GLOBAL_STATE: ___check_global_state()
| +- TORCH_FUNCTION_MODE_STACK: ___check_torch_function_mode_stack()
| +- GuardManager: source=L['x'], accessed_by=FrameLocalsGuardAccessor(key='x', framelocals_idx=0)
| | +- TENSOR_MATCH: check_tensor(L['x'], Tensor, DispatchKeySet(CPU, BackendSelect, ADInplaceOrView, AutogradCPU), torch.float32, device=None, requires_grad=False, size=[3, 3], stride=[3, 1])  # return x + 1  # mp/ipykernel_231194/1068332425.py:3 in fn
| | +- NO_HASATTR: hasattr(L['x'], '_dynamo_dynamic_indices') == False           # return x + 1  # mp/ipykernel_231194/1068332425.py:3 in fn

Guard eval latency = 295.42 us

```

Recompilations 
----------------------------------------------------------------

If the guards fail for every instance of previously compiled code, then `torch.compile`  must “recompile” the function,
requiring the original code to be traced again. In the example below, recompilation is necessary because the guard checking the tensor argument’s shape failed. 

```
@torch.compile
def fn(x):
    return x + 1

print(fn(torch.ones(3, 3)))
print(fn(torch.ones(4, 4)))

```

```
tensor([[2., 2., 2.],
        [2., 2., 2.],
        [2., 2., 2.]])
tensor([[2., 2., 2., 2.],
        [2., 2., 2., 2.],
        [2., 2., 2., 2.],
        [2., 2., 2., 2.]])

```

```
Recompiling function fn in /tmp/ipykernel_231194/420870727.py:1
    triggered by the following guard failure(s):
    - 3/0: tensor 'x' size mismatch at index 0. expected 3, actual 4

```

Dynamic Shapes 
----------------------------------------------------------------

`torch.compile`  initially assumes tensor shapes are static/constant and guards based on these assumptions. By using “dynamic shapes,”
we can get `torch.compile`  to produce compiled code that can accept tensor inputs with different shapes - we avoid recompiling every time shapes differ.
By default, automatic dynamic shapes are enabled in `torch.compile(dynamic=None)`  - if compilation fails due to shape mismatch,
recompilation is attempted with dynamic shapes. Dynamic shapes can also be fully enabled ( `dynamic=True`  ) or disabled ( `dynamic=False`  ). 

Below, we enable dynamic shapes and note that we no longer need to recompile. 

```
@torch.compile(dynamic=True)
def fn(x):
    return x + 1

print(fn(torch.ones(3, 3)))
print(fn(torch.ones(4, 4)))

```

```
tensor([[2., 2., 2.],
        [2., 2., 2.],
        [2., 2., 2.]])
tensor([[2., 2., 2., 2.],
        [2., 2., 2., 2.],
        [2., 2., 2., 2.],
        [2., 2., 2., 2.]])

```

```
create_env
create_symbol s77 = 3 for L['x'].size()[0] [2, int_oo] return x + 1  # mp/ipykernel_231194/1458103805.py:3 in fn (_dynamo/variables/builder.py:3382 in <lambda>), for more info run with TORCHDYNAMO_EXTENDED_DEBUG_CREATE_SYMBOL="s77" or to suppress this message run with TORCHDYNAMO_EXTENDED_ADVICE="0"
create_symbol s77 duck sized L['x'].size()[1]
eval False == False [statically known]
eval False == False [statically known]
produce_guards
track_symint L['x'].size()[0] s77 None
track_symint L['x'].size()[1] s77 None
track_symint L['x'].stride()[0] s77 None
track_symint L['x'].stride()[1] 1 None
track_symint L['x'].storage_offset() 0 None
Skipping guard L['x'].stride()[1] == 1
Skipping guard L['x'].storage_offset() == 0

```

For more information on dynamic shapes, see [The dynamic shapes manual](https://docs.google.com/document/d/1GgvOe7C8_NVOMLOCwDaYV1mXXyHMXY7ExoewHqooxrs/edit?tab=t.0#heading=h.fh8zzonyw8ng)  .

