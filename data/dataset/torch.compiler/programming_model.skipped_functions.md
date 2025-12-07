Skipped Functions 
======================================================================

**Summary:** 

* Sometimes, `torch.compile`  completely gives up compiling a function and runs it eagerly instead,
resulting in potentially lost optimization opportunities.
* There are ways to work around skipped functions in order to re-enable tracing around the problematic code.

Sometimes, `torch.compile`  with `fullgraph=False`  is unable to resume tracing when encountering a graph break
or other compiler error. In many of these cases, `torch.compile`  will skip compiling the function entirely and run it eagerly. 

Note that the skip is only applied to the current function and NOT any nested function calls. `torch.compile`  will still attempt to compile nested calls. 

```
def inner1(x):
    return x + 1
def inner2(x):
    return x + 2
@torch.compile
def fn(x):
    x = inner1(x)
    torch._dynamo.skip_frame()
    x = inner2(x)
fn(torch.randn(3))

```

```
ChromiumEventLogger initialized with id 3aedd3be-2d0d-4069-b5d5-e6520582fac6
torchdynamo start compiling fn /tmp/ipykernel_231646/2126697152.py:5, stack (elided 5 frames):
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
  File "/tmp/ipykernel_231646/2126697152.py", line 10, in <module>
    fn(torch.randn(3))

Step 1: torchdynamo start tracing fn /tmp/ipykernel_231646/2126697152.py:5
create_env
TRACE starts_line /tmp/ipykernel_231646/2126697152.py:5 in fn
    @torch.compile
TRACE RESUME 0 []
TRACE starts_line /tmp/ipykernel_231646/2126697152.py:7 in fn
        x = inner1(x)
TRACE LOAD_GLOBAL inner1 []
TRACE LOAD_FAST x [NullVariable, LazyVariableTracker()]
TRACE CALL 1 [NullVariable, LazyVariableTracker(), LazyVariableTracker()]
TRACE inlined call inner1 from /tmp/ipykernel_231646/2126697152.py:7 in fn
    x = inner1(x)
        ~~~~~~^^^
INLINING <code object inner1 at 0x7f0fe3401960, file "/tmp/ipykernel_231646/2126697152.py", line 1>, inlined according trace_rules.lookup inlined by default
wrap_to_fake L['x'] (3,) StatefulSymbolicContext(dynamic_sizes=[<DimDynamic.STATIC: 2>], dynamic_strides=[<DimDynamic.INFER_STRIDE: 4>], constraint_sizes=[None], constraint_strides=[None], specialize_on=[[]], view_base_context=None, tensor_source=LocalSource(local_name='x', is_input=True, dynamism=None, is_derefed_cell_contents=False), shape_env_to_source_to_symbol_cache={}) <class 'torch.Tensor'>
create_graph_input L_x_ L['x'] FakeTensor(..., size=(3,)) at debug_level 0 before=False
TRACE starts_line /tmp/ipykernel_231646/2126697152.py:1 in inner1 (inline depth: 1)
    def inner1(x):
TRACE RESUME 0 []
TRACE starts_line /tmp/ipykernel_231646/2126697152.py:2 in inner1 (inline depth: 1)
        return x + 1
TRACE LOAD_FAST x []
TRACE LOAD_CONST 1 [TensorVariable()]
TRACE BINARY_OP 0 [TensorVariable(), ConstantVariable(int: 1)]
TRACE FX call add from /tmp/ipykernel_231646/2126697152.py:2 in inner1 (inline depth: 1)
    return x + 1
           ~~^~~
TRACE RETURN_VALUE None [TensorVariable()]
DONE INLINING <code object inner1 at 0x7f0fe3401960, file "/tmp/ipykernel_231646/2126697152.py", line 1>
TRACE STORE_FAST x [TensorVariable()]
TRACE starts_line /tmp/ipykernel_231646/2126697152.py:8 in fn
        torch._dynamo.skip_frame()
TRACE LOAD_GLOBAL torch []
TRACE LOAD_ATTR _dynamo [LazyVariableTracker()]
TRACE LOAD_ATTR skip_frame [LazyVariableTracker()]
TRACE CALL 0 [NullVariable, LazyVariableTracker()]
Skipping frame Skip frame due to `torch._dynamo.skip_frame()`. Message: None fn                     /tmp/ipykernel_231646/2126697152.py 5
put_code_state: no cache key, skipping
run_gc_after_compile: running gc
torchdynamo start compiling inner1 /tmp/ipykernel_231646/2126697152.py:1, stack (elided 5 frames):
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
  File "/tmp/ipykernel_231646/2126697152.py", line 10, in <module>
    fn(torch.randn(3))
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/eval_frame.py", line 736, in compile_wrapper
    return fn(*args, **kwargs)

Step 1: torchdynamo start tracing inner1 /tmp/ipykernel_231646/2126697152.py:1
create_env
TRACE starts_line /tmp/ipykernel_231646/2126697152.py:1 in inner1
    def inner1(x):
TRACE RESUME 0 []
TRACE starts_line /tmp/ipykernel_231646/2126697152.py:2 in inner1
        return x + 1
TRACE LOAD_FAST x []
TRACE LOAD_CONST 1 [LazyVariableTracker()]
TRACE BINARY_OP 0 [LazyVariableTracker(), ConstantVariable(int: 1)]
wrap_to_fake L['x'] (3,) StatefulSymbolicContext(dynamic_sizes=[<DimDynamic.STATIC: 2>], dynamic_strides=[<DimDynamic.INFER_STRIDE: 4>], constraint_sizes=[None], constraint_strides=[None], specialize_on=[[]], view_base_context=None, tensor_source=LocalSource(local_name='x', is_input=True, dynamism=None, is_derefed_cell_contents=False), shape_env_to_source_to_symbol_cache={}) <class 'torch.Tensor'>
create_graph_input L_x_ L['x'] FakeTensor(..., size=(3,)) at debug_level 0 before=False
TRACE FX call add from /tmp/ipykernel_231646/2126697152.py:2 in inner1
    return x + 1
           ~~^~~
TRACE RETURN_VALUE None [TensorVariable()]
Step 1: torchdynamo done tracing inner1 (RETURN_VALUE)
RETURN_VALUE triggered compile
COMPILING GRAPH due to GraphCompileReason(reason='return_value', user_stack=[<FrameSummary file /tmp/ipykernel_231646/2126697152.py, line 2 in inner1>], graph_break=False)
TRACED GRAPH
 ===== __compiled_fn_2_2a686b5a_3112_4967_a7bf_9f977d4d2424 =====
 /home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/fx/_lazy_graph_module.py class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3][1]cpu"):
        l_x_ = L_x_
        
         # File: /tmp/ipykernel_231646/2126697152.py:2 in inner1, code: return x + 1
        add: "f32[3][1]cpu" = l_x_ + 1;  l_x_ = None
        return (add,)
        

Step 2: calling compiler function eager
Step 2: done compiler function eager
produce_guards
track_symint L['x'].size()[0] 3 None
track_symint L['x'].stride()[0] 1 None
track_symint L['x'].storage_offset() 0 None
Skipping guard L['x'].size()[0] == 3
Skipping guard L['x'].stride()[0] == 1
Skipping guard L['x'].storage_offset() == 0
GUARDS:

TREE_GUARD_MANAGER:
+- RootGuardManager
| +- LAMBDA_GUARD: torch._functorch.aot_autograd.utils.top_saved_tensors_hooks ids == None  # _dynamo/output_graph.py:633 in init_ambient_guards
| +- DEFAULT_DEVICE: utils_device.CURRENT_DEVICE == None                           # _dynamo/output_graph.py:621 in init_ambient_guards
| +- GLOBAL_STATE: ___check_global_state()
| +- TORCH_FUNCTION_MODE_STACK: ___check_torch_function_mode_stack()
| +- GuardManager: source=L['x'], accessed_by=FrameLocalsGuardAccessor(key='x', framelocals_idx=0)
| | +- TENSOR_MATCH: check_tensor(L['x'], Tensor, DispatchKeySet(CPU, BackendSelect, ADInplaceOrView, AutogradCPU), torch.float32, device=None, requires_grad=False, size=[3], stride=[1])  # return x + 1  # mp/ipykernel_231646/2126697152.py:2 in inner1
| | +- NO_HASATTR: hasattr(L['x'], '_dynamo_dynamic_indices') == False           # return x + 1  # mp/ipykernel_231646/2126697152.py:2 in inner1

Guard eval latency = 260.79 us
put_code_state: no cache key, skipping
run_gc_after_compile: running gc
skipping: inner (reason: in skipfiles, file: /home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_compile.py)
skipping: disable (reason: in skipfiles, file: /home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/decorators.py)
skipping: innermost_fn (reason: in skipfiles, file: /home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/eval_frame.py)
skipping: __init__ (reason: in skipfiles, file: /home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/eval_frame.py)
skipping: __init__ (reason: in skipfiles, file: /home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/eval_frame.py)
skipping: nothing (reason: in skipfiles, file: /home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/eval_frame.py)
skipping: __call__ (reason: in skipfiles, file: /home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/eval_frame.py)
skipping: _fn (reason: in skipfiles, file: /home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/eval_frame.py)
skipping: skip_frame (reason: in skipfiles, file: /home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/decorators.py)
torchdynamo start compiling inner2 /tmp/ipykernel_231646/2126697152.py:3, stack (elided 5 frames):
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
  File "/tmp/ipykernel_231646/2126697152.py", line 10, in <module>
    fn(torch.randn(3))
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/eval_frame.py", line 736, in compile_wrapper
    return fn(*args, **kwargs)

Step 1: torchdynamo start tracing inner2 /tmp/ipykernel_231646/2126697152.py:3
create_env
TRACE starts_line /tmp/ipykernel_231646/2126697152.py:3 in inner2
    def inner2(x):
TRACE RESUME 0 []
TRACE starts_line /tmp/ipykernel_231646/2126697152.py:4 in inner2
        return x + 2
TRACE LOAD_FAST x []
TRACE LOAD_CONST 2 [LazyVariableTracker()]
TRACE BINARY_OP 0 [LazyVariableTracker(), ConstantVariable(int: 2)]
wrap_to_fake L['x'] (3,) StatefulSymbolicContext(dynamic_sizes=[<DimDynamic.STATIC: 2>], dynamic_strides=[<DimDynamic.INFER_STRIDE: 4>], constraint_sizes=[None], constraint_strides=[None], specialize_on=[[]], view_base_context=None, tensor_source=LocalSource(local_name='x', is_input=True, dynamism=None, is_derefed_cell_contents=False), shape_env_to_source_to_symbol_cache={}) <class 'torch.Tensor'>
create_graph_input L_x_ L['x'] FakeTensor(..., size=(3,)) at debug_level 0 before=False
TRACE FX call add from /tmp/ipykernel_231646/2126697152.py:4 in inner2
    return x + 2
           ~~^~~
TRACE RETURN_VALUE None [TensorVariable()]
Step 1: torchdynamo done tracing inner2 (RETURN_VALUE)
RETURN_VALUE triggered compile
COMPILING GRAPH due to GraphCompileReason(reason='return_value', user_stack=[<FrameSummary file /tmp/ipykernel_231646/2126697152.py, line 4 in inner2>], graph_break=False)
TRACED GRAPH
 ===== __compiled_fn_4_401abcb7_fbcb_4ddc_a207_00a192f0f796 =====
 /home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/fx/_lazy_graph_module.py class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3][1]cpu"):
        l_x_ = L_x_
        
         # File: /tmp/ipykernel_231646/2126697152.py:4 in inner2, code: return x + 2
        add: "f32[3][1]cpu" = l_x_ + 2;  l_x_ = None
        return (add,)
        

Step 2: calling compiler function eager
Step 2: done compiler function eager
produce_guards
track_symint L['x'].size()[0] 3 None
track_symint L['x'].stride()[0] 1 None
track_symint L['x'].storage_offset() 0 None
Skipping guard L['x'].size()[0] == 3
Skipping guard L['x'].stride()[0] == 1
Skipping guard L['x'].storage_offset() == 0
GUARDS:

TREE_GUARD_MANAGER:
+- RootGuardManager
| +- LAMBDA_GUARD: torch._functorch.aot_autograd.utils.top_saved_tensors_hooks ids == None  # _dynamo/output_graph.py:633 in init_ambient_guards
| +- DEFAULT_DEVICE: utils_device.CURRENT_DEVICE == None                           # _dynamo/output_graph.py:621 in init_ambient_guards
| +- GLOBAL_STATE: ___check_global_state()
| +- TORCH_FUNCTION_MODE_STACK: ___check_torch_function_mode_stack()
| +- GuardManager: source=L['x'], accessed_by=FrameLocalsGuardAccessor(key='x', framelocals_idx=0)
| | +- TENSOR_MATCH: check_tensor(L['x'], Tensor, DispatchKeySet(CPU, BackendSelect, ADInplaceOrView, AutogradCPU), torch.float32, device=None, requires_grad=False, size=[3], stride=[1])  # return x + 2  # mp/ipykernel_231646/2126697152.py:4 in inner2
| | +- NO_HASATTR: hasattr(L['x'], '_dynamo_dynamic_indices') == False           # return x + 2  # mp/ipykernel_231646/2126697152.py:4 in inner2

Guard eval latency = 6.47 us
put_code_state: no cache key, skipping
run_gc_after_compile: running gc
skipping: remove (reason: in skipfiles, file: /home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/utils/weak.py)
skipping: __hash__ (reason: in skipfiles, file: /home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/utils/weak.py)

```

In the above example, `torch.compile`  will trace `fn`  (including `inner1`  ) up until the `skip_frame`  .
Then `fn`  is skipped and run eagerly - `inner1`  and `inner2`  are compiled when they are called. 

Skipping functions may result in lost optimization opportunities,
so it is important to check if code you want compiled is being skipped, and if so, to work around the skip. 

Graph Break in a Loop 
------------------------------------------------------------------------------

`torch.compile`  cannot resume tracing if a graph break occurs in a loop: 

```
@torch.compile
def fn(x):
    for i in range(5):
        x = x + 1
        if i == 3:
            torch._dynamo.graph_break()
    return x
fn(torch.randn(3))

```

```
torchdynamo start compiling fn /tmp/ipykernel_231646/2044822433.py:1, stack (elided 5 frames):
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
  File "/tmp/ipykernel_231646/2044822433.py", line 8, in <module>
    fn(torch.randn(3))

Step 1: torchdynamo start tracing fn /tmp/ipykernel_231646/2044822433.py:1
create_env
TRACE starts_line /tmp/ipykernel_231646/2044822433.py:1 in fn
    @torch.compile
TRACE RESUME 0 []
TRACE starts_line /tmp/ipykernel_231646/2044822433.py:3 in fn
        for i in range(5):
TRACE LOAD_GLOBAL range []
TRACE LOAD_CONST 5 [NullVariable, LazyVariableTracker()]
TRACE CALL 1 [NullVariable, LazyVariableTracker(), ConstantVariable(int: 5)]
TRACE GET_ITER None [RangeVariable()]
TRACE FOR_ITER 114 [ListIteratorVariable(length=5, index=0)]
TRACE STORE_FAST i [ListIteratorVariable(length=5, index=1), ConstantVariable(int: 0)]
TRACE starts_line /tmp/ipykernel_231646/2044822433.py:4 in fn
            x = x + 1
TRACE LOAD_FAST x [ListIteratorVariable(length=5, index=1)]
TRACE LOAD_CONST 1 [ListIteratorVariable(length=5, index=1), LazyVariableTracker()]
TRACE BINARY_OP 0 [ListIteratorVariable(length=5, index=1), LazyVariableTracker(), ConstantVariable(int: 1)]
wrap_to_fake L['x'] (3,) StatefulSymbolicContext(dynamic_sizes=[<DimDynamic.STATIC: 2>], dynamic_strides=[<DimDynamic.INFER_STRIDE: 4>], constraint_sizes=[None], constraint_strides=[None], specialize_on=[[]], view_base_context=None, tensor_source=LocalSource(local_name='x', is_input=True, dynamism=None, is_derefed_cell_contents=False), shape_env_to_source_to_symbol_cache={}) <class 'torch.Tensor'>
create_graph_input L_x_ L['x'] FakeTensor(..., size=(3,)) at debug_level 0 before=False
TRACE FX call add from /tmp/ipykernel_231646/2044822433.py:4 in fn
        x = x + 1
            ~~^~~
TRACE STORE_FAST x [ListIteratorVariable(length=5, index=1), TensorVariable()]
TRACE starts_line /tmp/ipykernel_231646/2044822433.py:5 in fn
            if i == 3:
TRACE LOAD_FAST i [ListIteratorVariable(length=5, index=1)]
TRACE LOAD_CONST 3 [ListIteratorVariable(length=5, index=1), ConstantVariable(int: 0)]
TRACE COMPARE_OP == [ListIteratorVariable(length=5, index=1), ConstantVariable(int: 0), ConstantVariable(int: 3)]
TRACE POP_JUMP_IF_TRUE 52 [ListIteratorVariable(length=5, index=1), ConstantVariable(bool: False)]
TRACE JUMP_BACKWARD 24 [ListIteratorVariable(length=5, index=1)]
TRACE starts_line /tmp/ipykernel_231646/2044822433.py:3 in fn
        for i in range(5):
TRACE FOR_ITER 114 [ListIteratorVariable(length=5, index=1)]
TRACE STORE_FAST i [ListIteratorVariable(length=5, index=2), ConstantVariable(int: 1)]
TRACE starts_line /tmp/ipykernel_231646/2044822433.py:4 in fn
            x = x + 1
TRACE LOAD_FAST x [ListIteratorVariable(length=5, index=2)]
TRACE LOAD_CONST 1 [ListIteratorVariable(length=5, index=2), TensorVariable()]
TRACE BINARY_OP 0 [ListIteratorVariable(length=5, index=2), TensorVariable(), ConstantVariable(int: 1)]
TRACE STORE_FAST x [ListIteratorVariable(length=5, index=2), TensorVariable()]
TRACE starts_line /tmp/ipykernel_231646/2044822433.py:5 in fn
            if i == 3:
TRACE LOAD_FAST i [ListIteratorVariable(length=5, index=2)]
TRACE LOAD_CONST 3 [ListIteratorVariable(length=5, index=2), ConstantVariable(int: 1)]
TRACE COMPARE_OP == [ListIteratorVariable(length=5, index=2), ConstantVariable(int: 1), ConstantVariable(int: 3)]
TRACE POP_JUMP_IF_TRUE 52 [ListIteratorVariable(length=5, index=2), ConstantVariable(bool: False)]
TRACE JUMP_BACKWARD 24 [ListIteratorVariable(length=5, index=2)]
TRACE starts_line /tmp/ipykernel_231646/2044822433.py:3 in fn
        for i in range(5):
TRACE FOR_ITER 114 [ListIteratorVariable(length=5, index=2)]
TRACE STORE_FAST i [ListIteratorVariable(length=5, index=3), ConstantVariable(int: 2)]
TRACE starts_line /tmp/ipykernel_231646/2044822433.py:4 in fn
            x = x + 1
TRACE LOAD_FAST x [ListIteratorVariable(length=5, index=3)]
TRACE LOAD_CONST 1 [ListIteratorVariable(length=5, index=3), TensorVariable()]
TRACE BINARY_OP 0 [ListIteratorVariable(length=5, index=3), TensorVariable(), ConstantVariable(int: 1)]
TRACE STORE_FAST x [ListIteratorVariable(length=5, index=3), TensorVariable()]
TRACE starts_line /tmp/ipykernel_231646/2044822433.py:5 in fn
            if i == 3:
TRACE LOAD_FAST i [ListIteratorVariable(length=5, index=3)]
TRACE LOAD_CONST 3 [ListIteratorVariable(length=5, index=3), ConstantVariable(int: 2)]
TRACE COMPARE_OP == [ListIteratorVariable(length=5, index=3), ConstantVariable(int: 2), ConstantVariable(int: 3)]
TRACE POP_JUMP_IF_TRUE 52 [ListIteratorVariable(length=5, index=3), ConstantVariable(bool: False)]
TRACE JUMP_BACKWARD 24 [ListIteratorVariable(length=5, index=3)]
TRACE starts_line /tmp/ipykernel_231646/2044822433.py:3 in fn
        for i in range(5):
TRACE FOR_ITER 114 [ListIteratorVariable(length=5, index=3)]
TRACE STORE_FAST i [ListIteratorVariable(length=5, index=4), ConstantVariable(int: 3)]
TRACE starts_line /tmp/ipykernel_231646/2044822433.py:4 in fn
            x = x + 1
TRACE LOAD_FAST x [ListIteratorVariable(length=5, index=4)]
TRACE LOAD_CONST 1 [ListIteratorVariable(length=5, index=4), TensorVariable()]
TRACE BINARY_OP 0 [ListIteratorVariable(length=5, index=4), TensorVariable(), ConstantVariable(int: 1)]
TRACE STORE_FAST x [ListIteratorVariable(length=5, index=4), TensorVariable()]
TRACE starts_line /tmp/ipykernel_231646/2044822433.py:5 in fn
            if i == 3:
TRACE LOAD_FAST i [ListIteratorVariable(length=5, index=4)]
TRACE LOAD_CONST 3 [ListIteratorVariable(length=5, index=4), ConstantVariable(int: 3)]
TRACE COMPARE_OP == [ListIteratorVariable(length=5, index=4), ConstantVariable(int: 3), ConstantVariable(int: 3)]
TRACE POP_JUMP_IF_TRUE 52 [ListIteratorVariable(length=5, index=4), ConstantVariable(bool: True)]
TRACE starts_line /tmp/ipykernel_231646/2044822433.py:6 in fn
                torch._dynamo.graph_break()
TRACE LOAD_GLOBAL torch [ListIteratorVariable(length=5, index=4)]
TRACE LOAD_ATTR _dynamo [ListIteratorVariable(length=5, index=4), LazyVariableTracker()]
TRACE LOAD_ATTR graph_break [ListIteratorVariable(length=5, index=4), LazyVariableTracker()]
TRACE CALL 0 [ListIteratorVariable(length=5, index=4), NullVariable, LazyVariableTracker()]
Graph break in user code at /tmp/ipykernel_231646/2044822433.py:6
Graph Break Reason: Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

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
  File "/tmp/ipykernel_231646/2044822433.py", line 8, in <module>
    fn(torch.randn(3))
  File "/tmp/ipykernel_231646/2044822433.py", line 6, in fn
    torch._dynamo.graph_break()

Skipping frame because there is a graph break in a for/while loop
<FrameSummary file /tmp/ipykernel_231646/2044822433.py, line 6 in fn>
Skipping frame Skipping frame because there is a graph break in a for/while loop
<FrameSummary file /tmp/ipykernel_231646/2044822433.py, line 6 in fn> fn                     /tmp/ipykernel_231646/2044822433.py 1
put_code_state: no cache key, skipping
run_gc_after_compile: running gc
skipping: graph_break (reason: in skipfiles, file: /home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/decorators.py)

```

```
tensor([5.6613, 5.3784, 4.9111])

```

In this example, we can avoid skipping by unrolling the loop: 

```
@torch.compile
def fn(x):
    def inner(i):
        nonlocal x
        x = x + 1
        if i == 3:
            torch._dynamo.graph_break()
    inner(0)
    inner(1)
    inner(2)
    inner(3)
    inner(4)
    return x
fn(torch.randn(3))

```

```
torchdynamo start compiling fn /tmp/ipykernel_231646/617960493.py:1, stack (elided 5 frames):
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
  File "/tmp/ipykernel_231646/617960493.py", line 14, in <module>
    fn(torch.randn(3))

Step 1: torchdynamo start tracing fn /tmp/ipykernel_231646/617960493.py:1
create_env
TRACE MAKE_CELL x []
TRACE starts_line /tmp/ipykernel_231646/617960493.py:1 in fn
    @torch.compile
TRACE RESUME 0 []
TRACE starts_line /tmp/ipykernel_231646/617960493.py:3 in fn
        def inner(i):
TRACE LOAD_CLOSURE x []
TRACE BUILD_TUPLE 1 [CellVariable()]
TRACE LOAD_CONST <code object inner at 0x7f0fd86ded30, file "/tmp/ipykernel_231646/617960493.py", line 3> [TupleVariable(length=1)]
TRACE MAKE_FUNCTION 8 [TupleVariable(length=1), ConstantVariable(code: <code object inner at 0x7f0fd86ded30, file "/tmp/ipykernel_231646/617960493.py", line 3>)]
TRACE STORE_FAST inner [NestedUserFunctionVariable()]
TRACE starts_line /tmp/ipykernel_231646/617960493.py:8 in fn
        inner(0)
TRACE PUSH_NULL None []
TRACE LOAD_FAST inner [NullVariable]
TRACE LOAD_CONST 0 [NullVariable, NestedUserFunctionVariable()]
TRACE CALL 1 [NullVariable, NestedUserFunctionVariable(), ConstantVariable(int: 0)]
TRACE inlined call inner from /tmp/ipykernel_231646/617960493.py:8 in fn
    inner(0)
    ~~~~~^^^
INLINING <code object inner at 0x7f0fd86ded30, file "/tmp/ipykernel_231646/617960493.py", line 3>, inlined according trace_rules.lookup inlined by default
TRACE COPY_FREE_VARS 1 []
TRACE starts_line /tmp/ipykernel_231646/617960493.py:3 in inner (inline depth: 1)
        def inner(i):
TRACE RESUME 0 []
TRACE starts_line /tmp/ipykernel_231646/617960493.py:5 in inner (inline depth: 1)
            x = x + 1
TRACE LOAD_DEREF x []
wrap_to_fake L['x'] (3,) StatefulSymbolicContext(dynamic_sizes=[<DimDynamic.STATIC: 2>], dynamic_strides=[<DimDynamic.INFER_STRIDE: 4>], constraint_sizes=[None], constraint_strides=[None], specialize_on=[[]], view_base_context=None, tensor_source=LocalSource(local_name='x', is_input=True, dynamism=None, is_derefed_cell_contents=True), shape_env_to_source_to_symbol_cache={}) <class 'torch.Tensor'>
create_graph_input L_x_ L['x'] FakeTensor(..., size=(3,)) at debug_level 0 before=False
TRACE LOAD_CONST 1 [TensorVariable()]
TRACE BINARY_OP 0 [TensorVariable(), ConstantVariable(int: 1)]
TRACE FX call add from /tmp/ipykernel_231646/617960493.py:5 in inner (inline depth: 1)
        x = x + 1
            ~~^~~
TRACE STORE_DEREF x [TensorVariable()]
TRACE starts_line /tmp/ipykernel_231646/617960493.py:6 in inner (inline depth: 1)
            if i == 3:
TRACE LOAD_FAST i []
TRACE LOAD_CONST 3 [ConstantVariable(int: 0)]
TRACE COMPARE_OP == [ConstantVariable(int: 0), ConstantVariable(int: 3)]
TRACE POP_JUMP_IF_FALSE 86 [ConstantVariable(bool: False)]
TRACE RETURN_CONST None []
DONE INLINING <code object inner at 0x7f0fd86ded30, file "/tmp/ipykernel_231646/617960493.py", line 3>
TRACE POP_TOP None [ConstantVariable(NoneType: None)]
TRACE starts_line /tmp/ipykernel_231646/617960493.py:9 in fn
        inner(1)
TRACE PUSH_NULL None []
TRACE LOAD_FAST inner [NullVariable]
TRACE LOAD_CONST 1 [NullVariable, NestedUserFunctionVariable()]
TRACE CALL 1 [NullVariable, NestedUserFunctionVariable(), ConstantVariable(int: 1)]
TRACE inlined call inner from /tmp/ipykernel_231646/617960493.py:9 in fn
    inner(1)
    ~~~~~^^^
INLINING <code object inner at 0x7f0fd86ded30, file "/tmp/ipykernel_231646/617960493.py", line 3>, inlined according trace_rules.lookup inlined by default
TRACE COPY_FREE_VARS 1 []
TRACE starts_line /tmp/ipykernel_231646/617960493.py:3 in inner (inline depth: 1)
        def inner(i):
TRACE RESUME 0 []
TRACE starts_line /tmp/ipykernel_231646/617960493.py:5 in inner (inline depth: 1)
            x = x + 1
TRACE LOAD_DEREF x []
TRACE LOAD_CONST 1 [TensorVariable()]
TRACE BINARY_OP 0 [TensorVariable(), ConstantVariable(int: 1)]
TRACE STORE_DEREF x [TensorVariable()]
TRACE starts_line /tmp/ipykernel_231646/617960493.py:6 in inner (inline depth: 1)
            if i == 3:
TRACE LOAD_FAST i []
TRACE LOAD_CONST 3 [ConstantVariable(int: 1)]
TRACE COMPARE_OP == [ConstantVariable(int: 1), ConstantVariable(int: 3)]
TRACE POP_JUMP_IF_FALSE 86 [ConstantVariable(bool: False)]
TRACE RETURN_CONST None []
DONE INLINING <code object inner at 0x7f0fd86ded30, file "/tmp/ipykernel_231646/617960493.py", line 3>
TRACE POP_TOP None [ConstantVariable(NoneType: None)]
TRACE starts_line /tmp/ipykernel_231646/617960493.py:10 in fn
        inner(2)
TRACE PUSH_NULL None []
TRACE LOAD_FAST inner [NullVariable]
TRACE LOAD_CONST 2 [NullVariable, NestedUserFunctionVariable()]
TRACE CALL 1 [NullVariable, NestedUserFunctionVariable(), ConstantVariable(int: 2)]
TRACE inlined call inner from /tmp/ipykernel_231646/617960493.py:10 in fn
    inner(2)
    ~~~~~^^^
INLINING <code object inner at 0x7f0fd86ded30, file "/tmp/ipykernel_231646/617960493.py", line 3>, inlined according trace_rules.lookup inlined by default
TRACE COPY_FREE_VARS 1 []
TRACE starts_line /tmp/ipykernel_231646/617960493.py:3 in inner (inline depth: 1)
        def inner(i):
TRACE RESUME 0 []
TRACE starts_line /tmp/ipykernel_231646/617960493.py:5 in inner (inline depth: 1)
            x = x + 1
TRACE LOAD_DEREF x []
TRACE LOAD_CONST 1 [TensorVariable()]
TRACE BINARY_OP 0 [TensorVariable(), ConstantVariable(int: 1)]
TRACE STORE_DEREF x [TensorVariable()]
TRACE starts_line /tmp/ipykernel_231646/617960493.py:6 in inner (inline depth: 1)
            if i == 3:
TRACE LOAD_FAST i []
TRACE LOAD_CONST 3 [ConstantVariable(int: 2)]
TRACE COMPARE_OP == [ConstantVariable(int: 2), ConstantVariable(int: 3)]
TRACE POP_JUMP_IF_FALSE 86 [ConstantVariable(bool: False)]
TRACE RETURN_CONST None []
DONE INLINING <code object inner at 0x7f0fd86ded30, file "/tmp/ipykernel_231646/617960493.py", line 3>
TRACE POP_TOP None [ConstantVariable(NoneType: None)]
TRACE starts_line /tmp/ipykernel_231646/617960493.py:11 in fn
        inner(3)
TRACE PUSH_NULL None []
TRACE LOAD_FAST inner [NullVariable]
TRACE LOAD_CONST 3 [NullVariable, NestedUserFunctionVariable()]
TRACE CALL 1 [NullVariable, NestedUserFunctionVariable(), ConstantVariable(int: 3)]
TRACE inlined call inner from /tmp/ipykernel_231646/617960493.py:11 in fn
    inner(3)
    ~~~~~^^^
INLINING <code object inner at 0x7f0fd86ded30, file "/tmp/ipykernel_231646/617960493.py", line 3>, inlined according trace_rules.lookup inlined by default
TRACE COPY_FREE_VARS 1 []
TRACE starts_line /tmp/ipykernel_231646/617960493.py:3 in inner (inline depth: 1)
        def inner(i):
TRACE RESUME 0 []
TRACE starts_line /tmp/ipykernel_231646/617960493.py:5 in inner (inline depth: 1)
            x = x + 1
TRACE LOAD_DEREF x []
TRACE LOAD_CONST 1 [TensorVariable()]
TRACE BINARY_OP 0 [TensorVariable(), ConstantVariable(int: 1)]
TRACE STORE_DEREF x [TensorVariable()]
TRACE starts_line /tmp/ipykernel_231646/617960493.py:6 in inner (inline depth: 1)
            if i == 3:
TRACE LOAD_FAST i []
TRACE LOAD_CONST 3 [ConstantVariable(int: 3)]
TRACE COMPARE_OP == [ConstantVariable(int: 3), ConstantVariable(int: 3)]
TRACE POP_JUMP_IF_FALSE 86 [ConstantVariable(bool: True)]
TRACE starts_line /tmp/ipykernel_231646/617960493.py:7 in inner (inline depth: 1)
                torch._dynamo.graph_break()
TRACE LOAD_GLOBAL torch []
TRACE LOAD_ATTR _dynamo [LazyVariableTracker()]
TRACE LOAD_ATTR graph_break [LazyVariableTracker()]
TRACE CALL 0 [NullVariable, LazyVariableTracker()]
empty checkpoint
FAILED INLINING <code object inner at 0x7f0fd86ded30, file "/tmp/ipykernel_231646/617960493.py", line 3>
Graph break in user code at /tmp/ipykernel_231646/617960493.py:7
Graph Break Reason: Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

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
  File "/tmp/ipykernel_231646/617960493.py", line 14, in <module>
    fn(torch.randn(3))
  File "/tmp/ipykernel_231646/617960493.py", line 11, in fn
    inner(3)
  File "/tmp/ipykernel_231646/617960493.py", line 7, in inner
    torch._dynamo.graph_break()

Restarting analysis due to _dynamo/symbolic_convert.py:223 in fail_and_restart_analysis
Step 1: torchdynamo start tracing fn /tmp/ipykernel_231646/617960493.py:1
create_env
TRACE MAKE_CELL x []
TRACE starts_line /tmp/ipykernel_231646/617960493.py:1 in fn
    @torch.compile
TRACE RESUME 0 []
TRACE starts_line /tmp/ipykernel_231646/617960493.py:3 in fn
        def inner(i):
TRACE LOAD_CLOSURE x []
TRACE BUILD_TUPLE 1 [CellVariable()]
TRACE LOAD_CONST <code object inner at 0x7f0fd86ded30, file "/tmp/ipykernel_231646/617960493.py", line 3> [TupleVariable(length=1)]
TRACE MAKE_FUNCTION 8 [TupleVariable(length=1), ConstantVariable(code: <code object inner at 0x7f0fd86ded30, file "/tmp/ipykernel_231646/617960493.py", line 3>)]
TRACE STORE_FAST inner [NestedUserFunctionVariable()]
TRACE starts_line /tmp/ipykernel_231646/617960493.py:8 in fn
        inner(0)
TRACE PUSH_NULL None []
TRACE LOAD_FAST inner [NullVariable]
TRACE LOAD_CONST 0 [NullVariable, NestedUserFunctionVariable()]
TRACE CALL 1 [NullVariable, NestedUserFunctionVariable(), ConstantVariable(int: 0)]
TRACE inlined call inner from /tmp/ipykernel_231646/617960493.py:8 in fn
    inner(0)
    ~~~~~^^^
INLINING <code object inner at 0x7f0fd86ded30, file "/tmp/ipykernel_231646/617960493.py", line 3>, inlined according trace_rules.lookup inlined by default
TRACE COPY_FREE_VARS 1 []
TRACE starts_line /tmp/ipykernel_231646/617960493.py:3 in inner (inline depth: 1)
        def inner(i):
TRACE RESUME 0 []
TRACE starts_line /tmp/ipykernel_231646/617960493.py:5 in inner (inline depth: 1)
            x = x + 1
TRACE LOAD_DEREF x []
wrap_to_fake L['x'] (3,) StatefulSymbolicContext(dynamic_sizes=[<DimDynamic.STATIC: 2>], dynamic_strides=[<DimDynamic.INFER_STRIDE: 4>], constraint_sizes=[None], constraint_strides=[None], specialize_on=[[]], view_base_context=None, tensor_source=LocalSource(local_name='x', is_input=True, dynamism=None, is_derefed_cell_contents=True), shape_env_to_source_to_symbol_cache={}) <class 'torch.Tensor'>
create_graph_input L_x_ L['x'] FakeTensor(..., size=(3,)) at debug_level 0 before=False
TRACE LOAD_CONST 1 [TensorVariable()]
TRACE BINARY_OP 0 [TensorVariable(), ConstantVariable(int: 1)]
TRACE FX call add from /tmp/ipykernel_231646/617960493.py:5 in inner (inline depth: 1)
        x = x + 1
            ~~^~~
TRACE STORE_DEREF x [TensorVariable()]
TRACE starts_line /tmp/ipykernel_231646/617960493.py:6 in inner (inline depth: 1)
            if i == 3:
TRACE LOAD_FAST i []
TRACE LOAD_CONST 3 [ConstantVariable(int: 0)]
TRACE COMPARE_OP == [ConstantVariable(int: 0), ConstantVariable(int: 3)]
TRACE POP_JUMP_IF_FALSE 86 [ConstantVariable(bool: False)]
TRACE RETURN_CONST None []
DONE INLINING <code object inner at 0x7f0fd86ded30, file "/tmp/ipykernel_231646/617960493.py", line 3>
TRACE POP_TOP None [ConstantVariable(NoneType: None)]
TRACE starts_line /tmp/ipykernel_231646/617960493.py:9 in fn
        inner(1)
TRACE PUSH_NULL None []
TRACE LOAD_FAST inner [NullVariable]
TRACE LOAD_CONST 1 [NullVariable, NestedUserFunctionVariable()]
TRACE CALL 1 [NullVariable, NestedUserFunctionVariable(), ConstantVariable(int: 1)]
TRACE inlined call inner from /tmp/ipykernel_231646/617960493.py:9 in fn
    inner(1)
    ~~~~~^^^
INLINING <code object inner at 0x7f0fd86ded30, file "/tmp/ipykernel_231646/617960493.py", line 3>, inlined according trace_rules.lookup inlined by default
TRACE COPY_FREE_VARS 1 []
TRACE starts_line /tmp/ipykernel_231646/617960493.py:3 in inner (inline depth: 1)
        def inner(i):
TRACE RESUME 0 []
TRACE starts_line /tmp/ipykernel_231646/617960493.py:5 in inner (inline depth: 1)
            x = x + 1
TRACE LOAD_DEREF x []
TRACE LOAD_CONST 1 [TensorVariable()]
TRACE BINARY_OP 0 [TensorVariable(), ConstantVariable(int: 1)]
TRACE STORE_DEREF x [TensorVariable()]
TRACE starts_line /tmp/ipykernel_231646/617960493.py:6 in inner (inline depth: 1)
            if i == 3:
TRACE LOAD_FAST i []
TRACE LOAD_CONST 3 [ConstantVariable(int: 1)]
TRACE COMPARE_OP == [ConstantVariable(int: 1), ConstantVariable(int: 3)]
TRACE POP_JUMP_IF_FALSE 86 [ConstantVariable(bool: False)]
TRACE RETURN_CONST None []
DONE INLINING <code object inner at 0x7f0fd86ded30, file "/tmp/ipykernel_231646/617960493.py", line 3>
TRACE POP_TOP None [ConstantVariable(NoneType: None)]
TRACE starts_line /tmp/ipykernel_231646/617960493.py:10 in fn
        inner(2)
TRACE PUSH_NULL None []
TRACE LOAD_FAST inner [NullVariable]
TRACE LOAD_CONST 2 [NullVariable, NestedUserFunctionVariable()]
TRACE CALL 1 [NullVariable, NestedUserFunctionVariable(), ConstantVariable(int: 2)]
TRACE inlined call inner from /tmp/ipykernel_231646/617960493.py:10 in fn
    inner(2)
    ~~~~~^^^
INLINING <code object inner at 0x7f0fd86ded30, file "/tmp/ipykernel_231646/617960493.py", line 3>, inlined according trace_rules.lookup inlined by default
TRACE COPY_FREE_VARS 1 []
TRACE starts_line /tmp/ipykernel_231646/617960493.py:3 in inner (inline depth: 1)
        def inner(i):
TRACE RESUME 0 []
TRACE starts_line /tmp/ipykernel_231646/617960493.py:5 in inner (inline depth: 1)
            x = x + 1
TRACE LOAD_DEREF x []
TRACE LOAD_CONST 1 [TensorVariable()]
TRACE BINARY_OP 0 [TensorVariable(), ConstantVariable(int: 1)]
TRACE STORE_DEREF x [TensorVariable()]
TRACE starts_line /tmp/ipykernel_231646/617960493.py:6 in inner (inline depth: 1)
            if i == 3:
TRACE LOAD_FAST i []
TRACE LOAD_CONST 3 [ConstantVariable(int: 2)]
TRACE COMPARE_OP == [ConstantVariable(int: 2), ConstantVariable(int: 3)]
TRACE POP_JUMP_IF_FALSE 86 [ConstantVariable(bool: False)]
TRACE RETURN_CONST None []
DONE INLINING <code object inner at 0x7f0fd86ded30, file "/tmp/ipykernel_231646/617960493.py", line 3>
TRACE POP_TOP None [ConstantVariable(NoneType: None)]
TRACE starts_line /tmp/ipykernel_231646/617960493.py:11 in fn
        inner(3)
TRACE PUSH_NULL None []
TRACE LOAD_FAST inner [NullVariable]
TRACE LOAD_CONST 3 [NullVariable, NestedUserFunctionVariable()]
TRACE CALL 1 [NullVariable, NestedUserFunctionVariable(), ConstantVariable(int: 3)]
COMPILING GRAPH due to GraphCompileReason(reason='Call to `torch._dynamo.graph_break()`n  Explanation: User-inserted graph break. Message: Nonen  Hint: Remove the `torch._dynamo.graph_break()` call.nn  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`n', user_stack=[<FrameSummary file /tmp/ipykernel_231646/617960493.py, line 11 in fn>, <FrameSummary file /tmp/ipykernel_231646/617960493.py, line 7 in inner>], graph_break=True)
TRACED GRAPH
 ===== __compiled_fn_8_332a4794_1ebc_40f3_9737_e90cc89d7420 =====
 /home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/fx/_lazy_graph_module.py class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3][1]cpu"):
        l_x_ = L_x_
        
         # File: /tmp/ipykernel_231646/617960493.py:5 in inner, code: x = x + 1
        x: "f32[3][1]cpu" = l_x_ + 1;  l_x_ = None
        
         # File: /tmp/ipykernel_231646/617960493.py:5 in inner, code: x = x + 1
        x_1: "f32[3][1]cpu" = x + 1;  x = None
        
         # File: /tmp/ipykernel_231646/617960493.py:5 in inner, code: x = x + 1
        x_2: "f32[3][1]cpu" = x_1 + 1;  x_1 = None
        return (x_2,)
        

Step 2: calling compiler function eager
Step 2: done compiler function eager
produce_guards
track_symint L['x'].size()[0] 3 None
track_symint L['x'].stride()[0] 1 None
track_symint L['x'].storage_offset() 0 None
Skipping guard L['x'].size()[0] == 3
Skipping guard L['x'].stride()[0] == 1
Skipping guard L['x'].storage_offset() == 0
GUARDS:

TREE_GUARD_MANAGER:
+- RootGuardManager
| +- LAMBDA_GUARD: torch._functorch.aot_autograd.utils.top_saved_tensors_hooks ids == None  # _dynamo/output_graph.py:633 in init_ambient_guards
| +- DEFAULT_DEVICE: utils_device.CURRENT_DEVICE == None                           # _dynamo/output_graph.py:621 in init_ambient_guards
| +- GLOBAL_STATE: ___check_global_state()
| +- TORCH_FUNCTION_MODE_STACK: ___check_torch_function_mode_stack()
| +- GuardManager: source=L['x'], accessed_by=FrameLocalsGuardAccessor(key='x', framelocals_idx=0)
| | +- TENSOR_MATCH: check_tensor(L['x'], Tensor, DispatchKeySet(CPU, BackendSelect, ADInplaceOrView, AutogradCPU), torch.float32, device=None, requires_grad=False, size=[3], stride=[1])  # x = x + 1  # mp/ipykernel_231646/617960493.py:5 in inner
| | +- NO_HASATTR: hasattr(L['x'], '_dynamo_dynamic_indices') == False           # x = x + 1  # mp/ipykernel_231646/617960493.py:5 in inner

Guard eval latency = 263.76 us
put_code_state: no cache key, skipping
run_gc_after_compile: running gc
skipping: _create_nested_fn (reason: in skipfiles, file: /home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/variables/functions.py)
torchdynamo start compiling inner /tmp/ipykernel_231646/617960493.py:3, stack (elided 5 frames):
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
  File "/tmp/ipykernel_231646/617960493.py", line 14, in <module>
    fn(torch.randn(3))
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/eval_frame.py", line 736, in compile_wrapper
    return fn(*args, **kwargs)

Step 1: torchdynamo start tracing inner /tmp/ipykernel_231646/617960493.py:3
create_env
TRACE COPY_FREE_VARS 1 []
TRACE starts_line /tmp/ipykernel_231646/617960493.py:3 in inner
        def inner(i):
TRACE RESUME 0 []
TRACE starts_line /tmp/ipykernel_231646/617960493.py:5 in inner
            x = x + 1
TRACE LOAD_DEREF x []
TRACE LOAD_CONST 1 [LazyVariableTracker()]
TRACE BINARY_OP 0 [LazyVariableTracker(), ConstantVariable(int: 1)]
wrap_to_fake L['x'] (3,) StatefulSymbolicContext(dynamic_sizes=[<DimDynamic.STATIC: 2>], dynamic_strides=[<DimDynamic.INFER_STRIDE: 4>], constraint_sizes=[None], constraint_strides=[None], specialize_on=[[]], view_base_context=None, tensor_source=LocalSource(local_name='x', is_input=False, dynamism=None, is_derefed_cell_contents=True), shape_env_to_source_to_symbol_cache={}) <class 'torch.Tensor'>
create_graph_input L_x_ L['x'] FakeTensor(..., size=(3,)) at debug_level 0 before=False
TRACE FX call add from /tmp/ipykernel_231646/617960493.py:5 in inner
        x = x + 1
            ~~^~~
TRACE STORE_DEREF x [TensorVariable()]
TRACE starts_line /tmp/ipykernel_231646/617960493.py:6 in inner
            if i == 3:
TRACE LOAD_FAST i []
TRACE LOAD_CONST 3 [LazyVariableTracker()]
TRACE COMPARE_OP == [LazyVariableTracker(), ConstantVariable(int: 3)]
TRACE POP_JUMP_IF_FALSE 86 [ConstantVariable(bool: True)]
TRACE starts_line /tmp/ipykernel_231646/617960493.py:7 in inner
                torch._dynamo.graph_break()
TRACE LOAD_GLOBAL torch []
TRACE LOAD_ATTR _dynamo [LazyVariableTracker()]
TRACE LOAD_ATTR graph_break [LazyVariableTracker()]
TRACE CALL 0 [NullVariable, LazyVariableTracker()]
Graph break (user stack suppressed due to duplicate graph break) in user code at /tmp/ipykernel_231646/617960493.py:7
Graph Break Reason: Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

Restarting analysis due to _dynamo/symbolic_convert.py:223 in fail_and_restart_analysis
Step 1: torchdynamo start tracing inner /tmp/ipykernel_231646/617960493.py:3
create_env
TRACE COPY_FREE_VARS 1 []
TRACE starts_line /tmp/ipykernel_231646/617960493.py:3 in inner
        def inner(i):
TRACE RESUME 0 []
TRACE starts_line /tmp/ipykernel_231646/617960493.py:5 in inner
            x = x + 1
TRACE LOAD_DEREF x []
TRACE LOAD_CONST 1 [LazyVariableTracker()]
TRACE BINARY_OP 0 [LazyVariableTracker(), ConstantVariable(int: 1)]
wrap_to_fake L['x'] (3,) StatefulSymbolicContext(dynamic_sizes=[<DimDynamic.STATIC: 2>], dynamic_strides=[<DimDynamic.INFER_STRIDE: 4>], constraint_sizes=[None], constraint_strides=[None], specialize_on=[[]], view_base_context=None, tensor_source=LocalSource(local_name='x', is_input=False, dynamism=None, is_derefed_cell_contents=True), shape_env_to_source_to_symbol_cache={}) <class 'torch.Tensor'>
create_graph_input L_x_ L['x'] FakeTensor(..., size=(3,)) at debug_level 0 before=False
TRACE FX call add from /tmp/ipykernel_231646/617960493.py:5 in inner
        x = x + 1
            ~~^~~
TRACE STORE_DEREF x [TensorVariable()]
TRACE starts_line /tmp/ipykernel_231646/617960493.py:6 in inner
            if i == 3:
TRACE LOAD_FAST i []
TRACE LOAD_CONST 3 [LazyVariableTracker()]
TRACE COMPARE_OP == [LazyVariableTracker(), ConstantVariable(int: 3)]
TRACE POP_JUMP_IF_FALSE 86 [ConstantVariable(bool: True)]
TRACE starts_line /tmp/ipykernel_231646/617960493.py:7 in inner
                torch._dynamo.graph_break()
TRACE LOAD_GLOBAL torch []
TRACE LOAD_ATTR _dynamo [LazyVariableTracker()]
TRACE LOAD_ATTR graph_break [LazyVariableTracker()]
TRACE CALL 0 [NullVariable, LazyVariableTracker()]
COMPILING GRAPH due to GraphCompileReason(reason='Call to `torch._dynamo.graph_break()`n  Explanation: User-inserted graph break. Message: Nonen  Hint: Remove the `torch._dynamo.graph_break()` call.nn  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`n', user_stack=[<FrameSummary file /tmp/ipykernel_231646/617960493.py, line 7 in inner>], graph_break=True)
TRACED GRAPH
 ===== __compiled_fn_12_d13477b8_2fbb_4a81_a805_b282e97ebf51 =====
 /home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/fx/_lazy_graph_module.py class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3][1]cpu"):
        l_x_ = L_x_
        
         # File: /tmp/ipykernel_231646/617960493.py:5 in inner, code: x = x + 1
        x: "f32[3][1]cpu" = l_x_ + 1;  l_x_ = None
        return (x,)
        

Step 2: calling compiler function eager
Step 2: done compiler function eager
produce_guards
track_symint L['x'].size()[0] 3 None
track_symint L['x'].stride()[0] 1 None
track_symint L['x'].storage_offset() 0 None
Skipping guard L['x'].size()[0] == 3
Skipping guard L['x'].stride()[0] == 1
Skipping guard L['x'].storage_offset() == 0
GUARDS:

TREE_GUARD_MANAGER:
+- RootGuardManager
| +- LAMBDA_GUARD: torch._functorch.aot_autograd.utils.top_saved_tensors_hooks ids == None  # _dynamo/output_graph.py:633 in init_ambient_guards
| +- DEFAULT_DEVICE: utils_device.CURRENT_DEVICE == None                           # _dynamo/output_graph.py:621 in init_ambient_guards
| +- GLOBAL_STATE: ___check_global_state()
| +- TORCH_FUNCTION_MODE_STACK: ___check_torch_function_mode_stack()
| +- GuardManager: source=L['i'], accessed_by=FrameLocalsGuardAccessor(key='i', framelocals_idx=0)
| | +- EQUALS_MATCH: L['i'] == 3                                                   # if i == 3:  # mp/ipykernel_231646/617960493.py:6 in inner
| +- GuardManager: source=L['x'], accessed_by=FrameLocalsGuardAccessor(key='x', framelocals_idx=1)
| | +- TENSOR_MATCH: check_tensor(L['x'], Tensor, DispatchKeySet(CPU, BackendSelect, ADInplaceOrView, AutogradCPU), torch.float32, device=None, requires_grad=False, size=[3], stride=[1])  # x = x + 1  # mp/ipykernel_231646/617960493.py:5 in inner
| | +- NO_HASATTR: hasattr(L['x'], '_dynamo_dynamic_indices') == False           # x = x + 1  # mp/ipykernel_231646/617960493.py:5 in inner
| +- GuardManager: source=G, accessed_by=GlobalsGuardAccessor
| | +- GuardManager: source=G['torch'], accessed_by=DictGetItemGuardAccessor('torch')
| | | +- ID_MATCH: ___check_obj_id(G['torch'], 139712098143824)                  # torch._dynamo.graph_break()  # mp/ipykernel_231646/617960493.py:7 in inner
| | | +- GuardManager: source=G['torch']._dynamo, accessed_by=GetAttrGuardAccessor(_dynamo)
| | | | +- ID_MATCH: ___check_obj_id(G['torch']._dynamo, 139706212821616)          # torch._dynamo.graph_break()  # mp/ipykernel_231646/617960493.py:7 in inner
| | | | +- GuardManager: source=G['torch']._dynamo.graph_break, accessed_by=GetAttrGuardAccessor(graph_break)
| | | | | +- ID_MATCH: ___check_obj_id(G['torch']._dynamo.graph_break, 139706044196288)  # torch._dynamo.graph_break()  # mp/ipykernel_231646/617960493.py:7 in inner

Guard eval latency = 33.36 us
put_code_state: no cache key, skipping
run_gc_after_compile: running gc
torchdynamo start compiling torch_dynamo_resume_in_inner_at_7 /tmp/ipykernel_231646/617960493.py:7, stack (elided 5 frames):
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
  File "/tmp/ipykernel_231646/617960493.py", line 14, in <module>
    fn(torch.randn(3))
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/eval_frame.py", line 736, in compile_wrapper
    return fn(*args, **kwargs)
  File "/tmp/ipykernel_231646/617960493.py", line 11, in fn
    inner(3)

Step 1: torchdynamo start tracing torch_dynamo_resume_in_inner_at_7 /tmp/ipykernel_231646/617960493.py:7
create_env
TRACE starts_line /tmp/ipykernel_231646/617960493.py:7 in torch_dynamo_resume_in_inner_at_7
                torch._dynamo.graph_break()
TRACE COPY_FREE_VARS 1 []
TRACE RESUME 0 []
TRACE LOAD_FAST ___stack0 []
TRACE JUMP_FORWARD 90 [LazyVariableTracker()]
TRACE POP_TOP None [LazyVariableTracker()]
TRACE RETURN_CONST None []
Skipping frame because no content in function call torch_dynamo_resume_in_inner_at_7                     /tmp/ipykernel_231646/617960493.py 7
put_code_state: no cache key, skipping
run_gc_after_compile: running gc
torchdynamo start compiling torch_dynamo_resume_in_fn_at_11 /tmp/ipykernel_231646/617960493.py:11, stack (elided 5 frames):
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
  File "/tmp/ipykernel_231646/617960493.py", line 14, in <module>
    fn(torch.randn(3))
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/eval_frame.py", line 736, in compile_wrapper
    return fn(*args, **kwargs)

Step 1: torchdynamo start tracing torch_dynamo_resume_in_fn_at_11 /tmp/ipykernel_231646/617960493.py:11
create_env
TRACE starts_line /tmp/ipykernel_231646/617960493.py:11 in torch_dynamo_resume_in_fn_at_11
        inner(3)
TRACE COPY_FREE_VARS 1 []
TRACE RESUME 0 []
TRACE LOAD_FAST ___stack0 []
TRACE JUMP_FORWARD 84 [LazyVariableTracker()]
TRACE POP_TOP None [LazyVariableTracker()]
TRACE starts_line /tmp/ipykernel_231646/617960493.py:12 in torch_dynamo_resume_in_fn_at_11
        inner(4)
TRACE PUSH_NULL None []
TRACE LOAD_FAST inner [NullVariable]
TRACE LOAD_CONST 4 [NullVariable, LazyVariableTracker()]
TRACE CALL 1 [NullVariable, LazyVariableTracker(), ConstantVariable(int: 4)]
TRACE inlined call inner from /tmp/ipykernel_231646/617960493.py:12 in torch_dynamo_resume_in_fn_at_11
    inner(4)
    ~~~~~^^^
INLINING <code object inner at 0x7f0fd86ded30, file "/tmp/ipykernel_231646/617960493.py", line 3>, inlined according trace_rules.lookup inlined by default
TRACE COPY_FREE_VARS 1 []
TRACE starts_line /tmp/ipykernel_231646/617960493.py:3 in inner (inline depth: 1)
        def inner(i):
TRACE RESUME 0 []
TRACE starts_line /tmp/ipykernel_231646/617960493.py:5 in inner (inline depth: 1)
            x = x + 1
TRACE LOAD_DEREF x []
TRACE LOAD_CONST 1 [LazyVariableTracker()]
TRACE BINARY_OP 0 [LazyVariableTracker(), ConstantVariable(int: 1)]
wrap_to_fake L['x'] (3,) StatefulSymbolicContext(dynamic_sizes=[<DimDynamic.STATIC: 2>], dynamic_strides=[<DimDynamic.INFER_STRIDE: 4>], constraint_sizes=[None], constraint_strides=[None], specialize_on=[[]], view_base_context=None, tensor_source=LocalSource(local_name='x', is_input=False, dynamism=None, is_derefed_cell_contents=True), shape_env_to_source_to_symbol_cache={}) <class 'torch.Tensor'>
create_graph_input L_x_ L['x'] FakeTensor(..., size=(3,)) at debug_level 0 before=False
TRACE FX call add from /tmp/ipykernel_231646/617960493.py:5 in inner (inline depth: 1)
        x = x + 1
            ~~^~~
TRACE STORE_DEREF x [TensorVariable()]
TRACE starts_line /tmp/ipykernel_231646/617960493.py:6 in inner (inline depth: 1)
            if i == 3:
TRACE LOAD_FAST i []
TRACE LOAD_CONST 3 [ConstantVariable(int: 4)]
TRACE COMPARE_OP == [ConstantVariable(int: 4), ConstantVariable(int: 3)]
TRACE POP_JUMP_IF_FALSE 86 [ConstantVariable(bool: False)]
TRACE RETURN_CONST None []
DONE INLINING <code object inner at 0x7f0fd86ded30, file "/tmp/ipykernel_231646/617960493.py", line 3>
TRACE POP_TOP None [ConstantVariable(NoneType: None)]
TRACE starts_line /tmp/ipykernel_231646/617960493.py:13 in torch_dynamo_resume_in_fn_at_11
        return x
TRACE LOAD_DEREF x []
TRACE RETURN_VALUE None [TensorVariable()]
Step 1: torchdynamo done tracing torch_dynamo_resume_in_fn_at_11 (RETURN_VALUE)
RETURN_VALUE triggered compile
COMPILING GRAPH due to GraphCompileReason(reason='return_value', user_stack=[<FrameSummary file /tmp/ipykernel_231646/617960493.py, line 13 in torch_dynamo_resume_in_fn_at_11>], graph_break=False)
TRACED GRAPH
 ===== __compiled_fn_16_a943b451_304c_4ed5_ab96_a1a0a0bf0ff0 =====
 /home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/fx/_lazy_graph_module.py class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3][1]cpu"):
        l_x_ = L_x_
        
         # File: /tmp/ipykernel_231646/617960493.py:5 in inner, code: x = x + 1
        x: "f32[3][1]cpu" = l_x_ + 1;  l_x_ = None
        return (x,)
        

Step 2: calling compiler function eager
Step 2: done compiler function eager
produce_guards
track_symint L['x'].size()[0] 3 None
track_symint L['x'].stride()[0] 1 None
track_symint L['x'].storage_offset() 0 None
Skipping guard L['x'].size()[0] == 3
Skipping guard L['x'].stride()[0] == 1
Skipping guard L['x'].storage_offset() == 0
GUARDS:

TREE_GUARD_MANAGER:
+- RootGuardManager
| +- LAMBDA_GUARD: torch._functorch.aot_autograd.utils.top_saved_tensors_hooks ids == None  # _dynamo/output_graph.py:633 in init_ambient_guards
| +- DEFAULT_DEVICE: utils_device.CURRENT_DEVICE == None                           # _dynamo/output_graph.py:621 in init_ambient_guards
| +- GLOBAL_STATE: ___check_global_state()
| +- TORCH_FUNCTION_MODE_STACK: ___check_torch_function_mode_stack()
| +- GuardManager: source=L['x'], accessed_by=FrameLocalsGuardAccessor(key='x', framelocals_idx=2)
| | +- TENSOR_MATCH: check_tensor(L['x'], Tensor, DispatchKeySet(CPU, BackendSelect, ADInplaceOrView, AutogradCPU), torch.float32, device=None, requires_grad=False, size=[3], stride=[1])  # x = x + 1  # mp/ipykernel_231646/617960493.py:5 in inner
| | +- NO_HASATTR: hasattr(L['x'], '_dynamo_dynamic_indices') == False           # x = x + 1  # mp/ipykernel_231646/617960493.py:5 in inner
| +- GuardManager: source=L['inner'], accessed_by=FrameLocalsGuardAccessor(key='inner', framelocals_idx=1)
| | +- GuardManager: source=L['inner'].__code__, accessed_by=GetAttrGuardAccessor(__code__)
| | | +- ID_MATCH: ___check_obj_id(L['inner'].__code__, 139706032319792)         # inner(4)  # mp/ipykernel_231646/617960493.py:12 in torch_dynamo_resume_in_fn_at_11

Guard eval latency = 712.48 us
put_code_state: no cache key, skipping
run_gc_after_compile: running gc
skipping because no torch.* remove             /home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/weakref.py 369

```

```
tensor([6.1123, 3.4859, 3.7100])

```

In general, resolving the graph break causing the skip will also resolve the skip.

Graph Break in a Context Manager 
----------------------------------------------------------------------------------------------------

Another common example of an unresumable graph break is a graph break in most context managers: 

```
class CustomCtxManager:
    def __enter__(self):
        pass
    def __exit__(self, exc_type, exc_value, traceback):
        pass
@torch.compile
def fn(x):
    with CustomCtxManager():
        x = x + 1
        torch._dynamo.graph_break()
        return x + 1
fn(torch.randn(3))

```

```
torchdynamo start compiling fn /tmp/ipykernel_231646/4148913404.py:6, stack (elided 5 frames):
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
  File "/tmp/ipykernel_231646/4148913404.py", line 12, in <module>
    fn(torch.randn(3))

Step 1: torchdynamo start tracing fn /tmp/ipykernel_231646/4148913404.py:6
create_env
TRACE starts_line /tmp/ipykernel_231646/4148913404.py:6 in fn
    @torch.compile
TRACE RESUME 0 []
TRACE starts_line /tmp/ipykernel_231646/4148913404.py:8 in fn
        with CustomCtxManager():
TRACE LOAD_GLOBAL CustomCtxManager []
TRACE CALL 0 [NullVariable, LazyVariableTracker()]
TRACE BEFORE_WITH None [GenericContextWrappingVariable(CustomCtxManager)]
TRACE inlined call __enter__ from /tmp/ipykernel_231646/4148913404.py:8 in fn
    with CustomCtxManager():
         ~~~~~~~~~~~~~~~~^^
INLINING <code object __enter__ at 0x7f111cb3ee80, file "/tmp/ipykernel_231646/4148913404.py", line 2>, inlined according trace_rules.lookup inlined by default
TRACE starts_line /tmp/ipykernel_231646/4148913404.py:2 in __enter__ (inline depth: 1)
        def __enter__(self):
TRACE RESUME 0 []
TRACE starts_line /tmp/ipykernel_231646/4148913404.py:3 in __enter__ (inline depth: 1)
            pass
TRACE RETURN_CONST None []
DONE INLINING <code object __enter__ at 0x7f111cb3ee80, file "/tmp/ipykernel_231646/4148913404.py", line 2>
TRACE POP_TOP None [WithExitFunctionVariable(), ConstantVariable(NoneType: None)]
TRACE starts_line /tmp/ipykernel_231646/4148913404.py:9 in fn
            x = x + 1
TRACE LOAD_FAST x [WithExitFunctionVariable()]
TRACE LOAD_CONST 1 [WithExitFunctionVariable(), LazyVariableTracker()]
TRACE BINARY_OP 0 [WithExitFunctionVariable(), LazyVariableTracker(), ConstantVariable(int: 1)]
wrap_to_fake L['x'] (3,) StatefulSymbolicContext(dynamic_sizes=[<DimDynamic.STATIC: 2>], dynamic_strides=[<DimDynamic.INFER_STRIDE: 4>], constraint_sizes=[None], constraint_strides=[None], specialize_on=[[]], view_base_context=None, tensor_source=LocalSource(local_name='x', is_input=True, dynamism=None, is_derefed_cell_contents=False), shape_env_to_source_to_symbol_cache={}) <class 'torch.Tensor'>
create_graph_input L_x_ L['x'] FakeTensor(..., size=(3,)) at debug_level 0 before=False
TRACE FX call add from /tmp/ipykernel_231646/4148913404.py:9 in fn
        x = x + 1
            ~~^~~
TRACE STORE_FAST x [WithExitFunctionVariable(), TensorVariable()]
TRACE starts_line /tmp/ipykernel_231646/4148913404.py:10 in fn
            torch._dynamo.graph_break()
TRACE LOAD_GLOBAL torch [WithExitFunctionVariable()]
TRACE LOAD_ATTR _dynamo [WithExitFunctionVariable(), LazyVariableTracker()]
TRACE LOAD_ATTR graph_break [WithExitFunctionVariable(), LazyVariableTracker()]
TRACE CALL 0 [WithExitFunctionVariable(), NullVariable, LazyVariableTracker()]
empty checkpoint
run_gc_after_compile: running gc
Graph break: skip: from user code at:
  File "/tmp/ipykernel_231646/4148913404.py", line 10, in fn
    torch._dynamo.graph_break()
Traceback (most recent call last):
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 834, in wrapper
    return inner_fn(self, inst)
           ^^^^^^^^^^^^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 2910, in CALL
    self._call(inst)
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 2904, in _call
    self.call_function(fn, args, kwargs)
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 1193, in call_function
    self.push(fn.call_function(self, args, kwargs))  # type: ignore[arg-type]
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/variables/lazy.py", line 201, in realize_and_forward
    return getattr(self.realize(), name)(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/variables/functions.py", line 1398, in call_function
    unimplemented_v2(
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/exc.py", line 528, in unimplemented_v2
    raise Unsupported(msg)
torch._dynamo.exc.Unsupported: Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 1272, in __call__
    result = self._inner_convert(
             ^^^^^^^^^^^^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 629, in __call__
    return _compile(
           ^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 1111, in _compile
    guarded_code = compile_inner(code, one_graph, hooks, transform)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_utils_internal.py", line 97, in wrapper_function
    return function(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 793, in compile_inner
    return _compile_inner(code, one_graph, hooks, transform)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 832, in _compile_inner
    out_code = transform_code_object(code, transform)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/bytecode_transformation.py", line 1424, in transform_code_object
    transformations(instructions, code_options)
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 267, in _fn
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 753, in transform
    tracer.run()
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 3497, in run
    super().run()
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 1363, in run
    while self.step():
          ^^^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 1267, in step
    self.dispatch_table[inst.opcode](self, inst)
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 840, in wrapper
    unimplemented_v2(
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/exc.py", line 527, in unimplemented_v2
    raise Unsupported(msg) from from_exc
torch._dynamo.exc.Unsupported: Graph break under GenericContextWrappingVariable
  Explanation: Attempted to graph break in an active context manager(s) that doesn't support graph breaking.
  Hint: Move the offending context manager(s) to outside the compiled region.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: Active generic context managers: [GenericContextWrappingVariable(CustomCtxManager)]

from user code:
   File "/tmp/ipykernel_231646/4148913404.py", line 10, in fn
    torch._dynamo.graph_break()

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"

WON'T CONVERT fn /tmp/ipykernel_231646/4148913404.py line 6 
due to: 
Traceback (most recent call last):
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 834, in wrapper
    return inner_fn(self, inst)
           ^^^^^^^^^^^^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 2910, in CALL
    self._call(inst)
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 2904, in _call
    self.call_function(fn, args, kwargs)
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 1193, in call_function
    self.push(fn.call_function(self, args, kwargs))  # type: ignore[arg-type]
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/variables/lazy.py", line 201, in realize_and_forward
    return getattr(self.realize(), name)(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/variables/functions.py", line 1398, in call_function
    unimplemented_v2(
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/exc.py", line 528, in unimplemented_v2
    raise Unsupported(msg)
torch._dynamo.exc.Unsupported: Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 1272, in __call__
    result = self._inner_convert(
             ^^^^^^^^^^^^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 629, in __call__
    return _compile(
           ^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 1111, in _compile
    guarded_code = compile_inner(code, one_graph, hooks, transform)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_utils_internal.py", line 97, in wrapper_function
    return function(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 793, in compile_inner
    return _compile_inner(code, one_graph, hooks, transform)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 832, in _compile_inner
    out_code = transform_code_object(code, transform)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/bytecode_transformation.py", line 1424, in transform_code_object
    transformations(instructions, code_options)
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 267, in _fn
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 753, in transform
    tracer.run()
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 3497, in run
    super().run()
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 1363, in run
    while self.step():
          ^^^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 1267, in step
    self.dispatch_table[inst.opcode](self, inst)
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 840, in wrapper
    unimplemented_v2(
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/exc.py", line 527, in unimplemented_v2
    raise Unsupported(msg) from from_exc
torch._dynamo.exc.Unsupported: Graph break under GenericContextWrappingVariable
  Explanation: Attempted to graph break in an active context manager(s) that doesn't support graph breaking.
  Hint: Move the offending context manager(s) to outside the compiled region.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: Active generic context managers: [GenericContextWrappingVariable(CustomCtxManager)]

from user code:
   File "/tmp/ipykernel_231646/4148913404.py", line 10, in fn
    torch._dynamo.graph_break()

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"

Traceback (most recent call last):
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 834, in wrapper
    return inner_fn(self, inst)
           ^^^^^^^^^^^^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 2910, in CALL
    self._call(inst)
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 2904, in _call
    self.call_function(fn, args, kwargs)
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 1193, in call_function
    self.push(fn.call_function(self, args, kwargs))  # type: ignore[arg-type]
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/variables/lazy.py", line 201, in realize_and_forward
    return getattr(self.realize(), name)(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/variables/functions.py", line 1398, in call_function
    unimplemented_v2(
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/exc.py", line 528, in unimplemented_v2
    raise Unsupported(msg)
torch._dynamo.exc.Unsupported: Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 1272, in __call__
    result = self._inner_convert(
             ^^^^^^^^^^^^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 629, in __call__
    return _compile(
           ^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 1111, in _compile
    guarded_code = compile_inner(code, one_graph, hooks, transform)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_utils_internal.py", line 97, in wrapper_function
    return function(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 793, in compile_inner
    return _compile_inner(code, one_graph, hooks, transform)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 832, in _compile_inner
    out_code = transform_code_object(code, transform)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/bytecode_transformation.py", line 1424, in transform_code_object
    transformations(instructions, code_options)
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 267, in _fn
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 753, in transform
    tracer.run()
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 3497, in run
    super().run()
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 1363, in run
    while self.step():
          ^^^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 1267, in step
    self.dispatch_table[inst.opcode](self, inst)
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 840, in wrapper
    unimplemented_v2(
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/exc.py", line 527, in unimplemented_v2
    raise Unsupported(msg) from from_exc
torch._dynamo.exc.Unsupported: Graph break under GenericContextWrappingVariable
  Explanation: Attempted to graph break in an active context manager(s) that doesn't support graph breaking.
  Hint: Move the offending context manager(s) to outside the compiled region.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: Active generic context managers: [GenericContextWrappingVariable(CustomCtxManager)]

from user code:
   File "/tmp/ipykernel_231646/4148913404.py", line 10, in fn
    torch._dynamo.graph_break()

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"

skipping because no torch.* __enter__             /tmp/ipykernel_231646/4148913404.py 2
skipping because no torch.* __exit__             /tmp/ipykernel_231646/4148913404.py 4

```

```
tensor([1.6601, 1.7491, 2.6870])

```

We can avoid skipping by moving the graph break outside of the context manager: 

```
@torch.compile
def fn(x):
    with CustomCtxManager():
        x = x + 1
    torch._dynamo.graph_break()
    with CustomCtxManager():
        return x + 1
fn(torch.randn(3))

```

```
torchdynamo start compiling fn /tmp/ipykernel_231646/2124425154.py:1, stack (elided 5 frames):
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
  File "/tmp/ipykernel_231646/2124425154.py", line 8, in <module>
    fn(torch.randn(3))

Step 1: torchdynamo start tracing fn /tmp/ipykernel_231646/2124425154.py:1
create_env
TRACE starts_line /tmp/ipykernel_231646/2124425154.py:1 in fn
    @torch.compile
TRACE RESUME 0 []
TRACE starts_line /tmp/ipykernel_231646/2124425154.py:3 in fn
        with CustomCtxManager():
TRACE LOAD_GLOBAL CustomCtxManager []
TRACE CALL 0 [NullVariable, LazyVariableTracker()]
TRACE BEFORE_WITH None [GenericContextWrappingVariable(CustomCtxManager)]
TRACE inlined call __enter__ from /tmp/ipykernel_231646/2124425154.py:3 in fn
    with CustomCtxManager():
         ~~~~~~~~~~~~~~~~^^
INLINING <code object __enter__ at 0x7f111cb3ee80, file "/tmp/ipykernel_231646/4148913404.py", line 2>, inlined according trace_rules.lookup inlined by default
TRACE starts_line /tmp/ipykernel_231646/4148913404.py:2 in __enter__ (inline depth: 1)
        def __enter__(self):
TRACE RESUME 0 []
TRACE starts_line /tmp/ipykernel_231646/4148913404.py:3 in __enter__ (inline depth: 1)
            pass
TRACE RETURN_CONST None []
DONE INLINING <code object __enter__ at 0x7f111cb3ee80, file "/tmp/ipykernel_231646/4148913404.py", line 2>
TRACE POP_TOP None [WithExitFunctionVariable(), ConstantVariable(NoneType: None)]
TRACE starts_line /tmp/ipykernel_231646/2124425154.py:4 in fn
            x = x + 1
TRACE LOAD_FAST x [WithExitFunctionVariable()]
TRACE LOAD_CONST 1 [WithExitFunctionVariable(), LazyVariableTracker()]
TRACE BINARY_OP 0 [WithExitFunctionVariable(), LazyVariableTracker(), ConstantVariable(int: 1)]
wrap_to_fake L['x'] (3,) StatefulSymbolicContext(dynamic_sizes=[<DimDynamic.STATIC: 2>], dynamic_strides=[<DimDynamic.INFER_STRIDE: 4>], constraint_sizes=[None], constraint_strides=[None], specialize_on=[[]], view_base_context=None, tensor_source=LocalSource(local_name='x', is_input=True, dynamism=None, is_derefed_cell_contents=False), shape_env_to_source_to_symbol_cache={}) <class 'torch.Tensor'>
create_graph_input L_x_ L['x'] FakeTensor(..., size=(3,)) at debug_level 0 before=False
TRACE FX call add from /tmp/ipykernel_231646/2124425154.py:4 in fn
        x = x + 1
            ~~^~~
TRACE STORE_FAST x [WithExitFunctionVariable(), TensorVariable()]
TRACE starts_line /tmp/ipykernel_231646/2124425154.py:3 in fn
        with CustomCtxManager():
TRACE LOAD_CONST None [WithExitFunctionVariable()]
TRACE LOAD_CONST None [WithExitFunctionVariable(), ConstantVariable(NoneType: None)]
TRACE LOAD_CONST None [WithExitFunctionVariable(), ConstantVariable(NoneType: None), ConstantVariable(NoneType: None)]
TRACE CALL 2 [WithExitFunctionVariable(), ConstantVariable(NoneType: None), ConstantVariable(NoneType: None), ConstantVariable(NoneType: None)]
TRACE inlined call __exit__ from /tmp/ipykernel_231646/2124425154.py:3 in fn
    with CustomCtxManager():
         ~~~~~~~~~~~~~~~~^^
INLINING <code object __exit__ at 0x7f0fd85612e0, file "/tmp/ipykernel_231646/4148913404.py", line 4>, inlined according trace_rules.lookup inlined by default
TRACE starts_line /tmp/ipykernel_231646/4148913404.py:4 in __exit__ (inline depth: 1)
        def __exit__(self, exc_type, exc_value, traceback):
TRACE RESUME 0 []
TRACE starts_line /tmp/ipykernel_231646/4148913404.py:5 in __exit__ (inline depth: 1)
            pass
TRACE RETURN_CONST None []
DONE INLINING <code object __exit__ at 0x7f0fd85612e0, file "/tmp/ipykernel_231646/4148913404.py", line 4>
TRACE POP_TOP None [ConstantVariable(NoneType: None)]
TRACE starts_line /tmp/ipykernel_231646/2124425154.py:5 in fn
        torch._dynamo.graph_break()
TRACE LOAD_GLOBAL torch []
TRACE LOAD_ATTR _dynamo [LazyVariableTracker()]
TRACE LOAD_ATTR graph_break [LazyVariableTracker()]
TRACE CALL 0 [NullVariable, LazyVariableTracker()]
Graph break in user code at /tmp/ipykernel_231646/2124425154.py:5
Graph Break Reason: Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

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
  File "/tmp/ipykernel_231646/2124425154.py", line 8, in <module>
    fn(torch.randn(3))
  File "/tmp/ipykernel_231646/2124425154.py", line 5, in fn
    torch._dynamo.graph_break()

Restarting analysis due to _dynamo/symbolic_convert.py:223 in fail_and_restart_analysis
Step 1: torchdynamo start tracing fn /tmp/ipykernel_231646/2124425154.py:1
create_env
TRACE starts_line /tmp/ipykernel_231646/2124425154.py:1 in fn
    @torch.compile
TRACE RESUME 0 []
TRACE starts_line /tmp/ipykernel_231646/2124425154.py:3 in fn
        with CustomCtxManager():
TRACE LOAD_GLOBAL CustomCtxManager []
TRACE CALL 0 [NullVariable, LazyVariableTracker()]
TRACE BEFORE_WITH None [GenericContextWrappingVariable(CustomCtxManager)]
TRACE inlined call __enter__ from /tmp/ipykernel_231646/2124425154.py:3 in fn
    with CustomCtxManager():
         ~~~~~~~~~~~~~~~~^^
INLINING <code object __enter__ at 0x7f111cb3ee80, file "/tmp/ipykernel_231646/4148913404.py", line 2>, inlined according trace_rules.lookup inlined by default
TRACE starts_line /tmp/ipykernel_231646/4148913404.py:2 in __enter__ (inline depth: 1)
        def __enter__(self):
TRACE RESUME 0 []
TRACE starts_line /tmp/ipykernel_231646/4148913404.py:3 in __enter__ (inline depth: 1)
            pass
TRACE RETURN_CONST None []
DONE INLINING <code object __enter__ at 0x7f111cb3ee80, file "/tmp/ipykernel_231646/4148913404.py", line 2>
TRACE POP_TOP None [WithExitFunctionVariable(), ConstantVariable(NoneType: None)]
TRACE starts_line /tmp/ipykernel_231646/2124425154.py:4 in fn
            x = x + 1
TRACE LOAD_FAST x [WithExitFunctionVariable()]
TRACE LOAD_CONST 1 [WithExitFunctionVariable(), LazyVariableTracker()]
TRACE BINARY_OP 0 [WithExitFunctionVariable(), LazyVariableTracker(), ConstantVariable(int: 1)]
wrap_to_fake L['x'] (3,) StatefulSymbolicContext(dynamic_sizes=[<DimDynamic.STATIC: 2>], dynamic_strides=[<DimDynamic.INFER_STRIDE: 4>], constraint_sizes=[None], constraint_strides=[None], specialize_on=[[]], view_base_context=None, tensor_source=LocalSource(local_name='x', is_input=True, dynamism=None, is_derefed_cell_contents=False), shape_env_to_source_to_symbol_cache={}) <class 'torch.Tensor'>
create_graph_input L_x_ L['x'] FakeTensor(..., size=(3,)) at debug_level 0 before=False
TRACE FX call add from /tmp/ipykernel_231646/2124425154.py:4 in fn
        x = x + 1
            ~~^~~
TRACE STORE_FAST x [WithExitFunctionVariable(), TensorVariable()]
TRACE starts_line /tmp/ipykernel_231646/2124425154.py:3 in fn
        with CustomCtxManager():
TRACE LOAD_CONST None [WithExitFunctionVariable()]
TRACE LOAD_CONST None [WithExitFunctionVariable(), ConstantVariable(NoneType: None)]
TRACE LOAD_CONST None [WithExitFunctionVariable(), ConstantVariable(NoneType: None), ConstantVariable(NoneType: None)]
TRACE CALL 2 [WithExitFunctionVariable(), ConstantVariable(NoneType: None), ConstantVariable(NoneType: None), ConstantVariable(NoneType: None)]
TRACE inlined call __exit__ from /tmp/ipykernel_231646/2124425154.py:3 in fn
    with CustomCtxManager():
         ~~~~~~~~~~~~~~~~^^
INLINING <code object __exit__ at 0x7f0fd85612e0, file "/tmp/ipykernel_231646/4148913404.py", line 4>, inlined according trace_rules.lookup inlined by default
TRACE starts_line /tmp/ipykernel_231646/4148913404.py:4 in __exit__ (inline depth: 1)
        def __exit__(self, exc_type, exc_value, traceback):
TRACE RESUME 0 []
TRACE starts_line /tmp/ipykernel_231646/4148913404.py:5 in __exit__ (inline depth: 1)
            pass
TRACE RETURN_CONST None []
DONE INLINING <code object __exit__ at 0x7f0fd85612e0, file "/tmp/ipykernel_231646/4148913404.py", line 4>
TRACE POP_TOP None [ConstantVariable(NoneType: None)]
TRACE starts_line /tmp/ipykernel_231646/2124425154.py:5 in fn
        torch._dynamo.graph_break()
TRACE LOAD_GLOBAL torch []
TRACE LOAD_ATTR _dynamo [LazyVariableTracker()]
TRACE LOAD_ATTR graph_break [LazyVariableTracker()]
TRACE CALL 0 [NullVariable, LazyVariableTracker()]
COMPILING GRAPH due to GraphCompileReason(reason='Call to `torch._dynamo.graph_break()`n  Explanation: User-inserted graph break. Message: Nonen  Hint: Remove the `torch._dynamo.graph_break()` call.nn  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`n', user_stack=[<FrameSummary file /tmp/ipykernel_231646/2124425154.py, line 5 in fn>], graph_break=True)
TRACED GRAPH
 ===== __compiled_fn_20_2c9f71c5_df28_43e9_9381_a6f51cbe5936 =====
 /home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/fx/_lazy_graph_module.py class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3][1]cpu"):
        l_x_ = L_x_
        
         # File: /tmp/ipykernel_231646/2124425154.py:4 in fn, code: x = x + 1
        x: "f32[3][1]cpu" = l_x_ + 1;  l_x_ = None
        return (x,)
        

Step 2: calling compiler function eager
Step 2: done compiler function eager
produce_guards
track_symint L['x'].size()[0] 3 None
track_symint L['x'].stride()[0] 1 None
track_symint L['x'].storage_offset() 0 None
Skipping guard L['x'].size()[0] == 3
Skipping guard L['x'].stride()[0] == 1
Skipping guard L['x'].storage_offset() == 0
GUARDS:

TREE_GUARD_MANAGER:
+- RootGuardManager
| +- LAMBDA_GUARD: torch._functorch.aot_autograd.utils.top_saved_tensors_hooks ids == None  # _dynamo/output_graph.py:633 in init_ambient_guards
| +- DEFAULT_DEVICE: utils_device.CURRENT_DEVICE == None                           # _dynamo/output_graph.py:621 in init_ambient_guards
| +- GLOBAL_STATE: ___check_global_state()
| +- TORCH_FUNCTION_MODE_STACK: ___check_torch_function_mode_stack()
| +- GuardManager: source=L['x'], accessed_by=FrameLocalsGuardAccessor(key='x', framelocals_idx=0)
| | +- TENSOR_MATCH: check_tensor(L['x'], Tensor, DispatchKeySet(CPU, BackendSelect, ADInplaceOrView, AutogradCPU), torch.float32, device=None, requires_grad=False, size=[3], stride=[1])  # x = x + 1  # mp/ipykernel_231646/2124425154.py:4 in fn
| | +- NO_HASATTR: hasattr(L['x'], '_dynamo_dynamic_indices') == False           # x = x + 1  # mp/ipykernel_231646/2124425154.py:4 in fn
| +- GuardManager: source=G, accessed_by=GlobalsGuardAccessor
| | +- GuardManager: source=G['torch'], accessed_by=DictGetItemGuardAccessor('torch')
| | | +- ID_MATCH: ___check_obj_id(G['torch'], 139712098143824)                  # torch._dynamo.graph_break()  # mp/ipykernel_231646/2124425154.py:5 in fn
| | | +- GuardManager: source=G['torch']._dynamo, accessed_by=GetAttrGuardAccessor(_dynamo)
| | | | +- ID_MATCH: ___check_obj_id(G['torch']._dynamo, 139706212821616)          # torch._dynamo.graph_break()  # mp/ipykernel_231646/2124425154.py:5 in fn
| | | | +- GuardManager: source=G['torch']._dynamo.graph_break, accessed_by=GetAttrGuardAccessor(graph_break)
| | | | | +- ID_MATCH: ___check_obj_id(G['torch']._dynamo.graph_break, 139706044196288)  # torch._dynamo.graph_break()  # mp/ipykernel_231646/2124425154.py:5 in fn
| | +- GuardManager: source=G['CustomCtxManager'], accessed_by=DictGetItemGuardAccessor('CustomCtxManager')
| | | +- ID_MATCH: ___check_obj_id(G['CustomCtxManager'], 94855574301696)        # with CustomCtxManager():  # mp/ipykernel_231646/2124425154.py:3 in fn

Guard eval latency = 114.22 us
put_code_state: no cache key, skipping
run_gc_after_compile: running gc
torchdynamo start compiling torch_dynamo_resume_in_fn_at_5 /tmp/ipykernel_231646/2124425154.py:5, stack (elided 5 frames):
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
  File "/tmp/ipykernel_231646/2124425154.py", line 8, in <module>
    fn(torch.randn(3))
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/eval_frame.py", line 736, in compile_wrapper
    return fn(*args, **kwargs)

Step 1: torchdynamo start tracing torch_dynamo_resume_in_fn_at_5 /tmp/ipykernel_231646/2124425154.py:5
create_env
TRACE starts_line /tmp/ipykernel_231646/2124425154.py:5 in torch_dynamo_resume_in_fn_at_5
        torch._dynamo.graph_break()
TRACE RESUME 0 []
TRACE LOAD_FAST ___stack0 []
TRACE JUMP_FORWARD 114 [LazyVariableTracker()]
TRACE POP_TOP None [LazyVariableTracker()]
TRACE starts_line /tmp/ipykernel_231646/2124425154.py:6 in torch_dynamo_resume_in_fn_at_5
        with CustomCtxManager():
TRACE LOAD_GLOBAL CustomCtxManager []
TRACE CALL 0 [NullVariable, LazyVariableTracker()]
TRACE BEFORE_WITH None [GenericContextWrappingVariable(CustomCtxManager)]
TRACE inlined call __enter__ from /tmp/ipykernel_231646/2124425154.py:6 in torch_dynamo_resume_in_fn_at_5
    with CustomCtxManager():
         ~~~~~~~~~~~~~~~~^^
INLINING <code object __enter__ at 0x7f111cb3ee80, file "/tmp/ipykernel_231646/4148913404.py", line 2>, inlined according trace_rules.lookup inlined by default
TRACE starts_line /tmp/ipykernel_231646/4148913404.py:2 in __enter__ (inline depth: 1)
        def __enter__(self):
TRACE RESUME 0 []
TRACE starts_line /tmp/ipykernel_231646/4148913404.py:3 in __enter__ (inline depth: 1)
            pass
TRACE RETURN_CONST None []
DONE INLINING <code object __enter__ at 0x7f111cb3ee80, file "/tmp/ipykernel_231646/4148913404.py", line 2>
TRACE POP_TOP None [WithExitFunctionVariable(), ConstantVariable(NoneType: None)]
TRACE starts_line /tmp/ipykernel_231646/2124425154.py:7 in torch_dynamo_resume_in_fn_at_5
            return x + 1
TRACE LOAD_FAST x [WithExitFunctionVariable()]
TRACE LOAD_CONST 1 [WithExitFunctionVariable(), LazyVariableTracker()]
TRACE BINARY_OP 0 [WithExitFunctionVariable(), LazyVariableTracker(), ConstantVariable(int: 1)]
wrap_to_fake L['x'] (3,) StatefulSymbolicContext(dynamic_sizes=[<DimDynamic.STATIC: 2>], dynamic_strides=[<DimDynamic.INFER_STRIDE: 4>], constraint_sizes=[None], constraint_strides=[None], specialize_on=[[]], view_base_context=None, tensor_source=LocalSource(local_name='x', is_input=True, dynamism=None, is_derefed_cell_contents=False), shape_env_to_source_to_symbol_cache={}) <class 'torch.Tensor'>
create_graph_input L_x_ L['x'] FakeTensor(..., size=(3,)) at debug_level 0 before=False
TRACE FX call add from /tmp/ipykernel_231646/2124425154.py:7 in torch_dynamo_resume_in_fn_at_5
        return x + 1
               ~~^~~
TRACE starts_line /tmp/ipykernel_231646/2124425154.py:6 in torch_dynamo_resume_in_fn_at_5
        with CustomCtxManager():
TRACE SWAP 2 [WithExitFunctionVariable(), TensorVariable()]
TRACE LOAD_CONST None [TensorVariable(), WithExitFunctionVariable()]
TRACE LOAD_CONST None [TensorVariable(), WithExitFunctionVariable(), ConstantVariable(NoneType: None)]
TRACE LOAD_CONST None [TensorVariable(), WithExitFunctionVariable(), ConstantVariable(NoneType: None), ConstantVariable(NoneType: None)]
TRACE CALL 2 [TensorVariable(), WithExitFunctionVariable(), ConstantVariable(NoneType: None), ConstantVariable(NoneType: None), ConstantVariable(NoneType: None)]
TRACE inlined call __exit__ from /tmp/ipykernel_231646/2124425154.py:6 in torch_dynamo_resume_in_fn_at_5
    with CustomCtxManager():
         ~~~~~~~~~~~~~~~~^^
INLINING <code object __exit__ at 0x7f0fd85612e0, file "/tmp/ipykernel_231646/4148913404.py", line 4>, inlined according trace_rules.lookup inlined by default
TRACE starts_line /tmp/ipykernel_231646/4148913404.py:4 in __exit__ (inline depth: 1)
        def __exit__(self, exc_type, exc_value, traceback):
TRACE RESUME 0 []
TRACE starts_line /tmp/ipykernel_231646/4148913404.py:5 in __exit__ (inline depth: 1)
            pass
TRACE RETURN_CONST None []
DONE INLINING <code object __exit__ at 0x7f0fd85612e0, file "/tmp/ipykernel_231646/4148913404.py", line 4>
TRACE POP_TOP None [TensorVariable(), ConstantVariable(NoneType: None)]
TRACE RETURN_VALUE None [TensorVariable()]
Step 1: torchdynamo done tracing torch_dynamo_resume_in_fn_at_5 (RETURN_VALUE)
RETURN_VALUE triggered compile
COMPILING GRAPH due to GraphCompileReason(reason='return_value', user_stack=[<FrameSummary file /tmp/ipykernel_231646/2124425154.py, line 6 in torch_dynamo_resume_in_fn_at_5>], graph_break=False)
TRACED GRAPH
 ===== __compiled_fn_23_231fb581_c24e_4b59_8c27_751c2a0e9cc5 =====
 /home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/fx/_lazy_graph_module.py class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3][1]cpu"):
        l_x_ = L_x_
        
         # File: /tmp/ipykernel_231646/2124425154.py:7 in torch_dynamo_resume_in_fn_at_5, code: return x + 1
        add: "f32[3][1]cpu" = l_x_ + 1;  l_x_ = None
        return (add,)
        

Step 2: calling compiler function eager
Step 2: done compiler function eager
produce_guards
track_symint L['x'].size()[0] 3 None
track_symint L['x'].stride()[0] 1 None
track_symint L['x'].storage_offset() 0 None
Skipping guard L['x'].size()[0] == 3
Skipping guard L['x'].stride()[0] == 1
Skipping guard L['x'].storage_offset() == 0
GUARDS:

TREE_GUARD_MANAGER:
+- RootGuardManager
| +- LAMBDA_GUARD: torch._functorch.aot_autograd.utils.top_saved_tensors_hooks ids == None  # _dynamo/output_graph.py:633 in init_ambient_guards
| +- DEFAULT_DEVICE: utils_device.CURRENT_DEVICE == None                           # _dynamo/output_graph.py:621 in init_ambient_guards
| +- GLOBAL_STATE: ___check_global_state()
| +- TORCH_FUNCTION_MODE_STACK: ___check_torch_function_mode_stack()
| +- GuardManager: source=L['x'], accessed_by=FrameLocalsGuardAccessor(key='x', framelocals_idx=1)
| | +- TENSOR_MATCH: check_tensor(L['x'], Tensor, DispatchKeySet(CPU, BackendSelect, ADInplaceOrView, AutogradCPU), torch.float32, device=None, requires_grad=False, size=[3], stride=[1])  # return x + 1  # mp/ipykernel_231646/2124425154.py:7 in torch_dynamo_resume_in_fn_at_5
| | +- NO_HASATTR: hasattr(L['x'], '_dynamo_dynamic_indices') == False           # return x + 1  # mp/ipykernel_231646/2124425154.py:7 in torch_dynamo_resume_in_fn_at_5
| +- GuardManager: source=G, accessed_by=GlobalsGuardAccessor
| | +- GuardManager: source=G['CustomCtxManager'], accessed_by=DictGetItemGuardAccessor('CustomCtxManager')
| | | +- ID_MATCH: ___check_obj_id(G['CustomCtxManager'], 94855574301696)        # with CustomCtxManager():  # mp/ipykernel_231646/2124425154.py:6 in torch_dynamo_resume_in_fn_at_5

Guard eval latency = 569.92 us
put_code_state: no cache key, skipping
run_gc_after_compile: running gc

```

```
tensor([2.9352, 2.3305, 2.1486])

```

There are some context managers where Dynamo can resume after a graph break.
Some of these can be found in `supported_ctx_manager_classes`  in `torch/_dynamo/variables/torch.py`  .
In general, any context manager represented by a `ContextWrappingVariable`  subclass in `torch/_dynamo/variables/ctx_manager.py`  support resuming after a graph break. For example: 

```
import contextlib
@torch.compile
def fn(x):
    with contextlib.nullcontext():
        with torch.no_grad():
            x = x + 1
            torch._dynamo.graph_break()
            return x + 1
fn(torch.randn(3))

```

```
torchdynamo start compiling fn /tmp/ipykernel_231646/3152636365.py:2, stack (elided 5 frames):
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
  File "/tmp/ipykernel_231646/3152636365.py", line 9, in <module>
    fn(torch.randn(3))

Step 1: torchdynamo start tracing fn /tmp/ipykernel_231646/3152636365.py:2
create_env
TRACE starts_line /tmp/ipykernel_231646/3152636365.py:2 in fn
    @torch.compile
TRACE RESUME 0 []
TRACE starts_line /tmp/ipykernel_231646/3152636365.py:4 in fn
        with contextlib.nullcontext():
TRACE LOAD_GLOBAL contextlib []
TRACE LOAD_ATTR nullcontext [LazyVariableTracker()]
TRACE CALL 0 [NullVariable, LazyVariableTracker()]
TRACE BEFORE_WITH None [NullContextVariable()]
TRACE POP_TOP None [WithExitFunctionVariable(), ConstantVariable(NoneType: None)]
TRACE starts_line /tmp/ipykernel_231646/3152636365.py:5 in fn
            with torch.no_grad():
TRACE LOAD_GLOBAL torch [WithExitFunctionVariable()]
TRACE LOAD_ATTR no_grad [WithExitFunctionVariable(), LazyVariableTracker()]
TRACE CALL 0 [WithExitFunctionVariable(), NullVariable, LazyVariableTracker()]
TRACE BEFORE_WITH None [WithExitFunctionVariable(), GradModeVariable()]
TRACE POP_TOP None [WithExitFunctionVariable(), WithExitFunctionVariable(), ConstantVariable(NoneType: None)]
TRACE starts_line /tmp/ipykernel_231646/3152636365.py:6 in fn
                x = x + 1
TRACE LOAD_FAST x [WithExitFunctionVariable(), WithExitFunctionVariable()]
TRACE LOAD_CONST 1 [WithExitFunctionVariable(), WithExitFunctionVariable(), LazyVariableTracker()]
TRACE BINARY_OP 0 [WithExitFunctionVariable(), WithExitFunctionVariable(), LazyVariableTracker(), ConstantVariable(int: 1)]
wrap_to_fake L['x'] (3,) StatefulSymbolicContext(dynamic_sizes=[<DimDynamic.STATIC: 2>], dynamic_strides=[<DimDynamic.INFER_STRIDE: 4>], constraint_sizes=[None], constraint_strides=[None], specialize_on=[[]], view_base_context=None, tensor_source=LocalSource(local_name='x', is_input=True, dynamism=None, is_derefed_cell_contents=False), shape_env_to_source_to_symbol_cache={}) <class 'torch.Tensor'>
create_graph_input L_x_ L['x'] FakeTensor(..., size=(3,)) at debug_level 0 before=False
TRACE FX call add from /tmp/ipykernel_231646/3152636365.py:6 in fn
            x = x + 1
                ~~^~~
TRACE STORE_FAST x [WithExitFunctionVariable(), WithExitFunctionVariable(), TensorVariable()]
TRACE starts_line /tmp/ipykernel_231646/3152636365.py:7 in fn
                torch._dynamo.graph_break()
TRACE LOAD_GLOBAL torch [WithExitFunctionVariable(), WithExitFunctionVariable()]
TRACE LOAD_ATTR _dynamo [WithExitFunctionVariable(), WithExitFunctionVariable(), LazyVariableTracker()]
TRACE LOAD_ATTR graph_break [WithExitFunctionVariable(), WithExitFunctionVariable(), LazyVariableTracker()]
TRACE CALL 0 [WithExitFunctionVariable(), WithExitFunctionVariable(), NullVariable, LazyVariableTracker()]
Graph break in user code at /tmp/ipykernel_231646/3152636365.py:7
Graph Break Reason: Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

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
  File "/tmp/ipykernel_231646/3152636365.py", line 9, in <module>
    fn(torch.randn(3))
  File "/tmp/ipykernel_231646/3152636365.py", line 7, in fn
    torch._dynamo.graph_break()

Restarting analysis due to _dynamo/symbolic_convert.py:223 in fail_and_restart_analysis
Step 1: torchdynamo start tracing fn /tmp/ipykernel_231646/3152636365.py:2
create_env
TRACE starts_line /tmp/ipykernel_231646/3152636365.py:2 in fn
    @torch.compile
TRACE RESUME 0 []
TRACE starts_line /tmp/ipykernel_231646/3152636365.py:4 in fn
        with contextlib.nullcontext():
TRACE LOAD_GLOBAL contextlib []
TRACE LOAD_ATTR nullcontext [LazyVariableTracker()]
TRACE CALL 0 [NullVariable, LazyVariableTracker()]
TRACE BEFORE_WITH None [NullContextVariable()]
TRACE POP_TOP None [WithExitFunctionVariable(), ConstantVariable(NoneType: None)]
TRACE starts_line /tmp/ipykernel_231646/3152636365.py:5 in fn
            with torch.no_grad():
TRACE LOAD_GLOBAL torch [WithExitFunctionVariable()]
TRACE LOAD_ATTR no_grad [WithExitFunctionVariable(), LazyVariableTracker()]
TRACE CALL 0 [WithExitFunctionVariable(), NullVariable, LazyVariableTracker()]
TRACE BEFORE_WITH None [WithExitFunctionVariable(), GradModeVariable()]
TRACE POP_TOP None [WithExitFunctionVariable(), WithExitFunctionVariable(), ConstantVariable(NoneType: None)]
TRACE starts_line /tmp/ipykernel_231646/3152636365.py:6 in fn
                x = x + 1
TRACE LOAD_FAST x [WithExitFunctionVariable(), WithExitFunctionVariable()]
TRACE LOAD_CONST 1 [WithExitFunctionVariable(), WithExitFunctionVariable(), LazyVariableTracker()]
TRACE BINARY_OP 0 [WithExitFunctionVariable(), WithExitFunctionVariable(), LazyVariableTracker(), ConstantVariable(int: 1)]
wrap_to_fake L['x'] (3,) StatefulSymbolicContext(dynamic_sizes=[<DimDynamic.STATIC: 2>], dynamic_strides=[<DimDynamic.INFER_STRIDE: 4>], constraint_sizes=[None], constraint_strides=[None], specialize_on=[[]], view_base_context=None, tensor_source=LocalSource(local_name='x', is_input=True, dynamism=None, is_derefed_cell_contents=False), shape_env_to_source_to_symbol_cache={}) <class 'torch.Tensor'>
create_graph_input L_x_ L['x'] FakeTensor(..., size=(3,)) at debug_level 0 before=False
TRACE FX call add from /tmp/ipykernel_231646/3152636365.py:6 in fn
            x = x + 1
                ~~^~~
TRACE STORE_FAST x [WithExitFunctionVariable(), WithExitFunctionVariable(), TensorVariable()]
TRACE starts_line /tmp/ipykernel_231646/3152636365.py:7 in fn
                torch._dynamo.graph_break()
TRACE LOAD_GLOBAL torch [WithExitFunctionVariable(), WithExitFunctionVariable()]
TRACE LOAD_ATTR _dynamo [WithExitFunctionVariable(), WithExitFunctionVariable(), LazyVariableTracker()]
TRACE LOAD_ATTR graph_break [WithExitFunctionVariable(), WithExitFunctionVariable(), LazyVariableTracker()]
TRACE CALL 0 [WithExitFunctionVariable(), WithExitFunctionVariable(), NullVariable, LazyVariableTracker()]
COMPILING GRAPH due to GraphCompileReason(reason='Call to `torch._dynamo.graph_break()`n  Explanation: User-inserted graph break. Message: Nonen  Hint: Remove the `torch._dynamo.graph_break()` call.nn  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`n', user_stack=[<FrameSummary file /tmp/ipykernel_231646/3152636365.py, line 7 in fn>], graph_break=True)
TRACED GRAPH
 ===== __compiled_fn_26_41567203_c867_4e94_af7e_058abf511094 =====
 /home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/fx/_lazy_graph_module.py class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3][1]cpu"):
        l_x_ = L_x_
        
        # No stacktrace found for following nodes
        _set_grad_enabled = torch._C._set_grad_enabled(False);  _set_grad_enabled = None
        
         # File: /tmp/ipykernel_231646/3152636365.py:6 in fn, code: x = x + 1
        x: "f32[3][1]cpu" = l_x_ + 1;  l_x_ = None
        
        # No stacktrace found for following nodes
        _set_grad_enabled_1 = torch._C._set_grad_enabled(True);  _set_grad_enabled_1 = None
        return (x,)
        

Step 2: calling compiler function eager
Step 2: done compiler function eager
produce_guards
track_symint L['x'].size()[0] 3 None
track_symint L['x'].stride()[0] 1 None
track_symint L['x'].storage_offset() 0 None
Skipping guard L['x'].size()[0] == 3
Skipping guard L['x'].stride()[0] == 1
Skipping guard L['x'].storage_offset() == 0
GUARDS:

TREE_GUARD_MANAGER:
+- RootGuardManager
| +- LAMBDA_GUARD: torch._functorch.aot_autograd.utils.top_saved_tensors_hooks ids == None  # _dynamo/output_graph.py:633 in init_ambient_guards
| +- DEFAULT_DEVICE: utils_device.CURRENT_DEVICE == None                           # _dynamo/output_graph.py:621 in init_ambient_guards
| +- GLOBAL_STATE: ___check_global_state()
| +- TORCH_FUNCTION_MODE_STACK: ___check_torch_function_mode_stack()
| +- GuardManager: source=L['x'], accessed_by=FrameLocalsGuardAccessor(key='x', framelocals_idx=0)
| | +- TENSOR_MATCH: check_tensor(L['x'], Tensor, DispatchKeySet(CPU, BackendSelect, ADInplaceOrView, AutogradCPU), torch.float32, device=None, requires_grad=False, size=[3], stride=[1])  # x = x + 1  # mp/ipykernel_231646/3152636365.py:6 in fn
| | +- NO_HASATTR: hasattr(L['x'], '_dynamo_dynamic_indices') == False           # x = x + 1  # mp/ipykernel_231646/3152636365.py:6 in fn
| +- GuardManager: source=G, accessed_by=GlobalsGuardAccessor
| | +- GuardManager: source=G['torch'], accessed_by=DictGetItemGuardAccessor('torch')
| | | +- ID_MATCH: ___check_obj_id(G['torch'], 139712098143824)                  # with torch.no_grad():  # mp/ipykernel_231646/3152636365.py:5 in fn
| | | +- GuardManager: source=G['torch']._dynamo, accessed_by=GetAttrGuardAccessor(_dynamo)
| | | | +- ID_MATCH: ___check_obj_id(G['torch']._dynamo, 139706212821616)          # torch._dynamo.graph_break()  # mp/ipykernel_231646/3152636365.py:7 in fn
| | | | +- GuardManager: source=G['torch']._dynamo.graph_break, accessed_by=GetAttrGuardAccessor(graph_break)
| | | | | +- ID_MATCH: ___check_obj_id(G['torch']._dynamo.graph_break, 139706044196288)  # torch._dynamo.graph_break()  # mp/ipykernel_231646/3152636365.py:7 in fn
| | | +- GuardManager: source=G['torch'].no_grad, accessed_by=GetAttrGuardAccessor(no_grad)
| | | | +- ID_MATCH: ___check_obj_id(G['torch'].no_grad, 94855522131216)           # with torch.no_grad():  # mp/ipykernel_231646/3152636365.py:5 in fn
| | +- GuardManager: source=G['contextlib'], accessed_by=DictGetItemGuardAccessor('contextlib')
| | | +- ID_MATCH: ___check_obj_id(G['contextlib'], 139712139833280)             # with contextlib.nullcontext():  # mp/ipykernel_231646/3152636365.py:4 in fn
| | | +- GuardManager: source=G['contextlib'].nullcontext, accessed_by=GetAttrGuardAccessor(nullcontext)
| | | | +- ID_MATCH: ___check_obj_id(G['contextlib'].nullcontext, 94855428429456)  # with contextlib.nullcontext():  # mp/ipykernel_231646/3152636365.py:4 in fn

Guard eval latency = 288.70 us
put_code_state: no cache key, skipping
run_gc_after_compile: running gc
skipping because no torch.* __init__             /home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/contextlib.py 772
skipping because no torch.* __enter__             /home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/contextlib.py 775
skipping: __init__ (reason: in skipfiles, file: /home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/autograd/grad_mode.py)
skipping: __enter__ (reason: in skipfiles, file: /home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/autograd/grad_mode.py)
skipping: __exit__ (reason: in skipfiles, file: /home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/autograd/grad_mode.py)
skipping because no torch.* __exit__             /home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/contextlib.py 778
torchdynamo start compiling torch_dynamo_resume_in_fn_at_7 /tmp/ipykernel_231646/3152636365.py:7, stack (elided 5 frames):
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
  File "/tmp/ipykernel_231646/3152636365.py", line 9, in <module>
    fn(torch.randn(3))
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/eval_frame.py", line 736, in compile_wrapper
    return fn(*args, **kwargs)

Step 1: torchdynamo start tracing torch_dynamo_resume_in_fn_at_7 /tmp/ipykernel_231646/3152636365.py:7
create_env
TRACE starts_line /tmp/ipykernel_231646/3152636365.py:7 in torch_dynamo_resume_in_fn_at_7
                torch._dynamo.graph_break()
TRACE RESUME 0 []
TRACE LOAD_FAST ___stack0 []
TRACE PUSH_NULL None [LazyVariableTracker()]
TRACE SWAP 2 [LazyVariableTracker(), NullVariable]
TRACE CALL 0 [NullVariable, LazyVariableTracker()]
TRACE NOP None [NullContextVariable()]
TRACE BEFORE_WITH None [NullContextVariable()]
TRACE POP_TOP None [WithExitFunctionVariable(), ConstantVariable(NoneType: None)]
TRACE NOP None [WithExitFunctionVariable()]
TRACE LOAD_FAST ___stack1 [WithExitFunctionVariable()]
TRACE PUSH_NULL None [WithExitFunctionVariable(), LazyVariableTracker()]
TRACE SWAP 2 [WithExitFunctionVariable(), LazyVariableTracker(), NullVariable]
TRACE LOAD_CONST False [WithExitFunctionVariable(), NullVariable, LazyVariableTracker()]
TRACE CALL 1 [WithExitFunctionVariable(), NullVariable, LazyVariableTracker(), ConstantVariable(bool: False)]
TRACE NOP None [WithExitFunctionVariable(), GradModeVariable()]
TRACE BEFORE_WITH None [WithExitFunctionVariable(), GradModeVariable()]
TRACE POP_TOP None [WithExitFunctionVariable(), WithExitFunctionVariable(), ConstantVariable(NoneType: None)]
TRACE NOP None [WithExitFunctionVariable(), WithExitFunctionVariable()]
TRACE LOAD_FAST ___stack2 [WithExitFunctionVariable(), WithExitFunctionVariable()]
TRACE JUMP_FORWARD 314 [WithExitFunctionVariable(), WithExitFunctionVariable(), LazyVariableTracker()]
TRACE POP_TOP None [WithExitFunctionVariable(), WithExitFunctionVariable(), LazyVariableTracker()]
TRACE starts_line /tmp/ipykernel_231646/3152636365.py:8 in torch_dynamo_resume_in_fn_at_7
                return x + 1
TRACE LOAD_FAST x [WithExitFunctionVariable(), WithExitFunctionVariable()]
TRACE LOAD_CONST 1 [WithExitFunctionVariable(), WithExitFunctionVariable(), LazyVariableTracker()]
TRACE BINARY_OP 0 [WithExitFunctionVariable(), WithExitFunctionVariable(), LazyVariableTracker(), ConstantVariable(int: 1)]
wrap_to_fake L['x'] (3,) StatefulSymbolicContext(dynamic_sizes=[<DimDynamic.STATIC: 2>], dynamic_strides=[<DimDynamic.INFER_STRIDE: 4>], constraint_sizes=[None], constraint_strides=[None], specialize_on=[[]], view_base_context=None, tensor_source=LocalSource(local_name='x', is_input=True, dynamism=None, is_derefed_cell_contents=False), shape_env_to_source_to_symbol_cache={}) <class 'torch.Tensor'>
create_graph_input L_x_ L['x'] FakeTensor(..., size=(3,)) at debug_level 0 before=False
TRACE FX call add from /tmp/ipykernel_231646/3152636365.py:8 in torch_dynamo_resume_in_fn_at_7
            return x + 1
                   ~~^~~
TRACE starts_line /tmp/ipykernel_231646/3152636365.py:5 in torch_dynamo_resume_in_fn_at_7
            with torch.no_grad():
TRACE SWAP 2 [WithExitFunctionVariable(), WithExitFunctionVariable(), TensorVariable()]
TRACE LOAD_CONST None [WithExitFunctionVariable(), TensorVariable(), WithExitFunctionVariable()]
TRACE LOAD_CONST None [WithExitFunctionVariable(), TensorVariable(), WithExitFunctionVariable(), ConstantVariable(NoneType: None)]
TRACE LOAD_CONST None [WithExitFunctionVariable(), TensorVariable(), WithExitFunctionVariable(), ConstantVariable(NoneType: None), ConstantVariable(NoneType: None)]
TRACE CALL 2 [WithExitFunctionVariable(), TensorVariable(), WithExitFunctionVariable(), ConstantVariable(NoneType: None), ConstantVariable(NoneType: None), ConstantVariable(NoneType: None)]
TRACE POP_TOP None [WithExitFunctionVariable(), TensorVariable(), ConstantVariable(NoneType: None)]
TRACE starts_line /tmp/ipykernel_231646/3152636365.py:4 in torch_dynamo_resume_in_fn_at_7
        with contextlib.nullcontext():
TRACE SWAP 2 [WithExitFunctionVariable(), TensorVariable()]
TRACE LOAD_CONST None [TensorVariable(), WithExitFunctionVariable()]
TRACE LOAD_CONST None [TensorVariable(), WithExitFunctionVariable(), ConstantVariable(NoneType: None)]
TRACE LOAD_CONST None [TensorVariable(), WithExitFunctionVariable(), ConstantVariable(NoneType: None), ConstantVariable(NoneType: None)]
TRACE CALL 2 [TensorVariable(), WithExitFunctionVariable(), ConstantVariable(NoneType: None), ConstantVariable(NoneType: None), ConstantVariable(NoneType: None)]
TRACE POP_TOP None [TensorVariable(), ConstantVariable(NoneType: None)]
TRACE RETURN_VALUE None [TensorVariable()]
Step 1: torchdynamo done tracing torch_dynamo_resume_in_fn_at_7 (RETURN_VALUE)
RETURN_VALUE triggered compile
COMPILING GRAPH due to GraphCompileReason(reason='return_value', user_stack=[<FrameSummary file /tmp/ipykernel_231646/3152636365.py, line 4 in torch_dynamo_resume_in_fn_at_7>], graph_break=False)
TRACED GRAPH
 ===== __compiled_fn_31_99c5cccf_c43c_457f_93b1_ce10af784e02 =====
 /home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/fx/_lazy_graph_module.py class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3][1]cpu"):
        l_x_ = L_x_
        
        # No stacktrace found for following nodes
        _set_grad_enabled = torch._C._set_grad_enabled(False);  _set_grad_enabled = None
        
         # File: /tmp/ipykernel_231646/3152636365.py:8 in torch_dynamo_resume_in_fn_at_7, code: return x + 1
        add: "f32[3][1]cpu" = l_x_ + 1;  l_x_ = None
        
        # No stacktrace found for following nodes
        _set_grad_enabled_1 = torch._C._set_grad_enabled(True);  _set_grad_enabled_1 = None
        return (add,)
        

Step 2: calling compiler function eager
Step 2: done compiler function eager
produce_guards
track_symint L['x'].size()[0] 3 None
track_symint L['x'].stride()[0] 1 None
track_symint L['x'].storage_offset() 0 None
Skipping guard L['x'].size()[0] == 3
Skipping guard L['x'].stride()[0] == 1
Skipping guard L['x'].storage_offset() == 0
GUARDS:

TREE_GUARD_MANAGER:
+- RootGuardManager
| +- LAMBDA_GUARD: torch._functorch.aot_autograd.utils.top_saved_tensors_hooks ids == None  # _dynamo/output_graph.py:633 in init_ambient_guards
| +- DEFAULT_DEVICE: utils_device.CURRENT_DEVICE == None                           # _dynamo/output_graph.py:621 in init_ambient_guards
| +- GLOBAL_STATE: ___check_global_state()
| +- TORCH_FUNCTION_MODE_STACK: ___check_torch_function_mode_stack()
| +- GuardManager: source=L['x'], accessed_by=FrameLocalsGuardAccessor(key='x', framelocals_idx=3)
| | +- TENSOR_MATCH: check_tensor(L['x'], Tensor, DispatchKeySet(CPU, BackendSelect, ADInplaceOrView, AutogradCPU), torch.float32, device=None, requires_grad=False, size=[3], stride=[1])  # return x + 1  # mp/ipykernel_231646/3152636365.py:8 in torch_dynamo_resume_in_fn_at_7
| | +- NO_HASATTR: hasattr(L['x'], '_dynamo_dynamic_indices') == False           # return x + 1  # mp/ipykernel_231646/3152636365.py:8 in torch_dynamo_resume_in_fn_at_7
| +- GuardManager: source=L['___stack0'], accessed_by=FrameLocalsGuardAccessor(key='___stack0', framelocals_idx=0)
| | +- ID_MATCH: ___check_obj_id(L['___stack0'], 94855428429456)               # torch._dynamo.graph_break()  # mp/ipykernel_231646/3152636365.py:7 in torch_dynamo_resume_in_fn_at_7
| +- GuardManager: source=L['___stack1'], accessed_by=FrameLocalsGuardAccessor(key='___stack1', framelocals_idx=1)
| | +- ID_MATCH: ___check_obj_id(L['___stack1'], 94855522137904)               # torch._dynamo.graph_break()  # mp/ipykernel_231646/3152636365.py:7 in torch_dynamo_resume_in_fn_at_7

Guard eval latency = 484.37 us
put_code_state: no cache key, skipping
run_gc_after_compile: running gc

```

```
tensor([2.7023, 1.8611, 1.7168])

```

Graph Break in a Try Block 
----------------------------------------------------------------------------------------

A graph break in a try block cannot be resumed: 

```
@torch.compile
def fn(x):
    try:
        x = x + 1
        torch._dynamo.graph_break()
        return x + 1
    except Exception as e:
        pass
fn(torch.randn(3))

```

```
torchdynamo start compiling fn /tmp/ipykernel_231646/2546874632.py:1, stack (elided 5 frames):
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
  File "/tmp/ipykernel_231646/2546874632.py", line 9, in <module>
    fn(torch.randn(3))

Step 1: torchdynamo start tracing fn /tmp/ipykernel_231646/2546874632.py:1
create_env
TRACE starts_line /tmp/ipykernel_231646/2546874632.py:1 in fn
    @torch.compile
TRACE RESUME 0 []
TRACE starts_line /tmp/ipykernel_231646/2546874632.py:3 in fn
        try:
TRACE NOP None []
TRACE starts_line /tmp/ipykernel_231646/2546874632.py:4 in fn
            x = x + 1
TRACE LOAD_FAST x []
TRACE LOAD_CONST 1 [LazyVariableTracker()]
TRACE BINARY_OP 0 [LazyVariableTracker(), ConstantVariable(int: 1)]
wrap_to_fake L['x'] (3,) StatefulSymbolicContext(dynamic_sizes=[<DimDynamic.STATIC: 2>], dynamic_strides=[<DimDynamic.INFER_STRIDE: 4>], constraint_sizes=[None], constraint_strides=[None], specialize_on=[[]], view_base_context=None, tensor_source=LocalSource(local_name='x', is_input=True, dynamism=None, is_derefed_cell_contents=False), shape_env_to_source_to_symbol_cache={}) <class 'torch.Tensor'>
create_graph_input L_x_ L['x'] FakeTensor(..., size=(3,)) at debug_level 0 before=False
TRACE FX call add from /tmp/ipykernel_231646/2546874632.py:4 in fn
        x = x + 1
            ~~^~~
TRACE STORE_FAST x [TensorVariable()]
TRACE starts_line /tmp/ipykernel_231646/2546874632.py:5 in fn
            torch._dynamo.graph_break()
TRACE LOAD_GLOBAL torch []
TRACE LOAD_ATTR _dynamo [LazyVariableTracker()]
TRACE LOAD_ATTR graph_break [LazyVariableTracker()]
TRACE CALL 0 [NullVariable, LazyVariableTracker()]
empty checkpoint
run_gc_after_compile: running gc
Graph break: skip: from user code at:
  File "/tmp/ipykernel_231646/2546874632.py", line 5, in fn
    torch._dynamo.graph_break()
Traceback (most recent call last):
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 1272, in __call__
    result = self._inner_convert(
             ^^^^^^^^^^^^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 629, in __call__
    return _compile(
           ^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 1111, in _compile
    guarded_code = compile_inner(code, one_graph, hooks, transform)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_utils_internal.py", line 97, in wrapper_function
    return function(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 793, in compile_inner
    return _compile_inner(code, one_graph, hooks, transform)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 832, in _compile_inner
    out_code = transform_code_object(code, transform)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/bytecode_transformation.py", line 1424, in transform_code_object
    transformations(instructions, code_options)
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 267, in _fn
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 753, in transform
    tracer.run()
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 3497, in run
    super().run()
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 1363, in run
    while self.step():
          ^^^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 1267, in step
    self.dispatch_table[inst.opcode](self, inst)
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 834, in wrapper
    return inner_fn(self, inst)
           ^^^^^^^^^^^^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 2910, in CALL
    self._call(inst)
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 2904, in _call
    self.call_function(fn, args, kwargs)
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 1193, in call_function
    self.push(fn.call_function(self, args, kwargs))  # type: ignore[arg-type]
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/variables/lazy.py", line 201, in realize_and_forward
    return getattr(self.realize(), name)(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/variables/functions.py", line 1398, in call_function
    unimplemented_v2(
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/exc.py", line 528, in unimplemented_v2
    raise Unsupported(msg)
torch._dynamo.exc.Unsupported: Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

from user code:
   File "/tmp/ipykernel_231646/2546874632.py", line 5, in fn
    torch._dynamo.graph_break()

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"

WON'T CONVERT fn /tmp/ipykernel_231646/2546874632.py line 1 
due to: 
Traceback (most recent call last):
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 1272, in __call__
    result = self._inner_convert(
             ^^^^^^^^^^^^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 629, in __call__
    return _compile(
           ^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 1111, in _compile
    guarded_code = compile_inner(code, one_graph, hooks, transform)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_utils_internal.py", line 97, in wrapper_function
    return function(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 793, in compile_inner
    return _compile_inner(code, one_graph, hooks, transform)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 832, in _compile_inner
    out_code = transform_code_object(code, transform)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/bytecode_transformation.py", line 1424, in transform_code_object
    transformations(instructions, code_options)
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 267, in _fn
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 753, in transform
    tracer.run()
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 3497, in run
    super().run()
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 1363, in run
    while self.step():
          ^^^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 1267, in step
    self.dispatch_table[inst.opcode](self, inst)
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 834, in wrapper
    return inner_fn(self, inst)
           ^^^^^^^^^^^^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 2910, in CALL
    self._call(inst)
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 2904, in _call
    self.call_function(fn, args, kwargs)
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 1193, in call_function
    self.push(fn.call_function(self, args, kwargs))  # type: ignore[arg-type]
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/variables/lazy.py", line 201, in realize_and_forward
    return getattr(self.realize(), name)(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/variables/functions.py", line 1398, in call_function
    unimplemented_v2(
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/exc.py", line 528, in unimplemented_v2
    raise Unsupported(msg)
torch._dynamo.exc.Unsupported: Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

from user code:
   File "/tmp/ipykernel_231646/2546874632.py", line 5, in fn
    torch._dynamo.graph_break()

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"

Traceback (most recent call last):
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 1272, in __call__
    result = self._inner_convert(
             ^^^^^^^^^^^^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 629, in __call__
    return _compile(
           ^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 1111, in _compile
    guarded_code = compile_inner(code, one_graph, hooks, transform)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_utils_internal.py", line 97, in wrapper_function
    return function(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 793, in compile_inner
    return _compile_inner(code, one_graph, hooks, transform)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 832, in _compile_inner
    out_code = transform_code_object(code, transform)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/bytecode_transformation.py", line 1424, in transform_code_object
    transformations(instructions, code_options)
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 267, in _fn
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 753, in transform
    tracer.run()
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 3497, in run
    super().run()
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 1363, in run
    while self.step():
          ^^^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 1267, in step
    self.dispatch_table[inst.opcode](self, inst)
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 834, in wrapper
    return inner_fn(self, inst)
           ^^^^^^^^^^^^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 2910, in CALL
    self._call(inst)
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 2904, in _call
    self.call_function(fn, args, kwargs)
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 1193, in call_function
    self.push(fn.call_function(self, args, kwargs))  # type: ignore[arg-type]
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/variables/lazy.py", line 201, in realize_and_forward
    return getattr(self.realize(), name)(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/variables/functions.py", line 1398, in call_function
    unimplemented_v2(
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/exc.py", line 528, in unimplemented_v2
    raise Unsupported(msg)
torch._dynamo.exc.Unsupported: Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

from user code:
   File "/tmp/ipykernel_231646/2546874632.py", line 5, in fn
    torch._dynamo.graph_break()

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"

```

```
tensor([1.7327, 2.2532, 1.5965])

```

We can avoid skipping by moving the graph break outside of the try block: 

```
@torch.compile
def fn(x):
    try:
        x = x + 1
    except Exception as e:
        pass
    torch._dynamo.graph_break()
    try:
        return x + 1
    except Exception as e:
        pass
fn(torch.randn(3))

```

```
torchdynamo start compiling fn /tmp/ipykernel_231646/3015389759.py:1, stack (elided 5 frames):
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
  File "/tmp/ipykernel_231646/3015389759.py", line 12, in <module>
    fn(torch.randn(3))

Step 1: torchdynamo start tracing fn /tmp/ipykernel_231646/3015389759.py:1
create_env
TRACE starts_line /tmp/ipykernel_231646/3015389759.py:1 in fn
    @torch.compile
TRACE RESUME 0 []
TRACE starts_line /tmp/ipykernel_231646/3015389759.py:3 in fn
        try:
TRACE NOP None []
TRACE starts_line /tmp/ipykernel_231646/3015389759.py:4 in fn
            x = x + 1
TRACE LOAD_FAST x []
TRACE LOAD_CONST 1 [LazyVariableTracker()]
TRACE BINARY_OP 0 [LazyVariableTracker(), ConstantVariable(int: 1)]
wrap_to_fake L['x'] (3,) StatefulSymbolicContext(dynamic_sizes=[<DimDynamic.STATIC: 2>], dynamic_strides=[<DimDynamic.INFER_STRIDE: 4>], constraint_sizes=[None], constraint_strides=[None], specialize_on=[[]], view_base_context=None, tensor_source=LocalSource(local_name='x', is_input=True, dynamism=None, is_derefed_cell_contents=False), shape_env_to_source_to_symbol_cache={}) <class 'torch.Tensor'>
create_graph_input L_x_ L['x'] FakeTensor(..., size=(3,)) at debug_level 0 before=False
TRACE FX call add from /tmp/ipykernel_231646/3015389759.py:4 in fn
        x = x + 1
            ~~^~~
TRACE STORE_FAST x [TensorVariable()]
TRACE starts_line /tmp/ipykernel_231646/3015389759.py:7 in fn
        torch._dynamo.graph_break()
TRACE LOAD_GLOBAL torch []
TRACE LOAD_ATTR _dynamo [LazyVariableTracker()]
TRACE LOAD_ATTR graph_break [LazyVariableTracker()]
TRACE CALL 0 [NullVariable, LazyVariableTracker()]
Graph break in user code at /tmp/ipykernel_231646/3015389759.py:7
Graph Break Reason: Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

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
  File "/tmp/ipykernel_231646/3015389759.py", line 12, in <module>
    fn(torch.randn(3))
  File "/tmp/ipykernel_231646/3015389759.py", line 7, in fn
    torch._dynamo.graph_break()

Restarting analysis due to _dynamo/symbolic_convert.py:223 in fail_and_restart_analysis
Step 1: torchdynamo start tracing fn /tmp/ipykernel_231646/3015389759.py:1
create_env
TRACE starts_line /tmp/ipykernel_231646/3015389759.py:1 in fn
    @torch.compile
TRACE RESUME 0 []
TRACE starts_line /tmp/ipykernel_231646/3015389759.py:3 in fn
        try:
TRACE NOP None []
TRACE starts_line /tmp/ipykernel_231646/3015389759.py:4 in fn
            x = x + 1
TRACE LOAD_FAST x []
TRACE LOAD_CONST 1 [LazyVariableTracker()]
TRACE BINARY_OP 0 [LazyVariableTracker(), ConstantVariable(int: 1)]
wrap_to_fake L['x'] (3,) StatefulSymbolicContext(dynamic_sizes=[<DimDynamic.STATIC: 2>], dynamic_strides=[<DimDynamic.INFER_STRIDE: 4>], constraint_sizes=[None], constraint_strides=[None], specialize_on=[[]], view_base_context=None, tensor_source=LocalSource(local_name='x', is_input=True, dynamism=None, is_derefed_cell_contents=False), shape_env_to_source_to_symbol_cache={}) <class 'torch.Tensor'>
create_graph_input L_x_ L['x'] FakeTensor(..., size=(3,)) at debug_level 0 before=False
TRACE FX call add from /tmp/ipykernel_231646/3015389759.py:4 in fn
        x = x + 1
            ~~^~~
TRACE STORE_FAST x [TensorVariable()]
TRACE starts_line /tmp/ipykernel_231646/3015389759.py:7 in fn
        torch._dynamo.graph_break()
TRACE LOAD_GLOBAL torch []
TRACE LOAD_ATTR _dynamo [LazyVariableTracker()]
TRACE LOAD_ATTR graph_break [LazyVariableTracker()]
TRACE CALL 0 [NullVariable, LazyVariableTracker()]
COMPILING GRAPH due to GraphCompileReason(reason='Call to `torch._dynamo.graph_break()`n  Explanation: User-inserted graph break. Message: Nonen  Hint: Remove the `torch._dynamo.graph_break()` call.nn  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`n', user_stack=[<FrameSummary file /tmp/ipykernel_231646/3015389759.py, line 7 in fn>], graph_break=True)
TRACED GRAPH
 ===== __compiled_fn_35_754040ab_996a_4694_88bb_01847adecb76 =====
 /home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/fx/_lazy_graph_module.py class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3][1]cpu"):
        l_x_ = L_x_
        
         # File: /tmp/ipykernel_231646/3015389759.py:4 in fn, code: x = x + 1
        x: "f32[3][1]cpu" = l_x_ + 1;  l_x_ = None
        return (x,)
        

Step 2: calling compiler function eager
Step 2: done compiler function eager
produce_guards
track_symint L['x'].size()[0] 3 None
track_symint L['x'].stride()[0] 1 None
track_symint L['x'].storage_offset() 0 None
Skipping guard L['x'].size()[0] == 3
Skipping guard L['x'].stride()[0] == 1
Skipping guard L['x'].storage_offset() == 0
GUARDS:

TREE_GUARD_MANAGER:
+- RootGuardManager
| +- LAMBDA_GUARD: torch._functorch.aot_autograd.utils.top_saved_tensors_hooks ids == None  # _dynamo/output_graph.py:633 in init_ambient_guards
| +- DEFAULT_DEVICE: utils_device.CURRENT_DEVICE == None                           # _dynamo/output_graph.py:621 in init_ambient_guards
| +- GLOBAL_STATE: ___check_global_state()
| +- TORCH_FUNCTION_MODE_STACK: ___check_torch_function_mode_stack()
| +- GuardManager: source=L['x'], accessed_by=FrameLocalsGuardAccessor(key='x', framelocals_idx=0)
| | +- TENSOR_MATCH: check_tensor(L['x'], Tensor, DispatchKeySet(CPU, BackendSelect, ADInplaceOrView, AutogradCPU), torch.float32, device=None, requires_grad=False, size=[3], stride=[1])  # x = x + 1  # mp/ipykernel_231646/3015389759.py:4 in fn
| | +- NO_HASATTR: hasattr(L['x'], '_dynamo_dynamic_indices') == False           # x = x + 1  # mp/ipykernel_231646/3015389759.py:4 in fn
| +- GuardManager: source=G, accessed_by=GlobalsGuardAccessor
| | +- GuardManager: source=G['torch'], accessed_by=DictGetItemGuardAccessor('torch')
| | | +- ID_MATCH: ___check_obj_id(G['torch'], 139712098143824)                  # torch._dynamo.graph_break()  # mp/ipykernel_231646/3015389759.py:7 in fn
| | | +- GuardManager: source=G['torch']._dynamo, accessed_by=GetAttrGuardAccessor(_dynamo)
| | | | +- ID_MATCH: ___check_obj_id(G['torch']._dynamo, 139706212821616)          # torch._dynamo.graph_break()  # mp/ipykernel_231646/3015389759.py:7 in fn
| | | | +- GuardManager: source=G['torch']._dynamo.graph_break, accessed_by=GetAttrGuardAccessor(graph_break)
| | | | | +- ID_MATCH: ___check_obj_id(G['torch']._dynamo.graph_break, 139706044196288)  # torch._dynamo.graph_break()  # mp/ipykernel_231646/3015389759.py:7 in fn

Guard eval latency = 727.15 us
put_code_state: no cache key, skipping
run_gc_after_compile: running gc
torchdynamo start compiling torch_dynamo_resume_in_fn_at_7 /tmp/ipykernel_231646/3015389759.py:7, stack (elided 5 frames):
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
  File "/tmp/ipykernel_231646/3015389759.py", line 12, in <module>
    fn(torch.randn(3))
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/eval_frame.py", line 736, in compile_wrapper
    return fn(*args, **kwargs)

Step 1: torchdynamo start tracing torch_dynamo_resume_in_fn_at_7 /tmp/ipykernel_231646/3015389759.py:7
create_env
TRACE starts_line /tmp/ipykernel_231646/3015389759.py:7 in torch_dynamo_resume_in_fn_at_7
        torch._dynamo.graph_break()
TRACE RESUME 0 []
TRACE LOAD_FAST ___stack0 []
TRACE JUMP_FORWARD 78 [LazyVariableTracker()]
TRACE POP_TOP None [LazyVariableTracker()]
TRACE starts_line /tmp/ipykernel_231646/3015389759.py:8 in torch_dynamo_resume_in_fn_at_7
        try:
TRACE NOP None []
TRACE starts_line /tmp/ipykernel_231646/3015389759.py:9 in torch_dynamo_resume_in_fn_at_7
            return x + 1
TRACE LOAD_FAST x []
TRACE LOAD_CONST 1 [LazyVariableTracker()]
TRACE BINARY_OP 0 [LazyVariableTracker(), ConstantVariable(int: 1)]
wrap_to_fake L['x'] (3,) StatefulSymbolicContext(dynamic_sizes=[<DimDynamic.STATIC: 2>], dynamic_strides=[<DimDynamic.INFER_STRIDE: 4>], constraint_sizes=[None], constraint_strides=[None], specialize_on=[[]], view_base_context=None, tensor_source=LocalSource(local_name='x', is_input=True, dynamism=None, is_derefed_cell_contents=False), shape_env_to_source_to_symbol_cache={}) <class 'torch.Tensor'>
create_graph_input L_x_ L['x'] FakeTensor(..., size=(3,)) at debug_level 0 before=False
TRACE FX call add from /tmp/ipykernel_231646/3015389759.py:9 in torch_dynamo_resume_in_fn_at_7
        return x + 1
               ~~^~~
TRACE RETURN_VALUE None [TensorVariable()]
Step 1: torchdynamo done tracing torch_dynamo_resume_in_fn_at_7 (RETURN_VALUE)
RETURN_VALUE triggered compile
COMPILING GRAPH due to GraphCompileReason(reason='return_value', user_stack=[<FrameSummary file /tmp/ipykernel_231646/3015389759.py, line 9 in torch_dynamo_resume_in_fn_at_7>], graph_break=False)
TRACED GRAPH
 ===== __compiled_fn_38_39512a7f_e27c_4e5f_98b5_7add8b936cc4 =====
 /home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/fx/_lazy_graph_module.py class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3][1]cpu"):
        l_x_ = L_x_
        
         # File: /tmp/ipykernel_231646/3015389759.py:9 in torch_dynamo_resume_in_fn_at_7, code: return x + 1
        add: "f32[3][1]cpu" = l_x_ + 1;  l_x_ = None
        return (add,)
        

Step 2: calling compiler function eager
Step 2: done compiler function eager
produce_guards
track_symint L['x'].size()[0] 3 None
track_symint L['x'].stride()[0] 1 None
track_symint L['x'].storage_offset() 0 None
Skipping guard L['x'].size()[0] == 3
Skipping guard L['x'].stride()[0] == 1
Skipping guard L['x'].storage_offset() == 0
GUARDS:

TREE_GUARD_MANAGER:
+- RootGuardManager
| +- LAMBDA_GUARD: torch._functorch.aot_autograd.utils.top_saved_tensors_hooks ids == None  # _dynamo/output_graph.py:633 in init_ambient_guards
| +- DEFAULT_DEVICE: utils_device.CURRENT_DEVICE == None                           # _dynamo/output_graph.py:621 in init_ambient_guards
| +- GLOBAL_STATE: ___check_global_state()
| +- TORCH_FUNCTION_MODE_STACK: ___check_torch_function_mode_stack()
| +- GuardManager: source=L['x'], accessed_by=FrameLocalsGuardAccessor(key='x', framelocals_idx=1)
| | +- TENSOR_MATCH: check_tensor(L['x'], Tensor, DispatchKeySet(CPU, BackendSelect, ADInplaceOrView, AutogradCPU), torch.float32, device=None, requires_grad=False, size=[3], stride=[1])  # return x + 1  # mp/ipykernel_231646/3015389759.py:9 in torch_dynamo_resume_in_fn_at_7
| | +- NO_HASATTR: hasattr(L['x'], '_dynamo_dynamic_indices') == False           # return x + 1  # mp/ipykernel_231646/3015389759.py:9 in torch_dynamo_resume_in_fn_at_7

Guard eval latency = 133.34 us
put_code_state: no cache key, skipping
run_gc_after_compile: running gc

```

```
tensor([2.3901, 2.9209, 1.2873])

```

Hitting a Recompilation Limit 
----------------------------------------------------------------------------------------------

See [Changing the Cache Size Limit.](programming_model.recompilation.html#programming-model-recompilation-changing-cache-size-limit)

Compiler Errors 
------------------------------------------------------------------

Some compiler errors will result in skipped functions.
Other compiler errors will result in a hard error rather than a skipped function.

Dealing with Skipped Functions 
------------------------------------------------------------------------------------------------

In general, you can resolve a skipped function by fixing the underlying graph break or error that
is causing the function to be skipped. 

If the graph break/error causing the skipped function is difficult to fix,
then consider isolating the graph break/error in its own function so that minimal things are skipped. 

```
def inner1(x):
    return x + 1
def inner2(x):
    return x + 2
@torch.compile
def fn(x):
    x = inner1(x)
    def problematic_code():
        torch._dynamo.skip_frame()
    problematic_code()
    x = inner2(x)
fn(torch.randn(3))

```

```
torchdynamo start compiling fn /tmp/ipykernel_231646/273153676.py:5, stack (elided 5 frames):
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
  File "/tmp/ipykernel_231646/273153676.py", line 12, in <module>
    fn(torch.randn(3))

Step 1: torchdynamo start tracing fn /tmp/ipykernel_231646/273153676.py:5
create_env
TRACE starts_line /tmp/ipykernel_231646/273153676.py:5 in fn
    @torch.compile
TRACE RESUME 0 []
TRACE starts_line /tmp/ipykernel_231646/273153676.py:7 in fn
        x = inner1(x)
TRACE LOAD_GLOBAL inner1 []
TRACE LOAD_FAST x [NullVariable, LazyVariableTracker()]
TRACE CALL 1 [NullVariable, LazyVariableTracker(), LazyVariableTracker()]
TRACE inlined call inner1 from /tmp/ipykernel_231646/273153676.py:7 in fn
    x = inner1(x)
        ~~~~~~^^^
INLINING <code object inner1 at 0x7f0fd4008d30, file "/tmp/ipykernel_231646/273153676.py", line 1>, inlined according trace_rules.lookup inlined by default
wrap_to_fake L['x'] (3,) StatefulSymbolicContext(dynamic_sizes=[<DimDynamic.STATIC: 2>], dynamic_strides=[<DimDynamic.INFER_STRIDE: 4>], constraint_sizes=[None], constraint_strides=[None], specialize_on=[[]], view_base_context=None, tensor_source=LocalSource(local_name='x', is_input=True, dynamism=None, is_derefed_cell_contents=False), shape_env_to_source_to_symbol_cache={}) <class 'torch.Tensor'>
create_graph_input L_x_ L['x'] FakeTensor(..., size=(3,)) at debug_level 0 before=False
TRACE starts_line /tmp/ipykernel_231646/273153676.py:1 in inner1 (inline depth: 1)
    def inner1(x):
TRACE RESUME 0 []
TRACE starts_line /tmp/ipykernel_231646/273153676.py:2 in inner1 (inline depth: 1)
        return x + 1
TRACE LOAD_FAST x []
TRACE LOAD_CONST 1 [TensorVariable()]
TRACE BINARY_OP 0 [TensorVariable(), ConstantVariable(int: 1)]
TRACE FX call add from /tmp/ipykernel_231646/273153676.py:2 in inner1 (inline depth: 1)
    return x + 1
           ~~^~~
TRACE RETURN_VALUE None [TensorVariable()]
DONE INLINING <code object inner1 at 0x7f0fd4008d30, file "/tmp/ipykernel_231646/273153676.py", line 1>
TRACE STORE_FAST x [TensorVariable()]
TRACE starts_line /tmp/ipykernel_231646/273153676.py:8 in fn
        def problematic_code():
TRACE LOAD_CONST <code object problematic_code at 0x7f0fd8557130, file "/tmp/ipykernel_231646/273153676.py", line 8> []
TRACE MAKE_FUNCTION 0 [ConstantVariable(code: <code object problematic_code at 0x7f0fd8557130, file "/tmp/ipykernel_231646/273153676.py", line 8>)]
TRACE STORE_FAST problematic_code [NestedUserFunctionVariable()]
TRACE starts_line /tmp/ipykernel_231646/273153676.py:10 in fn
        problematic_code()
TRACE PUSH_NULL None []
TRACE LOAD_FAST problematic_code [NullVariable]
TRACE CALL 0 [NullVariable, NestedUserFunctionVariable()]
TRACE inlined call problematic_code from /tmp/ipykernel_231646/273153676.py:10 in fn
    problematic_code()
    ~~~~~~~~~~~~~~~~^^
INLINING <code object problematic_code at 0x7f0fd8557130, file "/tmp/ipykernel_231646/273153676.py", line 8>, inlined according trace_rules.lookup inlined by default
TRACE starts_line /tmp/ipykernel_231646/273153676.py:8 in problematic_code (inline depth: 1)
        def problematic_code():
TRACE RESUME 0 []
TRACE starts_line /tmp/ipykernel_231646/273153676.py:9 in problematic_code (inline depth: 1)
            torch._dynamo.skip_frame()
TRACE LOAD_GLOBAL torch []
TRACE LOAD_ATTR _dynamo [LazyVariableTracker()]
TRACE LOAD_ATTR skip_frame [LazyVariableTracker()]
TRACE CALL 0 [NullVariable, LazyVariableTracker()]
SKIPPED INLINING <code object problematic_code at 0x7f0fd8557130, file "/tmp/ipykernel_231646/273153676.py", line 8>: Skip frame due to `torch._dynamo.skip_frame()`. Message: None
Graph break in user code at /tmp/ipykernel_231646/273153676.py:10
Graph Break Reason: SKIPPED INLINING <code object problematic_code at 0x7f0fd8557130, file "/tmp/ipykernel_231646/273153676.py", line 8>: Skip frame due to `torch._dynamo.skip_frame()`. Message: None
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
  File "/tmp/ipykernel_231646/273153676.py", line 12, in <module>
    fn(torch.randn(3))
  File "/tmp/ipykernel_231646/273153676.py", line 10, in fn
    problematic_code()

Restarting analysis due to _dynamo/symbolic_convert.py:223 in fail_and_restart_analysis
Step 1: torchdynamo start tracing fn /tmp/ipykernel_231646/273153676.py:5
create_env
TRACE starts_line /tmp/ipykernel_231646/273153676.py:5 in fn
    @torch.compile
TRACE RESUME 0 []
TRACE starts_line /tmp/ipykernel_231646/273153676.py:7 in fn
        x = inner1(x)
TRACE LOAD_GLOBAL inner1 []
TRACE LOAD_FAST x [NullVariable, LazyVariableTracker()]
TRACE CALL 1 [NullVariable, LazyVariableTracker(), LazyVariableTracker()]
TRACE inlined call inner1 from /tmp/ipykernel_231646/273153676.py:7 in fn
    x = inner1(x)
        ~~~~~~^^^
INLINING <code object inner1 at 0x7f0fd4008d30, file "/tmp/ipykernel_231646/273153676.py", line 1>, inlined according trace_rules.lookup inlined by default
wrap_to_fake L['x'] (3,) StatefulSymbolicContext(dynamic_sizes=[<DimDynamic.STATIC: 2>], dynamic_strides=[<DimDynamic.INFER_STRIDE: 4>], constraint_sizes=[None], constraint_strides=[None], specialize_on=[[]], view_base_context=None, tensor_source=LocalSource(local_name='x', is_input=True, dynamism=None, is_derefed_cell_contents=False), shape_env_to_source_to_symbol_cache={}) <class 'torch.Tensor'>
create_graph_input L_x_ L['x'] FakeTensor(..., size=(3,)) at debug_level 0 before=False
TRACE starts_line /tmp/ipykernel_231646/273153676.py:1 in inner1 (inline depth: 1)
    def inner1(x):
TRACE RESUME 0 []
TRACE starts_line /tmp/ipykernel_231646/273153676.py:2 in inner1 (inline depth: 1)
        return x + 1
TRACE LOAD_FAST x []
TRACE LOAD_CONST 1 [TensorVariable()]
TRACE BINARY_OP 0 [TensorVariable(), ConstantVariable(int: 1)]
TRACE FX call add from /tmp/ipykernel_231646/273153676.py:2 in inner1 (inline depth: 1)
    return x + 1
           ~~^~~
TRACE RETURN_VALUE None [TensorVariable()]
DONE INLINING <code object inner1 at 0x7f0fd4008d30, file "/tmp/ipykernel_231646/273153676.py", line 1>
TRACE STORE_FAST x [TensorVariable()]
TRACE starts_line /tmp/ipykernel_231646/273153676.py:8 in fn
        def problematic_code():
TRACE LOAD_CONST <code object problematic_code at 0x7f0fd8557130, file "/tmp/ipykernel_231646/273153676.py", line 8> []
TRACE MAKE_FUNCTION 0 [ConstantVariable(code: <code object problematic_code at 0x7f0fd8557130, file "/tmp/ipykernel_231646/273153676.py", line 8>)]
TRACE STORE_FAST problematic_code [NestedUserFunctionVariable()]
TRACE starts_line /tmp/ipykernel_231646/273153676.py:10 in fn
        problematic_code()
TRACE PUSH_NULL None []
TRACE LOAD_FAST problematic_code [NullVariable]
TRACE CALL 0 [NullVariable, NestedUserFunctionVariable()]
COMPILING GRAPH due to GraphCompileReason(reason='SKIPPED INLINING <code object problematic_code at 0x7f0fd8557130, file "/tmp/ipykernel_231646/273153676.py", line 8>: Skip frame due to `torch._dynamo.skip_frame()`. Message: None', user_stack=[<FrameSummary file /tmp/ipykernel_231646/273153676.py, line 10 in fn>], graph_break=True)
TRACED GRAPH
 ===== __compiled_fn_41_855d9452_152e_4ff4_98ae_0a2bb6157010 =====
 /home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/fx/_lazy_graph_module.py class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3][1]cpu"):
        l_x_ = L_x_
        
         # File: /tmp/ipykernel_231646/273153676.py:2 in inner1, code: return x + 1
        x: "f32[3][1]cpu" = l_x_ + 1;  l_x_ = None
        return (x,)
        

Step 2: calling compiler function eager
Step 2: done compiler function eager
produce_guards
track_symint L['x'].size()[0] 3 None
track_symint L['x'].stride()[0] 1 None
track_symint L['x'].storage_offset() 0 None
Skipping guard L['x'].size()[0] == 3
Skipping guard L['x'].stride()[0] == 1
Skipping guard L['x'].storage_offset() == 0
GUARDS:

TREE_GUARD_MANAGER:
+- RootGuardManager
| +- LAMBDA_GUARD: torch._functorch.aot_autograd.utils.top_saved_tensors_hooks ids == None  # _dynamo/output_graph.py:633 in init_ambient_guards
| +- DEFAULT_DEVICE: utils_device.CURRENT_DEVICE == None                           # _dynamo/output_graph.py:621 in init_ambient_guards
| +- GLOBAL_STATE: ___check_global_state()
| +- TORCH_FUNCTION_MODE_STACK: ___check_torch_function_mode_stack()
| +- GuardManager: source=L['x'], accessed_by=FrameLocalsGuardAccessor(key='x', framelocals_idx=0)
| | +- TENSOR_MATCH: check_tensor(L['x'], Tensor, DispatchKeySet(CPU, BackendSelect, ADInplaceOrView, AutogradCPU), torch.float32, device=None, requires_grad=False, size=[3], stride=[1])  # x = inner1(x)  # mp/ipykernel_231646/273153676.py:7 in fn
| | +- NO_HASATTR: hasattr(L['x'], '_dynamo_dynamic_indices') == False           # x = inner1(x)  # mp/ipykernel_231646/273153676.py:7 in fn
| +- GuardManager: source=G, accessed_by=GlobalsGuardAccessor
| | +- GuardManager: source=G['inner1'], accessed_by=DictGetItemGuardAccessor('inner1')
| | | +- GuardManager: source=G['inner1'].__code__, accessed_by=GetAttrGuardAccessor(__code__)
| | | | +- ID_MATCH: ___check_obj_id(G['inner1'].__code__, 139705958042928)        # x = inner1(x)  # mp/ipykernel_231646/273153676.py:7 in fn

Guard eval latency = 8.84 us
put_code_state: no cache key, skipping
run_gc_after_compile: running gc
torchdynamo start compiling problematic_code /tmp/ipykernel_231646/273153676.py:8, stack (elided 5 frames):
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
  File "/tmp/ipykernel_231646/273153676.py", line 12, in <module>
    fn(torch.randn(3))
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/eval_frame.py", line 736, in compile_wrapper
    return fn(*args, **kwargs)

Step 1: torchdynamo start tracing problematic_code /tmp/ipykernel_231646/273153676.py:8
create_env
TRACE starts_line /tmp/ipykernel_231646/273153676.py:8 in problematic_code
        def problematic_code():
TRACE RESUME 0 []
TRACE starts_line /tmp/ipykernel_231646/273153676.py:9 in problematic_code
            torch._dynamo.skip_frame()
TRACE LOAD_GLOBAL torch []
TRACE LOAD_ATTR _dynamo [LazyVariableTracker()]
TRACE LOAD_ATTR skip_frame [LazyVariableTracker()]
TRACE CALL 0 [NullVariable, LazyVariableTracker()]
Skipping frame Skip frame due to `torch._dynamo.skip_frame()`. Message: None problematic_code                     /tmp/ipykernel_231646/273153676.py 8
put_code_state: no cache key, skipping
run_gc_after_compile: running gc
torchdynamo start compiling torch_dynamo_resume_in_fn_at_10 /tmp/ipykernel_231646/273153676.py:10, stack (elided 5 frames):
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
  File "/tmp/ipykernel_231646/273153676.py", line 12, in <module>
    fn(torch.randn(3))
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/eval_frame.py", line 736, in compile_wrapper
    return fn(*args, **kwargs)

Step 1: torchdynamo start tracing torch_dynamo_resume_in_fn_at_10 /tmp/ipykernel_231646/273153676.py:10
create_env
TRACE starts_line /tmp/ipykernel_231646/273153676.py:10 in torch_dynamo_resume_in_fn_at_10
        problematic_code()
TRACE RESUME 0 []
TRACE LOAD_FAST ___stack0 []
TRACE JUMP_FORWARD 48 [LazyVariableTracker()]
TRACE POP_TOP None [LazyVariableTracker()]
TRACE starts_line /tmp/ipykernel_231646/273153676.py:11 in torch_dynamo_resume_in_fn_at_10
        x = inner2(x)
TRACE LOAD_GLOBAL inner2 []
TRACE LOAD_FAST x [NullVariable, LazyVariableTracker()]
TRACE CALL 1 [NullVariable, LazyVariableTracker(), LazyVariableTracker()]
TRACE inlined call inner2 from /tmp/ipykernel_231646/273153676.py:11 in torch_dynamo_resume_in_fn_at_10
    x = inner2(x)
        ~~~~~~^^^
INLINING <code object inner2 at 0x7f0fd4009070, file "/tmp/ipykernel_231646/273153676.py", line 3>, inlined according trace_rules.lookup inlined by default
wrap_to_fake L['x'] (3,) StatefulSymbolicContext(dynamic_sizes=[<DimDynamic.STATIC: 2>], dynamic_strides=[<DimDynamic.INFER_STRIDE: 4>], constraint_sizes=[None], constraint_strides=[None], specialize_on=[[]], view_base_context=None, tensor_source=LocalSource(local_name='x', is_input=True, dynamism=None, is_derefed_cell_contents=False), shape_env_to_source_to_symbol_cache={}) <class 'torch.Tensor'>
create_graph_input L_x_ L['x'] FakeTensor(..., size=(3,)) at debug_level 0 before=False
TRACE starts_line /tmp/ipykernel_231646/273153676.py:3 in inner2 (inline depth: 1)
    def inner2(x):
TRACE RESUME 0 []
TRACE starts_line /tmp/ipykernel_231646/273153676.py:4 in inner2 (inline depth: 1)
        return x + 2
TRACE LOAD_FAST x []
TRACE LOAD_CONST 2 [TensorVariable()]
TRACE BINARY_OP 0 [TensorVariable(), ConstantVariable(int: 2)]
TRACE FX call add from /tmp/ipykernel_231646/273153676.py:4 in inner2 (inline depth: 1)
    return x + 2
           ~~^~~
TRACE RETURN_VALUE None [TensorVariable()]
DONE INLINING <code object inner2 at 0x7f0fd4009070, file "/tmp/ipykernel_231646/273153676.py", line 3>
TRACE STORE_FAST x [TensorVariable()]
TRACE RETURN_CONST None []
Step 1: torchdynamo done tracing torch_dynamo_resume_in_fn_at_10 (RETURN_CONST)
RETURN_CONST triggered compile
COMPILING GRAPH due to GraphCompileReason(reason='return_value', user_stack=[<FrameSummary file /tmp/ipykernel_231646/273153676.py, line 11 in torch_dynamo_resume_in_fn_at_10>], graph_break=False)
TRACED GRAPH
 ===== __compiled_fn_45_e4eac564_2e1e_4049_a393_b7ba808b76ea =====
 /home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/fx/_lazy_graph_module.py class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3][1]cpu"):
        l_x_ = L_x_
        
         # File: /tmp/ipykernel_231646/273153676.py:4 in inner2, code: return x + 2
        x: "f32[3][1]cpu" = l_x_ + 2;  l_x_ = x = None
        return ()
        

Step 2: calling compiler function eager
Step 2: done compiler function eager
produce_guards
track_symint L['x'].size()[0] 3 None
track_symint L['x'].stride()[0] 1 None
track_symint L['x'].storage_offset() 0 None
Skipping guard L['x'].size()[0] == 3
Skipping guard L['x'].stride()[0] == 1
Skipping guard L['x'].storage_offset() == 0
GUARDS:

TREE_GUARD_MANAGER:
+- RootGuardManager
| +- LAMBDA_GUARD: torch._functorch.aot_autograd.utils.top_saved_tensors_hooks ids == None  # _dynamo/output_graph.py:633 in init_ambient_guards
| +- DEFAULT_DEVICE: utils_device.CURRENT_DEVICE == None                           # _dynamo/output_graph.py:621 in init_ambient_guards
| +- GLOBAL_STATE: ___check_global_state()
| +- TORCH_FUNCTION_MODE_STACK: ___check_torch_function_mode_stack()
| +- GuardManager: source=L['x'], accessed_by=FrameLocalsGuardAccessor(key='x', framelocals_idx=1)
| | +- TENSOR_MATCH: check_tensor(L['x'], Tensor, DispatchKeySet(CPU, BackendSelect, ADInplaceOrView, AutogradCPU), torch.float32, device=None, requires_grad=False, size=[3], stride=[1])  # x = inner2(x)  # mp/ipykernel_231646/273153676.py:11 in torch_dynamo_resume_in_fn_at_10
| | +- NO_HASATTR: hasattr(L['x'], '_dynamo_dynamic_indices') == False           # x = inner2(x)  # mp/ipykernel_231646/273153676.py:11 in torch_dynamo_resume_in_fn_at_10
| +- GuardManager: source=G, accessed_by=GlobalsGuardAccessor
| | +- GuardManager: source=G['inner2'], accessed_by=DictGetItemGuardAccessor('inner2')
| | | +- GuardManager: source=G['inner2'].__code__, accessed_by=GetAttrGuardAccessor(__code__)
| | | | +- ID_MATCH: ___check_obj_id(G['inner2'].__code__, 139705958043760)        # x = inner2(x)  # mp/ipykernel_231646/273153676.py:11 in torch_dynamo_resume_in_fn_at_10

Guard eval latency = 689.90 us
put_code_state: no cache key, skipping
run_gc_after_compile: running gc

```

