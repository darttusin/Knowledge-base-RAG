Disabling and Suppressing Errors 
====================================================================================================

For some model architectures, there are portions of the model which are particularly difficult to compile -
either there are many graph breaks, or there are crashes.
You may want to explicitly disable these portions of the model which are problematic so that you can apply `torch.compile`  to the parts that work. You can do this by using the `@torch.compiler.disable`  decorator.
When `torch.compile`  attempts to call a disabled function, it breaks the graph and skips tracing the disabled function,
resuming tracing after the call. By default, all recursive calls made from a disabled function are also disabled.
Use the `recursive=False`  option to allow compilation for recursive calls. 

```
def inner1(x):
    torch._dynamo.graph_break()  # not traced
    return x + 1  # not traced

@torch.compiler.disable
def outer1(x):
    x = x + 2  # not traced
    torch._dynamo.graph_break()  # not traced
    return inner1(x)

@torch.compile
def f(x):
    x = outer1(x)
    return x + 4  # traced

print(f(torch.ones(3)))

```

```
tensor([8., 8., 8.])

```

```
Graph break in user code at /tmp/ipykernel_231094/1421264493.py:13
Graph Break Reason: Skip calling `torch.compiler.disable()`d function
  Explanation: Skip calling function `<function outer1 at 0x7fb4ba976200>` since it was wrapped with `torch.compiler.disable` (reason: None)
  Hint: Remove the `torch.compiler.disable` call

  Developer debug context: <function outer1 at 0x7fb4ba976200>

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
  File "/tmp/ipykernel_231094/1421264493.py", line 16, in <module>
    print(f(torch.ones(3)))
  File "/tmp/ipykernel_231094/1421264493.py", line 13, in f
    x = outer1(x)

TRACED GRAPH
 ===== __compiled_fn_4_e76cc6ad_43b9_4810_9828_f0fb4ac5d792 =====
 /home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/fx/_lazy_graph_module.py class GraphModule(torch.nn.Module):
    def forward(self, L_stack0_: "f32[3][1]cpu"):
        l_stack0_ = L_stack0_
        
         # File: /tmp/ipykernel_231094/1421264493.py:14 in torch_dynamo_resume_in_f_at_13, code: return x + 4  # traced
        add: "f32[3][1]cpu" = l_stack0_ + 4;  l_stack0_ = None
        return (add,)
        

```

```
def inner2(x):
    torch._dynamo.graph_break()  # traced
    return x + 1  # traced

@torch.compiler.disable(recursive=False)
def outer2(x):
    x = x + 2  # not traced
    torch._dynamo.graph_break()  # not traced
    return inner2(x)

@torch.compile
def g(x):
    x = outer2(x)
    return x + 4  # traced

print(g(torch.ones(3)))

```

```
tensor([8., 8., 8.])

```

```
Graph break in user code at /tmp/ipykernel_231094/881423632.py:13
Graph Break Reason: Skip inlining `torch.compiler.disable()`d function
  Explanation: Skip inlining function <function outer2 at 0x7fb4ac83c040> since it was wrapped with `torch.compiler.disable` (reason: None)
  Hint: Remove the `torch.compiler.disable` call

  Developer debug context: <function outer2 at 0x7fb4ac83c040>

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
  File "/tmp/ipykernel_231094/881423632.py", line 16, in <module>
    print(g(torch.ones(3)))
  File "/tmp/ipykernel_231094/881423632.py", line 13, in g
    x = outer2(x)

Graph break in user code at /tmp/ipykernel_231094/881423632.py:2
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
  File "/tmp/ipykernel_231094/881423632.py", line 16, in <module>
    print(g(torch.ones(3)))
  File "/tmp/ipykernel_231094/881423632.py", line 13, in g
    x = outer2(x)
  File "/home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/_dynamo/external_utils.py", line 198, in nonrecursive_disable_wrapper
    return fn(*args, **kwargs)
  File "/tmp/ipykernel_231094/881423632.py", line 9, in outer2
    return inner2(x)
  File "/tmp/ipykernel_231094/881423632.py", line 2, in inner2
    torch._dynamo.graph_break()  # traced

TRACED GRAPH
 ===== __compiled_fn_12_6cf7e459_6c5e_41c6_a2ce_e3559518a1f7 =====
 /home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/fx/_lazy_graph_module.py class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3][1]cpu"):
        l_x_ = L_x_
        
         # File: /tmp/ipykernel_231094/881423632.py:3 in torch_dynamo_resume_in_inner2_at_2, code: return x + 1  # traced
        add: "f32[3][1]cpu" = l_x_ + 1;  l_x_ = None
        return (add,)
        

TRACED GRAPH
 ===== __compiled_fn_14_47a23a1f_587a_4357_95a0_620e4b9579c7 =====
 /home/p0xwave/.pyenv/versions/3.12.7/lib/python3.12/site-packages/torch/fx/_lazy_graph_module.py class GraphModule(torch.nn.Module):
    def forward(self, L_stack0_: "f32[3][1]cpu"):
        l_stack0_ = L_stack0_
        
         # File: /tmp/ipykernel_231094/881423632.py:14 in torch_dynamo_resume_in_g_at_13, code: return x + 4  # traced
        add: "f32[3][1]cpu" = l_stack0_ + 4;  l_stack0_ = None
        return (add,)
        

```

For example, one can use `torch.compiler.disable`  to disable `torch.compile`  on sparse architecture in
recommendation models, as the sparse arch is difficult to compile.
Preprocessing and logging functions are other examples of functions that typically cause
a lot of graph breaks and do not get value from being compiled. 

If you are experiencing compiler crashes and you want to continue regardless,
you can set `torch._dynamo.config.suppress_errors = True`  .
When the compiler crashes, we will just skip tracing the function and try again later. **This is not best practice** - it is better to eventually manually add `disable`  annotations as necessary.

