Common Graph Breaks 
==========================================================================

Below are some common graph breaks and some workarounds. 

Incorrect Code 
----------------------------------------------------------------

Your code might contain errors (meaning it doesn’t execute even without `torch.compile`  ). In the example below, there’s a typo in the `torch.sin`  call due to an extra argument. **Always disable `torch.compile` to check if the code runs correctly.** 

```
@torch.compile
def fn(x):
    y = torch.sin(x, x)
    return y

try:
    fn(torch.ones(3, 3))
except Exception as e:
    pass

```

```
Graph break in user code at /tmp/ipykernel_230947/343837593.py:3
Graph Break Reason: TypeError when making fake tensor call
  Explanation: 

  Developer debug context: TypeError <built-in method sin of type object at 0x7f7346f8b4a0>: sin() takes 1 positional argument but 2 were given

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
  File "/tmp/ipykernel_230947/343837593.py", line 7, in <module>
    fn(torch.ones(3, 3))
  File "/tmp/ipykernel_230947/343837593.py", line 3, in fn
    y = torch.sin(x, x)

```

Dynamo makes a best-effort attempt to hint if a graph break is caused by your code.
But it can still sometimes be difficult to tell from the logs if the graph break is caused by an error in your code,
is a more complicated graph break, or is a `torch.compile`  bug. In order to differentiate, we recommend trying to run your code without `torch.compile`  to see if you still get the error reported by the graph break.

Data-dependent operations 
---------------------------------------------------------------------------------------

`torch.compile`  graph breaks on data-dependent operations such as data-dependent control flow (if-statements, loops with tensors) and direct tensor data accesses ( `.item`  , `.data_ptr`  ). 

```
@torch.compile
def fn(x):
    y = x.sum()
    if y > 0:
        return x + y.item()
    return x - y.item()

print(fn(torch.ones(3, 3)))

```

```
tensor([[10., 10., 10.],
        [10., 10., 10.],
        [10., 10., 10.]])

```

```
Graph break in user code at /tmp/ipykernel_230947/3495555842.py:4
Graph Break Reason: Data-dependent branching
  Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow.
  Hint: This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through your code. Consider finding a workaround.
  Hint: Use `torch.cond` to express dynamic control flow.

  Developer debug context: attempted to jump with TensorVariable()

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
  File "/tmp/ipykernel_230947/3495555842.py", line 8, in <module>
    print(fn(torch.ones(3, 3)))
  File "/tmp/ipykernel_230947/3495555842.py", line 4, in fn
    if y > 0:

Graph break from `Tensor.item()`, consider setting:
    torch._dynamo.config.capture_scalar_outputs = True
or:
    env TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1
to include these operations in the captured graph.

Graph break: from user code at:
  File "/tmp/ipykernel_230947/3495555842.py", line 5, in torch_dynamo_resume_in_fn_at_4
    return x + y.item()

Graph break in user code at /tmp/ipykernel_230947/3495555842.py:5
Graph Break Reason: Unsupported Tensor.item() call with capture_scalar_outputs=False
  Explanation: Dynamo does not support tracing `Tensor.item()` with config.capture_scalar_outputs=False.
  Hint: Set `torch._dynamo.config.capture_scalar_outputs = True` or `export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1` to include these operations in the captured graph.

  Developer debug context: call_method TensorVariable() item () {}

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
  File "/tmp/ipykernel_230947/3495555842.py", line 8, in <module>
    print(fn(torch.ones(3, 3)))
  File "/tmp/ipykernel_230947/3495555842.py", line 5, in fn
    return x + y.item()

```

The general workaround for these graph breaks is to avoid doing data-dependent operations. Some specific workarounds are: 

* If your control flow doesn’t actually depend on data values, consider modifying your code to perform control flow on constants.

```
# old
x = torch.randn(3, 3)
@torch.compile
def fn(y):
    if x.sum() > 0:
        return y + x
    else:
        return y - x

print(fn(torch.ones(3, 3)))

```

```
tensor([[ 1.6040, -0.8507,  0.5823],
        [ 0.7290,  1.4197,  2.6653],
        [ 0.9986,  2.3169,  0.5217]])

```

```
Graph break in user code at /tmp/ipykernel_230947/2410325100.py:5
Graph Break Reason: Data-dependent branching
  Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow.
  Hint: This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through your code. Consider finding a workaround.
  Hint: Use `torch.cond` to express dynamic control flow.

  Developer debug context: attempted to jump with TensorVariable()

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
  File "/tmp/ipykernel_230947/2410325100.py", line 10, in <module>
    print(fn(torch.ones(3, 3)))
  File "/tmp/ipykernel_230947/2410325100.py", line 5, in fn
    if x.sum() > 0:

```

```
# new
x = torch.randn(3, 3)
cond = (x.sum() > 0).item()
@torch.compile
def fn(y):
    if cond:
        return y + x
    else:
        return y - x

print(fn(torch.ones(3, 3)))

```

```
tensor([[-1.8993,  2.2489,  2.0591],
        [-0.5374,  1.9662,  1.3910],
        [ 1.5428,  1.0726,  2.2536]])

```

* Use higher-order ops like [Control Flow - Cond](../cond.html#cond)  in place of data-dependent control flow

```
# old
@torch.compile
def fn(x):
    if x.sum() > 0:
        return x + 1
    return x - 1

print(fn(torch.ones(3, 3)))

```

```
tensor([[2., 2., 2.],
        [2., 2., 2.],
        [2., 2., 2.]])

```

```
Graph break in user code at /tmp/ipykernel_230947/520574912.py:4
Graph Break Reason: Data-dependent branching
  Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow.
  Hint: This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through your code. Consider finding a workaround.
  Hint: Use `torch.cond` to express dynamic control flow.

  Developer debug context: attempted to jump with TensorVariable()

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
  File "/tmp/ipykernel_230947/520574912.py", line 8, in <module>
    print(fn(torch.ones(3, 3)))
  File "/tmp/ipykernel_230947/520574912.py", line 4, in fn
    if x.sum() > 0:

```

```
# new
@torch.compile
def fn(x):
    return torch.cond(
        x.sum() > 0,
        lambda x: x + 1,
        lambda x: x - 1,
        (x,),
    )

print(fn(torch.ones(3, 3)))

```

```
tensor([[2., 2., 2.],
        [2., 2., 2.],
        [2., 2., 2.]])

```

* If you have a `.item()`  call, try `torch._dynamo.config.capture_scalar_outputs = True`  or `TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1`  .
* Wrap problematic parts of the function in a custom operator

Printing and logging 
----------------------------------------------------------------------------

Printing/logging/issuing warnings will result in a graph break.
You can try working around this by using `torch._dynamo.config.reorderable_logging_functions`  .
This config is used to reorder logging functions so that they are called at the end of the
traced function, thus avoiding a graph break.
However, the logged contents may differ if, for example, a mutation occurs. 

```
torch._dynamo.config.reorderable_logging_functions.add(print)

@torch.compile
def fn(x):
    x += 1
    print("log!")
    return torch.sin(x)

print(fn(torch.ones(3, 3)))

```

```
log!
tensor([[0.9093, 0.9093, 0.9093],
        [0.9093, 0.9093, 0.9093],
        [0.9093, 0.9093, 0.9093]])

```

